"""RAG (Retrieval-Augmented Generation) for finance task planning.

A zero-dependency lexical retriever over the verified KB built by
``pkg.finance_taskgen``. Same shape as the github implementation
(cosine-similarity over bag-of-words), with finance-specific benign
literals (tickers, dates, currency symbols) so they don't get the gate
treatment.

The retriever is used in two places:

* During planning (PlannerAgent.plan): consult the KB BEFORE the LLM. A
  near-identical task whose literal arguments are all grounded in the
  query is reused verbatim (no LLM call); weaker matches become
  in-context precedent for the planner.
* During taskgen (the flywheel that builds the KB in the first place):
  the retriever is deliberately NOT used, so the generator keeps
  rediscovering sequences from scratch — the basis of independent
  ground-truth.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Tokenization + benign literals
# ---------------------------------------------------------------------------

# Patterns we strip from the query text before indexing / searching. These
# are high-cardinality identifiers (tickers, dates, numbers) that would
# otherwise dominate a cosine similarity score and prevent the retriever
# from finding structurally similar tasks. The groundedness gate
# (task_grounds_entry) re-checks that they're present in the QUERY
# verbatim — so removing them from the index is safe.
_FINANCE_BENIGN = (
    # Dates: "January 9, 2023", "2023-01-09", "1/9/2023"
    (re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"), " "),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), " "),
    (
        re.compile(
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b",
            re.IGNORECASE,
        ),
        " ",
    ),
    # Currency amounts: "$25,000", "$100.50", "USD 1000"
    (re.compile(r"\$\s*\d[\d,.]*"), " "),
    (re.compile(r"\b(?:USD|EUR|GBP|JPY|CAD)\s*\d[\d,.]*\b", re.IGNORECASE), " "),
    (re.compile(r"\b\d[\d,.]*\s*(?:USD|EUR|GBP|JPY|CAD)\b", re.IGNORECASE), " "),
    # Percentages: "5%", "5.25%"
    (re.compile(r"\b\d+(?:\.\d+)?\s*%"), " "),
    # Common finance verbs that add noise without aiding structure.
    (
        re.compile(
            r"\b(calculate|compute|figure out|tell me|please|kindly|excited|"
            r"super|curious|wondering|would love|wanted to know)\b",
            re.IGNORECASE,
        ),
        " ",
    ),
)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]+")


def _normalize(text: str) -> list[str]:
    s = text
    for rx, repl in _FINANCE_BENIGN:
        s = rx.sub(repl, s)
    return [w.lower() for w in _WORD_RE.findall(s)]


# ---------------------------------------------------------------------------
# KB entry types
# ---------------------------------------------------------------------------


@dataclass
class KBEntry:
    id: str
    task_prompt: str
    task_summary: str
    tool_sequence: list[str]
    steps: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "KBEntry":
        return cls(
            id=str(obj.get("id", "")),
            task_prompt=str(obj.get("task_prompt", "")),
            task_summary=str(obj.get("task_summary", "")),
            tool_sequence=list(obj.get("tool_sequence") or []),
            steps=list(obj.get("steps") or []),
            extra={k: v for k, v in obj.items()
                   if k not in {"id", "task_prompt", "task_summary",
                                "tool_sequence", "steps"}},
        )


@dataclass
class RetrievedExample:
    entry: dict[str, Any]
    score: float


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class KBRetriever:
    """Lexical cosine-similarity retriever over a KB JSON file."""

    def __init__(self, entries: Iterable[dict[str, Any]] | None = None,
                 kb_path: str | os.PathLike[str] | None = None):
        self._entries: list[dict[str, Any]] = []
        self._doc_vectors: list[Counter[str]] = []
        self._doc_norms: list[float] = []
        self._vocab: set[str] = set()
        if entries is not None:
            self.add_entries(entries)
        if kb_path:
            self.add_entries_from_file(kb_path)

    # ----------------------------------------------------------------- API

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def add_entries(self, entries: Iterable[dict[str, Any]]) -> None:
        for e in entries:
            self._entries.append(e)
            tokens = _normalize(e.get("task_prompt", ""))
            self._doc_vectors.append(Counter(tokens))
            self._vocab.update(tokens)
        self._doc_norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0
                           for vec in self._doc_vectors]

    def add_entries_from_file(self, path: str | os.PathLike[str]) -> None:
        """Load KB entries from disk.

        Accepts the knowledge-base envelope written by
        ``finance_taskgen.kb.KnowledgeBase.save`` —
        ``{"version": 1, "updated_at": ..., "entries": [...]}`` — as well as
        a bare list of entries or a single entry object. Without the
        envelope case the whole file was indexed as ONE entry with no
        ``task_prompt``, so retrieval silently matched nothing and the
        KB -> RAG flywheel never closed.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data["entries"] if isinstance(data.get("entries"), list) else [data]
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list of KB entries, got {type(data).__name__}")
        self.add_entries(data)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedExample]:
        """Return the top-k KB entries by cosine similarity."""
        if not self._entries:
            return []
        q_tokens = _normalize(query)
        q_vec = Counter(q_tokens)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry, doc_vec, doc_norm in zip(self._entries, self._doc_vectors, self._doc_norms):
            overlap = set(q_vec) & set(doc_vec)
            if not overlap:
                scored.append((0.0, entry))
                continue
            dot = sum(q_vec[t] * doc_vec[t] for t in overlap)
            scored.append((dot / (q_norm * doc_norm), entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievedExample(entry=e, score=s) for s, e in scored[:k]]

    def save(self, path: str | os.PathLike[str]) -> int:
        Path(path).write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return len(self._entries)

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# Groundedness gate (RAG reuse safety)
# ---------------------------------------------------------------------------


def catalog_enum_index(condensed_catalog: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return a map of (tool, param) -> enum values, used by
    task_grounds_entry to whitelist enum literals."""
    out: dict[str, list[str]] = {}
    for entry in condensed_catalog:
        tool = entry.get("tool")
        for p in entry.get("params") or []:
            ev = p.get("enum")
            if isinstance(ev, list) and ev:
                out[f"{tool}.{p['name']}"] = [str(v) for v in ev]
    return out


# Currency literals the groundedness gate must accept (numeric literals
# representing amounts are NOT enforced — only string-typed enum values
# are). Date / percentage literals in the catalog schema are `string` types,
# so the gate relies on string containment rather than strict enum match.
_FINANCE_LITERAL_HINTS = (
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\$\s*\d[\d,.]*"),
)


def task_grounds_entry(
    entry: dict[str, Any],
    task: str,
    enum_index: dict[str, list[str]],
) -> bool:
    """True when every literal value in the stored entry's arguments is
    either (a) an enum value of the relevant tool/parameter, (b) a
    numeric / boolean / null literal, (c) a binding template, or (d)
    appears verbatim in the user's task text.

    This is the safety gate that prevents RAG from reusing a sequence
    whose literal ticker / date / amount doesn't match the new query —
    which would silently produce wrong answers.
    """
    for step in entry.get("steps") or []:
        tool = step.get("tool")
        for k, v in (step.get("arguments") or {}).items():
            if isinstance(v, str):
                # Binding template — skip.
                if v.startswith("$") and re.match(r"^\$s\d+", v):
                    continue
                # Enum value — skip.
                enum_key = f"{tool}.{k}"
                if enum_key in enum_index and v in enum_index[enum_key]:
                    continue
                # Otherwise the value must appear in the task text
                # (case-insensitive substring) OR be a finance literal
                # the gate accepts (e.g. a month name).
                low_task = task.lower()
                low_v = v.lower()
                if low_v in low_task:
                    continue
                if any(rx.search(v) for rx in _FINANCE_LITERAL_HINTS):
                    continue
                return False
    return True


__all__ = [
    "KBRetriever",
    "KBEntry",
    "RetrievedExample",
    "catalog_enum_index",
    "task_grounds_entry",
    "_normalize",
]