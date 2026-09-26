"""KB retrieval for the planner: a zero-dependency lexical RAG over the verified
task -> tool-sequence knowledge base that pkg/taskgen builds.

The planner consults this BEFORE asking the LLM (see planner.py). Two outcomes:

* A near-identical solved task whose stored arguments are ALL grounded in the new
  task's text -> the stored tool-call sequence is "referable": the planner reuses
  it directly, with no LLM planning round-trip.
* Otherwise the top matches are returned as worked examples to ground the LLM's
  own planning (retrieval-augmented), or nothing when the KB is empty / no match.

Retrieval is intentionally dependency-free (no embeddings API): cosine
similarity over term-frequency vectors of each entry's task prompt + summary +
tool names. The KB is small (tens to low hundreds of entries), so this is fast,
runs offline, and adds no new dependency. If the corpus grows large, swap the
scoring in :meth:`KBRetriever.retrieve` for embeddings — nothing else changes.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import REPO_ROOT


def default_kb_path() -> Path:
    """The KB the taskgen pipeline writes; override with AGENTICMCPE_KB_PATH."""
    env = os.environ.get("AGENTICMCPE_KB_PATH")
    if env:
        return Path(env)
    return REPO_ROOT / "pkg" / "taskgen" / "knowledge_base.json"


# Lightweight tokenizer: lowercase alphanumeric runs, minus a few stopwords and
# 1-char tokens. snake_case tool names ("browser_click") split into useful terms.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from",
    "by", "at", "as", "is", "are", "be", "it", "its", "this", "that", "these",
    "those", "my", "me", "you", "your", "our", "their", "i", "we", "they",
    "can", "could", "would", "should", "will", "do", "does", "please", "then",
    "also", "show", "get", "find", "list", "give",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) > 1 and t not in _STOPWORDS]


# --- groundedness: may the new task safely reuse an entry's literal arguments? -
_WS_RE = re.compile(r"\s+")
# Common scaffolding literals that recur without identifying a specific entity;
# requiring these to appear verbatim in the task would block valid reuse. For a
# browser corpus these are targeting/waiting scaffolding, not task entities.
_BENIGN_LITERALS = {
    "true", "false", "list", "new", "close", "select", "left", "right",
    "middle", "textbox", "checkbox", "radio", "combobox", "slider",
}

# find: targeting prefix — the needle itself must be grounded in the task or in
# page content, but the prefix is scaffolding.
_FIND_PREFIX_RE = re.compile(r"^find:")


def _norm_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip().lower()


def catalog_enum_index(
        catalog: list[dict[str, Any]]) -> dict[str, dict[str, frozenset[str]]]:
    """tool -> parameter -> the values that parameter's schema enumerates.

    Enum arguments come from the tool SCHEMA, not from the task: `action:"list"`
    for browser_tabs and every `button`/`type` discriminator are structural
    choices the planner must make regardless of wording, so a task never
    "grounds" them and requiring it blocks most reuse.

    Indexed per parameter rather than as one bag of strings, because enum values
    can collide with real entity names. Keyed this way, `button:"left"` is
    exempt while a search text "left" still has to be grounded in the task.
    """
    index: dict[str, dict[str, frozenset[str]]] = {}
    for tool in catalog:
        params = {
            str(p["name"]): frozenset(str(v).casefold() for v in p["enum"])
            for p in tool.get("params") or [] if p.get("enum")
        }
        if params:
            index[str(tool["tool"])] = params
    return index


def _iter_literal_strings(value: Any) -> Iterator[str]:
    """Yield every literal (non-$binding) string anywhere inside an arguments
    object, recursing into nested dicts/lists (e.g. browser_fill_form.fields)."""
    if isinstance(value, str):
        if not value.startswith("$"):
            yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_literal_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _iter_literal_strings(v)


def task_grounds_entry(
        entry: dict[str, Any], task: str,
        enum_index: dict[str, dict[str, frozenset[str]]] | None = None) -> bool:
    """True iff every entity-like literal argument in the entry's stored steps
    actually appears in the new task text. This is the safety gate for reusing a
    stored sequence: the stored plan's concrete inputs (URLs, search terms,
    texts to type, element captions) must all be present in THIS task, so
    replaying them is correct rather than a leftover from the original task.
    Bindings, very short tokens, refs (eNN) and benign scaffolding literals are
    exempt; everything else must match (whitespace-normalized substring).

    ``enum_index`` (see :func:`catalog_enum_index`) additionally exempts an
    argument whose OWN parameter schema enumerates its value — those are
    structural choices, never task entities, so they cannot be a stale leftover.
    """
    task_norm = _norm_ws(task)
    for step in entry.get("steps", []):
        arguments = step.get("arguments", {})
        enums = (enum_index or {}).get(str(step.get("tool", "")), {})
        for key, value in (arguments.items() if isinstance(arguments, dict)
                           else [(None, arguments)]):
            if isinstance(value, str) and key in enums:
                if value.strip().casefold() in enums[key]:
                    continue  # a value this very parameter enumerates
            for lit in _iter_literal_strings(value):
                s = _FIND_PREFIX_RE.sub("", lit.strip())
                # Snapshot refs are runtime handles, never task entities. A
                # stored plan that hardcodes one is session-bound, so a ref
                # blocks reuse (it cannot be grounded in any task text).
                if re.fullmatch(r"e\d+", s):
                    return False
                if len(s) < 4 or s.lower() in _BENIGN_LITERALS:
                    continue
                if _norm_ws(s) not in task_norm:
                    return False
    return True


@dataclass
class RetrievedExample:
    entry: dict[str, Any]
    score: float


class KBRetriever:
    """Lexical retriever over a taskgen knowledge_base.json. Safe when the file
    is missing or empty (``is_empty`` is True, ``retrieve`` returns [])."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path else default_kb_path()
        self.entries: list[dict[str, Any]] = []
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = list(raw.get("entries", []))
        # Precompute each entry's term-frequency vector + L2 norm once.
        self._vecs: list[tuple[Counter, float]] = []
        for e in self.entries:
            vec = Counter(_tokens(self._doc(e)))
            norm = math.sqrt(sum(c * c for c in vec.values())) or 1.0
            self._vecs.append((vec, norm))

    @staticmethod
    def _doc(entry: dict[str, Any]) -> str:
        parts = [str(entry.get("task_prompt", "")), str(entry.get("task_summary", ""))]
        parts.extend(str(t) for t in entry.get("tool_sequence", []))
        return " ".join(parts)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve(self, task: str, k: int = 3) -> list[RetrievedExample]:
        """Top-k KB entries by cosine similarity to ``task`` (descending)."""
        q = Counter(_tokens(task))
        if not q:
            return []
        qn = math.sqrt(sum(c * c for c in q.values())) or 1.0
        scored: list[RetrievedExample] = []
        for entry, (vec, norm) in zip(self.entries, self._vecs):
            dot = sum(c * vec.get(t, 0) for t, c in q.items())
            if dot <= 0:
                continue
            scored.append(RetrievedExample(entry=entry, score=dot / (qn * norm)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


__all__ = ["KBRetriever", "RetrievedExample", "task_grounds_entry",
           "catalog_enum_index", "default_kb_path"]
