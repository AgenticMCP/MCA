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
# 1-char tokens. snake_case tool names ("list_issues") split into useful terms.
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
# requiring these to appear verbatim in the task would block valid reuse.
# Postgres vocabulary: object types, health-check names, ranking criteria and
# ubiquitous schema names are structural choices, not task entities.
_BENIGN_LITERALS = {
    "public", "table", "view", "sequence", "extension", "all", "true", "false",
    "index", "connection", "vacuum", "replication", "buffer", "constraint",
    "total_time", "mean_time", "resources", "dta", "llm",
    "information_schema", "pg_catalog",
}


# Schema qualification: "public.orders" names the same table as "orders", so
# ground the object part and ignore the default-schema wrapper.
_REF_PREFIX_RE = re.compile(r"^public\.")


def _norm_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip().lower()


# --- SQL entity extraction -------------------------------------------------
# Arguments that carry SQL rather than a plain entity name. Grounding these as
# ONE literal is what made reuse unreachable on this server: a whole statement
# ("CREATE SCHEMA IF NOT EXISTS agenticmcpe_bench_courier") never appears
# verbatim in a natural-language prompt, so every SQL-bearing entry — 95% of
# the corpus — failed the gate. We ground the ENTITIES inside the statement
# instead: the object names it touches and the data values it carries. That
# keeps the safety property (a stored plan may not inject a value the new task
# never asked for) while making reuse reachable.
_SQL_ARGS = {("execute_sql", "sql"), ("explain_query", "sql"),
             ("analyze_query_indexes", "queries")}

_SQL_STRING_RE = re.compile(r"'((?:[^']|'')*)'")
_SQL_QUALIFIED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*)+")
_SQL_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_$]*\b")
# Two digits or more: single digits are overwhelmingly structural (LIMIT 1,
# top 3) while real data values — prices, counts, ids — are longer.
_SQL_NUMBER_RE = re.compile(r"\b\d{2,}\b")

# SQL grammar, built-in types and common functions: structural choices the
# planner makes regardless of wording, so they can never be a stale leftover.
_SQL_KEYWORDS = frozenset("""
select from where insert into values update set delete create table schema view
materialized index sequence type domain extension drop alter add column rename to
constraint primary key foreign references unique check not null default on conflict
do nothing and or in is as order by group having limit offset join left right inner
outer full cross using distinct union all any some between like ilike similar asc
desc nulls first last if cascade restrict generated always stored identity refresh
analyze explain vacuum show begin commit rollback returning true false case when
then else end with recursive exists count sum avg min max coalesce nullif greatest
least cast concat substring trim lower upper length round abs now current_date
current_time current_timestamp current_user current_database session_user version
date_trunc extract to_char string_agg array_agg jsonb_agg row_number over partition
text integer int int2 int4 int8 bigint smallint serial bigserial smallserial
boolean bool numeric decimal real float double precision money char character
varying varchar bpchar date timestamp timestamptz time timetz interval uuid json
jsonb bytea inet cidr array record void trigger language plpgsql sql immutable
stable volatile returns function procedure declare begin_ loop end_ raise notice
exception grant revoke usage select_ temp temporary unlogged including excluding
buffers costs verbose format wal settings summary analyse concurrently only
""".split())

# Catalog namespaces are structural: a task says "list the schemas", never
# "query information_schema.schemata", so requiring those names would block
# every catalog task.
_CATALOG_PREFIXES = ("pg_",)
_CATALOG_NAMESPACES = {"information_schema", "pg_catalog", "pg_toast"}
_CATALOG_OBJECTS = {
    "tables", "columns", "schemata", "views", "routines", "sequences",
    "table_constraints", "check_constraints", "key_column_usage",
    "constraint_column_usage", "referential_constraints", "table_privileges",
    "role_table_grants", "domains", "column_domain_usage", "parameters",
}


def _is_structural_ident(name: str) -> bool:
    low = name.lower()
    return (low in _SQL_KEYWORDS or low in _CATALOG_NAMESPACES
            or low in _CATALOG_OBJECTS or low.startswith(_CATALOG_PREFIXES))


# Relations a statement reads from / writes to, used to tell a pure catalog
# query apart from one that touches the user's own objects.
_SQL_TARGET_RE = re.compile(
    r"\b(?:from|join|into|update)\s+(?:only\s+)?"
    r"([A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*)?)", re.IGNORECASE)


def _catalog_only(sql: str) -> bool:
    """True when every relation the statement touches is a system catalog (or
    it touches none at all, e.g. `SELECT current_database()`).

    Such a statement carries no task-specific OBJECT names — its identifiers
    are catalog view/column names like `schema_name`, which a natural-language
    prompt never spells out. Its quoted values and numbers still matter
    (`WHERE table_schema = 'sales'` really does target one schema), so those
    are still grounded.
    """
    targets = [m.group(1) for m in _SQL_TARGET_RE.finditer(sql)]
    return all(_is_structural_ident(t.split(".")[0]) or _is_structural_ident(t)
               for t in targets)


def sql_entities(sql: str) -> list[str]:
    """The task-specific entities a SQL statement carries: quoted values, the
    object names it touches, and multi-digit numeric literals. SQL grammar,
    built-in types and catalog names are excluded as structural.

    Deliberately over-inclusive on DATA (quoted values and numbers are always
    returned): the gate fails closed, so a value the new task never mentions
    still blocks reuse.
    """
    out: list[str] = []
    for m in _SQL_STRING_RE.finditer(sql):
        val = m.group(1).replace("''", "'").strip()
        if val:
            out.append(val)
    body = _SQL_STRING_RE.sub(" ", sql)
    if _catalog_only(sql):
        # pure catalog read: identifiers are catalog names, not task entities
        out.extend(m.group(0) for m in _SQL_NUMBER_RE.finditer(body))
        return out
    # Identifiers, with quoted values removed so their words aren't re-counted
    # as identifiers.
    consumed: set[str] = set()
    for m in _SQL_QUALIFIED_RE.finditer(body):
        qualified = m.group(0)
        consumed.add(qualified)
        parts = qualified.split(".")
        if any(_is_structural_ident(p) for p in parts[:-1]):
            continue  # catalog- or schema-qualified structural reference
        for p in parts:
            if not _is_structural_ident(p):
                out.append(p)
    stripped = _SQL_QUALIFIED_RE.sub(" ", body)
    for m in _SQL_IDENT_RE.finditer(stripped):
        name = m.group(0)
        if not _is_structural_ident(name):
            out.append(name)
    out.extend(m.group(0) for m in _SQL_NUMBER_RE.finditer(body))
    return out


def catalog_enum_index(
        catalog: list[dict[str, Any]]) -> dict[str, dict[str, frozenset[str]]]:
    """tool -> parameter -> the values that parameter's schema enumerates.

    Enum arguments come from the tool SCHEMA, not from the task:
    `orderBy:"CREATED_AT"`, `detail:"full_patch"` and every `method`
    discriminator are structural choices the planner must make regardless of
    wording, so a task never "grounds" them and requiring it blocks most reuse.

    Indexed per parameter rather than as one bag of strings, because enum values
    do collide with real entity names — "rust" is both a dependency ecosystem
    and the repo in `rust-lang/rust`. Keyed this way, `ecosystem:"rust"` is
    exempt while `repo:"rust"` still has to be grounded in the task.
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
    object, recursing into nested dicts/lists (e.g. push_files.files)."""
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
    stored sequence: the stored plan's concrete inputs (schema names, table
    names, SQL statements, inserted values) must all be present in THIS task,
    so replaying them is correct rather than a leftover from the original task.
    Bindings, very short tokens and benign scaffolding literals (public, table,
    all, ...) are exempt; everything else must match (whitespace-normalized
    substring).

    ``enum_index`` (see :func:`catalog_enum_index`) additionally exempts an
    argument whose OWN parameter schema enumerates its value — those are
    structural choices, never task entities, so they cannot be a stale leftover.
    """
    task_norm = _norm_ws(task)
    for step in entry.get("steps", []):
        arguments = step.get("arguments", {})
        tool = str(step.get("tool", ""))
        enums = (enum_index or {}).get(tool, {})
        for key, value in (arguments.items() if isinstance(arguments, dict)
                           else [(None, arguments)]):
            if isinstance(value, str) and key in enums:
                if value.strip().casefold() in enums[key]:
                    continue  # a value this very parameter enumerates
            # SQL arguments are grounded ENTITY-BY-ENTITY, not as one opaque
            # blob: the statement itself is never quoted in a prompt, but the
            # tables it touches and the values it writes must be (see
            # sql_entities).
            if (tool, key) in _SQL_ARGS:
                for raw in _iter_literal_strings(value):
                    for ent in sql_entities(raw):
                        s = _REF_PREFIX_RE.sub("", ent.strip())
                        if len(s) < 4 or s.lower() in _BENIGN_LITERALS:
                            continue
                        if _norm_ws(s) not in task_norm:
                            return False
                continue
            for lit in _iter_literal_strings(value):
                s = _REF_PREFIX_RE.sub("", lit.strip())
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
           "catalog_enum_index", "default_kb_path", "sql_entities"]
