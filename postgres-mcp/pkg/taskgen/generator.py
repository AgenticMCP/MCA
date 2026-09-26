"""Task-generation agent: realistic PostgreSQL tasks grounded in the live catalog.

The generator does NOT invent tasks at random. It is forced through a
human-thinking scaffold (persona/motive -> concrete goal -> mental walkthrough
-> natural prompt) seeded with a scenario archetype, so the resulting prompt
reads like something a real user would ask, and its tool-call sequence is
recoverable by an independent planner. The generator's `expected_tools` is a
hypothesis used for coverage/diversity metadata only — ground truth is always
established downstream by execution + verification (see pipeline.py).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

from pkg.agenticmcpe.config import LLMClient, extract_json

# Generation wants VARIETY, so it runs the LLM hotter than the planner/verifier
# (which need temperature 0 for determinism). At temperature 0 the generator
# collapses to near-identical tasks for the same archetype+difficulty; a higher
# temperature, combined with the sampled persona/phrasing below, is what makes
# successive tasks read like different real people. Used by the CLI when it
# builds the generator's LLM client.
GENERATION_TEMPERATURE = 0.85

# Personas and phrasing styles are sampled per task and injected into the prompt
# so generated tasks vary in voice, register and length instead of converging on
# one "assistant task" template. They steer wording only — never the tool choice.
PERSONAS: list[str] = [
    "a backend developer sketching a schema for a new feature",
    "a database administrator auditing an unfamiliar database",
    "a data analyst who wants a quick, concrete answer",
    "a site-reliability engineer checking database health before a deploy",
    "a new team member finding their way around the company database",
    "a performance-minded engineer hunting a slow query",
    "a QA engineer seeding reproducible test data",
    "a student practising SQL on a scratch database",
    "a data engineer staging a small migration",
    "a product engineer wiring up reporting for a prototype",
]

PHRASING_STYLES: list[str] = [
    "terse and imperative — one direct sentence, no pleasantries",
    "conversational and polite — a couple of natural sentences, as if chatting",
    "context-first — a clause of background or motivation, then the concrete ask",
    "detailed and precise — names the exact schema, tables, columns and values up front",
    "slightly informal — natural wording, maybe a contraction or aside, still clear",
]

# Scenario archetypes: the human intents tasks are sampled from. `write` marks
# archetypes that require the write channel (excluded in readonly mode).
ARCHETYPES: list[dict[str, Any]] = [
    {"name": "schema-exploration", "write": False,
     "intent": "Get oriented in an unfamiliar database: list the schemas, "
               "see what tables/views/sequences/extensions live where, and "
               "drill into one object's structure (columns, constraints, "
               "indexes). Targets only what every PostgreSQL database has "
               "(system schemas and catalogs) — never assumes user tables."},
    {"name": "catalog-survey", "write": False,
     "intent": "Answer a concrete question from the system catalogs with "
               "read-only SQL: installed extensions and their versions, "
               "current database/user/version, configured settings (SHOW or "
               "pg_settings), table counts per schema from "
               "information_schema — data that exists in every database."},
    {"name": "query-plan-review", "write": False,
     "intent": "Understand how PostgreSQL would run a query: explain a "
               "concrete SELECT against a system catalog (pg_class, "
               "pg_settings, information_schema.tables) and read the plan, "
               "possibly comparing a filtered vs unfiltered variant."},
    {"name": "health-audit", "write": False,
     "intent": "Check the database's operational health before or after a "
               "change: run the built-in health analysis for one or more "
               "components (index, connection, vacuum, sequence, replication, "
               "buffer, constraint) and read a couple of related facts from "
               "the catalogs with read-only SQL."},
    {"name": "schema-bootstrap", "write": True,
     "intent": "Stand up a small working area from nothing: create a "
               "dedicated schema, create one or two tables with typed "
               "columns and a primary key, seed a handful of literal rows, "
               "and read them back to confirm."},
    {"name": "data-entry", "write": True,
     "intent": "Record concrete data: create a schema and table, insert "
               "several literal rows the task spells out exactly, then run a "
               "SELECT that proves the rows landed (count or ordered "
               "listing)."},
    {"name": "data-correction", "write": True,
     "intent": "Fix data that was entered wrong: create a schema and table, "
               "seed rows including one with a known-bad value, UPDATE that "
               "row to the corrected literal value with an absolute "
               "assignment, and SELECT to show the fix."},
    {"name": "table-evolution", "write": True,
     "intent": "Evolve a table already in use: create a schema and table, "
               "seed rows, ALTER TABLE to add a typed column, backfill it "
               "with an UPDATE, and read the final shape back (columns and "
               "rows)."},
    {"name": "view-reporting", "write": True,
     "intent": "Wire up a small report: create a schema and table, seed "
               "rows, create a view that filters or aggregates them "
               "(computed INSIDE one SQL statement), and query the view to "
               "get the report."},
    {"name": "index-tuning", "write": True,
     "intent": "Make a lookup fast: create a schema and table, seed rows, "
               "look at the query plan for a concrete WHERE, create an index "
               "on the filtered column, and confirm the INDEX EXISTS by "
               "inspecting the table's structure. The outcome to verify is "
               "the index's existence and definition — never that a plan "
               "chooses it (tiny seeded tables are legitimately seq-scanned "
               "no matter what indexes exist)."},
    {"name": "data-migration", "write": True,
     "intent": "Move rows where they belong: create a schema with two "
               "tables, seed the first, copy the qualifying rows into the "
               "second with INSERT ... SELECT, optionally delete them from "
               "the source, and verify both row counts."},
    {"name": "snapshot-report", "write": True,
     "intent": "Persist an answer taken from the live catalogs: create a "
               "schema, then CREATE TABLE ... AS SELECT a small, bounded "
               "catalog query (e.g. the installed extensions, or per-schema "
               "table counts), and read the snapshot back."},
    {"name": "constraint-guard", "write": True,
     "intent": "Protect data with constraints: create a schema and a table "
               "whose columns carry a primary key plus a CHECK or UNIQUE or "
               "NOT NULL constraint, insert rows that satisfy them, and "
               "inspect the object to confirm the constraints exist."},
    {"name": "cleanup-teardown", "write": True,
     "intent": "Leave the workspace tidy: create a schema with a couple of "
               "objects (table, view or sequence), verify they exist by "
               "listing, then drop what the task says is obsolete and list "
               "again to confirm only the survivors remain."},
    {"name": "sequence-numbering", "write": True,
     "intent": "Hand out stable identifiers: create a schema and a table "
               "whose id column defaults to a sequence (or identity), insert "
               "rows WITHOUT ids, and select them back to show the assigned "
               "numbering."},
    {"name": "performance-review", "write": False,
     "intent": "See where the database is spending its time: report the "
               "slowest or most resource-intensive recorded queries (by "
               "total time, mean time or resources), and alongside that do "
               "ONE simple companion read — a single-view catalog fact (a "
               "setting from pg_settings, a count from pg_stat_statements), "
               "a health component, the plan of a CONCRETE simple query the "
               "task spells out in full, or what the workload-wide index "
               "advisor recommends for the recorded workload. Recorded "
               "query texts are normalized with $n placeholders and can "
               "never be re-run or re-explained. Requires the "
               "pg_stat_statements extension, which this environment has."},
    {"name": "settings-audit", "write": False,
     "intent": "Check how the server is configured before a change: read a "
               "handful of NAMED settings (e.g. work_mem, shared_buffers, "
               "max_connections) via SHOW or single-view pg_settings "
               "SELECTs, note the current database/user/version, and finish "
               "with one health component that relates to the settings "
               "being checked (connection, buffer, vacuum)."},
    {"name": "extension-inventory", "write": False,
     "intent": "Audit what is installed: list the installed extensions with "
               "their versions, drill into ONE named extension's details, "
               "and read from pg_available_extensions (single-view SELECT, "
               "columns name/default_version/comment) whether a newer "
               "default version exists for it."},
    {"name": "explain-variants", "write": False,
     "intent": "Compare how PostgreSQL treats variants of the same catalog "
               "query: explain a concrete single-view catalog SELECT plain, "
               "then the same query with analyze=true (or a filtered vs "
               "unfiltered variant), and read the actual row count with a "
               "separate aggregate SELECT so the estimates can be compared "
               "with reality."},
    {"name": "json-documents", "write": True,
     "intent": "Store a few semi-structured records: create a schema and a "
               "table with an id and a jsonb column, insert 3-5 literal "
               "JSON documents the task spells out exactly, then query them "
               "back with jsonb operators (->> projections, a WHERE on one "
               "field, a count) computed inside single statements."},
    {"name": "dedup-cleanup", "write": True,
     "intent": "Remove duplicate rows: create a schema and table WITHOUT a "
               "unique constraint, seed rows that include exact duplicates "
               "on a business key, count them, delete the duplicates in ONE "
               "statement (e.g. DELETE USING keeping the smallest id), and "
               "show the deduplicated survivors ordered."},
    {"name": "bulk-index-variants", "write": True,
     "intent": "Build the right indexes for a bigger table: create a schema "
               "and table, bulk-seed a few thousand rows with one INSERT "
               "... SELECT generate_series statement, run ANALYZE, then "
               "create TWO different indexes (e.g. a plain btree on one "
               "column and a partial or expression index), and confirm both "
               "exist by inspecting the table's structure. Verify index "
               "EXISTENCE, never plan choice."},
    # --- lifecycle archetypes: designed for EXPERT-length chains ------------
    {"name": "data-lifecycle", "write": True,
     "intent": "Run a small dataset through its whole life in one sitting: "
               "create the schema, create a main table plus a lookup table, "
               "seed both with literal rows, add an index on the column the "
               "reports filter by, refresh statistics, create a view that "
               "joins/aggregates them, read the report back, correct one row "
               "with an absolute UPDATE, re-read to show the correction, "
               "inspect the table's final structure, and finish with a health "
               "check. Every phase must be observable afterwards."},
    {"name": "schema-refactor", "write": True,
     "intent": "Normalise a badly-shaped table without losing data: create "
               "the schema and one wide table, seed it with literal rows, "
               "create the two properly-shaped tables (parent + child with a "
               "REFERENCES foreign key), migrate the rows across with INSERT "
               "... SELECT, verify both row counts match the source, inspect "
               "the child's constraints to confirm the foreign key, drop the "
               "obsolete wide table, list the schema to show only the new "
               "shape remains, and close with a health check."},
    {"name": "audit-then-remediate", "write": True,
     "intent": "Turn an audit into a fix and prove the fix: survey the "
               "catalog (schemas, then objects in one of them), create a "
               "schema and a findings table, record a few literal findings, "
               "build the object the audit says is missing (an index or a "
               "CHECK constraint on a seeded table), re-inspect that object "
               "to show it now exists, mark the finding resolved with an "
               "UPDATE, read the findings back, and finish with a health "
               "check."},
    {"name": "size-inventory", "write": False,
     "intent": "Find out what is taking up space: list the objects in a "
               "system schema, then use single-view SELECTs over pg_class / "
               "pg_total_relation_size(...) to report the largest few "
               "relations (pg_size_pretty for readability), and relate that "
               "to a buffer or index health check."},
    {"name": "privilege-review", "write": False,
     "intent": "Review who can do what: read the roles from pg_roles and the "
               "table grants for one system schema from "
               "information_schema.role_table_grants (single-view SELECTs, "
               "bounded with LIMIT), and note the current user's own "
               "session role."},
    {"name": "vacuum-stats-review", "write": False,
     "intent": "Judge whether autovacuum is keeping up: read per-table "
               "vacuum/analyze statistics from pg_stat_user_tables or "
               "pg_stat_all_tables (n_live_tup, n_dead_tup, last_autovacuum "
               "— cast timestamps ::text), pair that with the vacuum health "
               "check, and read one relevant autovacuum setting."},
    {"name": "temporal-series", "write": True,
     "intent": "Record dated events and query a window: create a schema and "
               "a table with a date or timestamp column, insert a handful of "
               "literal rows on explicit dates the task spells out, then "
               "select a date RANGE back (cast temporal columns ::text in "
               "the output) and aggregate a count per period in one "
               "statement."},
    {"name": "text-search", "write": True,
     "intent": "Find records by text: create a schema and table with a text "
               "column, seed literal rows, then query with LIKE/ILIKE (or a "
               "case-insensitive pattern) for the matching subset, count the "
               "matches, and optionally add an index supporting the lookup "
               "and confirm the index exists."},
    {"name": "enum-domain-types", "write": True,
     "intent": "Constrain values with a custom type: create a schema, create "
               "a SCHEMA-QUALIFIED enum type (CREATE TYPE "
               "myschema.status AS ENUM (...)) or a domain with a CHECK, "
               "create a table using it, insert rows carrying valid values, "
               "and inspect the table's structure to confirm the column's "
               "type."},
    {"name": "foreign-key-graph", "write": True,
     "intent": "Model a parent/child relationship: create a schema with two "
               "tables where the child carries a REFERENCES foreign key to "
               "the parent, seed both with literal rows, run a JOIN that "
               "reports children per parent, and inspect the child table to "
               "confirm the foreign-key constraint exists."},
    {"name": "upsert-merge", "write": True,
     "intent": "Reconcile incoming data with what is stored: create a schema "
               "and a table with a unique business key, seed initial rows, "
               "then apply an INSERT ... ON CONFLICT (key) DO UPDATE that "
               "both updates an existing row and adds a new one, and read "
               "the reconciled table back ordered."},
    {"name": "matview-refresh", "write": True,
     "intent": "Precompute a summary: create a schema and base table, seed "
               "rows, create a MATERIALIZED VIEW aggregating them, insert "
               "another row, REFRESH the materialized view, and query it to "
               "show the refreshed totals."},
    {"name": "computed-columns", "write": True,
     "intent": "Let the database derive values: create a schema and a table "
               "with either a GENERATED ALWAYS AS (...) STORED column or an "
               "array (text[]/integer[]) column, insert literal rows, and "
               "query the derived/array values back (array operators or the "
               "generated column) to show what the database computed."},
    {"name": "index-advisor", "write": True,
     "intent": "Make a slow lookup fast with the tuning advisor: create ONE "
               "working table in the PUBLIC schema whose name starts with "
               "agenticmcpe_bench_ (advisor tasks use a prefixed table in "
               "public instead of a dedicated schema, because the advisor "
               "only sees tables on the search path), bulk-seed it with a "
               "few thousand generated rows in one INSERT ... SELECT "
               "generate_series statement, run ANALYZE, ask the advisor to "
               "analyze one or two concrete queries against that table "
               "(referenced UNQUALIFIED), then create the recommended index "
               "and confirm it exists or that the plan now uses it."},
]

# Tools that depend on optional PostgreSQL extensions the target database may
# not have — steps using them fail live unless the extension is installed.
# Declare installed extensions with AGENTICMCPE_PG_EXTENSIONS (comma-separated,
# e.g. "pg_stat_statements,hypopg") to unlock the corresponding tools.
_EXTENSION_TOOLS: dict[str, set[str]] = {
    "get_top_queries": {"pg_stat_statements"},
    "analyze_workload_indexes": {"pg_stat_statements", "hypopg"},
    "analyze_query_indexes": {"hypopg"},
}


def unsupported_tools() -> set[str]:
    installed = {e.strip().lower()
                 for e in os.environ.get("AGENTICMCPE_PG_EXTENSIONS", "").split(",")
                 if e.strip()}
    return {tool for tool, needs in _EXTENSION_TOOLS.items()
            if not needs <= installed}


# Kept for import parity with the GitHub original (computed at import time from
# the environment; call unsupported_tools() for the live value).
UNSUPPORTED_TOOLS = unsupported_tools()

# difficulty -> (min_steps, max_steps) for the mental walkthrough
DIFFICULTY_STEPS: dict[str, tuple[int, int]] = {
    "easy": (2, 3),
    "medium": (4, 6),
    "hard": (7, 10),
    # Multi-phase work: build -> populate -> tune -> transform -> verify ->
    # tidy. Only the lifecycle archetypes below sustain this length without
    # padding; on the ordinary archetypes it produces busywork, so pair
    # --difficulty expert with --category.
    "expert": (11, 18),
}

_GENERATOR_SYSTEM = """\
You are a TASK GENERATION agent. You invent realistic PostgreSQL tasks that a
human would ask an AI assistant to perform against a database. Each task is
later given to an INDEPENDENT planner that must rediscover the right tool-call
sequence, execute it against a real PostgreSQL database, and verify the
outcome; tasks that survive become benchmark ground truth.

Think like a real database user, in this exact order (the required
human-thinking structure):
1. PERSONA & MOTIVE — who is asking and why (a developer sketching a schema, a
   DBA auditing health, an analyst needing a number, ...).
2. CONCRETE GOAL — the specific outcome they want, with concrete names, types,
   values and limits (which schema, which table, which columns, which rows).
3. MENTAL WALKTHROUGH — how the human would do it by hand in psql, as ordered
   steps ("first I create X, that gives me Y, with Y I then ...").
   Each mental step must correspond to exactly ONE tool from the catalog, and
   each step's inputs must be obtainable from the task text or an earlier step.
4. TASK PROMPT — the request the persona would actually type, in natural
   language. It must be self-contained and unambiguous enough that the
   walkthrough can be reconstructed from the prompt alone.

Hard rules:
- expected_tools lists the catalog tool name of each mental step, in order.
  Use ONLY tool names from the provided catalog.
- The task prompt must NEVER contain tool names or the words "MCP"/"API"; it
  describes WHAT to achieve, not WHICH function to call. Talking about SQL
  concepts (tables, schemas, rows, indexes, queries) is natural and fine.
- The task must be fully self-contained: it may not assume ANY pre-existing
  user table, view or data. If it reads or changes something, it must create
  that thing first within the task. The ONLY pre-existing state it may rely
  on is what every PostgreSQL database has: the system catalogs
  (information_schema, pg_catalog), SHOW settings, the current
  database/user/version, and the built-in health checks.
- Every write goes into a DEDICATED schema the task itself creates, whose
  name starts with "agenticmcpe_bench_" (lowercase snake_case, e.g.
  agenticmcpe_bench_courier_dispatch). The task prompt must state that schema
  name explicitly. Never write into public or any schema the task did not
  create. Invent a FRESH, distinctive suffix for every task (two words, or a
  word plus digits) and never reuse a schema name that appears in the
  EXISTING TASKS list — a leftover schema with the same name but different
  table shapes makes CREATE TABLE IF NOT EXISTS silently adopt the wrong
  table and the task fail.
- Give every table/column lowercase snake_case names, and spell out literal
  row values EXACTLY (they are verified byte-for-byte after execution).
  Never ask for values that change run to run (now(), random(), the current
  date) as stored content.
- Any computation (counts, sums, filters, ordering) must be expressible
  INSIDE a single SQL statement. Steps can pass values along, but there is no
  calculator between steps — never ask for cross-step arithmetic that SQL
  itself cannot do in one statement.
- Keep result sizes bounded: seed at most 10 rows per table; ask for "up to
  N" items with N <= 10. EXCEPTION: index-advisor and bulk-index tasks may
  bulk-seed a few thousand rows, but ONLY via a single INSERT ... SELECT ...
  FROM generate_series(...) statement (never a literal row list), and their
  SELECTs must stay aggregate/limited. A bulk-seeded task may pin the exact
  TOTAL row count, but must NEVER promise exact per-category/bucket counts
  inside generated data (planners get modulo/CASE bucket arithmetic wrong,
  and the run fails verification) — ask for totals and simple observable
  aggregates instead.
- Index-advisor tasks follow a special convention (the advisor resolves
  tables via the search path): the working table lives in the PUBLIC schema
  with an agenticmcpe_bench_-prefixed TABLE name stated in the prompt, the
  queries to analyze reference it UNQUALIFIED, and an ANALYZE happens after
  seeding. All other write tasks keep the dedicated-schema rule above.
- The outcome must be verifiable afterwards by re-querying the database
  (objects exist, row counts, exact values — not vague impressions).
- READONLY mode: every step is a read-only inspection; the task must not ask
  to create, modify or delete anything, and any SQL it implies must be a
  SELECT/SHOW/EXPLAIN against the system catalogs.
- WRITE mode: writes happen ONLY inside the task's own agenticmcpe_bench_*
  schema, as above.
- Do not use tools that require optional extensions unless the catalog notes
  say they are available (the UNSUPPORTED list below is authoritative).
- Do not duplicate any of the EXISTING TASKS provided.

Naturalness (make it read like a real person, not a generated spec):
- Write the TASK PROMPT in the assigned persona's voice and phrasing style. Vary
  sentence length and register from task to task; never settle into one template.
- A real user says WHAT they want and maybe why — not an enumerated procedure.
  Never number the steps in the prompt, and never hint at how many calls or which
  tools it takes. The MENTAL WALKTHROUGH carries that structure, not the prompt.
- The prompt may include a short clause of natural context or motivation, but it
  must stay self-contained and unambiguous: the walkthrough must still be
  reconstructable from the prompt alone.

Output a SINGLE JSON object, no prose. ALL FOUR keys are REQUIRED, in this
exact order (expected_tools BEFORE task_prompt):
{
  "category": "<archetype name>",
  "thinking": ["<step 1 of the human walkthrough>", "..."],
  "expected_tools": ["<tool>", "..."],
  "task_prompt": "<the natural-language request>"
}
"""


class TaskGenError(RuntimeError):
    pass


@dataclass
class TaskSpec:
    category: str
    difficulty: str
    mode: str  # "readonly" | "write"
    thinking: list[str]
    task_prompt: str
    expected_tools: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "difficulty": self.difficulty,
            "mode": self.mode,
            "thinking": self.thinking,
            "task_prompt": self.task_prompt,
            "expected_tools": self.expected_tools,
            "warnings": self.warnings,
        }


class TaskGenerator:
    def __init__(self, llm: LLMClient, condensed_catalog: list[dict[str, Any]]):
        self.llm = llm
        self.catalog = condensed_catalog
        self._by_name = {c["tool"]: c for c in condensed_catalog}
        self._unsupported = unsupported_tools()
        # token-cheap view for the generation prompt (no parameter schemas)
        self._gen_catalog = [
            {"tool": c["tool"], "ro": c["read_only"], "summary": c["summary"]}
            for c in condensed_catalog
        ]

    # ------------------------------------------------------------------ public
    def generate(
        self,
        archetype: dict[str, Any],
        *,
        difficulty: str = "medium",
        undercovered: list[str] | None = None,
        avoid: list[str] | None = None,
        attempts: int = 2,
    ) -> TaskSpec:
        """Generate one validated TaskSpec; retries once on a rejected spec."""
        mode = "write" if archetype["write"] else "readonly"
        # Sample the voice once per task (kept stable across the retry so a
        # rejection only fixes content, not flavour); varies across tasks.
        persona_pool = random.sample(PERSONAS, k=min(3, len(PERSONAS)))
        style = random.choice(PHRASING_STYLES)
        feedback = ""
        last: TaskGenError | None = None
        for _ in range(attempts):
            user = self._user_prompt(archetype, mode, difficulty,
                                     undercovered or [], avoid or [], feedback,
                                     persona_pool, style)
            reply = self.llm.chat(_GENERATOR_SYSTEM, user, json_mode=True)
            try:
                obj = extract_json(reply)
                return self._validate(obj, archetype, mode, difficulty, avoid or [])
            except (ValueError, TaskGenError) as e:
                last = e if isinstance(e, TaskGenError) else TaskGenError(str(e))
                feedback = (f"\n\nYour previous attempt was REJECTED for this "
                            f"reason:\n{last}\nGenerate a corrected task.")
        assert last is not None
        raise last

    # ---------------------------------------------------------------- internal
    def _user_prompt(self, archetype: dict[str, Any], mode: str, difficulty: str,
                     undercovered: list[str], avoid: list[str], feedback: str,
                     persona_pool: list[str], style: str) -> str:
        lo, hi = DIFFICULTY_STEPS[difficulty]
        parts = [
            "TOOL CATALOG (authoritative; 'ro' = read-only):",
            json.dumps(self._gen_catalog, separators=(",", ":")),
            "",
            f"UNSUPPORTED tools on this database (never require them): "
            f"{sorted(self._unsupported)}",
            f"SCENARIO ARCHETYPE: {archetype['name']} — {archetype['intent']}",
            f"MODE: {mode}",
            f"DIFFICULTY: {difficulty} — the walkthrough should need {lo}-{hi} tool calls.",
            f"PERSONA — adopt whichever best fits this archetype: {persona_pool}",
            f"PHRASING STYLE — shape the prompt's register and length like this: {style}",
        ]
        if undercovered:
            parts.append(
                "UNDER-COVERED TOOLS (prefer exercising 1-2 of these when it fits "
                f"the archetype naturally; never force an unnatural fit): {undercovered[:15]}"
            )
        if avoid:
            parts.append("EXISTING TASKS (do not duplicate):")
            parts.append(json.dumps(avoid[-20:], ensure_ascii=False))
        parts.append("\nGenerate ONE task now.")
        return "\n".join(parts) + feedback

    def _validate(self, obj: Any, archetype: dict[str, Any], mode: str,
                  difficulty: str, avoid: list[str]) -> TaskSpec:
        if not isinstance(obj, dict):
            raise TaskGenError(f"generator output is not an object: {obj!r}")
        prompt = str(obj.get("task_prompt") or "").strip()
        # models occasionally emit the tool list under a shorthand key
        tools = (obj.get("expected_tools") or obj.get("tools")
                 or obj.get("tool_sequence") or [])
        if len(prompt) < 40:
            raise TaskGenError(
                f"task_prompt is missing or too short (got keys: {list(obj)[:8]})")
        if not isinstance(tools, list) or not tools:
            raise TaskGenError(
                f"expected_tools is missing or empty (got keys: {list(obj)[:8]}; "
                f"prompt head: {prompt[:60]!r})")
        unknown = [t for t in tools if t not in self._by_name]
        if unknown:
            raise TaskGenError(f"expected_tools not in catalog: {unknown}")
        unsupported = [t for t in tools if t in self._unsupported]
        if unsupported:
            raise TaskGenError(
                f"expected_tools use extension-dependent tools: {unsupported}")
        if mode == "readonly":
            # execute_sql is catalogued as the write tool, but it IS the way
            # to run read-only SELECT/SHOW/EXPLAIN — statement-level purity is
            # enforced downstream by the pipeline gate on the actual plan.
            writers = [t for t in tools
                       if not self._by_name[t]["read_only"] and t != "execute_sql"]
            if writers:
                raise TaskGenError(f"readonly mode but expected write tools: {writers}")
        else:
            if "agenticmcpe_bench_" not in prompt:
                raise TaskGenError(
                    "write task_prompt must name its agenticmcpe_bench_* schema")
        leaked = [t for t in self._by_name if t in prompt and "_" in t]
        if leaked:
            raise TaskGenError(f"task_prompt leaks literal tool names: {leaked}")
        norm = prompt.casefold()
        if any(norm == a.casefold() for a in avoid):
            raise TaskGenError("task_prompt duplicates an existing task")

        warnings: list[str] = []
        lo, hi = DIFFICULTY_STEPS[difficulty]
        if not lo <= len(tools) <= hi:
            warnings.append(
                f"expected {lo}-{hi} steps for {difficulty}, got {len(tools)}")
        return TaskSpec(
            category=str(obj.get("category") or archetype["name"]),
            difficulty=difficulty,
            mode=mode,
            thinking=[str(t) for t in (obj.get("thinking") or [])],
            task_prompt=prompt,
            expected_tools=[str(t) for t in tools],
            warnings=warnings,
        )


__all__ = ["ARCHETYPES", "DIFFICULTY_STEPS", "GENERATION_TEMPERATURE",
           "TaskGenError", "TaskGenerator", "TaskSpec", "UNSUPPORTED_TOOLS",
           "unsupported_tools"]
