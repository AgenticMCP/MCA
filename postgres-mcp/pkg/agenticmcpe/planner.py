"""Planner agent: task prompt -> grounded tool-call sequence (s0..sn) + summary.

The planner is *grounded*: it only ever sees the real postgres-mcp tool
catalog (names, one-line docs, parameter schemas), so it cannot invent tools or
parameters. After the LLM proposes a plan we validate every step against the
catalog (tool exists, required params present) before writing it out — cheap
local rejection beats a failed server round-trip.

Output artifacts (written under ``settings.work_dir``):

* ``plan.json``  — machine-readable sequence the executor replays verbatim.
* ``plan.md``    — human summary of the sequence with inputs/expected outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import LLMClient, Settings, extract_json
from .rag import (KBRetriever, RetrievedExample, catalog_enum_index,
                  task_grounds_entry)

# A binding is a whole-value string "$<step_id>" optionally followed by a path.
_BINDING_REF_RE = re.compile(r"^\$(?P<step>[A-Za-z_][A-Za-z0-9_-]*)(?:[.\[].*)?$")

# RAG (knowledge-base) retrieval thresholds, all tunable:
# - TOP_K        how many solved examples to retrieve.
# - STRONG       cosine >= this AND every literal grounded in the task => REUSE
#                the stored sequence directly, with no LLM planning call.
# - EXAMPLE_FLOOR below this a hit is noise and is not even shown to the LLM.
RAG_TOP_K = 3
RAG_STRONG_THRESHOLD = 0.86
RAG_EXAMPLE_FLOOR = 0.18

# JSON-Schema scalar type -> accepted python types, for local plan validation.
_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _binding_refs(value: Any) -> set[str]:
    """Collect the step ids referenced by binding templates inside ``value``."""
    refs: set[str] = set()
    if isinstance(value, str):
        m = _BINDING_REF_RE.match(value)
        if m:
            refs.add(m.group("step"))
    elif isinstance(value, dict):
        for v in value.values():
            refs |= _binding_refs(v)
    elif isinstance(value, list):
        for v in value:
            refs |= _binding_refs(v)
    return refs

_PLANNER_SYSTEM = """\
You are the PLANNER in an agentic workflow that replaces a human driving the \
PostgreSQL MCP server (postgres-mcp). You translate a natural-language task \
into a STRICT, ordered tool-call sequence using ONLY the provided tool catalog.

Rules:
- Use ONLY tools that appear in the catalog. Never invent tools or parameters.
- Only use parameter names that appear in a tool's `params`. Provide every \
required parameter.
- Steps are ordered and identified s0, s1, ... A later step may reference an \
earlier step's output with a binding string "$<id>.<json.path>" \
(e.g. "$s0[0].schema_name", "$s1.columns[0].column"). A binding that is the \
WHOLE value of a parameter keeps its native type (numbers stay numbers). A \
binding may ALSO be EMBEDDED inside a longer string: every "$sN.path" segment \
is substituted with the value rendered as text at execution time — e.g. SQL \
text "SELECT * FROM $s1[0].schema.$s1[0].name LIMIT 5". Build SQL and report \
content from prior outputs this way; NEVER hand-copy or guess a value that a \
prior step returns.
- Prefer the minimum number of steps that accomplishes the task. Do not add \
speculative or cleanup steps unless the task requires them.
- For each step include `post_action_properties`: a JSON object of checkable \
expectations about the result of that step, that a verifier can assert later. \
Use these keys when relevant:
    - "expect_fields": [list of field names the result object must contain]
    - "invariants": { "field": expected_fixed_value, ... }  (values that must equal the input/known constant)
    - "bounds": { "field": {"min": n, "max": m} }  (numeric boundaries)
    - "metamorphic": "short natural-language relation to re-check live, e.g. 'the table now exists in the schema and holds exactly 3 rows'"

PostgreSQL guardrails (follow exactly):
- `execute_sql` is the ONLY tool that can change the database. Every DDL/DML
  statement goes through it. All other tools are read-only inspections.
- ONE SQL statement per `execute_sql` step. Never chain statements with ";" in
  one step — separate steps keep failures isolated, bindable and auditable.
- NEVER assume a table, schema or column exists. When the task refers to
  EXISTING objects, first LOOK: `list_schemas`, `list_objects(schema_name)`, or
  `get_object_details(schema_name, object_name)` — then bind the names/columns
  the lookup returned. When the task has you CREATE the objects, no lookup is
  needed first.
- The same caution applies to SYSTEM views: read them with simple
  single-view SELECTs (pg_settings, pg_stat_user_tables, pg_indexes,
  pg_extension, information_schema.*) and only the columns you know. Do NOT
  hand-write multi-view catalog JOINs from memory — guessed join columns
  (e.g. joining pg_stat_user_tables to pg_index) fail deterministically.
  Keep such reads SIMPLE: project the columns directly
  (`SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY relname`).
  Do NOT hand-build vertical "field/value" pivots with UNION ALL — they
  multiply the chance of a wrong column name and their ORDER BY is rejected
  ("invalid UNION/INTERSECT/EXCEPT ORDER BY clause").
  Column facts that trip plans: pg_extension has extname/extversion (NOT
  name/version); information_schema.columns has udt_name (NOT udt);
  pg_available_extensions has
  name/default_version/installed_version/comment (there is NO 'installed'
  boolean — test installed_version IS NOT NULL); pg_settings has
  name/setting/unit. Inspect indexes via get_object_details or pg_indexes —
  never call pg_get_indexdef()/pg_get_* catalog functions by hand (their
  signatures are routinely guessed wrong).
- Result shapes for bindings (authoritative — never guess others):
  `list_schemas` returns a LIST of {schema_name, schema_owner, schema_type} ->
  bind "$sN[0].schema_name"; `list_objects` returns a LIST of {schema, name,
  type} for tables/views ({schema, name, data_type} for sequences; {name,
  version, relocatable} for extensions) -> bind "$sN[0].name";
  `get_object_details` returns an OBJECT {basic:{schema,name,type},
  columns:[{column,data_type,is_nullable,default}], constraints:[...],
  indexes:[{name,definition}]} -> bind "$sN.columns[0].column";
  `execute_sql` with a SELECT returns a LIST of row objects keyed by column
  name -> bind "$sN[0].<column_alias>"; `execute_sql` with DDL/DML returns the
  plain text "No results" -> bind NOTHING from it. `explain_query` and
  `analyze_db_health` return PLAIN TEXT reports -> only the whole-value "$sN"
  binding is meaningful. `get_top_queries` returns a LIST of {query, calls,
  total_exec_time, mean_exec_time, rows} -> bind "$sN[0].query". `analyze_workload_indexes` and
  `analyze_query_indexes` return an OBJECT {summary, recommendations:[
  {index_target_table, index_target_columns:[...], ...}]} -> bind e.g.
  "$sN.recommendations[0].index_target_columns[0]" (guard: recommendations
  may be empty when nothing helps).
- When a later step needs a value a SELECT returns, give the column a simple
  lowercase alias in the SELECT (e.g. `SELECT count(*) AS cnt ...` ->
  "$sN[0].cnt") and CAST non-text values you will re-inject into SQL text to
  ::text so the value round-trips exactly.
- Unquoted identifiers fold to lowercase in PostgreSQL. Use lowercase
  snake_case names for everything you create and never double-quote
  identifiers unless the task itself quotes them. Never use SQL reserved
  words as identifiers (user, session_user, current_user, order, group,
  table, check, primary, references, when) — pick a compound name like
  user_name / sort_order instead.
- Write statements must be IDEMPOTENT wherever the syntax allows: use
  `CREATE TABLE IF NOT EXISTS`, `CREATE SCHEMA IF NOT EXISTS`,
  `CREATE INDEX IF NOT EXISTS`, `CREATE OR REPLACE VIEW`, and
  `DROP ... IF EXISTS`. For INSERTs into a table with a primary key or unique
  constraint, add `ON CONFLICT DO NOTHING` when re-running the task must not
  duplicate rows. Prefer ABSOLUTE updates (`SET col = 'value'`) over relative
  ones (`SET col = col + 1`) — relative updates are unsafe to re-run.
- Schema-qualify every object you create or touch (e.g. `myschema.orders`,
  or `public.orders` when the task says nothing about a schema): the tools you
  verify with take an explicit schema name.
- When the task supplies literal content (quoted strings to insert, exact
  names, exact values), reproduce it VERBATIM in the SQL — preserve
  whitespace and apparent typos, do NOT "fix" or reformat it. The verifier
  compares the live rows against the task's quoted text.
- Bindings are LOOKUPS ONLY — there are NO expressions, comparisons, ternaries
  or arithmetic. When an action depends on COMPARING runtime values, do the
  comparison IN SQL (WHERE/CASE/ORDER BY inside one statement) — SQL is the
  expression language here. Only when a comparison must span steps include the
  read steps plus your best-guess literal; if the guess is wrong it fails, and
  on replan the per-step outputs show both values — then hardcode the correct
  literal.
- `explain_query` cannot combine analyze=true with hypothetical_indexes (the
  server rejects it). hypothetical_indexes requires the hypopg extension and
  `get_top_queries`/`analyze_workload_indexes` require pg_stat_statements —
  only plan them when the task clearly targets an environment that has those
  extensions; otherwise accomplish the task with `explain_query` (plain or
  analyze) and `execute_sql` SELECTs.
- The index advisors (`analyze_workload_indexes`, `analyze_query_indexes`)
  require UP-TO-DATE table statistics: when the plan created or bulk-loaded
  tables earlier, insert one `execute_sql` step running exactly `ANALYZE`
  before the advisor step, or it fails with "Statistics are not up-to-date".
  That statistics-refresh `ANALYZE` runs through `execute_sql` ONLY — never
  pass "ANALYZE" as the sql of `explain_query` (that tool wraps its input in
  EXPLAIN, and its analyze=true flag means EXPLAIN ANALYZE, something else
  entirely).
- Query texts returned by `get_top_queries` are NORMALIZED: literals are
  replaced with $1, $2 placeholders, and running/EXPLAINing such text fails
  with "there is no parameter $1". Never feed a recorded query back into
  `explain_query`/`execute_sql` unless its text is placeholder-free; when the
  task wants a plan, EXPLAIN a concrete query whose full text the task (or
  an earlier read) provides.
- The advisors (and `explain_query` hypothetical_indexes) resolve tables via
  the SEARCH PATH: a schema-qualified table inside the analyzed query fails
  with 'relation "x" does not exist'. Tables you want advised must live in
  the `public` schema and be referenced UNQUALIFIED in the queries passed to
  the advisor (everything else may stay schema-qualified as usual).
- An execute_sql failure surfaces as an error whose text starts with "Error:"
  followed by PostgreSQL's message (e.g. relation "x" does not exist / syntax
  error at or near ...). The tool does NOT roll anything back for you; each
  statement commits on success.
- A binding "$sN..." may only reference a step that appears EARLIER in this same
  plan. Never bind to a step id you did not include.
- Order matters: create schema -> create tables -> insert rows -> create
  indexes/views -> query/verify reads.
- Output a SINGLE JSON object, no prose, with this exact shape:
{
  "summary": "<one paragraph describing what the sequence does>",
  "steps": [
    {
      "id": "s0",
      "tool": "<tool name from catalog>",
      "arguments": { ... },
      "post_action_properties": { ... },
      "description": "<why this step>"
    }
  ]
}
"""

_REPLAN_NOTE = """\

The previous plan FAILED during execution. Below is the prior plan with the \
per-step outcome and, for read steps, what they returned.

Produce a FRESH plan (renumber from s0) that completes the ORIGINAL task FROM \
THE CURRENT STATE. Requirements:
- SUCCESS steps already happened AND COMMITTED; their schemas, tables, rows and \
indexes ALREADY EXIST in the database. Do NOT include steps that re-create or \
re-insert them — a re-run INSERT DUPLICATES its rows (SQL does not deduplicate \
for you).
- You MAY (and usually must) RE-INCLUDE read steps such as list_schemas, \
list_objects and SELECTs, because later steps bind to their outputs. Anything \
you bind to with "$sN" MUST be a step you include in this plan.
- Fix what failed using the read-step outputs below — e.g. if a SELECT or \
error message revealed the real schema, table or column name, use that EXACT \
name in the corrected statement.
- Resources may ALSO survive from an EARLIER run of this same task, not only \
from the steps above. An "already exists" style error means the object is \
ALREADY THERE: reference it, switch to IF NOT EXISTS / OR REPLACE, or look it \
up — NEVER add a plain CREATE for it again. A "duplicate key" error means the \
row is already inserted: do NOT insert it again.
- NEVER resubmit the failed plan unchanged — materially change the failing \
step (different SQL, a corrected binding, or an extra lookup step) based on \
the error text. If the error shows a literal "$sN..." string reaching the \
server, that binding's path was invalid: bind ONLY fields the step outputs \
below actually show.

EXECUTION REPORT:
{feedback}
"""


@dataclass
class PlanStep:
    id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    post_action_properties: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expect_output: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: dict[str, Any], default_id: str) -> "PlanStep":
        raw_args = dict(obj.get("arguments") or obj.get("args") or {})
        return cls(
            id=str(obj.get("id") or default_id),
            tool=obj["tool"],
            # Models emit `null` for optional params they want to skip; the
            # schema rejects nulls, so omitting is the only correct encoding.
            arguments={k: v for k, v in raw_args.items() if v is not None},
            post_action_properties=dict(obj.get("post_action_properties") or {}),
            description=obj.get("description", ""),
            expect_output=obj.get("expect_output"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "arguments": self.arguments,
            "post_action_properties": self.post_action_properties,
            "description": self.description,
            **({"expect_output": self.expect_output} if self.expect_output else {}),
        }


@dataclass
class Plan:
    task: str
    summary: str
    steps: list[PlanStep]
    provider: str = ""
    model: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "provider": self.provider,
            "model": self.model,
            "summary": self.summary,
            "warnings": self.warnings,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Plan":
        steps = [PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(obj.get("steps", []))]
        return cls(
            task=obj.get("task", ""),
            summary=obj.get("summary", ""),
            steps=steps,
            provider=obj.get("provider", ""),
            model=obj.get("model", ""),
            warnings=list(obj.get("warnings", [])),
        )


class PlannerError(RuntimeError):
    pass


class PlannerAgent:
    def __init__(self, llm: LLMClient, condensed_catalog: list[dict[str, Any]],
                 retriever: "KBRetriever | None" = None):
        self.llm = llm
        self.catalog = condensed_catalog
        self._by_name = {c["tool"]: c for c in condensed_catalog}
        # Enum/method values from the schema, exempted by the reuse gate below.
        self._enum_index = catalog_enum_index(condensed_catalog)
        # Optional RAG corpus of verified task->sequence mappings. When None the
        # planner behaves exactly as before (pure LLM planning). The taskgen
        # pipeline deliberately leaves this None so it keeps rediscovering
        # sequences independently — the basis of its ground truth.
        self.retriever = retriever

    # ------------------------------------------------------------------ public
    def plan(self, task: str, *, feedback: str | None = None) -> Plan:
        # --- RAG: consult the knowledge base BEFORE the LLM. Only on the initial
        # plan (a state-aware replan must reason about live execution state, not
        # a precedent), and only when a retriever is wired and non-empty. ---
        examples: list[RetrievedExample] = []
        if self.retriever is not None and not feedback and not self.retriever.is_empty:
            examples = self.retriever.retrieve(task, k=RAG_TOP_K)
            reused = self._maybe_reuse(task, examples)
            if reused is not None:
                return reused  # sequence "referable" -> no LLM call needed

        system = _PLANNER_SYSTEM
        catalog_json = json.dumps(self.catalog, separators=(",", ":"))
        user = f"TOOL CATALOG (authoritative):\n{catalog_json}\n\nTASK:\n{task}\n"
        user += self._format_examples(examples)
        if feedback:
            user += _REPLAN_NOTE.format(feedback=feedback)

        reply = self.llm.chat(system, user, json_mode=True)
        try:
            obj = extract_json(reply)
        except ValueError as e:
            raise PlannerError(f"planner produced unparseable output: {e}") from e
        if not isinstance(obj, dict) or "steps" not in obj:
            raise PlannerError(f"planner output missing 'steps': {obj!r}")

        steps = [
            PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(obj.get("steps", []))
        ]
        plan = Plan(
            task=task,
            summary=str(obj.get("summary", "")),
            steps=steps,
            provider=self.llm.settings.provider,
            model=self.llm.settings.model,
        )
        plan.warnings = self._validate(plan)
        return plan

    # ----------------------------------------------------------------- RAG path
    def _maybe_reuse(self, task: str, hits: list[RetrievedExample]) -> Plan | None:
        """If the top KB hit is a near-identical task whose literal arguments are
        all grounded in THIS task, reuse its verified sequence verbatim — that is
        what "the tool-call sequence can be referred" means. Returns the reused
        Plan, or None to fall through to LLM planning. The reused plan is still
        re-validated against the live catalog, so a stale entry (a tool/param
        that no longer exists) safely falls back instead of being trusted."""
        if not hits:
            return None
        top = hits[0]
        if top.score < RAG_STRONG_THRESHOLD or not task_grounds_entry(
                top.entry, task, self._enum_index):
            return None
        steps_raw = top.entry.get("steps") or []
        if not steps_raw:
            return None
        steps = [PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(steps_raw)]
        plan = Plan(
            task=task,
            summary=str(top.entry.get("task_summary", "")),
            steps=steps,
            provider=self.llm.settings.provider,
            model=self.llm.settings.model,
        )
        try:
            plan.warnings = self._validate(plan)
        except PlannerError:
            return None  # stored sequence no longer valid -> let the LLM replan
        plan.warnings.insert(
            0, f"RAG: reused verified KB entry {top.entry.get('id')} "
               f"(similarity={top.score:.2f}); no LLM planning call was made")
        return plan

    def _format_examples(self, hits: list[RetrievedExample]) -> str:
        """Render retrieved solved tasks as in-context precedent for the LLM.
        Empty string when there is nothing worth showing."""
        useful = [h for h in hits if h.score >= RAG_EXAMPLE_FLOOR]
        if not useful:
            return ""
        items = [
            {
                "similar_task": h.entry.get("task_prompt", ""),
                "tool_sequence": h.entry.get("tool_sequence", []),
                "steps": _compact_steps(h.entry.get("steps") or []),
            }
            for h in useful
        ]
        return (
            "\n\nSOLVED EXAMPLES retrieved from the verified knowledge base (most "
            "similar first). Treat them as PRECEDENT, not the answer: follow the "
            "tool choice and ORDER when the task shape matches, but RE-DERIVE every "
            "argument from THIS task and from your own earlier steps via $bindings "
            "— never copy another task's literal schemas, tables, SQL or values:\n"
            + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        )

    def write(self, plan: Plan, settings: Settings) -> tuple[str, str]:
        """Persist plan.json + plan.md. Returns their paths."""
        settings.ensure_work_dir()
        json_path = settings.work_dir / "plan.json"
        md_path = settings.work_dir / "plan.md"
        json_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self._render_md(plan), encoding="utf-8")
        return str(json_path), str(md_path)

    # ------------------------------------------------------------------ internal
    def _validate(self, plan: Plan) -> list[str]:
        """Reject hallucinated tools/params and dangling bindings before
        execution. Hard errors raise; soft issues are returned as warnings."""
        warnings: list[str] = []
        seen_ids: set[str] = set()
        for st in plan.steps:
            if st.id in seen_ids:
                raise PlannerError(f"duplicate step id {st.id!r}")
            # binding references must point to an EARLIER step in this plan
            for ref in _binding_refs(st.arguments):
                if ref not in seen_ids:
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): binding '${ref}...' references "
                        f"step {ref!r}, which is not an earlier step in this plan "
                        f"(earlier ids: {sorted(seen_ids)})"
                    )
            seen_ids.add(st.id)
            spec = self._by_name.get(st.tool)
            if spec is None:
                raise PlannerError(
                    f"step {st.id}: tool {st.tool!r} is not in the catalog"
                )
            known = {p["name"] for p in spec["params"]}
            required = {p["name"] for p in spec["params"] if p["required"]}
            present = set(st.arguments)
            missing = required - present
            if missing:
                raise PlannerError(
                    f"step {st.id} ({st.tool}): missing required params {sorted(missing)}"
                )
            unknown = present - known
            if unknown:
                warnings.append(
                    f"step {st.id} ({st.tool}): params not in schema {sorted(unknown)} "
                    "(server may reject)"
                )
            # enum/type check literal scalar args (bindings resolve later) —
            # catching these locally turns an execution-time validation
            # failure into a free planner self-correction.
            by_name = {p["name"]: p for p in spec["params"]}
            for k, v in st.arguments.items():
                p = by_name.get(k)
                if p is None or (isinstance(v, str) and _BINDING_REF_RE.match(v)):
                    continue
                if p.get("enum") and isinstance(v, (str, int)) and v not in p["enum"]:
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} not in enum "
                        f"{p['enum']}")
                expected = _TYPE_MAP.get(p.get("type"))
                if expected and not isinstance(v, expected):
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} must be of "
                        f"type {p['type']!r}")
                if p.get("type") == "integer" and isinstance(v, bool):
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} must be an integer")
            # postgres-specific soft check: multi-statement SQL in one step
            # defeats per-step isolation, bindings and idempotency handling.
            if st.tool == "execute_sql":
                sql = st.arguments.get("sql")
                if isinstance(sql, str) and _strips_to_multi_statement(sql):
                    warnings.append(
                        f"step {st.id}: sql contains multiple statements; use "
                        "one execute_sql step per statement")
        return warnings

    @staticmethod
    def _render_md(plan: Plan) -> str:
        lines = [
            f"# Plan for task\n",
            f"> {plan.task}\n",
            f"**Provider/model:** {plan.provider} / {plan.model}\n",
            f"## Summary\n\n{plan.summary}\n",
        ]
        if plan.warnings:
            lines.append("## Warnings\n")
            lines += [f"- {w}" for w in plan.warnings]
            lines.append("")
        lines.append("## Tool-call sequence\n")
        for st in plan.steps:
            lines.append(f"### {st.id} — `{st.tool}`")
            if st.description:
                lines.append(f"{st.description}\n")
            lines.append("```json")
            lines.append(json.dumps(st.arguments, indent=2))
            lines.append("```")
            if st.post_action_properties:
                lines.append("**Post-action properties:**")
                lines.append("```json")
                lines.append(json.dumps(st.post_action_properties, indent=2))
                lines.append("```")
            lines.append("")
        return "\n".join(lines)


def _strips_to_multi_statement(sql: str) -> bool:
    """Crude but safe multi-statement detector: a ';' followed by more
    non-whitespace outside of quotes. False negatives are harmless (the server
    executes them anyway); false positives are avoided by tracking quotes."""
    in_s = in_d = False
    for i, ch in enumerate(sql):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == ";" and not in_s and not in_d:
            if sql[i + 1:].strip():
                return True
    return False


def _compact_steps(steps: list[dict[str, Any]], max_val: int = 120) -> list[dict[str, Any]]:
    """Trim a KB entry's steps for the prompt: keep tool + arguments, truncating
    long string values (e.g. embedded SQL) so examples stay token-cheap."""
    out: list[dict[str, Any]] = []
    for s in steps:
        args: dict[str, Any] = {}
        for k, v in (s.get("arguments") or {}).items():
            if isinstance(v, str) and len(v) > max_val:
                args[k] = v[:max_val] + "…<truncated>"
            else:
                args[k] = v
        out.append({"tool": s.get("tool"), "arguments": args})
    return out


__all__ = ["PlanStep", "Plan", "PlannerAgent", "PlannerError"]
