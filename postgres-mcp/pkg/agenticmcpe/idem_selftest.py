"""Offline harness for the executor idempotency changes (no network, no LLM).

Run it after touching the executor's idempotency / ledger / binding paths:

    python3 pkg/agenticmcpe/idem_selftest.py

Drives ExecutionAgent with a scripted FakeClient and asserts each behaviour of
the PostgreSQL port: Python-literal result decoding (+bindings into it), the
"Error:" text -> MCPToolError convention, benign CREATE-exists / DROP-absent /
INSERT-duplicate-key adoption, the cross-attempt SQL ledger (replay skip with
in-plan-duplicate exemption), extension-missing fail-fast, replan hints, and
embedded SQL interpolation.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pkg.agenticmcpe.executor import (ExecutionAgent, normalize_sql,  # noqa: E402
                                      sql_kind, sql_object)
from pkg.agenticmcpe.planner import Plan, PlanStep  # noqa: E402
from pkg.mcp_wrapper import MCPToolError, ToolResult  # noqa: E402


class Text(str):
    """Marks a handler payload as the server's raw TEXT block (postgres-mcp
    formats results as str(python_object)); the fake decodes it through the
    real ToolResult, and raises MCPToolError on 'Error:' bodies exactly like
    the real client does."""


class FakeClient:
    """Scripted MCP client: handlers[tool] -> payload | Exception | callable."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []          # [(tool, args), ...]
        self.stderr_output = ""

    def get_tool(self, name):
        return SimpleNamespace(input_schema={"type": "object"})

    def call(self, tool, args):
        self.calls.append((tool, dict(args)))
        h = self.handlers[tool]
        out = h(args) if callable(h) else h
        if isinstance(out, Exception):
            raise out
        if isinstance(out, Text):
            tr = ToolResult(tool=tool, content=[{"type": "text", "text": str(out)}])
            if str(out).lstrip().startswith("Error:"):
                raise MCPToolError(tool, str(out).strip(), {})
            return tr
        return SimpleNamespace(data=out, content=[])


def agent(wd: Path | None = None):
    wd = wd or Path(tempfile.mkdtemp(prefix="idem-"))
    settings = SimpleNamespace(work_dir=wd, run_id=wd.name,
                               ensure_work_dir=lambda: None)
    return ExecutionAgent(settings, client=FakeClient({}), log_to_console=False)


def run(handlers, steps, wd: Path | None = None):
    ag = agent(wd)
    fake = FakeClient(handlers)
    ag._client = fake
    plan = Plan(task="t", summary="", steps=[PlanStep(**s) for s in steps])
    trace = ag.run(plan)
    return trace, fake


def err(tool, text):
    return MCPToolError(tool, text, {})


PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("ok   " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


# ------------------------- P0 sql classification helpers (pure functions)
check("P0 sql_kind: create", sql_kind("CREATE TABLE x (i int)") == "create")
check("P0 sql_kind: with-select", sql_kind("WITH a AS (SELECT 1) SELECT * FROM a") == "select")
check("P0 sql_kind: with-insert",
      sql_kind("WITH a AS (SELECT 1 AS v) INSERT INTO t SELECT v FROM a") == "insert")
check("P0 sql_object: qualified create",
      sql_object("CREATE TABLE IF NOT EXISTS inv.items (i int)") == ("table", "inv.items"))
check("P0 sql_object: drop schema",
      sql_object("DROP SCHEMA IF EXISTS bench CASCADE") == ("schema", "bench"))
check("P0 normalize_sql: whitespace+semicolon",
      normalize_sql("INSERT  INTO t\n VALUES (1) ;") == "INSERT INTO t VALUES (1)")

# ------------------------- P1 Python-literal decode feeds structured bindings
tr, fk = run(
    {"list_schemas": Text("[{'schema_name': 'public', 'schema_owner': 'pg', "
                          "'schema_type': 'User Schema'}]"),
     "list_objects": Text("[{'schema': 'public', 'name': 'orders', 'type': 'BASE TABLE'}]")},
    [{"id": "s0", "tool": "list_schemas", "arguments": {}},
     {"id": "s1", "tool": "list_objects",
      "arguments": {"schema_name": "$s0[0].schema_name", "object_type": "table"}}],
)
check("P1 literal decode: structured result_data",
      tr.success and isinstance(tr.steps[0].result_data, list)
      and tr.steps[0].result_data[0]["schema_name"] == "public")
check("P1 literal decode: binding resolved through it",
      fk.calls[1][1]["schema_name"] == "public")

# ------------------------- P2 "Error:" text -> tool_error with replan hint
tr, fk = run(
    {"execute_sql": Text('Error: relation "public.orderz" does not exist')},
    [{"id": "s0", "tool": "execute_sql",
      "arguments": {"sql": "SELECT * FROM public.orderz"}}],
)
check("P2 Error text: tool_error after retries",
      not tr.success and tr.steps[0].status == "tool_error"
      and len(tr.steps[0].attempts) == 3)
check("P2 Error text: hint steers replan to list_objects",
      "list_objects" in (tr.steps[0].error or ""))

# ------------------------- P3 CREATE already exists -> benign echo + binding
tr, fk = run(
    {"execute_sql": lambda a: (
        Text('Error: relation "items" already exists')
        if a["sql"].lstrip().lower().startswith("create") else Text("No results"))},
    [{"id": "s0", "tool": "execute_sql",
      "arguments": {"sql": "CREATE TABLE bench.items (i int)"}},
     {"id": "s1", "tool": "execute_sql",
      "arguments": {"sql": "INSERT INTO $s0.object_name VALUES (1)"}}],
)
check("P3 create-exists: idempotent success + echo",
      tr.success and tr.steps[0].result_data == {
          "status": "already_exists", "object_type": "table",
          "object_name": "bench.items"}
      and tr.steps[0].attempts[0].status == "success_idempotent")
check("P3 create-exists: embedded binding into the echo resolved",
      fk.calls[1][1]["sql"] == "INSERT INTO bench.items VALUES (1)")

# ------------------------- P4 DROP on absent object -> benign
tr, fk = run(
    {"execute_sql": Text('Error: table "bench.gone" does not exist')},
    [{"id": "s0", "tool": "execute_sql",
      "arguments": {"sql": "DROP TABLE bench.gone"}}],
)
check("P4 drop-absent: idempotent success",
      tr.success and tr.steps[0].result_data == {"status": "already_absent"})

# ------------------------- P5 INSERT duplicate key -> benign
tr, fk = run(
    {"execute_sql": Text('Error: duplicate key value violates unique constraint '
                         '"items_pkey"')},
    [{"id": "s0", "tool": "execute_sql",
      "arguments": {"sql": "INSERT INTO bench.items VALUES (1, 'a')"}}],
)
check("P5 duplicate key: idempotent success with table echo",
      tr.success and tr.steps[0].result_data == {
          "status": "row_exists", "table": "bench.items"})

# ------------------------- P6 real INSERT failure not masked
tr, fk = run(
    {"execute_sql": Text('Error: null value in column "id" violates not-null '
                         'constraint')},
    [{"id": "s0", "tool": "execute_sql",
      "arguments": {"sql": "INSERT INTO bench.items VALUES (NULL)"}}],
)
check("P6 real insert failure: tool_error, retried 3x",
      not tr.success and tr.steps[0].status == "tool_error"
      and len(tr.steps[0].attempts) == 3)

# ------------------------- P7 SQL ledger: cross-attempt replay skipped
wd = Path(tempfile.mkdtemp(prefix="idem-ledger-"))
INS = "INSERT INTO bench.items VALUES (1, 'first')"
tr, fk = run({"execute_sql": Text("No results")},
             [{"id": "s0", "tool": "execute_sql", "arguments": {"sql": INS}}],
             wd=wd)
check("P7 attempt 1: insert executed and ledgered",
      tr.success and (wd / "sql_ledger.jsonl").is_file()
      and [c[0] for c in fk.calls] == ["execute_sql"])
# a replan constructs a NEW agent over the SAME work dir
tr, fk = run({"execute_sql": Text("No results")},
             [{"id": "s0", "tool": "execute_sql",
               "arguments": {"sql": "INSERT  INTO bench.items\nVALUES (1, 'first');"}},
              {"id": "s1", "tool": "execute_sql",
               "arguments": {"sql": "INSERT INTO bench.items VALUES (2, 'second')"}}],
             wd=wd)
check("P7 attempt 2: identical insert pre-flight-skipped (normalized match)",
      tr.success and tr.steps[0].result_data == {
          "status": "already_executed", "kind": "insert"}
      and tr.steps[0].attempts[0].status == "success_idempotent")
check("P7 attempt 2: new insert still executed",
      [c[0] for c in fk.calls] == ["execute_sql"]
      and fk.calls[0][1]["sql"].endswith("(2, 'second')"))

# ------------------------- P7b ledger is per-RUN: reset clears prior history
from pkg.agenticmcpe.executor import reset_sql_ledger  # noqa: E402

check("P7b ledger file exists before reset", (wd / "sql_ledger.jsonl").is_file())
reset_sql_ledger(wd)
check("P7b reset removed the ledger", not (wd / "sql_ledger.jsonl").is_file())
tr, fk = run({"execute_sql": Text("No results")},
             [{"id": "s0", "tool": "execute_sql", "arguments": {"sql": INS}}],
             wd=wd)
check("P7b after reset the insert RE-EXECUTES (fresh run, stale effects gone)",
      tr.success and tr.steps[0].result_data == "No results"
      and [c[0] for c in fk.calls] == ["execute_sql"])
reset_sql_ledger(wd)  # tolerate a missing file
check("P7b reset is idempotent", not (wd / "sql_ledger.jsonl").is_file())

# ------------------------- P8 ledger exempts in-plan intentional duplicates
tr, fk = run({"execute_sql": Text("No results")},
             [{"id": "s0", "tool": "execute_sql",
               "arguments": {"sql": "INSERT INTO bench.log VALUES ('x')"}},
              {"id": "s1", "tool": "execute_sql",
               "arguments": {"sql": "INSERT INTO bench.log VALUES ('x')"}}])
check("P8 same-plan duplicate inserts both run",
      tr.success and [c[0] for c in fk.calls] == ["execute_sql", "execute_sql"])

# ------------------------- P9 reads are never ledgered or skipped
wd = Path(tempfile.mkdtemp(prefix="idem-read-"))
SEL = "SELECT count(*) AS c FROM bench.items"
run({"execute_sql": Text("[{'c': 1}]")},
    [{"id": "s0", "tool": "execute_sql", "arguments": {"sql": SEL}}], wd=wd)
tr, fk = run({"execute_sql": Text("[{'c': 2}]")},
             [{"id": "s0", "tool": "execute_sql", "arguments": {"sql": SEL}}], wd=wd)
check("P9 select re-runs across attempts (not skipped)",
      tr.success and tr.steps[0].result_data == [{"c": 2}]
      and [c[0] for c in fk.calls] == ["execute_sql"])
check("P9 select not written to ledger",
      not (wd / "sql_ledger.jsonl").is_file())

# ------------------------- P10 extension-missing message fails fast with hint
tr, fk = run(
    {"get_top_queries": Text("The 'pg_stat_statements' extension is required "
                             "to report slow queries, but it is not currently "
                             "installed.")},
    [{"id": "s0", "tool": "get_top_queries",
      "arguments": {"sort_by": "resources"}}],
)
check("P10 missing extension: immediate tool_error (no retries)",
      not tr.success and tr.steps[0].status == "tool_error"
      and len(tr.steps[0].attempts) == 1)
check("P10 missing extension: hint tells replanner to avoid the tool",
      "extension is not installed" in (tr.steps[0].error or ""))

# ------------------------- P10b get_top_queries prose-prefixed payload shim
tr, fk = run(
    {"get_top_queries": Text("Top 2 slowest queries by total execution time:\n"
                             "[{'query': 'SELECT 1', 'calls': 3, "
                             "'total_exec_time': 1.5}, "
                             "{'query': 'SELECT 2', 'calls': 1, "
                             "'total_exec_time': 0.5}]"),
     "explain_query": Text("Seq Scan on x")},
    [{"id": "s0", "tool": "get_top_queries",
      "arguments": {"sort_by": "total_time", "limit": 2}},
     {"id": "s1", "tool": "explain_query",
      "arguments": {"sql": "$s0[0].query"}}],
)
check("P10b top-queries shim: payload parsed out of the prose prefix",
      tr.success and isinstance(tr.steps[0].result_data, list)
      and tr.steps[0].result_data[0]["query"] == "SELECT 1")
check("P10b top-queries shim: '$s0[0].query' binding resolves",
      fk.calls[1][1]["sql"] == "SELECT 1")

# ------------------------- P11 embedded interpolation builds SQL from reads
tr, fk = run(
    {"list_objects": Text("[{'schema': 'public', 'name': 'orders', "
                          "'type': 'BASE TABLE'}]"),
     "execute_sql": Text("[{'cnt': 42}]")},
    [{"id": "s0", "tool": "list_objects",
      "arguments": {"schema_name": "public", "object_type": "table"}},
     {"id": "s1", "tool": "execute_sql",
      "arguments": {"sql": "SELECT count(*) AS cnt FROM "
                           "$s0[0].schema.$s0[0].name"}}],
)
check("P11 embedded interpolation: composed SQL",
      tr.success and fk.calls[1][1]["sql"]
      == "SELECT count(*) AS cnt FROM public.orders")

# ------------------------- P12 unresolvable binding fails fast, nothing sent
tr, fk = run(
    {"list_objects": Text("[]"), "execute_sql": Text("No results")},
    [{"id": "s0", "tool": "list_objects",
      "arguments": {"schema_name": "public", "object_type": "table"}},
     {"id": "s1", "tool": "execute_sql",
      "arguments": {"sql": "SELECT * FROM $s0[0].name"}}],
)
check("P12 empty list: binding fails as index-out-of-range, no bad SQL sent",
      not tr.success and tr.steps[1].status == "binding_error"
      and "out of range" in (tr.steps[1].error or "")
      and [c[0] for c in fk.calls] == ["list_objects"])

# ------------------------- P13 failure aborts and skips the rest
tr, fk = run(
    {"execute_sql": Text("Error: syntax error at or near \"FROMM\"")},
    [{"id": "s0", "tool": "execute_sql", "arguments": {"sql": "SELECT * FROMM t"}},
     {"id": "s1", "tool": "execute_sql", "arguments": {"sql": "SELECT 1"}}],
)
check("P13 prior failure: later step skipped",
      not tr.success and tr.steps[1].status == "skipped"
      and tr.failed_step == "s0")
check("P13 syntax hint present", "fix the SQL syntax" in (tr.steps[0].error or ""))

# ------------------------- P14 restricted-mode rejection carries hint
tr, fk = run(
    {"execute_sql": Text("Error: Only SELECT, ANALYZE, VACUUM, EXPLAIN, SHOW "
                         "and other read-only statements are allowed.")},
    [{"id": "s0", "tool": "execute_sql",
      "arguments": {"sql": "DROP TABLE bench.items"}}],
)
check("P14 read-only rejection: hint says restricted mode",
      not tr.success and "restricted (read-only) mode" in (tr.steps[0].error or ""))

# ---------------------------------------------------------------------- summary
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
