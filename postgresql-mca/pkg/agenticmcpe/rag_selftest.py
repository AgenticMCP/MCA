"""Offline harness for the RAG groundedness gate (no network, no LLM).

    python3 pkg/agenticmcpe/rag_selftest.py

The gate decides whether a stored, verified sequence may be REPLAYED verbatim
for a new task. It has to satisfy two opposing requirements, so this harness
tests both directions:

* REACHABLE — a task that really does ask for the stored work must pass.
  Grounding a whole SQL statement as one literal made this impossible on this
  server (a statement is never quoted in a prompt), which silently disabled
  reuse for 95% of the corpus; entity-level grounding fixes it.
* SAFE — a task that asks for DIFFERENT data must still be blocked, or reuse
  would inject values the user never requested (wrong rows, wrong schema).

Run it after touching `rag.sql_entities` / `rag.task_grounds_entry`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pkg.agenticmcpe.rag import sql_entities, task_grounds_entry  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("ok   " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail and not cond else ""))


def entry(*sqls: str) -> dict:
    return {"steps": [{"tool": "execute_sql", "arguments": {"sql": s}}
                      for s in sqls]}


# ---------------------------------------------------------------- extraction
e = set(sql_entities(
    "INSERT INTO shop.items (id, name, qty) VALUES (1, 'bolt', 100)"))
check("E1 pulls quoted data values", "bolt" in e)
check("E2 pulls table and schema names", {"shop", "items"} <= e)
check("E3 pulls multi-digit numbers", "100" in e)
check("E4 drops single digits (structural)", "1" not in e)
check("E5 drops SQL grammar", not ({"insert", "into", "values"} & {x.lower() for x in e}))

e2 = {x.lower() for x in sql_entities(
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")}
check("E6 drops catalog namespaces", "information_schema" not in e2 and "tables" not in e2)
e3 = {x.lower() for x in sql_entities("SELECT relname FROM pg_stat_user_tables")}
check("E7 drops pg_* catalog relations", not any(x.startswith("pg_") for x in e3))
e4 = {x.lower() for x in sql_entities("CREATE TABLE t (id integer PRIMARY KEY, note text)")}
check("E8 drops built-in type names", "integer" not in e4 and "text" not in e4)

# --------------------------------------------------------------- reachability
task = ("Set up a schema called agenticmcpe_bench_inv with a table items "
        "holding id, name and qty, and seed it with bolt at 100 and nut at 250.")
ent = entry("CREATE SCHEMA IF NOT EXISTS agenticmcpe_bench_inv",
            "CREATE TABLE agenticmcpe_bench_inv.items (id integer PRIMARY KEY, "
            "name text NOT NULL, qty integer)",
            "INSERT INTO agenticmcpe_bench_inv.items (id, name, qty) "
            "VALUES (1,'bolt',100),(2,'nut',250)")
check("R1 a task that really asks for the stored work is REACHABLE",
      task_grounds_entry(ent, task))
check("R2 pure catalog reads are reachable (no task-specific entities)",
      task_grounds_entry(entry(
          "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name"),
          "List every schema in this database."))

# ---------------------------------------------------------------------- safety
check("S1 different DATA VALUES block reuse",
      not task_grounds_entry(ent, task.replace("bolt at 100", "washer at 75")),
      "a task asking for other rows must not replay the stored INSERT")
check("S2 different SCHEMA blocks reuse",
      not task_grounds_entry(ent, task.replace("agenticmcpe_bench_inv",
                                               "agenticmcpe_bench_other")))
check("S3 different TABLE name blocks reuse",
      not task_grounds_entry(ent, task.replace("table items", "table widgets")))
check("S4 different COLUMN name blocks reuse",
      not task_grounds_entry(ent, task.replace("id, name and qty",
                                               "id, label and amount")))
check("S5 a wholly unrelated task blocks reuse",
      not task_grounds_entry(ent, "Show me the slowest queries on this server."))
check("S6 non-SQL args still grounded whole",
      not task_grounds_entry(
          {"steps": [{"tool": "get_object_details",
                      "arguments": {"schema_name": "sales", "object_name": "invoices"}}]},
          "Describe the orders table in the shop schema."))

# ------------------------------------------------------------------- summary
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
