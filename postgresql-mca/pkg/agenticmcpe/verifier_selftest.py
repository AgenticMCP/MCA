"""Offline harness for the verifier's tolerant comparators and self-repair
round (no network; the LLM is scripted).

    python3 pkg/agenticmcpe/verifier_selftest.py

Covers the structural reliability mechanisms added after measuring a
~1-2% per-check bug rate in LLM-authored dynamic checks (which compounds into
spurious task failures on 40-80-check trajectories):

* vals_equal / rows_equal — value comparisons that don't fail on formatting
  ('100' vs 100, '230.0' vs 230, identifier case);
* the repair round — one LLM pass over its own failing checks, which must fix
  buggy checks but must NOT be able to erase deterministic failures or real
  mismatches;
* sql_write_evaluators splitting a batched (';'-joined) execute_sql call into
  every statement it contains, not just the first — a shared-battery audit
  found a batched "DROP SCHEMA IF EXISTS x; CREATE SCHEMA x;" misclassified
  as leaving x absent, because only the DROP half was ever seen;
* _isolate_exceptions wrapping each top-level statement of the LLM-authored
  dynamic body in its own try/except, so one crashing statement (a bad key
  access, an unexpected result shape) can no longer silently abort every
  check written after it — the same audits found single exceptions costing
  1-10 checks that simply never ran (shrinking `total` instead of failing
  loudly).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import textwrap
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.pop("DATABASE_URI", None)
os.environ.pop("AGENTICMCPE_DATABASE_URI", None)

from pkg.agenticmcpe.cli import _trace_from_dict  # noqa: E402
from pkg.agenticmcpe.planner import Plan  # noqa: E402
from pkg.agenticmcpe.verifier import (  # noqa: E402
    VerifierAgent, _FORMAT_STATIC, _HELPERS, _isolate_exceptions)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("ok   " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail and not cond else ""))


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, system, user, json_mode=False):
        self.calls += 1
        return self.replies.pop(0)


# ------------------------------------------------------------ comparators
ns = {"json": json, "time": __import__("time")}
exec(_HELPERS, ns)
ve, rq = ns["vals_equal"], ns["rows_equal"]
check("C1 numeric string == int", ve("100", 100) and ve("230.0", 230))
check("C2 case/space-insensitive strings", ve(" Web ", "web"))
check("C3 unequal stays unequal", not ve("101", 100))
check("C4 rows: dict vs tuple, tolerant", rq([{"a": "100", "b": "x"}], [(100, "X")]))
check("C5 rows: mismatch detected", not rq([{"a": "1"}], [(2,)]))
check("C6 rows: unordered mode", rq([{"a": 2}, {"a": 1}], [(1,), (2,)], ordered=False))
check("C7 rows: None never passes", not rq(None, [(1,)]))
check("C8 rows: length mismatch", not rq([{"a": 1}], [(1,), (2,)]))

# ------------------------------------------------------------ repair round
PLAN = {"task": "t", "steps": [{"id": "s0", "tool": "list_schemas", "arguments": {}}]}
GOOD = {"task": "t", "success": True, "steps": [{
    "id": "s0", "tool": "list_schemas", "status": "success",
    "arguments_resolved": {},
    "attempts": [{"n": 1, "status": "success", "error": None, "duration_s": 0.1}],
    "result_data": [{"schema_name": "public"}]}]}


def settings(trace):
    wd = pathlib.Path(tempfile.mkdtemp(prefix="vselftest-"))
    (wd / "plan.json").write_text(json.dumps(PLAN))
    (wd / "trace.json").write_text(json.dumps(trace))
    return SimpleNamespace(work_dir=wd, run_id=wd.name, repo_root=REPO,
                           server_cmd=[], database=SimpleNamespace(uri=""),
                           ensure_work_dir=lambda: None)


P = Plan.from_dict(PLAN)
T = _trace_from_dict(GOOD)

s = settings(GOOD)
llm = FakeLLM(['check("dynamic","totals_match", "100"==100, "brittle ==")',
               'check("dynamic","totals_match", vals_equal("100",100), "tolerant")'])
rep = VerifierAgent(s, llm).verify(P, T)
v = json.loads((s.work_dir / "verification.json").read_text())
check("R1 repair fixes a buggy check", rep.ok and llm.calls == 2)
check("R2 repair metadata + attempt archived",
      v.get("repair", {}).get("repaired") is True
      and (s.work_dir / "verify.attempt1.py").is_file())

bad = json.loads(json.dumps(GOOD))
bad["success"] = False
bad["steps"][0]["status"] = "tool_error"
bad["steps"][0]["error"] = "boom"
s = settings(bad)
llm = FakeLLM(['check("dynamic","x", True, "fine")', "NOT-CALLED"])
rep = VerifierAgent(s, llm).verify(P, _trace_from_dict(bad))
check("R3 deterministic failure is not repair-eligible",
      not rep.ok and llm.calls == 1
      and "repair" not in json.loads((s.work_dir / "verification.json").read_text()))

s = settings(GOOD)
llm = FakeLLM(['check("dynamic","y", False, "fail")', "def broken(((("])
rep = VerifierAgent(s, llm).verify(P, T)
check("R4 unusable repair keeps attempt 1",
      not rep.ok and llm.calls == 2
      and "repair" not in json.loads((s.work_dir / "verification.json").read_text()))

s = settings(GOOD)
llm = FakeLLM(['check("dynamic","z", False, "real mismatch")',
               'check("dynamic","z", False, "confirmed mismatch")'])
rep = VerifierAgent(s, llm).verify(P, T)
check("R5 genuine mismatch survives the repair round",
      not rep.ok
      and json.loads((s.work_dir / "verification.json").read_text())["repair"]["repaired"])

# classifier edges
va = VerifierAgent.__new__(VerifierAgent)
mk = lambda r: type("R", (), {"results": r})()
check("R6 sql_write names never repair-eligible",
      not va._only_llm_dynamic_failures(
          mk([{"category": "dynamic", "name": "s0:schema_exists:public.x",
               "passed": False}])))
check("R7 mixed format+dynamic not repair-eligible",
      not va._only_llm_dynamic_failures(
          mk([{"category": "dynamic", "name": "totals", "passed": False},
              {"category": "format", "name": "s1:status", "passed": False}])))

# ------------------------------------------------------ batched-SQL split
class _FakePG:
    """pg.call() stub: every query here is a 'SELECT count(*) ...' existence
    probe; `answer` is the count it should report."""
    def __init__(self, answer):
        self.answer = answer

    def call(self, tool, args):
        return SimpleNamespace(data=[{"c": self.answer}])


_sql_results: list[dict] = []
sql_ns = {
    "re": __import__("re"),
    "check": lambda category, name, passed, detail="": _sql_results.append(
        {"name": name, "passed": bool(passed)}),
    "TRACE": {"steps": [{
        "id": "s0", "tool": "execute_sql", "status": "success",
        "arguments_resolved": {
            "sql": "DROP SCHEMA IF EXISTS agenticmcpe_bench_x CASCADE; "
                   "CREATE SCHEMA agenticmcpe_bench_x;"}}]},
}
exec(_HELPERS, sql_ns)
exec(_FORMAT_STATIC, sql_ns)
sql_ns["sql_write_evaluators"](_FakePG(answer=1))
schema_checks = [r for r in _sql_results if "agenticmcpe_bench_x" in r["name"]]
check("S1 batched DROP;CREATE classifies the LAST statement (exists, not absent)",
      len(schema_checks) == 1 and schema_checks[0]["passed"]
      and "exists" in schema_checks[0]["name"],
      str(schema_checks))

# ------------------------------------------------------ exception isolation
iso_body = ('check("dynamic", "a", True, "ok")\n'
            'x = {}["missing"]\n'
            'check("dynamic", "b", True, "reached")\n')
isolated = _isolate_exceptions(iso_body)
_iso_results: list[dict] = []
iso_ns = {"check": lambda category, name, passed, detail="": _iso_results.append(
    {"name": name, "passed": bool(passed)})}
exec("def dynamic_evaluators(pg):\n" + textwrap.indent(isolated, "    "), iso_ns)
iso_ns["dynamic_evaluators"](None)
names = {r["name"]: r["passed"] for r in _iso_results}
check("I1 a later check still runs after an earlier statement crashes",
      names.get("a") is True and names.get("b") is True, str(names))
check("I2 the crashing statement itself is reported as a failing check",
      any(not p and "isolated_exception" in n for n, p in names.items()),
      str(names))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
