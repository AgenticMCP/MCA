"""Offline self-test for the rectification loop.

Proves the three behaviours that separate this package from ``pkg.taskgen``
without touching an LLM, a browser or the real KB: the extra replan triggers
(plan gate, verification), the replan budget, and the acceptance WINDOW.

    python -m pkg.taskgen_rectify.selftest
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from pkg.agenticmcpe.executor import ExecutionTrace, StepExecution
from pkg.agenticmcpe.planner import Plan, PlanStep
from pkg.agenticmcpe.verifier import VerificationReport

from pkg.taskgen.kb import KnowledgeBase

from .pipeline import (RectifyPipeline, _gate_feedback, _plan_signature,
                       _verify_feedback, _with_history)
from .selector import RectifySpec, reason_bucket

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


# --------------------------------------------------------------------- doubles
def _plan(tools: list[str], *, url: str = "https://arxiv.org") -> Plan:
    steps = [
        PlanStep(id=f"s{i}", tool=t,
                 arguments={"url": url} if t == "browser_navigate" else {})
        for i, t in enumerate(tools)
    ]
    return Plan(task="t", summary="s", steps=steps, provider="p", model="m")


def _trace(plan: Plan, *, success: bool, failed_step: str | None = None) -> ExecutionTrace:
    tr = ExecutionTrace(task=plan.task, success=success)
    for st in plan.steps:
        if not success and st.id == failed_step:
            tr.steps.append(StepExecution(id=st.id, tool=st.tool, status="failed",
                                          arguments_resolved=dict(st.arguments),
                                          error="boom"))
            tr.failed_step = st.id
            break
        tr.steps.append(StepExecution(id=st.id, tool=st.tool, status="success",
                                      arguments_resolved=dict(st.arguments),
                                      result_data={"url": "https://arxiv.org",
                                                   "title": "x"}))
    return tr


def _report(passed: int, total: int) -> VerificationReport:
    results = [{"category": "dynamic", "name": f"c{i}", "passed": i < passed,
                "detail": "d"} for i in range(total)]
    return VerificationReport(task="t", total=total, passed=passed,
                              failed=total - passed, results=results,
                              script_path="verify.py",
                              exit_code=0 if passed == total else 1)


class _FakePipeline(RectifyPipeline):
    """Drives ``_run_rectify`` against a scripted list of round outcomes.

    Each entry of ``script`` is one of: "gate", "exec", "verify", "ok".
    """

    #: Tools of each successive plan the fake planner returns; the last entry
    #: repeats forever. One entry => the planner is stuck on one plan.
    DEFAULT_PLANS = [["browser_navigate", "browser_snapshot", "browser_close"]]

    def __init__(self, kb: KnowledgeBase, script: list[str],
                 plan_script: list[list[str]] | None = None,
                 coverage_fail: bool = False, **kw: Any):
        super().__init__(kb, log_to_console=False, **kw)
        self.script = script
        self.plan_script = plan_script or self.DEFAULT_PLANS
        # When set, a "passing" report carries an inverted check, so the run
        # reaches the coverage gate and is turned away by it.
        self.coverage_fail = coverage_fail
        self.round = 0
        self.feedbacks: list[str | None] = []
        self.temps: list[float] = []

    def _outcome(self) -> str:
        i = min(self.round, len(self.script) - 1)
        return self.script[i]

    # -- stub out everything that would reach the network ------------------
    def _run_rectify(self, spec, settings):  # noqa: ANN001
        import pkg.taskgen_rectify.pipeline as mod

        outer = self

        class _P:
            # `plan_script` gives the tools of each successive plan; the last
            # entry repeats, which is how a stuck planner behaves.
            llm = type("_L", (), {"settings": None})()

            def plan(_s, task, feedback=None):  # noqa: ANN001
                outer.feedbacks.append(feedback)
                i = min(len(outer.feedbacks) - 1, len(outer.plan_script) - 1)
                return _plan(outer.plan_script[i])

            def write(_s, plan, settings):  # noqa: ANN001
                return ("", "")

        class _V:
            def verify(_s, plan, trace):  # noqa: ANN001
                if outer._outcome() != "ok":
                    return _report(2, 3)
                if outer.coverage_fail:
                    return VerificationReport(
                        task="t", total=1, passed=1, failed=0, exit_code=0,
                        script_path="", results=[{
                            "category": "dynamic", "name": "snapshot has LHR",
                            "passed": True,
                            "detail": "LHR not found in recorded snapshots"}])
                return _report(3, 3)

        class _E:
            def __init__(_s, settings, log_to_console=False):  # noqa: ANN001
                pass

            def run(_s, plan):  # noqa: ANN001
                ok = self._outcome() != "exec"
                return _trace(plan, success=ok, failed_step="s1")

        orig = (mod.PlannerAgent, mod.VerifierAgent, mod.ExecutionAgent,
                mod.load_catalog, mod.condense_catalog, mod.LLMClient)
        mod.PlannerAgent = lambda *a, **k: _P()
        mod.VerifierAgent = lambda *a, **k: _V()
        mod.ExecutionAgent = _E
        mod.load_catalog = lambda s: []
        mod.condense_catalog = lambda c: []
        mod.LLMClient = lambda s: type("_L", (), {"settings": s})()
        try:
            return super()._run_rectify(spec, settings)
        finally:
            (mod.PlannerAgent, mod.VerifierAgent, mod.ExecutionAgent,
             mod.load_catalog, mod.condense_catalog, mod.LLMClient) = orig

    def _gate_plan(self, plan, spec):  # noqa: ANN001
        return "step s0: not allowed" if self._outcome() == "gate" else None

    def _next_round(self, settings, replans, deadline):  # noqa: ANN001
        nxt = super()._next_round(settings, replans, deadline)
        if nxt is not None:
            self.round += 1
        return nxt

    def _hot_llm(self, cold, idx):  # noqa: ANN001
        from .pipeline import _REPEAT_TEMPERATURES
        self.temps.append(_REPEAT_TEMPERATURES[min(idx, len(_REPEAT_TEMPERATURES) - 1)])
        return cold

    def _archive_attempt(self, settings, idx):  # noqa: ANN001
        pass


def _spec() -> RectifySpec:
    return RectifySpec(category="paper-lookup", difficulty="unknown", mode="readonly",
                       thinking=[], task_prompt="Find X on arxiv and close.",
                       expected_tools=[], source_reject_reason="execution failed at s3",
                       source_run_id="bench-1", original_tool_sequence=["a", "b"])


_KB_SEQ = [0]


def _run(script: list[str], tmp: Path, plan_script: list[list[str]] | None = None,
         coverage_fail: bool = False, **kw: Any) -> tuple[Any, _FakePipeline]:
    _KB_SEQ[0] += 1
    kb = KnowledgeBase(tmp / f"kb_{_KB_SEQ[0]}.json")
    pipe = _FakePipeline(kb, script, plan_script, coverage_fail,
                         rejects_path=tmp / "rej.jsonl", **kw)
    return pipe.run_spec(_spec()), pipe


# ----------------------------------------------------------------------- tests
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rectify-selftest-"))

    print("\n[1] verification failure triggers a replan (base pipeline rejects)")
    res, pipe = _run(["verify", "verify", "verify", "ok"], tmp)
    check("R1 solved after verification-driven replans", res.solved, res.reason)
    check("R2 accepted into KB", res.accepted, res.reason)
    check("R3 replans == 3", res.replans == 3, f"got {res.replans}")
    check("R4 one round record per attempt", len(res.rounds) == 4,
          f"got {len(res.rounds)}")
    check("R5 outcomes are 3x verification_failed then solved",
          [r["outcome"] for r in res.rounds] ==
          ["verification_failed"] * 3 + ["solved"])
    check("R6 replan feedback names the failed checks",
          any(f and "FAILED CHECKS" in f for f in pipe.feedbacks))
    check("R7 feedback says execution was fine",
          any(f and "EXECUTED SUCCESSFULLY" in f for f in pipe.feedbacks))

    print("\n[2] plan-gate failure triggers a replan (base pipeline rejects)")
    res, pipe = _run(["gate", "gate", "ok"], tmp)
    check("G1 accepted after gate-driven replans", res.accepted, res.reason)
    check("G2 replans == 2", res.replans == 2, f"got {res.replans}")
    check("G3 outcomes start with plan_gate",
          [r["outcome"] for r in res.rounds] == ["plan_gate", "plan_gate", "solved"])
    check("G4 gate feedback quotes the gate reason",
          any(f and "BLOCKED BEFORE EXECUTION" in f for f in pipe.feedbacks))
    check("G5 a gate round records no execution",
          "execution_success" not in res.rounds[0])

    print("\n[3] execution failure still triggers a replan")
    res, _ = _run(["exec", "exec", "ok"], tmp)
    check("E1 accepted after exec-driven replans", res.accepted, res.reason)
    check("E2 outcomes start with execution_failed",
          res.rounds[0]["outcome"] == "execution_failed")
    check("E3 failed step recorded", res.rounds[0]["failed_step"] == "s1")

    print("\n[4] acceptance WINDOW: first-time passers are NOT KB-worthy")
    res, _ = _run(["ok"], tmp)
    check("W1 solved with 0 replans", res.solved and res.replans == 0)
    check("W2 NOT accepted", not res.accepted, res.reason)
    check("W3 rejected as below_replan_floor",
          res.reason.startswith("below_replan_floor"), res.reason)

    res, _ = _run(["verify", "ok"], tmp)
    check("W4 1 replan is still below the floor",
          res.solved and res.replans == 1 and not res.accepted, res.reason)

    res, _ = _run(["verify", "verify", "ok"], tmp)
    check("W5 2 replans clears the floor", res.accepted and res.replans == 2,
          res.reason)

    print("\n[5] replan budget is enforced")
    res, _ = _run(["verify"] * 30, tmp, max_replans=15)
    check("B1 not accepted when never solved", not res.accepted)
    check("B2 stopped at exactly 15 replans", res.replans == 15, f"got {res.replans}")
    check("B3 16 rounds recorded (initial + 15)", len(res.rounds) == 16,
          f"got {len(res.rounds)}")
    check("B4 reject reason mentions verification",
          "verification failed" in res.reason, res.reason)

    res, _ = _run(["exec"] * 30, tmp, max_replans=4)
    check("B5 custom budget honoured", res.replans == 4, f"got {res.replans}")

    print("\n[6] accepted entry carries rectification provenance")
    res, _ = _run(["exec", "verify", "ok"], tmp)
    e = res.entry or {}
    check("P1 origin=rectified", e.get("origin") == "rectified")
    check("P2 replans recorded on the entry", e.get("replans") == 2)
    check("P3 source reject reason kept",
          e.get("rectification", {}).get("source_reject_reason") ==
          "execution failed at s3")
    check("P4 mixed failure modes recorded",
          e.get("rectification", {}).get("failure_modes") ==
          ["execution_failed", "verification_failed", "solved"])
    check("P5 expected_match is not a fake comparison",
          str(e.get("expected_match", "")).startswith("n/a"))
    check("P6 tool_sequence is the FINAL working plan",
          e.get("tool_sequence") ==
          ["browser_navigate", "browser_snapshot", "browser_close"])
    check("P7 verification result recorded",
          e.get("verification") == {"passed": 3, "total": 3})

    print("\n[7] rejects are logged with their round history")
    rej = (tmp / "rej.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in rej if x.strip()]
    check("J1 rejects file written", len(recs) >= 4, f"{len(recs)} records")
    check("J2 below-floor rejects marked solved",
          any(r["solved"] and r["reason"].startswith("below_replan_floor")
              for r in recs))
    check("J3 reject records carry rounds",
          all("rounds" in r for r in recs))

    print("\n[8] repeated-plan guard (a stuck planner must not burn the budget)")
    # Planner is stuck on one plan; execution always fails. Every round after
    # the first should notice the repetition and re-ask at a hotter temperature.
    res, pipe = _run(["exec"] * 5, tmp, max_replans=3)
    check("D1 repeat retries were made", bool(pipe.temps), f"temps={pipe.temps}")
    check("D2 temperature escalates across retries",
          pipe.temps[:3] == [0.3, 0.6, 0.9], f"got {pipe.temps[:3]}")
    check("D3 retries capped per round",
          all(r.get("repeat_retries", 0) <= 3 for r in res.rounds),
          str([r.get("repeat_retries") for r in res.rounds]))
    check("D4 round 0 has no repeat (nothing tried yet)",
          "repeat_retries" not in res.rounds[0])
    check("D5 later rounds record the repeats",
          res.rounds[1].get("repeat_retries") == 3,
          str(res.rounds[1].get("repeat_retries")))
    check("D6 repeat feedback names the repetition",
          any(f and "ALREADY BEEN TRIED" in f for f in pipe.feedbacks))
    check("D7 budget still respected despite retries", res.replans == 3,
          f"got {res.replans}")

    # A planner that moves on must NOT be nagged about repeats.
    res, pipe = _run(["exec", "exec", "ok"], tmp,
                     plan_script=[["browser_navigate", "browser_close"],
                                  ["browser_navigate", "browser_snapshot",
                                   "browser_close"],
                                  ["browser_navigate", "browser_snapshot",
                                   "browser_find", "browser_close"]])
    check("D8 distinct plans trigger no repeat retries", not pipe.temps,
          f"temps={pipe.temps}")
    check("D9 still accepted", res.accepted and res.replans == 2, res.reason)

    print("\n[9] cross-round history digest")
    rounds = [
        {"round": 0, "outcome": "execution_failed", "detail": "s4: boom",
         "tools": ["browser_navigate", "browser_click"]},
        {"round": 1, "outcome": "verification_failed", "detail": "9/10",
         "tools": ["browser_navigate", "browser_snapshot"],
         "failed_checks": ["dynamic:has_year"]},
    ]
    digest = _with_history("BASE-FEEDBACK", rounds)
    check("H1 every prior attempt is listed",
          digest.count("- attempt ") == 2)
    check("H2 outcomes carried", "EXECUTION_FAILED" in digest
          and "VERIFICATION_FAILED" in digest)
    check("H3 failing check names carried", "dynamic:has_year" in digest)
    check("H4 base feedback preserved", digest.endswith("BASE-FEEDBACK"))
    check("H5 no history on the first replan",
          _with_history("BASE", []) == "BASE")

    print("\n[10] plan signature ignores prose, tracks behaviour")
    a = _plan(["browser_navigate", "browser_close"])
    b = _plan(["browser_navigate", "browser_close"])
    b.summary = "totally different words"
    for s in b.steps:
        s.description = "reworded"
    c = _plan(["browser_navigate", "browser_close"], url="https://example.com")
    check("N1 reworded plan is the SAME plan",
          _plan_signature(a) == _plan_signature(b))
    check("N2 different argument is a DIFFERENT plan",
          _plan_signature(a) != _plan_signature(c))

    print("\n[11] task-coverage gate (verified, but did it DO the task?)")
    from .coverage import degeneracy_reasons, required_deliverables

    def rep(results):
        return VerificationReport(task="t", total=len(results),
                                  passed=len(results), failed=0,
                                  results=results, script_path="", exit_code=0)

    def dyn(name, detail):
        return {"category": "dynamic", "name": name, "passed": True,
                "detail": detail}

    # Each case below is taken from a real accepted-then-quarantined entry.
    inverted = rep([dyn("snapshot contains LHR origin label",
                        "LHR not found in recorded snapshots")])
    check("C1 inverted check caught (passes while reporting absence)",
          any("inverted" in r for r in
              degeneracy_reasons("read off the cheapest price", inverted)))

    hatch = rep([dyn("results_snapshot_has_price_or_no_results",
                     "results snapshot contains a price-shaped number or the "
                     "no-results message")])
    check("C2 escape-hatch OR caught",
          any("unfalsifiable" in r for r in
              degeneracy_reasons("read off the cheapest price", hatch)))

    skipped = rep([dyn("search_was_NOT_executed_to_results",
                       "final url is still the search form")])
    check("C3 work-skipped check caught",
          any("did not happen" in r for r in
              degeneracy_reasons("run a search and read the fare", skipped)))

    formonly = rep([dyn("snapshot_contains_from_field", "From field present"),
                    dyn("snapshot_contains_search_button", "Search button present")])
    check("C4 uncovered deliverable caught (verified the FORM, not the price)",
          any("asks for a price" in r for r in
              degeneracy_reasons("read off the cheapest one-way price", formonly)))

    genuine = rep([dyn("price_visible_in_results",
                       "cheapest fare GBP 142 appears in the results snapshot")])
    check("C5 a real content check is NOT flagged",
          not degeneracy_reasons("read off the cheapest one-way price", genuine),
          str(degeneracy_reasons("read off the cheapest one-way price", genuine)))

    # The prompt and the check may word the same deliverable differently.
    wording = rep([dyn("marie_curie_birth_date_in_infobox",
                       "birth date 7 November 1867 appears in the infobox")])
    check("C6 'date of birth' in prompt matches 'birth date' in check",
          not degeneracy_reasons("read off her date of birth", wording),
          str(degeneracy_reasons("read off her date of birth", wording)))

    check("C7 no dynamic checks => no deliverable verdict",
          not degeneracy_reasons("read off the price",
                                 rep([{"category": "format", "name": "s0:status",
                                       "passed": True, "detail": "success"}])))
    check("C8 a FAILING inverted check is not double-counted",
          not degeneracy_reasons("read off the price", VerificationReport(
              task="t", total=1, passed=0, failed=1,
              results=[{"category": "dynamic", "name": "x",
                        "passed": False, "detail": "LHR not found"}],
              script_path="", exit_code=1)))
    check("C9 deliverables detected from the prompt",
          set(required_deliverables("tell me the duration and distance")) ==
          {"duration", "distance"},
          str(required_deliverables("tell me the duration and distance")))

    print("\n[12] coverage failure drives a replan, and can end a run")
    res, pipe = _run(["ok"] * 20, tmp, max_replans=3, coverage_fail=True)
    check("C10 not accepted when every round is degenerate", not res.accepted)
    check("C11 reason names coverage", "coverage failed" in res.reason, res.reason)
    check("C12 rounds recorded as coverage_failed",
          all(r["outcome"] == "coverage_failed" for r in res.rounds),
          str([r["outcome"] for r in res.rounds]))
    check("C13 coverage reasons kept on the round",
          bool(res.rounds[0].get("coverage_reasons")))
    check("C14 coverage feedback tells the planner it did not do the task",
          any(f and "NOT ACTUALLY DO THE TASK" in f for f in pipe.feedbacks))

    print("\n[13] selector reason buckets")
    check("S1 execution bucket",
          reason_bucket("execution failed at step s3") == "execution failed")
    check("S2 verification bucket",
          reason_bucket("verification failed: 12/14 checks passed") ==
          "verification failed")
    check("S3 signature bucket kept distinct",
          reason_bucket("tool signature already has 3 KB entries") ==
          "tool signature")

    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} checks passed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
