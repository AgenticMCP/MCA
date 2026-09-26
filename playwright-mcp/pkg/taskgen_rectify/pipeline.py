"""Rectification pipeline: rejected task -> many replans -> accepted KB entry.

Differences from ``pkg.taskgen.pipeline.BenchmarkPipeline``, which this
subclasses (the original is not modified):

1. **Every failure mode replans, not just execution.** The base pipeline
   replans only when execution fails; a plan-gate rejection or a failed
   verification ends the run. Here all three feed a replan:

       plan gate    -> "your plan was blocked before it ran, because ..."
       execution    -> the base state-aware execution report
       verification -> "it ran, but the evidence you captured does not prove
                        the task; these checks failed ..."

   Verification-driven replanning is the point of the package: a long task
   that executes cleanly but records too little evidence is the most common
   and most recoverable kind of rejection.

2. **A large replan budget** (default 15 vs 2).

3. **An acceptance WINDOW, not a floor.** A run is written to the KB only if
   ``MIN_REPLANS <= replans <= max_replans``. Succeeding with 0 or 1 replans
   means the task was not really rectified — it is recorded as
   ``below_replan_floor`` and deliberately dropped, so the KB collects only
   sequences that survived genuine correction.

4. **Every attempt is archived** (``attempt{i}-plan.json`` etc.) so a 15-round
   rectification stays inspectable, and the entry carries the full round
   history.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pkg.agenticmcpe.catalog import condense_catalog, load_catalog
from pkg.agenticmcpe.config import LLMClient, Settings
from pkg.agenticmcpe.executor import ExecutionAgent, ExecutionTrace
from pkg.agenticmcpe.orchestrator import _replan_feedback
from pkg.agenticmcpe.planner import Plan, PlannerAgent, PlannerError
from pkg.agenticmcpe.verifier import VerificationReport, VerifierAgent

from pkg.taskgen.generator import TaskSpec
from pkg.taskgen.kb import KnowledgeBase
from pkg.taskgen.pipeline import BenchmarkPipeline, PipelineResult

from .coverage import coverage_feedback, degeneracy_reasons
from .selector import RectifySpec

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_base_rectified.json"
DEFAULT_REJECTS_PATH = Path(__file__).resolve().parent / "rejects.jsonl"

#: A run must have needed at least this many replans to count as rectified.
MIN_REPLANS = 2

#: Artifacts rolled into ``attempt{i}-*`` before the next round overwrites them.
_ATTEMPT_ARTIFACTS = ("plan.json", "plan.md", "trace.json", "trace.jsonl",
                      "execution.log", "server.log", "verify.py",
                      "verification.json")

#: How many times a round may re-ask the planner when it hands back a plan that
#: was already tried and already failed. At temperature 0 a planner given
#: similar-looking feedback reproduces its last answer, which would silently
#: burn the replan budget on identical runs; the retries escalate instead.
_MAX_REPEAT_RETRIES = 3

#: Temperature used for those retries (the cold default keeps repeating).
_REPEAT_TEMPERATURES = (0.3, 0.6, 0.9)


@dataclass
class RectifyResult(PipelineResult):
    """A PipelineResult plus the rectification history."""
    replans: int = 0
    rounds: list[dict[str, Any]] = field(default_factory=list)
    #: True when the task ended up fully executed AND verified, regardless of
    #: whether the replan window let it into the KB.
    solved: bool = False


class RectifyPipeline(BenchmarkPipeline):
    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        provider: str | None = None,
        max_replans: int = 15,
        min_replans: int = MIN_REPLANS,
        max_per_signature: int = 3,
        log_to_console: bool = True,
        rejects_path: Path | None = None,
        task_timeout_s: float = 3600.0,
    ):
        super().__init__(
            kb,
            provider=provider,
            allow_replans=True,          # replanning is the whole point
            max_replans=max_replans,
            max_per_signature=max_per_signature,
            log_to_console=log_to_console,
            rejects_path=rejects_path or DEFAULT_REJECTS_PATH,
        )
        self.min_replans = min_replans
        self.task_timeout_s = task_timeout_s

    # ------------------------------------------------------------------ public
    def run_spec(self, spec: TaskSpec) -> RectifyResult:  # type: ignore[override]
        if self.kb.has_prompt(spec.task_prompt):
            return self._reject_r(spec, "", "duplicate task_prompt already in KB")

        settings = Settings.load(provider=self.provider,
                                 run_id=time.strftime("rectify-%Y%m%d-%H%M%S"))
        settings.ensure_work_dir()
        try:
            return self._run_rectify(spec, settings)
        except Exception as e:  # keep a batch alive; record why
            return self._reject_r(spec, settings.run_id, f"pipeline error: {e!r}")

    # ---------------------------------------------------------------- internal
    def _run_rectify(self, spec: TaskSpec, settings: Settings) -> RectifyResult:
        catalog = load_catalog(settings)
        llm = LLMClient(settings.llm)
        # RAG stays OFF, as in the base pipeline: a rectified sequence must be
        # rediscovered by this run, not copied out of the KB it will join.
        planner = PlannerAgent(llm, condense_catalog(catalog))
        verifier = VerifierAgent(settings, llm, tool_names=[t.name for t in catalog])

        deadline = time.monotonic() + self.task_timeout_s
        rounds: list[dict[str, Any]] = []
        # Signature -> how that exact plan already failed. Every entry is a
        # dead end: the loop only exits on success, so nothing in here worked.
        tried: dict[str, dict[str, Any]] = {}
        feedback: str | None = None
        replans = 0
        plan: Plan | None = None
        trace: ExecutionTrace | None = None
        report: VerificationReport | None = None

        while True:
            started = time.monotonic()
            label = "plan" if replans == 0 else f"replan {replans}/{self.max_replans}"

            # --- plan, refusing plans already known to fail ---
            try:
                plan, repeats = self._plan_unseen(planner, settings, spec, feedback,
                                                  tried, label)
            except PlannerError as e:
                return self._reject_r(spec, settings.run_id,
                                      f"planner error after {replans} replan(s): {e}",
                                      replans=replans, rounds=rounds)
            signature = _plan_signature(plan)
            if self.log:
                print(f"[rectify][{label}] {len(plan.steps)} step(s): "
                      f"{' -> '.join(s.tool for s in plan.steps)}")

            # --- gate (pre-execution safety), now a replan trigger ---
            gate = self._gate_plan(plan, spec)
            if gate:
                self._round(rounds, replans, plan, "plan_gate", gate, started,
                            repeats=repeats)
                tried[signature] = {"round": replans, "outcome": "plan_gate",
                                    "detail": gate}
                nxt = self._next_round(settings, replans, deadline)
                if nxt is None:
                    return self._reject_r(spec, settings.run_id,
                                          f"plan gate after {replans} replan(s): {gate}",
                                          replans=replans, rounds=rounds)
                replans = nxt
                feedback = _with_history(_gate_feedback(plan, gate), rounds)
                continue

            # --- execute ---
            trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
            if not trace.success:
                detail = f"failed at step {trace.failed_step}"
                self._round(rounds, replans, plan, "execution_failed", detail, started,
                            trace=trace, repeats=repeats)
                tried[signature] = {"round": replans, "outcome": "execution_failed",
                                    "detail": _failure_detail(trace)}
                nxt = self._next_round(settings, replans, deadline)
                if nxt is None:
                    return self._reject_r(
                        spec, settings.run_id,
                        f"execution failed at step {trace.failed_step} after "
                        f"{replans} replan(s)", replans=replans, rounds=rounds)
                replans = nxt
                feedback = _with_history(_replan_feedback(plan, trace), rounds)
                continue

            # --- verify (also a replan trigger, unlike the base pipeline) ---
            report = verifier.verify(plan, trace)
            if self.log:
                print(f"[rectify][{label}] verification: {report.passed}/{report.total}"
                      f" -> {'OK' if report.ok else 'FAIL'}")
            if not report.ok:
                detail = f"{report.passed}/{report.total} checks passed"
                self._round(rounds, replans, plan, "verification_failed", detail,
                            started, trace=trace, report=report, repeats=repeats)
                tried[signature] = {"round": replans, "outcome": "verification_failed",
                                    "detail": _verification_detail(report)}
                nxt = self._next_round(settings, replans, deadline)
                if nxt is None:
                    return self._reject_r(
                        spec, settings.run_id,
                        f"verification failed: {detail} after {replans} replan(s)",
                        replans=replans, rounds=rounds)
                replans = nxt
                feedback = _with_history(_verify_feedback(spec, plan, trace, report),
                                         rounds)
                continue

            # --- coverage: passing the checks is not the same as doing the
            # task. Under a 15-replan budget the planner finds plans that
            # satisfy a trace-derived verifier by doing less; those are
            # anti-precedents and must not reach the KB. ---
            cov = degeneracy_reasons(spec.task_prompt, report)
            if cov:
                detail = "; ".join(cov)[:240]
                if self.log:
                    print(f"[rectify][{label}] verified {report.passed}/"
                          f"{report.total} but DID NOT DO THE TASK:")
                    for c in cov:
                        print(f"[rectify][{label}]   - {c}")
                rec = self._round(rounds, replans, plan, "coverage_failed", detail,
                                  started, trace=trace, report=report,
                                  repeats=repeats)
                rec["coverage_reasons"] = cov
                tried[signature] = {"round": replans, "outcome": "coverage_failed",
                                    "detail": detail}
                nxt = self._next_round(settings, replans, deadline)
                if nxt is None:
                    return self._reject_r(
                        spec, settings.run_id,
                        f"coverage failed after {replans} replan(s): {detail}",
                        replans=replans, rounds=rounds)
                replans = nxt
                feedback = _with_history(coverage_feedback(spec.task_prompt, cov),
                                         rounds)
                continue

            self._round(rounds, replans, plan, "solved",
                        f"{report.passed}/{report.total} checks passed", started,
                        trace=trace, report=report, repeats=repeats)
            break

        assert plan is not None and trace is not None and report is not None

        # --- acceptance WINDOW: rectified only, first-time passers excluded ---
        if replans < self.min_replans:
            return self._reject_r(
                spec, settings.run_id,
                f"below_replan_floor: solved with {replans} replan(s), needs "
                f">= {self.min_replans} to count as rectified",
                replans=replans, rounds=rounds, solved=True)

        signature = tuple(s.tool for s in plan.steps)
        if self.kb.signature_count(signature) >= self.max_per_signature:
            return self._reject_r(
                spec, settings.run_id,
                f"tool signature already has {self.max_per_signature} KB entries",
                replans=replans, rounds=rounds, solved=True)

        entry = self._entry(spec, plan, trace, report, replans, settings)
        entry.update(_rectification_meta(spec, rounds, replans))
        if entry["quality_warnings"] and self.log:
            for w in entry["quality_warnings"]:
                print(f"[rectify][quality] {w}")
        self.kb.add(entry)
        return RectifyResult(True, "accepted", entry=entry, run_id=settings.run_id,
                             replans=replans, rounds=rounds, solved=True)

    # ---------------------------------------------------------------- planning
    def _plan_unseen(self, planner: PlannerAgent, settings: Settings,
                     spec: TaskSpec, feedback: str | None,
                     tried: dict[str, dict[str, Any]],
                     label: str) -> tuple[Plan, int]:
        """Plan, re-asking while the planner hands back a known-failing plan.

        Everything in ``tried`` already failed, so re-executing it can only
        reproduce the same failure. Each retry names the repetition explicitly
        and raises the sampling temperature, because a cold planner given
        near-identical feedback reproduces its previous answer verbatim.
        """
        plan = self._plan(planner, spec.task_prompt, feedback=feedback)
        repeats = 0
        cold = planner.llm
        try:
            while _plan_signature(plan) in tried and repeats < _MAX_REPEAT_RETRIES:
                prior = tried[_plan_signature(plan)]
                if self.log:
                    print(f"[rectify][{label}] planner repeated the plan that "
                          f"already failed in round {prior['round']}; re-asking "
                          f"at temperature {_REPEAT_TEMPERATURES[repeats]}")
                planner.llm = self._hot_llm(cold, repeats)
                try:
                    plan = self._plan(
                        planner, spec.task_prompt,
                        feedback=_repeat_feedback(feedback, plan, prior, tried))
                except PlannerError:
                    break  # keep the plan we have; executing it beats crashing
                repeats += 1
        finally:
            planner.llm = cold
        planner.write(plan, settings)
        return plan, repeats

    def _hot_llm(self, cold: LLMClient, idx: int) -> LLMClient:
        from dataclasses import replace
        temp = _REPEAT_TEMPERATURES[min(idx, len(_REPEAT_TEMPERATURES) - 1)]
        return LLMClient(replace(cold.settings, temperature=temp))

    # ------------------------------------------------------------- round admin
    def _next_round(self, settings: Settings, replans: int,
                    deadline: float) -> int | None:
        """Archive this round and return the next replan number, or None when
        the budget or the wall-clock cap is spent."""
        if replans >= self.max_replans:
            if self.log:
                print(f"[rectify] replan budget exhausted ({self.max_replans})")
            return None
        if time.monotonic() >= deadline:
            if self.log:
                print(f"[rectify] task timeout ({self.task_timeout_s:.0f}s) reached "
                      f"after {replans} replan(s)")
            return None
        self._archive_attempt(settings, replans)
        return replans + 1

    def _archive_attempt(self, settings: Settings, idx: int) -> None:
        for name in _ATTEMPT_ARTIFACTS:
            src = settings.work_dir / name
            if src.exists():
                src.replace(settings.work_dir / f"attempt{idx}-{name}")

    def _round(self, rounds: list[dict[str, Any]], replans: int, plan: Plan,
               outcome: str, detail: str, started: float, *,
               trace: ExecutionTrace | None = None,
               report: VerificationReport | None = None,
               repeats: int = 0) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "round": replans,
            "outcome": outcome,
            "detail": detail,
            "steps": len(plan.steps),
            "tools": [s.tool for s in plan.steps],
            "seconds": round(time.monotonic() - started, 1),
        }
        if repeats:
            rec["repeat_retries"] = repeats
        if trace is not None:
            rec["execution_success"] = trace.success
            rec["failed_step"] = trace.failed_step
        if report is not None:
            rec["verification"] = {"passed": report.passed, "total": report.total}
            rec["failed_checks"] = [
                f"{r.get('category')}:{r.get('name')}"
                for r in report.results if not r.get("passed")
            ][:20]
        rounds.append(rec)
        return rec

    # -------------------------------------------------------------- rejection
    def _reject_r(self, spec: TaskSpec, run_id: str, reason: str, *,
                  replans: int = 0, rounds: list[dict[str, Any]] | None = None,
                  solved: bool = False) -> RectifyResult:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "category": spec.category,
            "mode": spec.mode,
            "task_prompt": spec.task_prompt,
            "run_id": run_id,
            "replans": replans,
            "solved": solved,
            "rounds": rounds or [],
        }
        with self.rejects_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.log:
            print(f"[rectify] REJECTED: {reason}")
        return RectifyResult(False, reason, run_id=run_id, replans=replans,
                             rounds=rounds or [], solved=solved)


# ------------------------------------------------------------------- feedback
def _plan_signature(plan: Plan) -> str:
    """Identity of a plan as something to EXECUTE: the tools and the arguments
    they are called with. Step ids, descriptions and the summary are prose and
    do not change what happens in the browser."""
    return json.dumps([[s.tool, s.arguments] for s in plan.steps],
                      sort_keys=True, default=str)


def _failure_detail(trace: ExecutionTrace) -> str:
    for ts in trace.steps:
        if ts.status == "failed" and ts.error:
            return f"{ts.id} {ts.tool}: {ts.error[:200]}"
    return f"failed at {trace.failed_step}"


def _verification_detail(report: VerificationReport) -> str:
    names = [str(r.get("name")) for r in report.results if not r.get("passed")]
    return (f"{report.passed}/{report.total} checks; failed: "
            f"{', '.join(names[:6])}" + (" ..." if len(names) > 6 else ""))


def _with_history(feedback: str, rounds: list[dict[str, Any]]) -> str:
    """Prepend every previous round's outcome to the feedback.

    The base replan feedback describes only the LAST attempt, which is all a
    2-replan budget needs. Over 15 rounds that is amnesia: the planner re-tries
    an approach it already exhausted because nothing tells it that it did. This
    digest is the loop's memory.
    """
    if not rounds:
        return feedback
    lines = [f"ATTEMPT HISTORY — {len(rounds)} previous attempt(s) at this task, "
             "ALL of them dead ends. Do not repeat any of them:"]
    for r in rounds:
        lines.append(
            f"- attempt {r['round']} ({len(r['tools'])} steps: "
            f"{' -> '.join(r['tools'])}) -> {r['outcome'].upper()}: {r['detail']}")
        failed = r.get("failed_checks")
        if failed:
            lines.append(f"    failing checks: {', '.join(failed[:6])}")
    lines.append("")
    lines.append("Your next plan must differ from EVERY attempt above in the part "
                 "that failed — a different target, a different route to the page, "
                 "or extra observation steps. Keep whatever the history shows was "
                 "working.")
    lines.append("")
    return "\n".join(lines) + feedback


def _repeat_feedback(feedback: str | None, plan: Plan, prior: dict[str, Any],
                     tried: dict[str, dict[str, Any]]) -> str:
    """Sent when the planner hands back a plan that is already known to fail."""
    head = [
        "STOP — YOU JUST RESUBMITTED A PLAN THAT HAS ALREADY BEEN TRIED AND "
        f"FAILED (attempt {prior['round']}, {prior['outcome']}): {prior['detail']}",
        "",
        "The plan you just produced was:",
    ]
    head += [f"- {s.id} {s.tool}({_brief_args(s.arguments)})" for s in plan.steps]
    head += [
        "",
        f"{len(tried)} distinct plan(s) have now failed on this task. Re-running "
        "any of them produces the same failure — the browser is deterministic "
        "about these errors. You MUST change the approach, not the wording:",
        "- change the FAILING step's target (a different caption copied verbatim "
        "from the snapshot excerpt, or a listed ref);",
        "- or reach the page a different way (a direct canonical URL instead of "
        "driving the site's search widget, or the reverse);",
        "- or add the observation steps whose absence the checks complained about.",
        "",
    ]
    return "\n".join(head) + (feedback or "")


def _gate_feedback(plan: Plan, gate: str) -> str:
    """The plan never ran — it was blocked by the pre-execution safety gate."""
    return (
        "Your plan was BLOCKED BEFORE EXECUTION by the benchmark safety gate, "
        f"for this reason:\n  {gate}\n\n"
        "Your blocked plan was:\n"
        + "\n".join(f"- {s.id} {s.tool}({_brief_args(s.arguments)})" for s in plan.steps)
        + "\n\nRules the gate enforces: no arbitrary-JS / upload / drag tools, and "
          "every literal navigation URL must be on the allowed-sites list. Rewrite "
          "the plan so the blocked step is replaced by an allowed equivalent — "
          "reach the same information through a permitted site, or through "
          "snapshot/find/click on a page you are already allowed to open."
    )


def _verify_feedback(spec: TaskSpec, plan: Plan, trace: ExecutionTrace,
                     report: VerificationReport) -> str:
    """Execution succeeded; the EVIDENCE did not prove the task was done.

    The planner's instinct after a failure is to change targets. Here nothing
    was broken, so the feedback leads with that and points at the checks, so
    the next plan adds the observation steps that record the missing facts.
    """
    failed = [r for r in report.results if not r.get("passed")]
    lines = [
        "Your plan EXECUTED SUCCESSFULLY — every step ran without error. It was "
        "rejected by VERIFICATION: the trace it produced does not prove the task "
        f"was carried out. {report.passed}/{report.total} checks passed; the "
        f"{len(failed)} below failed.",
        "",
        "FAILED CHECKS:",
    ]
    for r in failed[:25]:
        detail = str(r.get("detail", ""))
        if len(detail) > 300:
            detail = detail[:300] + "..."
        lines.append(f"- [{r.get('category')}] {r.get('name')}: {detail}")
    if len(failed) > 25:
        lines.append(f"- ... and {len(failed) - 25} more")
    if report.exit_code != 0 or report.total == 0:
        lines.append(f"- verification script exit code {report.exit_code} "
                     f"({report.total} checks produced)")

    lines += [
        "",
        "WHAT TO CHANGE. The checks read ONLY the recorded trace — a fact that "
        "was on screen but not captured in a step's result does not exist to the "
        "verifier. Revise the plan so that EVERY fact the task asks for is "
        "recorded in some step's output:",
        "- Re-read the task and list each item it asks you to find. Give each one "
        "its own observation step (`browser_snapshot`, or `browser_find` with the "
        "text that labels it) AFTER the page showing it is open.",
        "- If the task names several entities (rows, papers, legs of a route, "
        "prices), capture EACH of them — a check that wants five values is not "
        "satisfied by one snapshot of the first.",
        "- Keep the steps that already worked; add or re-target the observation "
        "steps, do not rebuild the whole route from nothing.",
        "- Finish with a snapshot that shows the collected information, then "
        "`browser_close`.",
        "",
        "For reference, here is how the successful run unfolded — reuse what "
        "worked and extend it:",
        "",
        _replan_feedback(plan, trace),
    ]
    return "\n".join(lines)


def _brief_args(args: dict[str, Any], limit: int = 48) -> str:
    out = []
    for k, v in list(args.items())[:4]:
        s = v if isinstance(v, str) else json.dumps(v, default=str)
        out.append(f"{k}={s if len(s) <= limit else s[:limit] + '...'}")
    return ", ".join(out)


def _rectification_meta(spec: TaskSpec, rounds: list[dict[str, Any]],
                        replans: int) -> dict[str, Any]:
    """Provenance that separates a rectified entry from a first-time passer."""
    src = spec if isinstance(spec, RectifySpec) else None
    return {
        "origin": "rectified",
        # expected_tools is a GENERATOR hypothesis; a reject carries none, so
        # the comparison the base entry makes is meaningless here.
        "expected_match": "n/a (rectified from a reject, no generator hypothesis)",
        "replans": replans,
        "rectification": {
            "source_reject_reason": src.source_reject_reason if src else "",
            "source_run_id": src.source_run_id if src else "",
            "original_tool_sequence": list(src.original_tool_sequence) if src else [],
            "rounds": rounds,
            "failure_modes": [r["outcome"] for r in rounds],
        },
    }


__all__ = ["DEFAULT_KB_PATH", "DEFAULT_REJECTS_PATH", "MIN_REPLANS",
           "RectifyPipeline", "RectifyResult"]
