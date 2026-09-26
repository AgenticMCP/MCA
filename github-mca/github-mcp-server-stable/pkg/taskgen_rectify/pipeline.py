"""Rectify pipeline: rejected task -> bounded replan loop -> consolidated
from-scratch plan -> fresh execution -> verification -> KB entry.

Built on ``pkg.taskgen.pipeline.BenchmarkPipeline`` (untouched; reused for
its pre-execution safety gate, bench-repo cleanup, rejection log and entry
format). What changes:

* Replans are the point, not a rejection reason. The loop runs up to
  ``max_replans`` (default 10) state-aware replans, stopping early only when
  the same failure repeats ``stall_limit`` times in a row — temperature-0
  planning that failed identically three times is not going to recover, and
  the remaining budget would be wasted LLM calls.
* A run that needed replans is NOT recorded as its final, partial plan (the
  reason taskgen rejects such runs: a replan continues "from current state").
  Its attempts are merged into one from-scratch sequence
  (``orchestrator.merge_effective_plan``), which is then executed again as a
  fresh attempt and verified. The KB entry is that proven sequence, reusable
  verbatim like any first-time pass. In write mode the bench repos are torn
  down first (``--cleanup``) so the re-execution really starts from nothing;
  without ``--cleanup`` it runs over the leftover state and non-idempotent
  steps may fail — reported as such, never silently accepted.
* Every attempt's artifacts are archived (``attempt{i}-plan.json`` ...), and
  the attempt history is written to ``rectify.json`` and into the entry's
  ``rectify`` block, so a rectified entry shows what failed and how it was
  fixed — the signal this KB is built to carry.
* ``promote_run`` finishes a run that ``--min-replans`` rejected BEFORE the
  consolidation step (the gate runs first to save the verifier call): it
  rebuilds the attempts from the archived artifacts, merges them, executes
  the merged plan fresh, verifies and stores — the same proof, run later, with
  the original replan history kept. Used when the threshold is lowered.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pkg.agenticmcpe.catalog import condense_catalog, load_catalog
from pkg.agenticmcpe.config import ConfigError, LLMClient, Settings
from pkg.agenticmcpe.cli import _trace_from_dict
from pkg.agenticmcpe.executor import ExecutionAgent, ExecutionTrace, load_plan
from pkg.agenticmcpe.orchestrator import _replan_feedback, merge_effective_plan
from pkg.agenticmcpe.planner import Plan, PlannerAgent, PlannerError
from pkg.agenticmcpe.verifier import VerifierAgent
from pkg.taskgen.generator import TaskSpec
from pkg.taskgen.kb import KnowledgeBase
from pkg.taskgen.pipeline import BenchmarkPipeline, PipelineResult

from .sources import RejectedTask

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_base_rectified.json"
DEFAULT_REJECTS_PATH = Path(__file__).resolve().parent / "rejects_rectify.jsonl"

# Renamed to attempt{i}-* before the next attempt overwrites them.
_ATTEMPT_ARTIFACTS = ("plan.json", "plan.md", "trace.json", "trace.jsonl",
                      "execution.log", "server.log")
# Seconds for GitHub to settle after the bench repos are deleted, before the
# consolidated plan re-creates them under the same names.
_TEARDOWN_SETTLE = 10
# GitHub's secondary rate limit (repo creation trips it after a few hundred
# repos in a batch, and the retry-after escalates to an hour). The plan is not
# wrong, so honour the hint and re-run the SAME plan instead of replanning —
# waiting is the only correct move, bounded per task so a broken account
# cannot hang the batch forever.
_RATE_LIMIT_RE = re.compile(r"secondary rate limit.*?Retry after (?:(\d+)m)?(\d+)s", re.I | re.S)
_MAX_RATE_WAITS = 6


class RectifyPipeline(BenchmarkPipeline):
    def __init__(self, kb: KnowledgeBase, *, provider: str | None = None,
                 max_replans: int = 10, stall_limit: int = 3, min_replans: int = 3,
                 cleanup: bool = False, max_per_signature: int = 3,
                 log_to_console: bool = True, rejects_path: Path | None = None):
        super().__init__(kb, provider=provider, allow_replans=True,
                         max_replans=max_replans, cleanup=cleanup,
                         max_per_signature=max_per_signature,
                         log_to_console=log_to_console,
                         rejects_path=rejects_path or DEFAULT_REJECTS_PATH)
        self.stall_limit = stall_limit
        # The KB is for rectifications: a run that passed with fewer replans
        # than this is not stored (first-time passes belong to taskgen).
        self.min_replans = min_replans
        self._attempts: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ public
    def run_task(self, task: RejectedTask) -> PipelineResult:
        spec = task.to_spec()
        if self.kb.has_prompt(spec.task_prompt):
            return self._reject(spec, "", "duplicate task_prompt already in KB")
        settings = _fresh_settings(self.provider)
        settings.ensure_work_dir()
        self._run_started = time.time()  # fork-cleanup epoch (BenchmarkPipeline)
        self._attempts = []
        try:
            result = self._rectify(task, spec, settings)
        except ConfigError:
            raise  # a missing key/binary is not this task's failure: abort the batch
        except Exception as e:  # keep the batch alive; record why
            result = self._reject(spec, settings.run_id, f"pipeline error: {e!r}")
        summary = {
            "task": spec.task_prompt,
            "source": task.source(),
            "accepted": result.accepted,
            "reason": result.reason,
            "kb_id": result.entry["id"] if result.entry else None,
            "attempts": self._attempts,
        }
        (settings.work_dir / "rectify.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return result

    # ---------------------------------------------------------------- internal
    def _rectify(self, task: RejectedTask, spec: TaskSpec,
                 settings: Settings) -> PipelineResult:
        catalog = load_catalog(settings)
        condensed = condense_catalog(catalog)
        read_only = {c["tool"]: c["read_only"] for c in condensed}
        llm = LLMClient(settings.llm)
        # No retriever: like taskgen, the sequence must be rediscovered
        # independently of the KB it will be added to.
        planner = PlannerAgent(llm, condensed)

        # --- attempt 0 ---
        try:
            plan = self._plan(planner, spec.task_prompt)
        except PlannerError as e:
            return self._reject(spec, settings.run_id, f"planner error: {e}")
        gate = self._gate_plan(plan, spec, read_only)
        if gate:
            return self._reject(spec, settings.run_id, f"plan gate: {gate}")
        planner.write(plan, settings)
        self._say(f"attempt 0: {_seq(plan)}")
        created = self._created_repos(plan)
        trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
        created |= self._forked_repos(trace)
        executed: list[tuple[Plan, ExecutionTrace]] = [(plan, trace)]
        self._attempts.append(_attempt_record(plan, trace))

        # --- replan loop: bounded by max_replans, cut short on a stall ---
        replans = 0
        rate_waits = 0
        last_sig = _failure_signature(trace)
        repeats = 1
        while not trace.success and replans < self.max_replans:
            wait = _rate_limit_wait(trace)
            if wait is not None and rate_waits < _MAX_RATE_WAITS:
                rate_waits += 1
                self._attempts[-1]["rate_limited"] = True
                self._say(f"GitHub secondary rate limit at {trace.failed_step}; waiting "
                          f"{wait}s and re-running the same plan ({rate_waits}/{_MAX_RATE_WAITS})")
                time.sleep(wait)
                trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
                created |= self._forked_repos(trace)
                executed.append((plan, trace))
                self._attempts.append(_attempt_record(plan, trace))
                sig = _failure_signature(trace)
                repeats = repeats + 1 if sig == last_sig else 1
                last_sig = sig
                continue
            if repeats >= self.stall_limit:
                return self._finish_reject(
                    spec, settings, created,
                    f"stalled: same failure {repeats}x in a row at "
                    f"{trace.failed_step} ({last_sig[0]}): {_short_error(trace)}")
            replans += 1
            self._say(f"replan {replans}/{self.max_replans} after failure at "
                      f"{trace.failed_step}: {_short_error(trace)}")
            try:
                plan = self._plan(planner, spec.task_prompt,
                                  feedback=_replan_feedback(plan, trace))
            except PlannerError as e:
                return self._finish_reject(spec, settings, created,
                                           f"replanner error: {e}")
            gate = self._gate_plan(plan, spec, read_only)
            if gate:
                return self._finish_reject(spec, settings, created,
                                           f"replan gate: {gate}")
            _archive_attempt(settings, replans - 1)
            planner.write(plan, settings)
            self._say(f"attempt {replans}: {_seq(plan)}")
            created |= self._created_repos(plan)
            trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
            created |= self._forked_repos(trace)
            executed.append((plan, trace))
            self._attempts.append(_attempt_record(plan, trace))
            sig = _failure_signature(trace)
            repeats = repeats + 1 if sig == last_sig else 1
            last_sig = sig
        if not trace.success:
            return self._finish_reject(
                spec, settings, created,
                f"execution failed after {replans} replan(s) at step "
                f"{trace.failed_step}: {_short_error(trace)}")
        if replans < self.min_replans:
            return self._finish_reject(
                spec, settings, created,
                f"needed only {replans} replan(s); below --min-replans "
                f"{self.min_replans}" + (" (first-time pass, not a rectification)"
                                         if replans == 0 else ""))

        # --- consolidate the attempts into ONE from-scratch plan and prove it ---
        if replans:
            plan = merge_effective_plan(executed)
            gate = self._gate_plan(plan, spec, read_only)
            if gate:
                return self._finish_reject(spec, settings, created,
                                           f"consolidated plan gate: {gate}")
            self._say(f"consolidated {len(plan.steps)} steps from {len(executed)} "
                      f"attempts; re-executing from scratch: {_seq(plan)}")
            if self.cleanup and spec.mode == "write" and created:
                self._maybe_cleanup(spec, settings, created)
                time.sleep(_TEARDOWN_SETTLE)
            _archive_attempt(settings, replans)
            planner.write(plan, settings)
            trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
            created |= self._forked_repos(trace)
            self._attempts.append({**_attempt_record(plan, trace), "consolidated": True})
            if not trace.success:
                hint = ("" if self.cleanup or spec.mode != "write" else
                        " (write task re-executed over leftover state; use --cleanup)")
                return self._finish_reject(
                    spec, settings, created,
                    f"consolidated plan failed at step {trace.failed_step}: "
                    f"{_short_error(trace)}{hint}")

        # --- verify (the acceptance oracle), then persist ---
        verifier = VerifierAgent(settings, llm, tool_names=[t.name for t in catalog])
        report = verifier.verify(plan, trace)
        self._say(f"verification: {report.passed}/{report.total} "
                  f"-> {'OK' if report.ok else 'FAIL'}")
        if not report.ok:
            failing = [r.get("name") for r in report.results if not r.get("passed")]
            return self._finish_reject(
                spec, settings, created,
                f"verification failed: {report.passed}/{report.total} checks "
                f"passed; failing: {failing[:8]}")
        signature = tuple(s.tool for s in plan.steps)
        if self.kb.signature_count(signature) >= self.max_per_signature:
            return self._finish_reject(
                spec, settings, created,
                f"tool signature already has {self.max_per_signature} KB entries")

        entry = self._entry(spec, plan, trace, report, replans, settings)
        entry["rectify"] = {"source": task.source(), "attempts": self._attempts}
        return self._store(entry, task, spec, settings, created)

    def _store(self, entry: dict[str, Any], task: RejectedTask, spec: TaskSpec,
               settings: Settings, created: set[str]) -> PipelineResult:
        # A replan may satisfy the verifier by DROPPING a step the prompt asked
        # for (e.g. "details of the first notification" when there are none).
        # Recorded as metadata so KB consumers can filter, never a rejection.
        signature = set(entry["tool_sequence"])
        dropped = [t for t in dict.fromkeys(task.tools) if t not in signature]
        if dropped:
            entry["quality_warnings"].append(
                f"rectified sequence omits tool(s) the rejected attempt planned: "
                f"{dropped}; check the prompt still gets everything it asked for")
        for w in entry["quality_warnings"]:
            self._say(f"[quality] {w}")
        self.kb.add(entry)
        self._maybe_cleanup(spec, settings, created)
        return PipelineResult(True, "accepted", entry=entry, run_id=settings.run_id)

    # ------------------------------------------------------------- promotion
    def promote_run(self, run_dir: Path, task: RejectedTask, min_replans: int) -> PipelineResult:
        """Finish a run the min-replans gate rejected before consolidation
        (see module docstring). Writes the proof into a NEW rectify-* run
        directory and marks the original ``rectify.json`` with ``promoted_to``."""
        spec = task.to_spec()
        info = json.loads((run_dir / "rectify.json").read_text(encoding="utf-8"))
        m = re.match(r"needed only (\d+) replan\(s\)", info.get("reason", ""))
        replans = int(m.group(1)) if m else -1
        if replans < min_replans:
            return PipelineResult(False, f"promote: {run_dir.name} needed {replans} "
                                         f"replan(s), below {min_replans}")
        if self.kb.has_prompt(spec.task_prompt):
            info["promoted_to"] = "already-in-kb"  # never list it again
            (run_dir / "rectify.json").write_text(
                json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            return PipelineResult(False, "promote: duplicate task_prompt already in KB")
        try:
            executed = _load_attempts(run_dir, replans)
        except (OSError, ValueError, KeyError) as e:
            return PipelineResult(False, f"promote: cannot load {run_dir.name}: {e!r}")
        settings = _fresh_settings(self.provider)
        settings.ensure_work_dir()
        self._run_started = time.time()
        self._attempts = list(info.get("attempts", []))
        try:
            result = self._promote(task, spec, settings, executed, replans, run_dir.name)
        except ConfigError:
            raise
        except Exception as e:
            result = self._reject(spec, settings.run_id, f"promote: pipeline error: {e!r}")
        summary = {
            "task": spec.task_prompt, "source": task.source(),
            "promoted_from": run_dir.name, "accepted": result.accepted,
            "reason": result.reason,
            "kb_id": result.entry["id"] if result.entry else None,
            "attempts": self._attempts,
        }
        (settings.work_dir / "rectify.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        info["promoted_to"] = settings.run_id
        info["promoted"] = result.accepted
        (run_dir / "rectify.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return result

    def _promote(self, task: RejectedTask, spec: TaskSpec, settings: Settings,
                 executed: list[tuple[Plan, ExecutionTrace]], replans: int,
                 promoted_from: str) -> PipelineResult:
        catalog = load_catalog(settings)
        condensed = condense_catalog(catalog)
        read_only = {c["tool"]: c["read_only"] for c in condensed}
        llm = LLMClient(settings.llm)
        plan = merge_effective_plan(executed)
        if plan is None or not plan.steps:
            return self._reject(spec, settings.run_id, "promote: nothing to consolidate")
        gate = self._gate_plan(plan, spec, read_only)
        if gate:
            return self._reject(spec, settings.run_id, f"promote: consolidated plan gate: {gate}")
        (settings.work_dir / "plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        (settings.work_dir / "plan.md").write_text(PlannerAgent._render_md(plan), encoding="utf-8")
        self._say(f"promoting {promoted_from} ({replans} replans): re-executing "
                  f"{len(plan.steps)} consolidated steps from scratch: {_seq(plan)}")
        created = self._created_repos(plan)
        trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
        created |= self._forked_repos(trace)
        waits = 0
        while not trace.success and waits < _MAX_RATE_WAITS:
            wait = _rate_limit_wait(trace)
            if wait is None:
                break
            waits += 1
            self._say(f"GitHub secondary rate limit; waiting {wait}s ({waits}/{_MAX_RATE_WAITS})")
            time.sleep(wait)
            trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
            created |= self._forked_repos(trace)
        self._attempts.append({**_attempt_record(plan, trace), "consolidated": True})
        if not trace.success:
            return self._finish_reject(
                spec, settings, created,
                f"promote: consolidated plan failed at step {trace.failed_step}: "
                f"{_short_error(trace)}")
        verifier = VerifierAgent(settings, llm, tool_names=[t.name for t in catalog])
        report = verifier.verify(plan, trace)
        self._say(f"verification: {report.passed}/{report.total} "
                  f"-> {'OK' if report.ok else 'FAIL'}")
        if not report.ok:
            failing = [r.get("name") for r in report.results if not r.get("passed")]
            return self._finish_reject(
                spec, settings, created,
                f"promote: verification failed: {report.passed}/{report.total} checks "
                f"passed; failing: {failing[:8]}")
        signature = tuple(s.tool for s in plan.steps)
        if self.kb.signature_count(signature) >= self.max_per_signature:
            return self._finish_reject(
                spec, settings, created,
                f"tool signature already has {self.max_per_signature} KB entries")
        entry = self._entry(spec, plan, trace, report, replans, settings)
        entry["rectify"] = {"source": task.source(), "attempts": self._attempts,
                            "promoted_from": promoted_from}
        return self._store(entry, task, spec, settings, created)

    def _say(self, msg: str) -> None:
        if self.log:
            print(f"[rectify] {msg}")


@dataclass
class Promotable:
    run_dir: Path
    replans: int
    source_run_id: str


def find_promotable(runs_dir: Path, min_replans: int) -> list[Promotable]:
    """rectify-* runs that the gate rejected for too few replans, with at least
    ``min_replans`` of them and not promoted yet (in run-id order)."""
    out: list[Promotable] = []
    for d in sorted(runs_dir.glob("rectify-*")):
        rj = d / "rectify.json"
        if not rj.is_file():
            continue
        try:
            info = json.loads(rj.read_text(encoding="utf-8"))
        except ValueError:
            continue
        m = re.match(r"needed only (\d+) replan\(s\)", str(info.get("reason", "")))
        if not m or info.get("promoted_to") or int(m.group(1)) < min_replans:
            continue
        src = str((info.get("source") or {}).get("run_id") or "")
        if src:
            out.append(Promotable(d, int(m.group(1)), src))
    return out


def _load_attempts(run_dir: Path, replans: int) -> list[tuple[Plan, ExecutionTrace]]:
    """The run's attempts in order: attempt{i}-* for the replanned ones, then
    the final plan.json/trace.json (which must have succeeded)."""
    names = [(f"attempt{i}-plan.json", f"attempt{i}-trace.json") for i in range(replans)]
    names.append(("plan.json", "trace.json"))
    pairs = [(load_plan(str(run_dir / p)),
              _trace_from_dict(json.loads((run_dir / t).read_text(encoding="utf-8"))))
             for p, t in names]
    if not pairs[-1][1].success:
        raise ValueError("final attempt did not succeed")
    return pairs


# ------------------------------------------------------------------ helpers
def _fresh_settings(provider: str | None) -> Settings:
    """Settings for a new rectify-* run directory that does not exist yet
    (two tasks failing fast in the same second must not share one)."""
    base = time.strftime("rectify-%Y%m%d-%H%M%S")
    settings = Settings.load(provider=provider, run_id=base)
    n = 1
    while settings.work_dir.exists():
        n += 1
        settings = Settings.load(provider=provider, run_id=f"{base}-{n}")
    return settings


def _seq(plan: Plan) -> str:
    return " -> ".join(s.tool for s in plan.steps)


def _failure_detail(trace: ExecutionTrace) -> tuple[str | None, str | None, str]:
    step = next((s for s in trace.steps if s.id == trace.failed_step), None)
    if step is None:
        return None, None, ""
    return step.tool, step.status, str(step.error or "")


def _short_error(trace: ExecutionTrace) -> str:
    return _failure_detail(trace)[2][:200]


def _failure_signature(trace: ExecutionTrace) -> tuple[str | None, str | None, str] | None:
    """What failed, with URLs and numbers blanked so "index 0 out of range" or
    a rate-limit retry-after count as the same failure."""
    if trace.success:
        return None
    tool, status, err = _failure_detail(trace)
    err = re.sub(r"https?://\S+", "<url>", err)
    err = re.sub(r"\d+", "N", err)
    return tool, status, err[:160]


def _rate_limit_wait(trace: ExecutionTrace) -> int | None:
    """Seconds to wait when the failed step hit GitHub's secondary rate limit
    (the server's message carries "Retry after 1m20s"); None otherwise."""
    if trace.success:
        return None
    err = _failure_detail(trace)[2]
    if "secondary rate limit" not in err.lower():
        return None
    m = _RATE_LIMIT_RE.search(err)
    secs = int(m.group(1) or 0) * 60 + int(m.group(2)) if m else 60
    return min(secs + 10, 3600)


def _attempt_record(plan: Plan, trace: ExecutionTrace) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "tools": [s.tool for s in plan.steps],
        "success": trace.success,
        "failed_step": trace.failed_step,
    }
    if not trace.success:
        rec["failed_tool"], rec["failed_status"], err = _failure_detail(trace)
        rec["error"] = err[:300]
    return rec


def _archive_attempt(settings: Settings, idx: int) -> None:
    """Rename the current attempt's artifacts to attempt{idx}-* so the next
    plan/execution does not overwrite them (as the orchestrator does)."""
    for name in _ATTEMPT_ARTIFACTS:
        src = settings.work_dir / name
        if src.exists():
            src.rename(settings.work_dir / f"attempt{idx}-{name}")


__all__ = ["DEFAULT_KB_PATH", "DEFAULT_REJECTS_PATH", "Promotable", "RectifyPipeline",
           "find_promotable"]
