"""Benchmark pipeline: generated task -> agenticmcpe run -> accepted KB entry.

Ground-truth policy: an entry is accepted ONLY when the planner's sequence
executed end-to-end in a real browser AND the verifier's evaluators passed.
By default runs that needed a replan are rejected too — a replanned plan is
not a from-scratch sequence for the task.

Unlike ``agenticmcpe.Orchestrator``, the loop here gates every plan (initial
and replanned) BEFORE execution. The GitHub original gated on read-only
purity and a repo-name namespace; in a browser neither concept applies
(navigation itself is a "write", and no resources are created), so the
browser-safe gate is: no benchmark-unsupported tools (arbitrary JS, uploads,
drag), and every literal navigation URL on the allowed-sites list. There is
nothing to clean up afterwards — a browse session leaves no durable state.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pkg.agenticmcpe.catalog import condense_catalog, load_catalog
from pkg.agenticmcpe.config import LLMClient, Settings
from pkg.agenticmcpe.executor import ExecutionAgent
from pkg.agenticmcpe.orchestrator import _replan_feedback
from pkg.agenticmcpe.planner import Plan, PlannerAgent, PlannerError
from pkg.agenticmcpe.verifier import VerifierAgent

from .generator import ALLOWED_SITES, UNSUPPORTED_TOOLS, TaskSpec
from .kb import KnowledgeBase

DEFAULT_REJECTS_PATH = Path(__file__).resolve().parent / "rejects.jsonl"


@dataclass
class PipelineResult:
    accepted: bool
    reason: str
    entry: dict[str, Any] | None = None
    run_id: str = ""


class BenchmarkPipeline:
    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        provider: str | None = None,
        allow_replans: bool = False,
        max_replans: int = 2,
        max_per_signature: int = 3,
        log_to_console: bool = True,
        rejects_path: Path | None = None,
    ):
        self.kb = kb
        self.provider = provider
        self.allow_replans = allow_replans
        self.max_replans = max_replans
        self.max_per_signature = max_per_signature
        self.log = log_to_console
        self.rejects_path = rejects_path or DEFAULT_REJECTS_PATH

    # ------------------------------------------------------------------ public
    def run_spec(self, spec: TaskSpec) -> PipelineResult:
        if self.kb.has_prompt(spec.task_prompt):
            return self._reject(spec, "", "duplicate task_prompt already in KB")

        settings = Settings.load(provider=self.provider,
                                 run_id=time.strftime("bench-%Y%m%d-%H%M%S"))
        settings.ensure_work_dir()
        try:
            return self._run(spec, settings)
        except Exception as e:  # keep the generation loop alive; record why
            return self._reject(spec, settings.run_id, f"pipeline error: {e!r}")

    # ---------------------------------------------------------------- internal
    def _run(self, spec: TaskSpec, settings: Settings) -> PipelineResult:
        catalog = load_catalog(settings)
        condensed = condense_catalog(catalog)
        llm = LLMClient(settings.llm)
        # RAG deliberately NOT wired here: the pipeline must rediscover
        # sequences independently — that is the basis of its ground truth.
        planner = PlannerAgent(llm, condensed)

        # --- plan (with bounded self-correction, as the orchestrator does) ---
        try:
            plan = self._plan(planner, spec.task_prompt)
        except PlannerError as e:
            return self._reject(spec, settings.run_id, f"planner error: {e}")
        planner.write(plan, settings)
        gate = self._gate_plan(plan, spec)
        if gate:
            return self._reject(spec, settings.run_id, f"plan gate: {gate}")
        if self.log:
            print(f"[taskgen] plan: {' -> '.join(s.tool for s in plan.steps)}")

        # --- execute, replanning only if replans are allowed ---
        trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
        replans = 0
        budget = self.max_replans if self.allow_replans else 0
        while not trace.success and replans < budget:
            replans += 1
            try:
                plan = self._plan(planner, spec.task_prompt,
                                  feedback=_replan_feedback(plan, trace))
            except PlannerError as e:
                return self._reject(spec, settings.run_id, f"replanner error: {e}")
            planner.write(plan, settings)
            gate = self._gate_plan(plan, spec)
            if gate:
                return self._reject(spec, settings.run_id, f"replan gate: {gate}")
            trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)

        if not trace.success:
            return self._reject(spec, settings.run_id,
                                f"execution failed at step {trace.failed_step}")
        if replans > 0 and not self.allow_replans:
            return self._reject(
                spec, settings.run_id,
                "needed replans; final plan is not a from-scratch sequence")

        # --- verify (the acceptance oracle) ---
        verifier = VerifierAgent(settings, llm, tool_names=[t.name for t in catalog])
        report = verifier.verify(plan, trace)
        if self.log:
            print(f"[taskgen] verification: {report.passed}/{report.total} "
                  f"-> {'OK' if report.ok else 'FAIL'}")
        if not report.ok:
            return self._reject(
                spec, settings.run_id,
                f"verification failed: {report.passed}/{report.total} checks passed")

        # --- dedup by tool signature, then persist ---
        signature = tuple(s.tool for s in plan.steps)
        if self.kb.signature_count(signature) >= self.max_per_signature:
            return self._reject(
                spec, settings.run_id,
                f"tool signature already has {self.max_per_signature} KB entries")

        entry = self._entry(spec, plan, trace, report, replans, settings)
        if entry["quality_warnings"] and self.log:
            for w in entry["quality_warnings"]:
                print(f"[taskgen][quality] {w}")
        self.kb.add(entry)
        return PipelineResult(True, "accepted", entry=entry, run_id=settings.run_id)

    def _plan(self, planner: PlannerAgent, task: str,
              feedback: str | None = None, attempts: int = 2) -> Plan:
        fb = feedback
        last: PlannerError | None = None
        for _ in range(attempts):
            try:
                return planner.plan(task, feedback=fb)
            except PlannerError as e:
                last = e
                fb = (feedback or "") + (
                    f"\n\nYour previous attempt was REJECTED for this reason:\n{e}\n"
                    "Output a corrected, valid plan.")
        assert last is not None
        raise last

    def _gate_plan(self, plan: Plan, spec: TaskSpec) -> str | None:
        """Safety gate applied BEFORE any execution. Returns a reason or None."""
        for st in plan.steps:
            if st.tool in UNSUPPORTED_TOOLS:
                return (f"step {st.id}: benchmark-unsupported tool {st.tool!r} "
                        f"(arbitrary JS / uploads / drag are out of scope)")
            urls: list[str] = []
            if st.tool in ("browser_navigate", "browser_tabs"):
                u = st.arguments.get("url")
                if isinstance(u, str) and not u.startswith("$"):
                    urls.append(u)
            for u in urls:
                host = (urlparse(u).hostname or "").lower()
                allowed = any(host == h or host.endswith("." + h.removeprefix("www."))
                              or host == h.removeprefix("www.")
                              for h in ALLOWED_SITES)
                if not allowed:
                    return (f"step {st.id}: navigates to {u!r} — host {host!r} "
                            f"is not on the allowed-sites list")
        return None

    def _entry(self, spec: TaskSpec, plan: Plan, trace: Any, report: Any,
               replans: int, settings: Settings) -> dict[str, Any]:
        actual = [s.tool for s in plan.steps]
        if spec.expected_tools == actual:
            match = "exact"
        elif _is_subsequence(spec.expected_tools, actual):
            match = "subsequence"
        else:
            match = "divergent"
        return {
            "id": self.kb.next_id(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "category": spec.category,
            "difficulty": spec.difficulty,
            "mode": spec.mode,
            "task_prompt": spec.task_prompt,
            "task_summary": plan.summary,
            "tool_sequence": actual,
            "steps": [
                {"id": s.id, "tool": s.tool, "arguments": s.arguments,
                 "post_action_properties": s.post_action_properties,
                 "description": s.description}
                for s in plan.steps
            ],
            # Generator-side metadata (a HYPOTHESIS, like expected_tools — not the
            # verified truth): the human mental walkthrough that produced the task,
            # kept as a worked rationale a RAG consumer can show alongside steps.
            "thinking": spec.thinking,
            "expected_tools": spec.expected_tools,
            "expected_match": match,
            "quality_warnings": _quality_warnings(spec, plan, trace),
            "replans": replans,
            "verification": {"passed": report.passed, "total": report.total},
            "provider": plan.provider,
            "model": plan.model,
            "run_id": settings.run_id,
        }

    # -------------------------------------------------------------- rejection
    def _reject(self, spec: TaskSpec, run_id: str, reason: str) -> PipelineResult:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "category": spec.category,
            "mode": spec.mode,
            "task_prompt": spec.task_prompt,
            "run_id": run_id,
        }
        with self.rejects_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.log:
            print(f"[taskgen] REJECTED: {reason}")
        return PipelineResult(False, reason, run_id=run_id)


# Common literals that legitimately recur without being derived from a result.
# Browser scaffolding: targeting refs are session handles (never "guessed"
# values), and these words recur across tasks without naming an entity.
_BENIGN_LITERALS = {"true", "false", "list", "new", "close", "select",
                    "textbox", "checkbox", "radio", "combobox", "slider",
                    "left", "right", "middle"}


def _quality_warnings(spec: TaskSpec, plan: Plan, trace: Any) -> list[str]:
    """Flag literal string arguments that match a PRIOR step's result but do
    not come from the task prompt — the planner likely hardcoded a value it
    should have derived (e.g. a URL read off a page). Heuristic: recorded as
    metadata so KB consumers can filter, never a rejection."""
    warnings: list[str] = []
    prompt_fold = spec.task_prompt.casefold()
    trace_by_id = {s.id: s for s in trace.steps}
    prior_blobs: list[tuple[str, str]] = []
    for st in plan.steps:
        for key, val in st.arguments.items():
            if not isinstance(val, str) or val.startswith("$"):
                continue
            if val.startswith("find:") or _REF_RE_FULL(val):
                continue  # element targeting, not a guessed data value
            if len(val) < 5 or val.casefold() in _BENIGN_LITERALS:
                continue
            if val.casefold() in prompt_fold:
                continue
            for pid, blob in prior_blobs:
                if val in blob:
                    warnings.append(
                        f"{st.id}.{key}={val!r} matches {pid}'s result but is not "
                        "in the task prompt; likely should be a $-binding")
                    break
        ts = trace_by_id.get(st.id)
        if ts is not None and ts.result_data is not None:
            prior_blobs.append((st.id, json.dumps(ts.result_data, default=str)))
    return warnings


def _REF_RE_FULL(s: str) -> bool:
    import re
    return re.fullmatch(r"e\d+", s) is not None


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


__all__ = ["BenchmarkPipeline", "PipelineResult"]
