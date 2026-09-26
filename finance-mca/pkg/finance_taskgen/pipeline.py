"""Benchmark pipeline: generated task -> agenticmcpe run -> accepted KB entry.

Ground-truth policy: an entry is accepted ONLY when the planner's
sequence executed end-to-end against the live server AND the verifier's
evaluators passed. By default runs that needed a replan are rejected
too — a replanned plan completes "from current state" (it just retries
reads that already succeeded), so it is not a correct from-scratch
sequence for the task.

Differences from the github taskgen pipeline:

* No plan gate. Every yfinance tool is read-only — there is no
  authoritative check to apply beyond the planner's local validation.
* No fork/cleanup machinery. Nothing is created on a remote service.
* `allow_replans` defaults to False (matches the github behavior;
  replanned plans are not "from-scratch" sequences).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pkg.finance_agenticmcpe.config import AgenticConfig, default_config
from pkg.finance_agenticmcpe.executor import ExecutionAgent
from pkg.finance_agenticmcpe.llm import LLMClient
from pkg.finance_agenticmcpe.planner import Plan, PlannerAgent, PlannerError
from pkg.finance_agenticmcpe.verifier import VerifierAgent

from .generator import TaskSpec
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
        config: AgenticConfig | None = None,
        llm: LLMClient | None = None,
        allow_replans: bool = False,
        max_replans: int = 2,
        max_per_signature: int = 3,
        log_to_console: bool = True,
        rejects_path: Path | None = None,
    ):
        self.kb = kb
        self.config = config or default_config()
        self.llm = llm or LLMClient.from_env(env_var=self.config.env_anthropic_api_key)
        self.allow_replans = allow_replans
        self.max_replans = max_replans
        self.max_per_signature = max_per_signature
        self.log = log_to_console
        self.rejects_path = rejects_path or DEFAULT_REJECTS_PATH

    # ------------------------------------------------------------------ public

    def run_spec(self, spec: TaskSpec) -> PipelineResult:
        if self.kb.has_prompt(spec.task_prompt):
            return self._reject(spec, "", "duplicate task_prompt already in KB")

        run_id = time.strftime("bench-%Y%m%d-%H%M%S")
        work_dir = os.path.join("runs", run_id)
        os.makedirs(work_dir, exist_ok=True)
        try:
            return self._run(spec, run_id, work_dir)
        except Exception as e:  # noqa: BLE001
            return self._reject(spec, run_id, f"pipeline error: {e!r}")

    # ---------------------------------------------------------------- internal

    def _run(self, spec: TaskSpec, run_id: str, work_dir: str) -> PipelineResult:
        # Discover tools once for this run. Use the source extractor so
        # the pipeline doesn't depend on a live server being already up
        # at the moment the planner is consulted.
        from pkg.finance_agenticmcpe.catalog import discover
        tools = discover(
            source_path=self.config.tool_definition_source,
            command=self.config.server_command,
            args=self.config.server_args,
            cwd=self.config.server_cwd,
            env=self.config.server_env,
            prefer="live",
        )
        planner = PlannerAgent(llm=self.llm, tools=tools)

        try:
            plan_obj = self._plan(planner, spec.task_prompt)
        except PlannerError as e:
            return self._reject(spec, run_id, f"planner error: {e}")

        plan_path = os.path.join(work_dir, "plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_obj.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        if self.log:
            print(f"[taskgen] plan: {' -> '.join(s.tool for s in plan_obj.steps)}")

        # --- execute, replanning only if replans are allowed ---
        trace = ExecutionAgent(
            self.config, work_dir=work_dir, log_to_console=self.log,
        ).run(plan_obj)
        replans = 0
        budget = self.max_replans if self.allow_replans else 0
        while not trace.success and replans < budget:
            replans += 1
            feedback = trace.error_context or "execution failed"
            try:
                plan_obj = self._plan(planner, spec.task_prompt, feedback=feedback)
            except PlannerError as e:
                return self._reject(spec, run_id, f"replanner error: {e}")
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan_obj.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            trace = ExecutionAgent(
                self.config, work_dir=work_dir, log_to_console=self.log,
            ).run(plan_obj)

        if not trace.success:
            return self._reject(
                spec, run_id, f"execution failed at step {trace.failed_step}",
            )
        if replans > 0 and not self.allow_replans:
            return self._reject(
                spec, run_id, "needed replans; final plan is not a from-scratch sequence",
            )

        # --- verify (the acceptance oracle) ---
        verifier = VerifierAgent(
            self.config, tools=tools, llm=self.llm,
        )
        report = verifier.verify(plan_obj, trace)
        ver_path = os.path.join(work_dir, "verification.json")
        with open(ver_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        n_pass = sum(1 for c in report.checks if c.passed)
        if self.log:
            print(
                f"[taskgen] verification: {n_pass}/{len(report.checks)} "
                f"-> {'OK' if report.passed else 'FAIL'}"
            )
        if not report.passed:
            return self._reject(
                spec, run_id,
                f"verification failed: {n_pass}/{len(report.checks)} checks passed",
            )

        # --- dedup by tool signature, then persist ---
        signature = tuple(s.tool for s in plan_obj.steps)
        if self.kb.signature_count(signature) >= self.max_per_signature:
            return self._reject(
                spec, run_id,
                f"tool signature already has {self.max_per_signature} KB entries",
            )

        entry = self._entry(spec, plan_obj, trace, report, replans, run_id)
        if entry["quality_warnings"] and self.log:
            for w in entry["quality_warnings"]:
                print(f"[taskgen][quality] {w}")
        self.kb.add(entry)
        return PipelineResult(True, "accepted", entry=entry, run_id=run_id)

    def _plan(
        self,
        planner: PlannerAgent,
        task: str,
        feedback: str | None = None,
        attempts: int = 2,
    ) -> Plan:
        last: PlannerError | None = None
        for _ in range(attempts):
            try:
                return planner.plan(task, feedback=feedback)
            except PlannerError as e:
                last = e
                feedback = (
                    (feedback or "")
                    + f"\n\nYour previous attempt was REJECTED for this reason:\n{e}\n"
                    + "Output a corrected, valid plan."
                )
        assert last is not None
        raise last

    def _entry(
        self,
        spec: TaskSpec,
        plan_obj: Plan,
        trace: Any,
        report: Any,
        replans: int,
        run_id: str,
    ) -> dict[str, Any]:
        actual = [s.tool for s in plan_obj.steps]
        if spec.expected_tools == actual:
            match = "exact"
        elif _is_subsequence(spec.expected_tools, actual):
            match = "subsequence"
        else:
            match = "divergent"
        n_pass = sum(1 for c in report.checks if c.passed)
        return {
            "id": self.kb.next_id(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "category": spec.category,
            "difficulty": spec.difficulty,
            "mode": spec.mode,
            "task_prompt": spec.task_prompt,
            "task_summary": plan_obj.summary,
            "tool_sequence": actual,
            "steps": [
                {
                    "id": s.id,
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "post_action_properties": s.post_action_properties,
                    "description": s.description,
                }
                for s in plan_obj.steps
            ],
            "thinking": spec.thinking,
            "expected_tools": spec.expected_tools,
            "expected_match": match,
            "quality_warnings": _quality_warnings(spec, plan_obj, trace),
            "replans": replans,
            "verification": {"passed": report.passed, "total": len(report.checks),
                             "passed_n": n_pass},
            "provider": plan_obj.provider,
            "model": plan_obj.model,
            "run_id": run_id,
        }

    # ---------------------------------------------------------------- rejection

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


def _quality_warnings(
    spec: TaskSpec, plan_obj: Plan, trace: Any
) -> list[str]:
    """Flag literal string arguments that match a PRIOR step's result but
    do not come from the task prompt — the planner likely guessed a value
    it should have bound with $sN. Recorded as metadata so KB consumers
    can filter, never a rejection."""
    warnings: list[str] = []
    prompt_fold = spec.task_prompt.casefold()
    trace_by_id = {s.id: s for s in getattr(trace, "steps", [])}
    prior_blobs: list[tuple[str, str]] = []
    for st in plan_obj.steps:
        for key, val in st.arguments.items():
            if not isinstance(val, str) or val.startswith("$"):
                continue
            if len(val) < 5 or val.casefold() in _BENIGN_LITERALS:
                continue
            if val.casefold() in prompt_fold:
                continue
            for pid, blob in prior_blobs:
                if val in blob:
                    warnings.append(
                        f"{st.id}.{key}={val!r} matches {pid}'s result but is "
                        "not in the task prompt; likely should be a $-binding"
                    )
                    break
        ts = trace_by_id.get(st.id)
        if ts is not None and ts.result_data is not None:
            prior_blobs.append(
                (st.id, json.dumps(ts.result_data, default=str))
            )
    return warnings


# Common literals that legitimately recur without being derived from a result.
_BENIGN_LITERALS = {
    "main", "master", "open", "closed", "all", "true", "false",
    "annual", "quarterly",
}


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


__all__ = ["BenchmarkPipeline", "PipelineResult"]