"""Orchestrator: plan -> execute -> (replan on failure) -> verify.

Ties the three agents together. The replan loop is the only place the plan can
change: the executor never alters the sequence, it just reports an
``error_context`` that the planner uses to produce a revised plan.

Browser caveat: every execution attempt runs a FRESH browser (the session dies
with its client), so a replanned plan must re-establish page state from
scratch — the replan prompt says so, and re-done navigations/interactions in a
new attempt are expected, not duplicates.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .catalog import condense_catalog, load_catalog
from .config import LLMClient, Settings
from .executor import ExecutionAgent, ExecutionTrace
from .planner import Plan, PlannerAgent, PlannerError, PlanStep
from .rag import KBRetriever
from .verifier import VerificationReport, VerifierAgent


@dataclass
class RunResult:
    task: str
    plan: Plan
    trace: ExecutionTrace
    report: VerificationReport | None
    replans: int

    @property
    def ok(self) -> bool:
        return self.trace.success and (self.report is None or self.report.ok)


class Orchestrator:
    def __init__(self, settings: Settings, *, log_to_console: bool = True,
                 use_rag: bool = True):
        self.settings = settings
        self.llm = LLMClient(settings.llm)
        self.log_to_console = log_to_console
        # RAG: the planner checks the verified KB before calling the LLM. On by
        # default; disable with use_rag=False or AGENTICMCPE_DISABLE_RAG. A
        # missing/empty KB is treated as "no RAG" rather than an error.
        self.retriever: KBRetriever | None = None
        if use_rag and not os.environ.get("AGENTICMCPE_DISABLE_RAG"):
            r = KBRetriever()
            if not r.is_empty:
                self.retriever = r

    def run(self, task: str, *, max_replans: int = 2, verify: bool = True) -> RunResult:
        settings = self.settings
        settings.ensure_work_dir()
        usage0 = dict(self.llm.usage)
        if (settings.work_dir / "run.json").exists() and self.log_to_console:
            print(f"[orchestrator][warn] {settings.work_dir} already holds a "
                  "previous run; its artifacts will be overwritten (use a fresh "
                  "--run-id to keep them)")

        catalog = load_catalog(settings)
        planner = PlannerAgent(self.llm, condense_catalog(catalog),
                               retriever=self.retriever)
        if self.retriever is not None and self.log_to_console:
            print(f"[orchestrator] RAG enabled: {len(self.retriever)} verified "
                  f"KB example(s) available to the planner")

        # --- plan ---
        plan = self._plan(planner, task)
        planner.write(plan, settings)
        if self.log_to_console:
            print(f"[planner] {len(plan.steps)} step(s): "
                  f"{' -> '.join(s.tool for s in plan.steps)}")
            for w in plan.warnings:
                print(f"[planner][warn] {w}")

        # --- execute, with bounded replan on failure ---
        executor = ExecutionAgent(settings, log_to_console=self.log_to_console)
        trace = executor.run(plan)
        replans = 0
        attempts = [_attempt_summary(plan, trace)]
        executed: list[tuple[Plan, ExecutionTrace]] = [(plan, trace)]
        while not trace.success and replans < max_replans:
            replans += 1
            if self.log_to_console:
                print(f"[orchestrator] execution failed; replanning "
                      f"({replans}/{max_replans}) with state-aware feedback")
            try:
                new_plan = self._plan(planner, task, feedback=_replan_feedback(plan, trace))
            except PlannerError as e:
                if self.log_to_console:
                    print(f"[orchestrator] replanning failed, stopping: {e}")
                break
            # A replanned plan restarts from a fresh browser; archive the
            # previous attempt's artifacts before planner.write/executor
            # overwrite them, so the full step history stays inspectable.
            self._archive_attempt(replans - 1)
            plan = new_plan
            planner.write(plan, settings)
            executor = ExecutionAgent(settings, log_to_console=self.log_to_console)
            trace = executor.run(plan)
            attempts.append(_attempt_summary(plan, trace))
            executed.append((plan, trace))

        # --- consolidated end-to-end plan (only meaningful after replans) ---
        effective = None
        if replans and trace.success:
            effective = merge_effective_plan(executed)
            if effective is not None:
                (settings.work_dir / "plan.effective.json").write_text(
                    json.dumps(effective.to_dict(), indent=2), encoding="utf-8")
                (settings.work_dir / "plan.effective.md").write_text(
                    PlannerAgent._render_md(effective), encoding="utf-8")
                if self.log_to_console:
                    print(f"[orchestrator] consolidated end-to-end sequence "
                          f"({len(effective.steps)} steps across {len(executed)} "
                          f"attempts) -> plan.effective.json")

        # --- verify ---
        usage_plan = dict(self.llm.usage)
        report = None
        if verify:
            verifier = VerifierAgent(settings, self.llm,
                                     tool_names=[t.name for t in catalog])
            report = verifier.verify(plan, trace)
            if self.log_to_console:
                print(f"[verifier] {report.passed}/{report.total} checks passed "
                      f"(exit={report.exit_code})")

        # LLM cost of this run, split by phase. Planning covers the initial
        # plan, self-correction retries and every replan; the executor makes
        # no LLM calls; verification is the dynamic-check codegen.
        usage_end = dict(self.llm.usage)
        self.last_usage = {
            "planning": {k: usage_plan[k] - usage0[k] for k in usage0},
            "verification": {k: usage_end[k] - usage_plan[k] for k in usage0},
            "total": {k: usage_end[k] - usage0[k] for k in usage0},
        }

        result = RunResult(task=task, plan=plan, trace=trace, report=report, replans=replans)
        self._write_summary(result, attempts, effective_written=effective is not None)
        return result

    def _archive_attempt(self, idx: int) -> None:
        """Rename the current attempt's artifacts to attempt{idx}-* before the
        next plan/execution overwrites them."""
        for name in ("plan.json", "plan.md", "trace.json", "trace.jsonl",
                     "execution.log", "server.log"):
            src = self.settings.work_dir / name
            if src.exists():
                src.rename(self.settings.work_dir / f"attempt{idx}-{name}")
        if self.log_to_console:
            print(f"[orchestrator] archived attempt {idx} artifacts -> "
                  f"attempt{idx}-plan.json / attempt{idx}-trace.json")

    def _plan(self, planner: PlannerAgent, task: str, feedback: str | None = None,
              attempts: int = 2) -> Plan:
        """Plan with a small retry: a rejected plan (bad tool/param/binding) is
        fed back so the planner can self-correct instead of crashing the run."""
        fb = feedback
        last: PlannerError | None = None
        for _ in range(attempts):
            try:
                return planner.plan(task, feedback=fb)
            except PlannerError as e:
                last = e
                if self.log_to_console:
                    print(f"[planner] rejected own plan, retrying: {e}")
                fb = (feedback or "") + (
                    f"\n\nYour previous attempt was REJECTED for this reason:\n{e}\n"
                    "Output a corrected, valid plan."
                )
        assert last is not None
        raise last

    def _write_summary(self, result: RunResult, attempts: list[dict[str, Any]] | None = None,
                       *, effective_written: bool = False) -> None:
        summary: dict[str, Any] = {
            "task": result.task,
            "ok": result.ok,
            "replans": result.replans,
            # one entry per executed plan; earlier attempts' artifacts live in
            # attempt{i}-plan.json / attempt{i}-trace.json, the final in plan.json
            "attempts": attempts or [],
            "execution": {
                "success": result.trace.success,
                "failed_step": result.trace.failed_step,
                "steps": [
                    {"id": s.id, "tool": s.tool, "status": s.status,
                     "attempts": len(s.attempts)}
                    for s in result.trace.steps
                ],
            },
            "verification": (
                {
                    "ok": result.report.ok,
                    "passed": result.report.passed,
                    "total": result.report.total,
                    "by_category": _by_category(result.report),
                }
                if result.report else None
            ),
            "llm_usage": getattr(self, "last_usage", None),
            "artifacts": {
                "plan": "plan.json",
                "plan_md": "plan.md",
                **({"plan_effective": "plan.effective.json"} if effective_written else {}),
                "trace": "trace.json",
                "trace_jsonl": "trace.jsonl",
                "execution_log": "execution.log",
                "server_log": "server.log",
                "verifier_script": "verify.py",
                "verification": "verification.json",
                "catalog": "catalog.json",
            },
        }
        (self.settings.work_dir / "run.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )


# Whole-value binding string: "$<step_id>" optionally followed by ".path"/"[idx]".
_BINDING_RE = re.compile(r"^\$(?P<step>[A-Za-z_][A-Za-z0-9_-]*)(?P<rest>[.\[].*)?$",
                         re.DOTALL)


def _rewrite_bindings(value: Any, idmap: dict[str, str]) -> Any:
    """Rewrite "$sN..." references inside ``value`` per ``idmap``."""
    if isinstance(value, str):
        m = _BINDING_RE.match(value)
        if m and m.group("step") in idmap:
            return f"${idmap[m.group('step')]}{m.group('rest') or ''}"
        return value
    if isinstance(value, dict):
        return {k: _rewrite_bindings(v, idmap) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_bindings(v, idmap) for v in value]
    return value


def merge_effective_plan(executed: list[tuple[Plan, ExecutionTrace]]) -> Plan | None:
    """Merge a replanned run's attempts into ONE from-scratch sequence.

    Takes every attempt's SUCCESSFUL steps in execution order, renumbers them
    s0..sN, rewrites "$sN" bindings to the new numbering, and drops a later
    step that exactly repeats an earlier one (same tool, same rewritten
    arguments — e.g. the re-included browser_navigate of a replan). Failed/
    skipped steps are excluded; their corrected replacements come from the
    later attempt.
    """
    if len(executed) < 2:
        return None
    merged: list[PlanStep] = []
    seen: dict[tuple[str, str], str] = {}  # (tool, canonical args) -> merged id
    for plan, trace in executed:
        tr_by_id = {s.id: s for s in trace.steps}
        idmap: dict[str, str] = {}
        for st in plan.steps:
            ts = tr_by_id.get(st.id)
            if ts is None or ts.status != "success":
                continue
            new_args = _rewrite_bindings(st.arguments, idmap)
            key = (st.tool, json.dumps(new_args, sort_keys=True, default=str))
            if key in seen:
                idmap[st.id] = seen[key]
                continue
            new_id = f"s{len(merged)}"
            idmap[st.id] = new_id
            seen[key] = new_id
            merged.append(PlanStep(
                id=new_id,
                tool=st.tool,
                arguments=new_args,
                post_action_properties=st.post_action_properties,
                description=st.description,
                expect_output=st.expect_output,
            ))
    last_plan = executed[-1][0]
    return Plan(
        task=executed[0][0].task,
        summary=(f"Consolidated end-to-end sequence: the successful steps of "
                 f"{len(executed)} attempts merged into one from-scratch plan "
                 f"(bindings renumbered, duplicate steps dropped)."),
        steps=merged,
        provider=last_plan.provider,
        model=last_plan.model,
    )


def _attempt_summary(plan: Plan, trace: ExecutionTrace) -> dict[str, Any]:
    tr_by_id = {s.id: s for s in trace.steps}
    return {
        "steps": len(plan.steps),
        "tools": [s.tool for s in plan.steps],
        "success": trace.success,
        "failed_step": trace.failed_step,
        "statuses": {s.id: (tr_by_id[s.id].status if s.id in tr_by_id else "not_run")
                     for s in plan.steps},
    }


def _short(v: Any, limit: int = 48) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


def _effect_hint(tool: str, args: dict[str, Any], data: Any) -> str:
    """A terse note on what a succeeded step produced. For actions: what page
    state resulted. For observations: a preview of what was seen, so the
    planner can correct targets/waits on replan."""
    url = data.get("url") if isinstance(data, dict) else None
    title = data.get("title") if isinstance(data, dict) else None
    where = f"page now {url} ({_short(title, 60)})" if url else "page state unchanged"
    if tool == "browser_navigate":
        return f"navigated OK — {where}"
    if tool in ("browser_snapshot", "browser_find"):
        text = data.get("text") if isinstance(data, dict) else None
        head = ""
        if isinstance(text, str):
            # the yaml body is what a replanner needs to pick refs/texts from
            body = text.partition("```yaml")[2] or text
            head = f"; snapshot head: {_short(body.strip(), 220)}"
        return f"OBSERVED {where}{head}"
    if tool in ("browser_click", "browser_type", "browser_press_key",
                "browser_select_option", "browser_hover", "browser_drag",
                "browser_drop", "browser_fill_form"):
        tgt = args.get("target") or args.get("element") or ""
        return (f"action on {_short(tgt, 40)} DONE — {where}. (Browser state "
                f"does NOT survive into a new attempt; a replan must re-reach "
                f"this state.)")
    if tool == "browser_wait_for":
        return "wait satisfied"
    if tool == "browser_close":
        return "browser closed"
    if tool == "browser_tabs":
        return f"tabs {args.get('action')}: {where}"
    text = data.get("text") if isinstance(data, dict) else data
    if isinstance(text, str):
        return f"returned: {_short(text, 180)}"
    return "done"


def _replan_feedback(plan: Plan, trace: ExecutionTrace) -> str:
    """Render the prior plan + per-step outcome for state-aware replanning."""
    tr_by_id = {s.id: s for s in trace.steps}
    lines: list[str] = []
    for ps in plan.steps:
        ts = tr_by_id.get(ps.id)
        status = (ts.status if ts else "not_run").upper()
        argbrief = ", ".join(f"{k}={_short(v)}" for k, v in list(ps.arguments.items())[:4])
        line = f"- {ps.id} {ps.tool}({argbrief}): {status}"
        if ts and ts.status == "success":
            line += f"  -> {_effect_hint(ps.tool, ts.arguments_resolved or {}, ts.result_data)}"
        elif ts and ts.error:
            line += f"  -> ERROR: {ts.error[:4200]}"
        lines.append(line)
    lines.append("")
    lines.append("The new attempt starts from a BLANK browser: re-include the "
                 "navigation and interactions needed to reach the failure "
                 "point, then fix the failing step using the outputs above.")
    return "\n".join(lines)


def _by_category(report: VerificationReport) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in report.results:
        cat = r.get("category", "?")
        bucket = out.setdefault(cat, {"passed": 0, "failed": 0})
        bucket["passed" if r.get("passed") else "failed"] += 1
    return out


# Default self-check task: a stable public page, observation-only.
SELFCHECK_TASK = (
    "Open https://example.com, confirm the page's main heading is "
    "'Example Domain', then close the browser."
)


def selfcheck(settings: Settings, *, task: str | None = None) -> RunResult:
    """End-to-end live self-check on a safe, observation-only task."""
    orch = Orchestrator(settings)
    return orch.run(task or SELFCHECK_TASK, max_replans=1, verify=True)


__all__ = ["Orchestrator", "RunResult", "merge_effective_plan", "selfcheck",
           "SELFCHECK_TASK"]
