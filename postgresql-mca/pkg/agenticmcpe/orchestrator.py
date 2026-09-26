"""Orchestrator: plan -> execute -> (replan on failure) -> verify.

Ties the three agents together. The replan loop is the only place the plan can
change: the executor never alters the sequence, it just reports an
``error_context`` that the planner uses to produce a revised plan.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .catalog import condense_catalog, load_catalog
from .config import LLMClient, Settings
from .executor import (ExecutionAgent, ExecutionTrace, reset_sql_ledger,
                       sql_kind, sql_object)
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
        # A fresh run must not inherit the previous run's SQL ledger: it
        # records what was executed, not what still EXISTS, so a stale entry
        # skips a write whose effect has since been dropped (see
        # reset_sql_ledger). Replan attempts within THIS run still share it.
        reset_sql_ledger(settings.work_dir)
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
            # A replanned plan covers only the REMAINING work; archive the
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
        report = None
        if verify:
            verifier = VerifierAgent(settings, self.llm,
                                     tool_names=[t.name for t in catalog])
            report = verifier.verify(plan, trace)
            if self.log_to_console:
                print(f"[verifier] {report.passed}/{report.total} checks passed "
                      f"(exit={report.exit_code})")

        result = RunResult(task=task, plan=plan, trace=trace, report=report, replans=replans)
        self._write_summary(result, attempts, effective_written=effective is not None)
        return result

    def _archive_attempt(self, idx: int) -> None:
        """Rename the current attempt's artifacts to attempt{idx}-* before the
        next plan/execution overwrites them. (sql_ledger.jsonl deliberately
        stays in place — the next attempt's pre-flight dedup reads it.)"""
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
            "artifacts": {
                "plan": "plan.json",
                "plan_md": "plan.md",
                **({"plan_effective": "plan.effective.json"} if effective_written else {}),
                "trace": "trace.json",
                "trace_jsonl": "trace.jsonl",
                "execution_log": "execution.log",
                "server_log": "server.log",
                "sql_ledger": "sql_ledger.jsonl",
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
    arguments — e.g. the re-included list_schemas of a replan). Failed/skipped
    steps are excluded; their corrected replacements come from the later
    attempt.
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


def _short(v: Any, limit: int = 64) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


def _effect_hint(tool: str, args: dict[str, Any], data: Any) -> str:
    """A terse note on what a succeeded step produced. For writes: what now
    exists / already ran (don't redo it). For reads: a preview of the returned
    value, so the planner can correct names/bindings on replan."""
    if tool == "execute_sql":
        sql = str(args.get("sql") or "")
        kind = sql_kind(sql)
        obj = sql_object(sql)
        if kind == "create":
            what = f"{obj[0]} {obj[1]}" if obj else "object"
            return (f"{what} now EXISTS and is COMMITTED — do NOT create it "
                    f"again (IF NOT EXISTS also fine)")
        if kind == "drop":
            what = f"{obj[0]} {obj[1]}" if obj else "object"
            return f"{what} is now ABSENT — do NOT drop it again"
        if kind == "insert":
            return ("rows already INSERTED and COMMITTED — do NOT re-run this "
                    "INSERT (it would duplicate them)")
        if kind in ("update", "delete", "truncate", "alter", "grant", "revoke"):
            return f"{kind.upper()} already APPLIED and COMMITTED — do NOT re-run it"
        # read statements: preview so a replan can fix names/shapes
        if isinstance(data, list):
            return (f"returned {len(data)} row(s); first: "
                    f"{_short(data[0] if data else None, 160)}")
        if isinstance(data, str):
            return f"returned TEXT: {_short(data, 160)}"
        return f"returned: {_short(data, 160)}"
    if tool == "list_schemas" and isinstance(data, list):
        names = [d.get("schema_name") for d in data if isinstance(d, dict)][:8]
        return f"returned {len(data)} schema(s): {names}"
    if tool == "list_objects" and isinstance(data, list):
        names = [d.get("name") for d in data if isinstance(d, dict)][:8]
        return (f"returned {len(data)} {args.get('object_type', 'table')}(s) in "
                f"'{args.get('schema_name')}': {names}")
    if tool == "get_object_details" and isinstance(data, dict):
        cols = [c.get("column") for c in data.get("columns") or []
                if isinstance(c, dict)][:10]
        return f"columns of {args.get('schema_name')}.{args.get('object_name')}: {cols}"
    if isinstance(data, str):
        return f"returned TEXT: {_short(data, 180)}"
    if isinstance(data, dict):
        return f"returned object keys={list(data.keys())[:6]}"
    if isinstance(data, list):
        return f"returned list len={len(data)}"
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
            line += f"  -> ERROR: {ts.error[:300]}"
        lines.append(line)
    lines.append("")
    lines.append("SUCCESS steps are COMMITTED — their schemas/tables/rows already "
                 "exist; do not recreate or re-insert them. Re-plan from the first "
                 "FAILED step onward for the original task.")
    return "\n".join(lines)


def _by_category(report: VerificationReport) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in report.results:
        cat = r.get("category", "?")
        bucket = out.setdefault(cat, {"passed": 0, "failed": 0})
        bucket["passed" if r.get("passed") else "failed"] += 1
    return out


# Default self-check task: read-only, safe to run against any live database
# (touches only information_schema / pg_catalog state that always exists).
SELFCHECK_TASK = (
    "List all schemas in the database, then run a query that returns the "
    "current database name, the current user, and the PostgreSQL server "
    "version as three columns named dbname, usr, and version."
)


def selfcheck(settings: Settings, *, task: str | None = None) -> RunResult:
    """End-to-end live self-check on a safe, read-only task."""
    orch = Orchestrator(settings)
    return orch.run(task or SELFCHECK_TASK, max_replans=1, verify=True)


__all__ = ["Orchestrator", "RunResult", "merge_effective_plan", "selfcheck",
           "SELFCHECK_TASK"]
