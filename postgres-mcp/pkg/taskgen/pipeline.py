"""Benchmark pipeline: generated task -> agenticmcpe run -> accepted KB entry.

Ground-truth policy: an entry is accepted ONLY when the planner's sequence
executed end-to-end against a real PostgreSQL database AND the verifier's
evaluators passed. By default runs that needed a replan are rejected too — a
replanned plan completes "from current state" (it skips objects the failed
attempt already created), so it is not a correct from-scratch sequence for the
task.

Unlike ``agenticmcpe.Orchestrator``, the loop here gates every plan (initial
and replanned) BEFORE execution: readonly tasks may only run read-only
statements, and write tasks may only touch schemas carrying the benchmark
prefix. Cleanup drops those schemas (CASCADE) — and ONLY those — afterwards.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pkg.agenticmcpe.catalog import condense_catalog, load_catalog
from pkg.agenticmcpe.config import LLMClient, Settings
from pkg.agenticmcpe.executor import ExecutionAgent, sql_kind, sql_object
from pkg.agenticmcpe.orchestrator import _replan_feedback
from pkg.agenticmcpe.planner import Plan, PlannerAgent, PlannerError
from pkg.agenticmcpe.verifier import VerifierAgent
from pkg.mcp_wrapper import MCPClient

from .generator import TaskSpec, unsupported_tools
from .kb import KnowledgeBase

# Schemas created by write-mode benchmark tasks must carry this prefix: it
# keeps them identifiable and makes cleanup (DROP SCHEMA ... CASCADE) safe.
BENCH_SCHEMA_PREFIX = "agenticmcpe_bench_"

# SQL statement kinds a readonly task's execute_sql steps may use — mirrors
# what the server's own restricted mode (SafeSqlDriver) permits: pure reads
# plus the ANALYZE/VACUUM maintenance statements (needed e.g. before the
# index advisors), which touch no user data.
_READONLY_KINDS = {"select", "show", "explain", "values", "table",
                   "analyze", "vacuum"}

_BENCH_SCHEMA_RE = re.compile(rf"\b({re.escape(BENCH_SCHEMA_PREFIX)}[a-z0-9_]*)",
                              re.IGNORECASE)

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
        cleanup: bool = False,
        max_per_signature: int = 3,
        log_to_console: bool = True,
        rejects_path: Path | None = None,
    ):
        self.kb = kb
        self.provider = provider
        self.allow_replans = allow_replans
        self.max_replans = max_replans
        self.cleanup = cleanup
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
        read_only = {c["tool"]: c["read_only"] for c in condensed}
        llm = LLMClient(settings.llm)
        planner = PlannerAgent(llm, condensed)

        # --- plan (with bounded self-correction, as the orchestrator does) ---
        try:
            plan = self._plan(planner, spec.task_prompt)
        except PlannerError as e:
            return self._reject(spec, settings.run_id, f"planner error: {e}")
        planner.write(plan, settings)
        gate = self._gate_plan(plan, spec, read_only)
        if gate:
            return self._reject(spec, settings.run_id, f"plan gate: {gate}")
        if self.log:
            print(f"[taskgen] plan: {' -> '.join(s.tool for s in plan.steps)}")

        # --- execute, replanning only if replans are allowed ---
        bench_schemas = self._bench_schemas(plan)
        trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
        replans = 0
        budget = self.max_replans if self.allow_replans else 0
        while not trace.success and replans < budget:
            replans += 1
            try:
                plan = self._plan(planner, spec.task_prompt,
                                  feedback=_replan_feedback(plan, trace))
            except PlannerError as e:
                return self._finish_reject(spec, settings, bench_schemas,
                                           f"replanner error: {e}")
            planner.write(plan, settings)
            gate = self._gate_plan(plan, spec, read_only)
            if gate:
                return self._finish_reject(spec, settings, bench_schemas,
                                           f"replan gate: {gate}")
            bench_schemas |= self._bench_schemas(plan)
            trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)

        if not trace.success:
            return self._finish_reject(
                spec, settings, bench_schemas,
                f"execution failed at step {trace.failed_step}")
        if replans > 0 and not self.allow_replans:
            return self._finish_reject(
                spec, settings, bench_schemas,
                "needed replans; final plan is not a from-scratch sequence")

        # --- verify (the acceptance oracle) ---
        verifier = VerifierAgent(settings, llm, tool_names=[t.name for t in catalog])
        report = verifier.verify(plan, trace)
        if self.log:
            print(f"[taskgen] verification: {report.passed}/{report.total} "
                  f"-> {'OK' if report.ok else 'FAIL'}")
        if not report.ok:
            return self._finish_reject(
                spec, settings, bench_schemas,
                f"verification failed: {report.passed}/{report.total} checks passed")

        # --- dedup by tool signature, then persist ---
        signature = tuple(s.tool for s in plan.steps)
        if self.kb.signature_count(signature) >= self.max_per_signature:
            return self._finish_reject(
                spec, settings, bench_schemas,
                f"tool signature already has {self.max_per_signature} KB entries")

        entry = self._entry(spec, plan, trace, report, replans, settings)
        if entry["quality_warnings"] and self.log:
            for w in entry["quality_warnings"]:
                print(f"[taskgen][quality] {w}")
        self.kb.add(entry)
        self._maybe_cleanup(spec, settings, bench_schemas)
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

    def _gate_plan(self, plan: Plan, spec: TaskSpec,
                   read_only: dict[str, bool]) -> str | None:
        """Safety gate applied BEFORE any execution. Returns a reason or None."""
        unsupported = unsupported_tools()
        for st in plan.steps:
            if st.tool in unsupported:
                return (f"step {st.id}: tool {st.tool!r} needs a PostgreSQL "
                        f"extension this database does not declare "
                        f"(AGENTICMCPE_PG_EXTENSIONS)")
        if spec.mode == "readonly":
            for st in plan.steps:
                if st.tool == "execute_sql":
                    sql = str(st.arguments.get("sql", ""))
                    kind = sql_kind(sql)
                    if kind not in _READONLY_KINDS:
                        return (f"step {st.id}: readonly task but sql is a "
                                f"{kind.upper()} statement")
                elif not read_only.get(st.tool, False):
                    return f"readonly task but plan contains write tool: {st.tool}"
            return None
        # write mode: every write statement must live inside a bench schema
        for st in plan.steps:
            if st.tool != "execute_sql":
                continue
            sql = str(st.arguments.get("sql", ""))
            kind = sql_kind(sql)
            if kind in _READONLY_KINDS:
                continue
            if kind == "create":
                obj = sql_object(sql)
                if obj and obj[0] == "schema" and not obj[1].strip('"').lower() \
                        .startswith(BENCH_SCHEMA_PREFIX):
                    return (f"step {st.id}: creates schema {obj[1]!r} without "
                            f"the {BENCH_SCHEMA_PREFIX!r} prefix")
            if not _BENCH_SCHEMA_RE.search(sql):
                return (f"step {st.id}: write statement does not target an "
                        f"{BENCH_SCHEMA_PREFIX}* schema: {sql[:80]!r}")
        return None

    @staticmethod
    def _bench_schemas(plan: Plan) -> set[str]:
        """Every benchmark-prefixed schema name the plan's SQL mentions —
        the cleanup candidates. The prefix requirement makes this safe: no
        pre-existing schema can legally carry it."""
        out: set[str] = set()
        for st in plan.steps:
            if st.tool != "execute_sql":
                continue
            for m in _BENCH_SCHEMA_RE.finditer(str(st.arguments.get("sql", ""))):
                out.add(m.group(1).lower())
        return out

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

    def _finish_reject(self, spec: TaskSpec, settings: Settings,
                       bench_schemas: set[str], reason: str) -> PipelineResult:
        """Reject after execution may have happened: clean up bench schemas first."""
        self._maybe_cleanup(spec, settings, bench_schemas)
        return self._reject(spec, settings.run_id, reason)

    # ---------------------------------------------------------------- cleanup
    def _maybe_cleanup(self, spec: TaskSpec, settings: Settings,
                       bench_schemas: set[str]) -> None:
        """Best-effort DROP SCHEMA ... CASCADE of benchmark schemas this run
        touched (write mode, --cleanup only). Deliberately strict: a schema is
        only ever dropped when its name carries the benchmark prefix — nothing
        pre-existing can legally carry it, so nothing pre-existing can be hit."""
        if not (self.cleanup and spec.mode == "write" and bench_schemas):
            return
        try:
            with MCPClient(
                database_uri=settings.database.current(),
                server_cmd=settings.server_cmd or None,
                access_mode="unrestricted",
            ) as pg:
                for name in sorted(bench_schemas):
                    if not name.startswith(BENCH_SCHEMA_PREFIX):
                        continue  # paranoia: never drop anything unprefixed
                    if not re.fullmatch(r"[a-z0-9_]+", name):
                        continue  # identifier chars only — nothing to escape
                    # A bench-prefixed identifier is either a dedicated schema
                    # or an advisor-task TABLE in public (the index advisors
                    # only see search_path tables). IF EXISTS makes the
                    # non-applicable drop a no-op.
                    for sql in (f"DROP SCHEMA IF EXISTS {name} CASCADE",
                                f"DROP TABLE IF EXISTS public.{name} CASCADE"):
                        try:
                            pg.call("execute_sql", {"sql": sql})
                        except Exception as e:
                            if self.log:
                                print(f"[taskgen] cleanup: {sql!r} failed: {e}")
                    if self.log:
                        print(f"[taskgen] cleanup: dropped {name}")
        except Exception as e:
            if self.log:
                print(f"[taskgen] cleanup skipped: {e}")


# Common literals that legitimately recur without being derived from a result.
_BENIGN_LITERALS = {"public", "table", "view", "sequence", "extension", "all",
                    "true", "false", "index", "connection", "vacuum",
                    "replication", "buffer", "constraint", "total_time",
                    "mean_time", "resources", "dta", "llm"}


def _quality_warnings(spec: TaskSpec, plan: Plan, trace: Any) -> list[str]:
    """Flag literal string arguments that match a PRIOR step's result but do
    not come from the task prompt — the planner likely guessed a value it
    should have bound with $sN (e.g. hardcoding a discovered table name).
    Heuristic: recorded as metadata so KB consumers can filter, never a
    rejection."""
    warnings: list[str] = []
    prompt_fold = spec.task_prompt.casefold()
    trace_by_id = {s.id: s for s in trace.steps}
    prior_blobs: list[tuple[str, str]] = []
    for st in plan.steps:
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
                        f"{st.id}.{key}={val!r} matches {pid}'s result but is not "
                        "in the task prompt; likely should be a $-binding")
                    break
        ts = trace_by_id.get(st.id)
        if ts is not None and ts.result_data is not None:
            prior_blobs.append((st.id, json.dumps(ts.result_data, default=str)))
    return warnings


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


__all__ = ["BENCH_SCHEMA_PREFIX", "BenchmarkPipeline", "PipelineResult"]
