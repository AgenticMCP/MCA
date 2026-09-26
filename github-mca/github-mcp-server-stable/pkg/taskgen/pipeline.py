"""Benchmark pipeline: generated task -> agenticmcpe run -> accepted KB entry.

Ground-truth policy: an entry is accepted ONLY when the planner's sequence
executed end-to-end against real GitHub AND the verifier's evaluators passed.
By default runs that needed a replan are rejected too — a replanned plan
completes "from current state" (it skips resources the failed attempt already
created), so it is not a correct from-scratch sequence for the task.

Unlike ``agenticmcpe.Orchestrator``, the loop here gates every plan (initial
and replanned) BEFORE execution: readonly tasks may only execute read-only
tools, and write tasks may only create benchmark-prefixed repositories owned
by the authenticated user.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pkg.agenticmcpe.catalog import condense_catalog, load_catalog
from pkg.agenticmcpe.config import LLMClient, Settings
from pkg.agenticmcpe.executor import ExecutionAgent
from pkg.agenticmcpe.orchestrator import _replan_feedback
from pkg.agenticmcpe.planner import Plan, PlannerAgent, PlannerError
from pkg.agenticmcpe.verifier import VerifierAgent

from .generator import TaskSpec
from .kb import KnowledgeBase

# Repositories created by write-mode benchmark tasks must carry this prefix:
# it keeps them identifiable and makes cleanup safe.
BENCH_REPO_PREFIX = "agenticmcpe-bench-"

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
        # Epoch cutoff for fork cleanup: a fork is only ours to delete if GitHub
        # created it after this run began. Epoch (not the local-time run_id)
        # so the comparison against GitHub's UTC created_at is timezone-proof.
        self._run_started = time.time()
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
        created_repos = self._created_repos(plan)
        trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
        created_repos |= self._forked_repos(trace)
        replans = 0
        budget = self.max_replans if self.allow_replans else 0
        while not trace.success and replans < budget:
            replans += 1
            try:
                plan = self._plan(planner, spec.task_prompt,
                                  feedback=_replan_feedback(plan, trace))
            except PlannerError as e:
                return self._finish_reject(spec, settings, created_repos,
                                           f"replanner error: {e}")
            planner.write(plan, settings)
            gate = self._gate_plan(plan, spec, read_only)
            if gate:
                return self._finish_reject(spec, settings, created_repos,
                                           f"replan gate: {gate}")
            created_repos |= self._created_repos(plan)
            trace = ExecutionAgent(settings, log_to_console=self.log).run(plan)
            created_repos |= self._forked_repos(trace)

        if not trace.success:
            return self._finish_reject(
                spec, settings, created_repos,
                f"execution failed at step {trace.failed_step}")
        if replans > 0 and not self.allow_replans:
            return self._finish_reject(
                spec, settings, created_repos,
                "needed replans; final plan is not a from-scratch sequence")

        # --- verify (the acceptance oracle) ---
        verifier = VerifierAgent(settings, llm, tool_names=[t.name for t in catalog])
        report = verifier.verify(plan, trace)
        if self.log:
            print(f"[taskgen] verification: {report.passed}/{report.total} "
                  f"-> {'OK' if report.ok else 'FAIL'}")
        if not report.ok:
            return self._finish_reject(
                spec, settings, created_repos,
                f"verification failed: {report.passed}/{report.total} checks passed")

        # --- dedup by tool signature, then persist ---
        signature = tuple(s.tool for s in plan.steps)
        if self.kb.signature_count(signature) >= self.max_per_signature:
            return self._finish_reject(
                spec, settings, created_repos,
                f"tool signature already has {self.max_per_signature} KB entries")

        entry = self._entry(spec, plan, trace, report, replans, settings)
        if entry["quality_warnings"] and self.log:
            for w in entry["quality_warnings"]:
                print(f"[taskgen][quality] {w}")
        self.kb.add(entry)
        self._maybe_cleanup(spec, settings, created_repos)
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
        if spec.mode == "readonly":
            writers = [s.tool for s in plan.steps if not read_only.get(s.tool, False)]
            if writers:
                return f"readonly task but plan contains write tools: {writers}"
            return None
        for st in plan.steps:
            if st.tool == "create_repository":
                name = str(st.arguments.get("name", ""))
                if not name.startswith(BENCH_REPO_PREFIX):
                    return (f"step {st.id}: created repo {name!r} lacks the "
                            f"{BENCH_REPO_PREFIX!r} prefix")
            elif st.tool == "fork_repository":
                # The one write tool whose literal third-party `owner` is safe:
                # forking does not touch the source repo, it copies it into an
                # account of ours. Require that account to be the authenticated
                # user (no `organization`), so --cleanup can find the fork again.
                org = st.arguments.get("organization")
                if org is not None and not (isinstance(org, str) and org.startswith("$")):
                    return (f"step {st.id}: forks into organization {org!r}; "
                            f"must fork into the authenticated user's account")
            elif not read_only.get(st.tool, False) and "owner" in st.arguments:
                owner = st.arguments["owner"]
                if not (isinstance(owner, str) and owner.startswith("$")):
                    return (f"step {st.id} ({st.tool}): write step targets literal "
                            f"owner {owner!r}; must bind the authenticated user")
        return None

    @staticmethod
    def _created_repos(plan: Plan) -> set[str]:
        return {str(s.arguments.get("name", "")) for s in plan.steps
                if s.tool == "create_repository"
                and str(s.arguments.get("name", "")).startswith(BENCH_REPO_PREFIX)}

    @staticmethod
    def _forked_repos(trace: Any) -> set[str]:
        """Repo names this run forked. Read from the TRACE, not the plan, so
        only forks that actually succeeded are considered for cleanup. A fork
        keeps the source repo's name unless the call overrode it."""
        out: set[str] = set()
        for step in getattr(trace, "steps", []):
            if getattr(step, "tool", "") != "fork_repository":
                continue
            if getattr(step, "status", "") != "success":
                continue
            args = getattr(step, "arguments_resolved", None) or {}
            name = str(args.get("name") or args.get("repo") or "").strip()
            if name:
                out.add(name)
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
                       created_repos: set[str], reason: str) -> PipelineResult:
        """Reject after execution may have happened: clean up bench repos first."""
        self._maybe_cleanup(spec, settings, created_repos)
        return self._reject(spec, settings.run_id, reason)

    # ---------------------------------------------------------------- cleanup
    def _maybe_cleanup(self, spec: TaskSpec, settings: Settings,
                       created_repos: set[str]) -> None:
        """Best-effort deletion of benchmark repos this run created (write mode,
        --cleanup only). Requires the token to have the delete_repo scope."""
        if not (self.cleanup and spec.mode == "write" and created_repos):
            return
        try:
            token = settings.tokens.current()
            login = _gh_api("GET", "https://api.github.com/user", token)["login"]
            for name in sorted(created_repos):
                try:
                    if not name.startswith(BENCH_REPO_PREFIX) and not self._is_our_fork(
                            login, name, token):
                        continue  # pre-existing or not a fork -> never ours to delete
                    _gh_api("DELETE", f"https://api.github.com/repos/{login}/{name}",
                            token)
                    if self.log:
                        print(f"[taskgen] cleanup: deleted {login}/{name}")
                except Exception as e:
                    if self.log:
                        print(f"[taskgen] cleanup: could not delete {name}: {e}")
        except Exception as e:
            if self.log:
                print(f"[taskgen] cleanup skipped: {e}")

    def _is_our_fork(self, login: str, name: str, token: str) -> bool:
        """Whether ``login/name`` is a fork THIS run created, and so is safe to
        delete. Deliberately strict — three conditions must all hold, because a
        false positive destroys a repository the user owns:

        1. the repo exists (a 404 means nothing to clean up),
        2. GitHub reports it as a fork (never touch an ordinary repository),
        3. it was created after this run started — `fork_repository` silently
           returns the EXISTING fork when the user already had one, and that
           one predates us and must survive.
        """
        try:
            repo = _gh_api("GET", f"https://api.github.com/repos/{login}/{name}", token)
        except Exception:
            return False
        if not repo.get("fork"):
            return False
        created = str(repo.get("created_at") or "")
        try:
            epoch = time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        except ValueError:
            return False
        # Small negative tolerance for clock skew between GitHub and this host.
        if epoch < getattr(self, "_run_started", time.time()) - 120:
            if self.log:
                print(f"[taskgen] cleanup: keeping {login}/{name} "
                      f"(fork predates this run, created {created})")
            return False
        return True


def _gh_api(method: str, url: str, token: str) -> Any:
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "agenticmcpe-taskgen",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


# Common literals that legitimately recur without being derived from a result.
_BENIGN_LITERALS = {"main", "master", "readme.md", "open", "closed", "all",
                    "true", "false"}


def _quality_warnings(spec: TaskSpec, plan: Plan, trace: Any) -> list[str]:
    """Flag literal string arguments that match a PRIOR step's result but do
    not come from the task prompt — the planner likely guessed a value it
    should have bound with $sN (e.g. hardcoding a release tag). Heuristic:
    recorded as metadata so KB consumers can filter, never a rejection."""
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


__all__ = ["BENCH_REPO_PREFIX", "BenchmarkPipeline", "PipelineResult"]
