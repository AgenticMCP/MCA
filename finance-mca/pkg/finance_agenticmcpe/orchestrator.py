"""Orchestrator: planner → executor → verifier loop with bounded replan.

Wires the three agents together, persists per-attempt artifacts, and
exposes a single `run` entry point. Mirrors the github orchestrator's
shape but is simpler: no GitHub-specific fork / settle-wait machinery,
no idempotency reconciliation, no SHA re-validation. The yfinance server
is read-only — every call is either a successful fetch or a hard failure.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from pkg.finance_mcp_wrapper import (
    MCPClient,
    MCPError,
)
from pkg.finance_mcp_wrapper.sequence import (
    BindingError,
    execute_sequence,
    resolve_bindings,
    validate_against_schema,
)
from pkg.finance_mcp_wrapper.types import Tool

from .config import AgenticConfig, default_config
from .executor import ExecutionAgent, ExecutionTrace, build_client, load_plan
from .llm import LLMClient, Message
from .planner import Plan, PlannerAgent, PlannerError
from .rag import KBRetriever
from .verifier import VerificationReport, VerifierAgent


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class AttemptRecord:
    n: int
    plan_path: str
    trace_path: str
    success: bool
    error_context: str | None = None
    verification_path: str | None = None


@dataclass
class RunResult:
    run_id: str
    task: str
    success: bool
    final_plan: Plan | None
    final_trace: ExecutionTrace | None
    final_verification: VerificationReport | None
    attempts: list[AttemptRecord] = field(default_factory=list)
    work_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "success": self.success,
            "work_dir": self.work_dir,
            "attempts": [
                {
                    "n": a.n,
                    "plan_path": a.plan_path,
                    "trace_path": a.trace_path,
                    "success": a.success,
                    "error_context": a.error_context,
                    "verification_path": a.verification_path,
                }
                for a in self.attempts
            ],
            "final_plan": self.final_plan.to_dict() if self.final_plan else None,
            "final_trace": self.final_trace.to_dict() if self.final_trace else None,
            "final_verification": self.final_verification.to_dict()
            if self.final_verification
            else None,
        }


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------


def plan(
    task: str,
    config: AgenticConfig | None = None,
    *,
    feedback: str | None = None,
    tools: list[Tool] | None = None,
    llm: LLMClient | None = None,
    retriever: KBRetriever | None = None,
) -> Plan:
    """One-shot planning helper. Useful when you only want a plan without
    executing or verifying (e.g. for human review)."""
    config = config or default_config()
    if tools is None:
        from .catalog import discover
        tools = discover(
            source_path=config.tool_definition_source,
            command=config.server_command,
            args=config.server_args,
            cwd=config.server_cwd,
            env=config.server_env,
            prefer="live" if not config.extract_tools_from_source else "source",
        )
    if llm is None:
        llm = LLMClient.from_env(env_var=config.env_anthropic_api_key)
    agent = PlannerAgent(llm=llm, tools=tools, retriever=retriever)
    return agent.plan(task, feedback=feedback)


def execute_sequence(
    plan_or_path: Plan | str,
    config: AgenticConfig | None = None,
    *,
    work_dir: str | None = None,
    client: MCPClient | None = None,
) -> ExecutionTrace:
    """Execute a plan against the server, with retries. Independent of
    planner/verifier so you can replay a stored plan.json verbatim."""
    config = config or default_config()
    if isinstance(plan_or_path, str):
        p = load_plan(plan_or_path)
    else:
        p = plan_or_path
    agent = ExecutionAgent(config, client=client, work_dir=work_dir)
    return agent.run(p)


def verify(
    plan_or_path: Plan | str,
    trace_or_path: ExecutionTrace | str,
    config: AgenticConfig | None = None,
    *,
    tools: list[Tool] | None = None,
    llm: LLMClient | None = None,
    client: MCPClient | None = None,
) -> VerificationReport:
    """Verify a completed trace without re-running. Independent of
    planner/executor so you can re-verify stored artifacts."""
    config = config or default_config()
    if isinstance(plan_or_path, str):
        p = load_plan(plan_or_path)
    else:
        p = plan_or_path
    if isinstance(trace_or_path, str):
        with open(trace_or_path, encoding="utf-8") as f:
            t = ExecutionTrace(**{k: v for k, v in json.load(f).items()
                                  if k in ExecutionTrace.__dataclass_fields__})
    else:
        t = trace_or_path
    if llm is None:
        llm = LLMClient.from_env(env_var=config.env_anthropic_api_key)
    agent = VerifierAgent(config, tools=tools, llm=llm, client=client)
    return agent.verify(p, t)


def load_default_retriever(config: AgenticConfig) -> KBRetriever | None:
    """Load the verified KB as the planner's RAG corpus, or ``None``.

    Best-effort by design: a missing, empty, or unreadable corpus degrades
    to pure LLM planning rather than failing the run. Override by passing
    ``retriever=`` explicitly, or disable with ``rag_enabled=False``.

    Note the taskgen pipeline does NOT go through the Orchestrator — it
    builds its planner directly with no retriever — so growing the KB stays
    an independent rediscovery loop rather than one that cites itself.
    """
    path = config.resolve_kb_path()
    if path is None:
        return None
    try:
        retriever = KBRetriever(kb_path=path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None if retriever.is_empty else retriever


def _verification_feedback(verification: VerificationReport | None) -> str:
    """Replan feedback for an attempt whose steps all ran but whose
    verification rejected the outcome."""
    if verification is None:
        return (
            "Execution completed but verification could not be produced. "
            "Re-derive the plan from the task."
        )
    failures = verification.failures
    if not failures:
        return "The attempt was rejected without a recorded reason."
    lines = [
        "Every step EXECUTED successfully, but verification rejected the "
        "result. Fix the plan so these checks hold — usually the step's "
        "`post_action_properties` claimed fields or shapes the tool does "
        "not actually return, or the sequence did not fetch everything the "
        "task asks for. Failed checks:",
    ]
    for c in failures[:20]:
        lines.append(f"- [{c.layer}] {c.name}: {c.detail or 'failed'}")
    if len(failures) > 20:
        lines.append(f"- ...and {len(failures) - 20} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Plans, executes, and verifies a single task with bounded replan."""

    def __init__(
        self,
        config: AgenticConfig | None = None,
        *,
        tools: list[Tool] | None = None,
        llm: LLMClient | None = None,
        retriever: KBRetriever | None = None,
        client: MCPClient | None = None,
    ):
        self.config = config or default_config()
        self.llm = llm or LLMClient.from_env(env_var=self.config.env_anthropic_api_key)
        self.tools = tools
        # RAG is on by default: consult the verified KB before the LLM.
        # Replans deliberately bypass it (PlannerAgent skips retrieval when
        # feedback is present) — a replan must reason about observed state,
        # not precedent.
        self.retriever = retriever if retriever is not None else load_default_retriever(self.config)
        self._client = client
        if self.tools is None:
            from .catalog import discover
            self.tools = discover(
                source_path=self.config.tool_definition_source,
                command=self.config.server_command,
                args=self.config.server_args,
                cwd=self.config.server_cwd,
                env=self.config.server_env,
                prefer="live",
            )

    # ----------------------------------------------------------------- API

    def run(
        self,
        task: str,
        *,
        run_id: str | None = None,
        work_dir: str | None = None,
        stream: bool = False,
    ) -> RunResult:
        """Run a single task end-to-end. Returns a `RunResult` with the
        successful attempt's plan, trace, and verification. On failure,
        ``error_context`` carries the last attempt's planner-replan hint."""
        run_id = run_id or f"finance_run_{int(time.time())}"
        work_dir = work_dir or os.path.join("runs", run_id)
        os.makedirs(work_dir, exist_ok=True)

        result = RunResult(
            run_id=run_id, task=task, success=False,
            final_plan=None, final_trace=None, final_verification=None,
            work_dir=work_dir,
        )

        # Spin up a single MCPClient and re-use it across attempts.
        client = self._client or build_client(self.config)
        own_client = self._client is None
        if own_client and not getattr(client, "_initialized", True):
            client.start()

        try:
            feedback: str | None = None
            for attempt_n in range(self.config.max_replans + 1):
                plan_obj = self._plan(task, feedback=feedback, work_dir=work_dir)
                plan_path = os.path.join(work_dir, f"attempt{attempt_n}-plan.json")
                if attempt_n > 0:
                    plan_path = os.path.join(work_dir, f"attempt{attempt_n}-plan.json")
                self._write_plan(plan_obj, plan_path)
                # Run the executor for this attempt.
                exec_agent = ExecutionAgent(
                    self.config,
                    client=client,
                    work_dir=work_dir,
                    log_to_console=stream,
                )
                trace = exec_agent.run(plan_obj)
                trace_path = os.path.join(work_dir, f"attempt{attempt_n}-trace.json")
                # The executor writes trace.json inside work_dir; rename.
                src = os.path.join(work_dir, "trace.json")
                if os.path.exists(src):
                    os.replace(src, trace_path)

                # Record attempt. Try to verify; record the verification too.
                ver_path = None
                verification = None
                try:
                    verification = self._verify(plan_obj, trace, client)
                    ver_path = os.path.join(work_dir, f"attempt{attempt_n}-verification.json")
                    with open(ver_path, "w", encoding="utf-8") as f:
                        json.dump(verification.to_dict(), f, indent=2, default=str)
                except Exception as e:  # noqa: BLE001 — verifier layer is best-effort.
                    ver_path = os.path.join(work_dir, f"attempt{attempt_n}-verification-error.txt")
                    with open(ver_path, "w", encoding="utf-8") as f:
                        f.write(f"verifier raised: {e!r}")

                ok = trace.success and (verification is None or verification.passed)
                result.attempts.append(
                    AttemptRecord(
                        n=attempt_n,
                        plan_path=plan_path,
                        trace_path=trace_path,
                        success=ok,
                        error_context=trace.error_context,
                        verification_path=ver_path,
                    )
                )
                if ok:
                    result.final_plan = plan_obj
                    result.final_trace = trace
                    result.final_verification = verification
                    result.success = True
                    break
                # Replan: feed back whichever stage actually rejected the
                # attempt. Reporting a verification failure as "execution
                # failed" left the planner with nothing to act on.
                feedback = trace.error_context or _verification_feedback(verification)
            else:
                # All attempts failed; surface the last attempt.
                result.final_plan = plan_obj  # type: ignore[possibly-undefined]
                result.final_trace = trace
                result.final_verification = verification

            # Persist run.json
            with open(os.path.join(work_dir, "run.json"), "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        finally:
            if own_client:
                client.close()
        return result

    # --------------------------------------------------------------- internal

    def _plan(
        self,
        task: str,
        *,
        feedback: str | None,
        work_dir: str,
        attempts: int = 2,
    ) -> Plan:
        """Plan one attempt, giving the planner a self-correction retry.

        Local validation (unknown tool, missing required param, binding to a
        step that isn't earlier in the plan) raises rather than returning a
        plan, so without this retry a single malformed LLM reply aborted the
        entire run instead of consuming one bounded attempt.
        """
        planner = PlannerAgent(llm=self.llm, tools=self.tools, retriever=self.retriever)
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

    def _verify(
        self, plan_obj: Plan, trace: ExecutionTrace, client: MCPClient
    ) -> VerificationReport:
        verifier = VerifierAgent(
            self.config,
            tools=self.tools,
            llm=self.llm,
            client=client,
        )
        return verifier.verify(plan_obj, trace)

    def _write_plan(self, plan_obj: Plan, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan_obj.to_dict(), f, indent=2, ensure_ascii=False, default=str)


__all__ = [
    "Orchestrator",
    "RunResult",
    "AttemptRecord",
    "load_default_retriever",
    "plan",
    "verify",
    "execute_sequence",
]