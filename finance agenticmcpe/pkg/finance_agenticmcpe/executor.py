"""Execution agent: replays a plan's tool-call sequence verbatim against the
yahoo_finance MCP server (via pkg.finance_mcp_wrapper).

Contract (mirrors the github executor):

* The sequence is executed STRICTLY in order and is NEVER altered. The
  executor does not add, drop, or reorder steps.
* Each step gets up to N attempts on *transient* failures
  (tool/protocol/transport errors). Deterministic failures
  (binding/schema) fail fast — a retry cannot help; that goes back to
  the planner.
* Full logging: a human ``execution.log``, a structured ``trace.json``,
  and the server's own captured stderr.
* On persistent failure the executor returns an ``error_context``
  string for the planner to revise the plan; it does not try to "fix"
  the plan itself.

Differences from the github executor:

* No idempotency machinery. The yfinance server is read-only: every
  tool returns the same answer on a re-call, so neither on-error
  healing nor pre-flight dedup applies. There's nothing to reconcile.
* No settle-wait machinery. There is no API-level eventual consistency
  on read-only endpoints.
* No SHA extraction, no path-disambiguation healing, no GitHub-shaped
  result-shape fixes — those are git-specific.

The result_shape_fixes module handles a small set of yfinance-specific
quirks: the server sometimes returns a tuple instead of a list, plain
text where JSON is expected, etc.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from pkg.finance_mcp_wrapper import (
    MCPClient,
    MCPProtocolError,
    MCPToolError,
    MCPTransportError,
)
from pkg.finance_mcp_wrapper.sequence import (
    BindingError,
    resolve_bindings,
    validate_against_schema,
)

from .config import AgenticConfig
from .planner import Plan, PlanStep

MAX_ATTEMPTS = 3
# Statuses for which retrying makes sense (could be flaky network or a
# transient server hiccup).
_TRANSIENT = {"tool_error", "protocol_error", "transport_error"}


@dataclass
class Attempt:
    n: int
    status: str
    error: str | None
    duration_s: float


@dataclass
class StepExecution:
    id: str
    tool: str
    status: str = "pending"
    arguments_resolved: dict[str, Any] | None = None
    attempts: list[Attempt] = field(default_factory=list)
    result_data: Any = None
    result_content: list[dict[str, Any]] | None = None
    error: str | None = None
    error_details: list[str] | None = None
    post_action_properties: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class ExecutionTrace:
    task: str
    success: bool = False
    steps: list[StepExecution] = field(default_factory=list)
    failed_step: str | None = None
    error_context: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    server_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Result-shape fixes (small, yfinance-specific surface area)
# ---------------------------------------------------------------------------


def _normalize_result(tool: str, data: Any) -> Any:
    """A few light fixes for quirks in the yfinance server's outputs:

    * ``get_historical_stock_prices`` returns a tuple in some yfinance
      versions; coerce to a list so downstream bindings like ``$sN[0].open``
      work uniformly.
    * Some tools occasionally embed JSON in a quoted wrapper; if the raw
      string is parseable JSON, use the parsed value.

    The orchestrator surfaces these via ``Trace`` so they show up in the
    trace.jsonl telemetry.
    """
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return data


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def build_client(config: AgenticConfig, *, client: MCPClient | None = None) -> MCPClient:
    """Construct (not start) an MCPClient wired to the configured server.

    Pass ``client=`` to inject an already-started client (the
    orchestrator does this when re-using a single connection across
    replans).
    """
    if client is not None:
        return client
    return MCPClient(
        command=config.server_command,
        args=list(config.server_args),
        cwd=config.server_cwd,
        env=dict(config.server_env),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ExecutionAgent:
    """Replays a `Plan` step-by-step against a yfinance MCP server."""

    def __init__(
        self,
        config: AgenticConfig,
        *,
        client: MCPClient | None = None,
        max_attempts: int | None = None,
        work_dir: str | None = None,
        log_to_console: bool = True,
    ):
        self.config = config
        self._client = client
        self._owns_client = client is None
        self.max_attempts = max_attempts or config.max_step_retries or MAX_ATTEMPTS
        self.work_dir = work_dir
        self.log = self._make_logger(log_to_console)
        self._jsonl: Any = None
        if work_dir:
            import os
            os.makedirs(work_dir, exist_ok=True)
            self._jsonl = open(os.path.join(work_dir, "trace.jsonl"), "a", encoding="utf-8")

    # -------------------------------------------------------------- logging

    def _make_logger(self, to_console: bool) -> logging.Logger:
        logger = logging.getLogger("finance_agenticmcpe.exec")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
        if self.work_dir:
            import os
            fh = logging.FileHandler(
                os.path.join(self.work_dir, "execution.log"), encoding="utf-8"
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        if to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
        logger.propagate = False
        return logger

    def _event(self, **fields: Any) -> None:
        if self._jsonl:
            self._jsonl.write(
                json.dumps({"ts": time.time(), **fields}, default=str) + "\n"
            )
            self._jsonl.flush()

    # ---------------------------------------------------------------- public

    def run(self, plan: Plan) -> ExecutionTrace:
        trace = ExecutionTrace(task=plan.task, started_at=time.time())
        client = self._client or build_client(self.config)
        self.log.info("=== EXECUTION START: %d step(s) ===", len(plan.steps))
        self._event(event="execution_start", steps=len(plan.steps), task=plan.task)

        started_here = False
        try:
            if not getattr(client, "_initialized", True):
                client.start()
                started_here = True
            outputs: dict[str, Any] = {}
            aborted = False
            for step in plan.steps:
                se = self._run_step(client, step, outputs, aborted)
                trace.steps.append(se)
                if se.status == "success":
                    outputs[step.id] = se.result_data
                elif se.status != "skipped":
                    aborted = True
                    trace.failed_step = step.id
            trace.success = all(s.status == "success" for s in trace.steps) and bool(trace.steps)
            if not trace.success and trace.failed_step:
                trace.error_context = self._error_context(trace)
            trace.server_stderr = getattr(client, "stderr", "")
        finally:
            if started_here and self._owns_client:
                client.close()
            trace.finished_at = time.time()
            self._finalize(trace)
        return trace

    # --------------------------------------------------------------- internal

    def _run_step(
        self,
        client: Any,
        step: PlanStep,
        outputs: dict[str, Any],
        aborted: bool,
    ) -> StepExecution:
        se = StepExecution(
            id=step.id,
            tool=step.tool,
            description=step.description,
            post_action_properties=step.post_action_properties,
        )
        if aborted:
            se.status = "skipped"
            se.error = "a previous step failed; sequence aborted"
            self.log.warning("[%s] SKIPPED (%s): prior failure", step.id, step.tool)
            self._event(event="step_skipped", id=step.id, tool=step.tool)
            return se

        # 1. binding resolution (deterministic — no retry)
        try:
            resolved = resolve_bindings(step.arguments, outputs)
        except BindingError as e:
            se.status = "binding_error"
            se.error = str(e)
            self.log.error("[%s] BINDING ERROR (%s): %s", step.id, step.tool, e)
            self._event(event="binding_error", id=step.id, error=str(e))
            return se
        if not isinstance(resolved, dict):
            se.status = "binding_error"
            se.error = f"resolved arguments are {type(resolved).__name__}, expected object"
            return se
        se.arguments_resolved = resolved

        # 2. schema validation (deterministic — no retry)
        tool = client.get_tool(step.tool)
        if tool is None:
            se.status = "validation_error"
            se.error = f"unknown tool {step.tool!r}"
            self.log.error("[%s] UNKNOWN TOOL %r", step.id, step.tool)
            return se
        errs = validate_against_schema(resolved, tool.input_schema)
        if errs:
            se.status = "validation_error"
            se.error = f"input failed schema validation ({len(errs)} issue(s))"
            se.error_details = errs
            self.log.error(
                "[%s] VALIDATION ERROR (%s): %s",
                step.id, step.tool, "; ".join(errs),
            )
            self._event(event="validation_error", id=step.id, errors=errs)
            return se

        # 3. invoke with retries on transient failure
        self.log.info(
            "[%s] CALL %s args=%s",
            step.id, step.tool, json.dumps(resolved, default=str),
        )
        for n in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                result = client.call(step.tool, resolved)
                dur = time.time() - t0
                data = _normalize_result(step.tool, result.data)
                # The yfinance tools report failure by RETURNING an error
                # string, not by setting isError — so a bad ticker or an
                # empty date range would otherwise be recorded as a success
                # carrying prose instead of data.
                server_error = _server_error_text(data)
                if server_error:
                    raise MCPToolError(tool=step.tool, text=server_error, raw={})
                se.attempts.append(Attempt(n, "success", None, dur))
                se.status = "success"
                se.result_data = data
                se.result_content = result.content
                self.log.info("[%s] OK (attempt %d, %.2fs)", step.id, n, dur)
                self._event(
                    event="step_success",
                    id=step.id,
                    attempt=n,
                    data_preview=_preview(data),
                )
                return se
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                dur = time.time() - t0
                msg = getattr(e, "text", None) or str(e)
                # yfinance read-side hint: a yfinance NoDataError arrives as
                # a tool-level message like "No data found, symbol may be
                # delisted" or "No price data found, symbol may be delisted".
                # These are usually a wrong ticker or an out-of-range date —
                # actionable for the planner, not transient. We surface the
                # hint in the error so the replanner can swap the argument.
                hint = _finance_error_hint(msg)
                if hint and hint not in msg:
                    msg = f"{msg} | HINT: {hint}"
                kind = {
                    MCPToolError: "tool_error",
                    MCPProtocolError: "protocol_error",
                    MCPTransportError: "transport_error",
                }[type(e)]
                se.attempts.append(Attempt(n, kind, msg, dur))
                se.status = kind
                se.error = msg
                self.log.warning(
                    "[%s] %s on attempt %d/%d: %s",
                    step.id, kind.upper(), n, self.max_attempts, msg,
                )
                self._event(
                    event="step_attempt_failed",
                    id=step.id,
                    attempt=n,
                    kind=kind,
                    error=msg,
                )
                if kind not in _TRANSIENT or n == self.max_attempts:
                    self.log.error(
                        "[%s] FAILED after %d attempt(s): %s", step.id, n, msg,
                    )
                    self._event(
                        event="step_failed", id=step.id, kind=kind, error=msg,
                    )
                    return se
                time.sleep(min(2 ** (n - 1), 5))  # small backoff
        return se

    def _error_context(self, trace: ExecutionTrace) -> str:
        fs = next((s for s in trace.steps if s.id == trace.failed_step), None)
        if fs is None:
            return "execution failed"
        parts = [
            f"Step {fs.id} (tool '{fs.tool}') failed with status '{fs.status}'.",
            f"Error: {fs.error}",
        ]
        if fs.error_details:
            parts.append("Details: " + "; ".join(fs.error_details))
        if fs.arguments_resolved is not None:
            parts.append(
                "Resolved arguments: " + json.dumps(fs.arguments_resolved, default=str)
            )
        # Surface the most recent successful step's data preview so the
        # replanner can pick up from it.
        for s in reversed(trace.steps):
            if s.status == "success" and s.result_data is not None:
                parts.append(
                    f"Last successful step ({s.id} / {s.tool}) output preview: "
                    + _preview(s.result_data, limit=400)
                )
                break
        return " ".join(parts)

    def _finalize(self, trace: ExecutionTrace) -> None:
        if self._jsonl:
            self._jsonl.close()
            self._jsonl = None
        if self.work_dir:
            import os
            out = os.path.join(self.work_dir, "trace.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(trace.to_dict(), f, indent=2, default=str)
            status = "SUCCESS" if trace.success else f"FAILED at {trace.failed_step}"
            self.log.info(
                "=== EXECUTION %s (%.2fs) -> %s ===",
                status, trace.finished_at - trace.started_at, out,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FINANCE_ERROR_HINTS: tuple[tuple[str, str], ...] = (
    (
        "no data found",
        "ticker may be delisted or the date range may be empty; "
        "verify the ticker (case-insensitive) and widen the date range.",
    ),
    (
        "no price data found",
        "ticker may be delisted or the date range may be empty; "
        "verify the ticker and the date range covers a trading day.",
    ),
    (
        "symbol may be delisted",
        "ticker likely delisted; verify against another source or "
        "swap to a current ticker.",
    ),
    (
        "failed to decrypt",
        "yfinance occasionally returns this when its CDN is rate-limiting; "
        "replan with a slightly different date range or retry on a fresh subprocess.",
    ),
    (
        "delisted",
        "ticker is delisted; pick a successor or use a broader date range.",
    ),
)


# The server's failure modes are plain-text returns (see servers/yahoo_finance
# /server.py): "Company ticker X not found.", "Error: getting ... for X: ...",
# "No news found for company ...", "Error: No options available for the date".
_SERVER_ERROR_PREFIXES = ("error:",)
_SERVER_ERROR_MARKERS = (
    "not found.",
    "no news found for company",
    "no options available for the date",
)


def _server_error_text(data: Any) -> str:
    """Return the error message when a tool result is one of the server's
    plain-text failure strings, else ""."""
    if not isinstance(data, str):
        return ""
    s = data.strip()
    if not s or len(s) > 400:  # real text payloads (news) are long
        return ""
    low = s.lower()
    if low.startswith(_SERVER_ERROR_PREFIXES) or any(m in low for m in _SERVER_ERROR_MARKERS):
        return s
    return ""


def _finance_error_hint(msg: str) -> str:
    low = msg.lower()
    for needle, hint in _FINANCE_ERROR_HINTS:
        if needle in low:
            return hint
    return ""


def _preview(data: Any, limit: int = 240) -> str:
    if isinstance(data, str):
        s = data
    else:
        s = json.dumps(data, default=str)
    return s[:limit] + ("..." if len(s) > limit else "")


def load_plan(path: str) -> Plan:
    with open(path, encoding="utf-8") as f:
        return Plan.from_dict(json.load(f))


__all__ = [
    "ExecutionAgent",
    "ExecutionTrace",
    "StepExecution",
    "Attempt",
    "build_client",
    "load_plan",
    "MAX_ATTEMPTS",
]