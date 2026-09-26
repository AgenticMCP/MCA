"""Execution agent: replays a plan's tool-call sequence verbatim against the
real postgres-mcp server (via pkg.mcp_wrapper), with retries and full logging.

Contract (from the workflow spec):
* The sequence is executed STRICTLY in order and is NEVER altered. The executor
  does not add, drop, or reorder steps.
* Each step gets up to 3 attempts on *transient* failures (tool/protocol/
  transport errors). Deterministic failures (binding/schema) fail fast — a retry
  cannot help; that goes straight back to the planner.
* Everything is logged: a human ``execution.log``, a structured ``trace.json``,
  and the server's captured stderr (``server.log``).
* On persistent failure the executor returns an ``error_context`` string for the
  planner to revise the plan; it does not try to "fix" the plan itself.
* Idempotent state reconciliation (re-runs / replans), the PostgreSQL profile:
  - CREATE-ish statements rejected with "already exists" (duplicate_table /
    duplicate_schema / duplicate_object ...) are adopted as idempotent
    successes — the desired object is there. The planner is told to prefer
    IF NOT EXISTS / OR REPLACE so this path is rarely needed.
  - DROP statements rejected with "does not exist" are adopted too — absence
    IS the desired end state.
  - INSERTs rejected with "duplicate key value violates unique constraint"
    are adopted — the keyed row already exists (content correctness is the
    verifier's job, which re-queries the live rows).
  - DML that PostgreSQL accepts silently on a re-run (an INSERT without a
    unique key duplicates its rows; a relative UPDATE re-applies) is guarded
    by the **SQL ledger**: every successful write statement is recorded in
    ``work_dir/sql_ledger.jsonl``, and a later attempt (a replan constructs a
    fresh ExecutionAgent over the same work dir) pre-flight-skips an INSERT/
    UPDATE/DELETE whose normalized SQL exactly matches a statement an EARLIER
    attempt already executed. In-plan intentional duplicates are exempt (the
    ledger only blocks cross-attempt replays).
* There is NO settle-wait machinery: PostgreSQL is strongly consistent — a
  committed write is visible to the next statement — so the GitHub-side
  eventual-consistency polling has no analog here.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from pkg.mcp_wrapper import (
    MCPClient,
    MCPProtocolError,
    MCPToolError,
    MCPTransportError,
)
from pkg.mcp_wrapper.sequence import (
    BindingError,
    resolve_bindings,
    validate_against_schema,
)

from .config import Settings
from .planner import Plan, PlanStep

MAX_ATTEMPTS = 3
# Statuses for which retrying makes sense (could be a flaky connection).
_TRANSIENT = {"tool_error", "protocol_error", "transport_error"}

# SQL statement kinds considered read-only: never ledgered, never skipped.
# ANALYZE/VACUUM are repeatable maintenance (the server's restricted mode
# allows them) — replaying them is harmless, so they are not ledgered either.
_READ_KINDS = {"select", "show", "explain", "values", "table",
               "analyze", "vacuum"}
# Kinds whose exact re-execution across attempts is unsafe (silent duplication
# or re-application) — these are the ledger's pre-flight-skip scope.
_LEDGER_SKIP_KINDS = {"insert", "update", "delete"}

# Error markers for the three benign end-state cases (matched lowercased).
_EXISTS_MARKER = "already exists"
_ABSENT_MARKER = "does not exist"
_DUPKEY_MARKER = "duplicate key value violates unique constraint"

# Extension-dependency messages some tools return as a NORMAL text result (not
# an "Error:"): the intended analysis did not happen, so the step must fail —
# deterministically, without burning retries — and steer the replan.
_EXTENSION_MISSING_MARKERS = (
    "hypopg extension is not installed",
    "hypopg extension to be installed",
    "pg_stat_statements",
)

_WS_RE = re.compile(r"\s+")

# CREATE/DROP object naming, for benign-result echoes and effect hints.
_CREATE_RE = re.compile(
    r"^\s*create\s+(?:or\s+replace\s+)?(?:unique\s+)?(?:temp(?:orary)?\s+)?"
    r"(table|view|materialized\s+view|schema|sequence|index|function|extension"
    r"|type|domain)\s+"
    r"(?:concurrently\s+)?(?:if\s+not\s+exists\s+)?([\w.\"$]+)", re.IGNORECASE)
_DROP_RE = re.compile(
    r"^\s*drop\s+(table|view|materialized\s+view|schema|sequence|index|function"
    r"|extension|type|domain)\s+"
    r"(?:concurrently\s+)?(?:if\s+exists\s+)?([\w.\"$]+)", re.IGNORECASE)
_INSERT_RE = re.compile(r"^\s*insert\s+into\s+([\w.\"$]+)", re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    """Whitespace-collapsed, trailing-semicolon-stripped form used as the SQL
    ledger identity. Case is preserved — string literals are case-sensitive,
    and a planner that changes anything at all produces a different statement."""
    return _WS_RE.sub(" ", sql).strip().rstrip(";").strip()


def sql_kind(sql: str) -> str:
    """The statement's leading keyword, lowercased ('create', 'insert', ...).
    A WITH ... prelude is classified by what it feeds: DML when the body
    contains insert/update/delete, else select."""
    head = sql.lstrip().split(None, 1)
    kind = head[0].lower().rstrip(";") if head else ""
    if kind == "with":
        low = sql.lower()
        for k in ("insert", "update", "delete"):
            if re.search(rf"\)\s*{k}\s|\s{k}\s+into\s|\s{k}\s+from\s", low):
                return k
        return "select"
    return kind


def sql_object(sql: str) -> tuple[str, str] | None:
    """(object_type, object_name) for CREATE/DROP statements, else None."""
    for rx in (_CREATE_RE, _DROP_RE):
        m = rx.match(sql)
        if m:
            return _WS_RE.sub(" ", m.group(1).lower()), m.group(2).strip('"')
    return None


def _is_write_kind(kind: str) -> bool:
    return bool(kind) and kind not in _READ_KINDS


def _parse_prefixed_payload(text: str) -> Any | None:
    """Structured payload from a text that PREFIXES a Python/JSON repr with a
    prose line (get_top_queries does this). None when nothing decodes."""
    from pkg.mcp_wrapper.types import parse_result_text
    for opener in ("[", "{"):
        i = text.find(opener)
        if i > 0:
            parsed = parse_result_text(text[i:])
            if not isinstance(parsed, str):
                return parsed
    return None


def _benign_exists_result(sql: str) -> dict[str, Any]:
    """Minimal truthful result for a CREATE whose object already existed: echo
    what identifies it. A bare None here would break later "$sN..." bindings
    into the idempotent step (BindingError -> pointless replan)."""
    obj = sql_object(sql)
    out: dict[str, Any] = {"status": "already_exists"}
    if obj:
        out["object_type"], out["object_name"] = obj
    return out


def _replan_hint(tool: str, resolved: dict[str, Any], msg: str) -> str:
    """Translate well-known PostgreSQL failures into actionable guidance
    carried in the error text the replanner reads."""
    low = msg.lower()
    if ("workload analysis" in low
            and ("raw_bloat" in low or "invalid input syntax" in low
                 or "hypopg:" in low or "explain plan" in low)):
        return (" | HINT: the workload advisor choked while parsing the "
                "RECORDED query history (another tool's own internal query, "
                "not this task's work). Re-plan: first run execute_sql with "
                "exactly SELECT pg_stat_statements_reset(); then re-run the "
                "advisor — its input is the recorded workload, which the "
                "reset clears.")
    if (tool in ("analyze_query_indexes", "analyze_workload_indexes")
            and "relation" in low and _ABSENT_MARKER in low):
        return (" | HINT: the index advisors resolve tables through the "
                "SEARCH PATH, so they cannot see a table inside a custom "
                "schema. Re-plan with the working table created in the "
                "PUBLIC schema (a bench-prefixed TABLE name) and referenced "
                "UNQUALIFIED in the analyzed queries.")
    if "relation" in low and _ABSENT_MARKER in low:
        return (" | HINT: the table/view name or schema is wrong, or it was "
                "never created. Look first: list_schemas, then "
                "list_objects(schema_name), and use the EXACT schema-qualified "
                "name they return. Unquoted identifiers fold to lowercase.")
    if "column" in low and _ABSENT_MARKER in low:
        return (" | HINT: the column name is wrong. Call get_object_details("
                "schema, table) and use a column it actually lists.")
    if "syntax error" in low:
        return (" | HINT: fix the SQL syntax. Keep ONE statement per "
                "execute_sql step and do not include a trailing explanation.")
    if ("read-only" in low or "only select" in low or "not allowed" in low
            or "error validating query" in low):
        return (" | HINT: the server is in restricted (read-only) mode — "
                "write statements cannot run in this session.")
    if "cannot use analyze and hypothetical" in low:
        return (" | HINT: explain_query rejects analyze=true combined with "
                "hypothetical_indexes; drop one of them.")
    for marker in _EXTENSION_MISSING_MARKERS:
        if marker in low:
            return (" | HINT: a required PostgreSQL extension is not installed "
                    "in this database. Re-plan WITHOUT the tool that needs it "
                    "(use explain_query / plain SQL instead).")
    return ""


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
        d = asdict(self)
        return d


def build_client(settings: Settings, *, read_only: bool = False) -> MCPClient:
    """Construct (not start) an MCPClient wired to the resolved server command,
    with captured server stderr pointed at ``work_dir/server.log``. Read-only
    means spawning the server in its restricted access mode."""
    settings.ensure_work_dir()
    return MCPClient(
        database_uri=settings.database.current(),
        server_cmd=settings.server_cmd or None,
        access_mode="restricted" if read_only else "unrestricted",
        log_file=settings.work_dir / "server.log",
    )


class ExecutionAgent:
    def __init__(self, settings: Settings, *, client: MCPClient | None = None,
                 max_attempts: int = MAX_ATTEMPTS, log_to_console: bool = True):
        self.settings = settings
        self._client = client  # injectable (real MCPClient, or a fake for tests)
        self._owns_client = client is None
        self.max_attempts = max_attempts
        self.log = self._make_logger(log_to_console)
        self._jsonl = (settings.work_dir / "trace.jsonl").open("a", encoding="utf-8") \
            if settings.work_dir else None
        # SQL ledger: normalized write statements EARLIER attempts of this run
        # already executed (loaded from work_dir/sql_ledger.jsonl). A replanned
        # attempt that re-includes an identical INSERT/UPDATE/DELETE is
        # pre-flight-skipped instead of silently duplicating its effect.
        # Statements executed by THIS attempt go to _own_sql, not _prior_sql,
        # so intentional in-plan duplicates still run.
        self._prior_sql: set[str] = set()
        self._own_sql: set[str] = set()
        self._ledger_path = (settings.work_dir / "sql_ledger.jsonl"
                             if settings.work_dir else None)
        if self._ledger_path is not None and self._ledger_path.is_file():
            for line in self._ledger_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("sql_norm"), str):
                    self._prior_sql.add(rec["sql_norm"])

    # ------------------------------------------------------------------ logging
    def _make_logger(self, to_console: bool) -> logging.Logger:
        self.settings.ensure_work_dir()
        logger = logging.getLogger(f"agenticmcpe.exec.{self.settings.run_id}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
        fh = logging.FileHandler(self.settings.work_dir / "execution.log", encoding="utf-8")
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
            self._jsonl.write(json.dumps({"ts": time.time(), **fields}, default=str) + "\n")
            self._jsonl.flush()

    # ------------------------------------------------------------------ public
    def run(self, plan: Plan) -> ExecutionTrace:
        trace = ExecutionTrace(task=plan.task, started_at=time.time())
        client = self._client or build_client(self.settings)
        self.log.info("=== EXECUTION START: %d step(s) ===", len(plan.steps))
        self._event(event="execution_start", steps=len(plan.steps), task=plan.task)

        started_here = False
        try:
            if not getattr(client, "_started", True):  # real MCPClient lazy start
                client.start()
                started_here = True
            outputs: dict[str, Any] = {}
            aborted = False
            for step in plan.steps:
                se = self._run_step(client, step, outputs, aborted)
                trace.steps.append(se)
                if se.status == "success":
                    outputs[step.id] = se.result_data
                    self._note_sql_write(step, se)
                elif se.status != "skipped":
                    aborted = True
                    trace.failed_step = step.id
            trace.success = all(s.status == "success" for s in trace.steps) and bool(trace.steps)
            if not trace.success and trace.failed_step:
                trace.error_context = self._error_context(trace)
            trace.server_stderr = getattr(client, "stderr_output", "")
        finally:
            if started_here and self._owns_client:
                client.close()
            trace.finished_at = time.time()
            self._finalize(trace)
        return trace

    # ------------------------------------------------------------------ internal
    def _run_step(self, client: Any, step: PlanStep, outputs: dict[str, Any],
                  aborted: bool) -> StepExecution:
        se = StepExecution(
            id=step.id, tool=step.tool, description=step.description,
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
            self.log.error("[%s] VALIDATION ERROR (%s): %s", step.id, step.tool, "; ".join(errs))
            self._event(event="validation_error", id=step.id, errors=errs)
            return se

        # 2.5 pre-flight state reconciliation: an INSERT/UPDATE/DELETE whose
        # normalized SQL an EARLIER attempt of this run already executed is
        # adopted instead of re-run — SQL re-applies DML silently (no "already
        # exists" error to heal from), so replays would pile up duplicate rows.
        pf = self._preflight_ledger(step, resolved, se)
        if pf is not None:
            return pf

        # 3. invoke with retries on transient failure
        self.log.info("[%s] CALL %s args=%s", step.id, step.tool, json.dumps(resolved, default=str))
        sql = str(resolved.get("sql") or "") if step.tool == "execute_sql" else ""
        kind = sql_kind(sql) if sql else ""
        for n in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                result = client.call(step.tool, resolved)
                dur = time.time() - t0
                # Result sanity: some analysis tools report a MISSING EXTENSION
                # as a normal text result. The intended analysis did not happen,
                # so this is a deterministic failure — fail fast with a steering
                # hint so the replanner drops the tool, instead of retrying.
                data = result.data
                if (step.tool in ("explain_query", "get_top_queries",
                                  "analyze_workload_indexes", "analyze_query_indexes")
                        and isinstance(data, str)
                        and any(m in data.lower() for m in _EXTENSION_MISSING_MARKERS)
                        and ("not installed" in data.lower()
                             or "install" in data.lower())):
                    msg = data.strip()[:400] + _replan_hint(step.tool, resolved, data)
                    se.attempts.append(Attempt(n, "tool_error", msg, dur))
                    se.status = "tool_error"
                    se.error = msg
                    se.result_data = data
                    self.log.warning("[%s] %s reports a missing extension -> "
                                     "failing for replan", step.id, step.tool)
                    self._event(event="step_failed", id=step.id, kind="tool_error",
                                error=msg)
                    return se
                # ...and the index-tuning advisors report failures as a DICT
                # whose only payload is an "error" key (e.g. "Statistics are
                # not up-to-date ... run 'ANALYZE;' first") — again a normal
                # result, again a deterministic failure of the intended
                # analysis. Fail fast with the message steering the replan.
                if (step.tool in ("analyze_workload_indexes", "analyze_query_indexes",
                                  "get_top_queries")
                        and isinstance(data, dict) and data.get("error")
                        and not (set(data) - {"error", "_langfuse_trace", "dta_traces"})):
                    err = str(data["error"]).strip()[:400]
                    if "analyze" in err.lower():
                        err += (" | HINT: add ONE prior execute_sql step running "
                                "exactly ANALYZE; then retry this tool.")
                    se.attempts.append(Attempt(n, "tool_error", err, dur))
                    se.status = "tool_error"
                    se.error = err
                    se.result_data = data
                    self.log.warning("[%s] %s returned an error payload -> "
                                     "failing for replan", step.id, step.tool)
                    self._event(event="step_failed", id=step.id, kind="tool_error",
                                error=err)
                    return se
                se.attempts.append(Attempt(n, "success", None, dur))
                se.status = "success"
                se.result_data = data
                se.result_content = result.content
                # get_top_queries prefixes its Python-repr payload with a
                # prose line ("Top N slowest queries by ...:\n[{...}]"), which
                # defeats the literal decode and leaves the natural binding
                # "$sN[0].query" unresolvable. Surface the parsed list; the
                # raw text stays available in result_content.
                if step.tool == "get_top_queries" and isinstance(data, str):
                    parsed = _parse_prefixed_payload(data)
                    if parsed is not None:
                        se.result_data = parsed
                self.log.info("[%s] OK (attempt %d, %.2fs)", step.id, n, dur)
                self._event(event="step_success", id=step.id, attempt=n,
                            data_preview=_preview(se.result_data))
                return se
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                dur = time.time() - t0
                msg = getattr(e, "text", None) or str(e)
                low = msg.lower()
                if isinstance(e, MCPToolError) and step.tool == "execute_sql":
                    # (idempotency) CREATE on an existing object: the desired
                    # end state holds. Echo identifying info so later bindings
                    # into this step still resolve.
                    if kind == "create" and _EXISTS_MARKER in low:
                        se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                        se.status = "success"
                        se.result_data = _benign_exists_result(sql)
                        self.log.warning("[%s] %s already exists -> success "
                                         "(idempotent)", step.id,
                                         (sql_object(sql) or ("object", "?"))[1])
                        self._event(event="step_idempotent", id=step.id,
                                    tool=step.tool, reason="already_exists")
                        return se
                    # (idempotency) DROP on an already-absent object: absence IS
                    # the desired end state.
                    if kind == "drop" and _ABSENT_MARKER in low:
                        se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                        se.status = "success"
                        se.result_data = {"status": "already_absent"}
                        self.log.warning("[%s] drop target already absent -> "
                                         "success (idempotent)", step.id)
                        self._event(event="step_idempotent", id=step.id,
                                    tool=step.tool, reason="already_absent")
                        return se
                    # (idempotency) INSERT hitting a unique/PK constraint: the
                    # keyed row is already there (typically a previous run of
                    # the same task). Content correctness stays the verifier's
                    # job — it re-queries the live rows.
                    if kind == "insert" and _DUPKEY_MARKER in low:
                        se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                        se.status = "success"
                        m = _INSERT_RE.match(sql)
                        se.result_data = {"status": "row_exists",
                                          **({"table": m.group(1).strip('"')} if m else {})}
                        self.log.warning("[%s] insert hit an existing key -> "
                                         "success (idempotent)", step.id)
                        self._event(event="step_idempotent", id=step.id,
                                    tool=step.tool, reason="duplicate_key")
                        return se
                # Replan steering: append actionable guidance for well-known
                # PostgreSQL failures to the error text the replanner reads.
                if isinstance(e, MCPToolError):
                    hint = _replan_hint(step.tool, resolved, msg)
                    if hint:
                        msg += hint
                kind_status = {
                    MCPToolError: "tool_error",
                    MCPProtocolError: "protocol_error",
                    MCPTransportError: "transport_error",
                }[type(e)]
                se.attempts.append(Attempt(n, kind_status, msg, dur))
                se.status = kind_status
                se.error = msg
                self.log.warning("[%s] %s on attempt %d/%d: %s",
                                  step.id, kind_status.upper(), n, self.max_attempts, msg)
                self._event(event="step_attempt_failed", id=step.id, attempt=n,
                            kind=kind_status, error=msg)
                if kind_status not in _TRANSIENT or n == self.max_attempts:
                    self.log.error("[%s] FAILED after %d attempt(s): %s", step.id, n, msg)
                    self._event(event="step_failed", id=step.id, kind=kind_status, error=msg)
                    return se
                time.sleep(min(2 ** (n - 1), 5))  # small backoff
        return se

    # ------------------------------------------------------- SQL ledger
    def _preflight_ledger(self, step: PlanStep, resolved: dict[str, Any],
                          se: StepExecution) -> StepExecution | None:
        """Adopt an INSERT/UPDATE/DELETE an earlier attempt already executed
        (exact normalized-SQL match) instead of re-running it. Returns the
        completed StepExecution, or None to proceed with the normal call.

        Scope is deliberately DML-only: CREATE/DROP replays are already safe
        (IF NOT EXISTS guardrail + the benign-exists/absent error paths), and
        skipping DDL from a ledger could break a legitimate drop-then-recreate
        sequence."""
        if step.tool != "execute_sql":
            return None
        sql = str(resolved.get("sql") or "")
        kind = sql_kind(sql)
        if kind not in _LEDGER_SKIP_KINDS:
            return None
        norm = normalize_sql(sql)
        if norm not in self._prior_sql or norm in self._own_sql:
            return None
        se.attempts.append(Attempt(1, "success_idempotent", None, 0.0))
        se.status = "success"
        se.result_data = {"status": "already_executed", "kind": kind}
        self.log.warning("[%s] identical %s statement already executed by an "
                         "earlier attempt -> adopting it (pre-flight skip, no "
                         "duplicate effect)", step.id, kind.upper())
        self._event(event="step_preflight_skip", id=step.id, tool=step.tool,
                    reason="sql_already_executed", kind=kind)
        return se

    def _note_sql_write(self, step: PlanStep, se: StepExecution) -> None:
        """Record a successful write statement in the SQL ledger (and in this
        attempt's own set, which exempts it from its own pre-flight skip)."""
        if step.tool != "execute_sql":
            return
        sql = str((se.arguments_resolved or {}).get("sql") or "")
        kind = sql_kind(sql)
        if not _is_write_kind(kind):
            return
        norm = normalize_sql(sql)
        self._own_sql.add(norm)
        if self._ledger_path is not None:
            try:
                with self._ledger_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.time(), "kind": kind,
                                        "sql_norm": norm}) + "\n")
            except OSError:
                pass

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
            parts.append("Resolved arguments: " + json.dumps(fs.arguments_resolved, default=str))
        return " ".join(parts)

    def _finalize(self, trace: ExecutionTrace) -> None:
        if self._jsonl:
            self._jsonl.close()
        out = self.settings.work_dir / "trace.json"
        out.write_text(json.dumps(trace.to_dict(), indent=2, default=str), encoding="utf-8")
        status = "SUCCESS" if trace.success else f"FAILED at {trace.failed_step}"
        self.log.info("=== EXECUTION %s (%.2fs) -> %s ===", status,
                      trace.finished_at - trace.started_at, out)


def _preview(data: Any, limit: int = 240) -> str:
    s = json.dumps(data, default=str) if not isinstance(data, str) else data
    return s[:limit] + ("..." if len(s) > limit else "")


def reset_sql_ledger(work_dir: Any) -> None:
    """Drop the SQL ledger so a FRESH run starts with no execution history.

    The ledger exists to stop a REPLAN from replaying an INSERT/UPDATE/DELETE
    an earlier attempt of the SAME run already executed. It is keyed on
    recorded history, not on live state, so if it survives into a later run of
    the same task — the work dir is reused whenever a --run-id repeats — it
    suppresses writes whose effects no longer exist (e.g. the schema was
    dropped in between), silently producing an empty database and a plan that
    "succeeded" without doing anything. Callers that own a run boundary (the
    orchestrator) must therefore clear it before the first attempt.
    """
    from pathlib import Path
    p = Path(work_dir) / "sql_ledger.jsonl"
    try:
        p.unlink()
    except (FileNotFoundError, OSError):
        pass


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
    "reset_sql_ledger",
    "normalize_sql",
    "sql_kind",
    "sql_object",
    "MAX_ATTEMPTS",
]
