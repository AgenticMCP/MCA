"""Execution agent: replays a plan's tool-call sequence verbatim against the
real playwright-mcp server (via pkg.mcp_wrapper), with retries and full logging.

Contract (from the workflow spec):
* The sequence is executed STRICTLY in order and is NEVER altered. The executor
  does not add, drop, or reorder steps.
* Deterministic failures (binding/schema/stale ref) fail fast — a retry cannot
  help; that goes straight back to the planner. Transient failures retry up to
  3 times — but ONLY for tools that are safe to repeat: navigation, snapshots
  and other observations. Interactions (click, type, drag, fill, dialogs, JS)
  are NON-idempotent in a browser — a replayed click can double-act — so they
  get exactly ONE attempt; their failures always go to the replanner, which
  re-establishes page state from scratch (every attempt runs a fresh browser).
* Everything is logged: a human ``execution.log``, a structured ``trace.json``,
  and the server's captured stderr (``server.log``).
* Playwright-specific state reconciliation:
  - ``target: "find:<text>"`` is resolved deterministically against the MOST
    RECENT snapshot of this session (exactly one matching line -> its
    [ref=eNN]; zero or several -> BindingError listing the candidates, which
    fails fast into a replan). This is how a plan can express an interaction
    whose ref cannot be known at plan time (refs are runtime-only handles).
  - Every result is surfaced as ``{text, url, title}`` (url/title parsed from
    the "### Page" section) so "$sN.url"-style bindings work over the
    otherwise text-only results.
  - A stale-ref failure ("Ref eNN not found") attaches a FRESH snapshot
    excerpt to the error text, so the replanner can pick the correct ref
    without an extra observation round-trip.
  - "No open page" failures on observation tools self-heal by re-navigating
    to the last successfully visited URL (navigation is convergent).
  - ``browser_close`` on an already-closed browser is adopted as an
    idempotent success (the desired end state holds).
  - ``browser_handle_dialog`` when no dialog is open is likewise adopted as
    an idempotent success — the desired end state (no blocking dialog)
    already holds.
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
# Statuses for which retrying makes sense (could be a flaky network / slow page).
_TRANSIENT = {"tool_error", "protocol_error", "transport_error"}

# Interactions that are NOT safe to replay: a second click/keystroke on a live
# page can double-act (submit twice, toggle back). These get ONE attempt; the
# recovery path is a replan that re-establishes state in a fresh browser.
_UNSAFE_TO_REPEAT = {
    "browser_click", "browser_type", "browser_press_key", "browser_drag",
    "browser_drop", "browser_fill_form", "browser_select_option",
    "browser_file_upload", "browser_handle_dialog", "browser_evaluate",
    "browser_run_code_unsafe",
}

# Observation tools that may self-heal a "no open page" failure by
# re-navigating to the last visited URL first (navigation is convergent).
_OBSERVATION_TOOLS = {
    "browser_snapshot", "browser_find", "browser_take_screenshot",
    "browser_console_messages", "browser_network_requests",
    "browser_network_request", "browser_wait_for",
}

# browser_close on an already-closed browser: the desired end state holds.
_IDEMPOTENT_TOOLS = {"browser_close"}
_BENIGN_CLOSED_MARKERS = ("no open", "already closed", "not connected",
                          "browser has been closed", "no current page")

# Result-text markers.
_PAGE_URL_RE = re.compile(r"^- Page URL: (.+)$", re.MULTILINE)
_PAGE_TITLE_RE = re.compile(r"^- Page Title: (.+)$", re.MULTILINE)
_REF_LINE_RE = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")  # also frame-prefixed refs (f1e24)
_STALE_REF_RE = re.compile(r"Ref \S+ not found", re.IGNORECASE)
# An EMPTY accessibility snapshot ("### Snapshot\n```yaml\n\n```"): the page
# has not rendered yet (or serves a bot wall). Observed on booking.com — the
# post-navigate snapshot file was 0 bytes while one taken 6s later held the
# full 18KB form.
_EMPTY_SNAP_RE = re.compile(r"### Snapshot\s*```yaml\s*```")
# The server stamps the HTTP status into navigate results when the load was
# not OK ("- HTTP status: 404"). A 4xx/5xx navigation "succeeds" as a tool
# call but can never deliver the task's content (observed: guessed blog URLs
# -> 404 pages snapshotted as if they were the article; nba.com -> 403).
_HTTP_STATUS_RE = re.compile(r"^- HTTP status: (\d+)$", re.MULTILINE)
# Seconds between settle-and-retake attempts on an empty snapshot (module
# level so the offline selftest can zero it).
_SETTLE_SECONDS = 2

# Role-aware find: disambiguation — when several snapshot lines match, a tool
# that TYPES/SELECTS can only ever act on an input-like element, so prefer
# those lines. (Observed: find:'Search' on huggingface.co matched the search
# textbox plus two links whose captions contain "Search".) For clicking, the
# preference is interactive elements — "@pointer" additionally admits any
# line the snapshot marks [cursor=pointer].
_TYPABLE_ROLES = frozenset({"textbox", "searchbox", "combobox", "spinbutton",
                            "textarea"})
_CLICKABLE_ROLES = frozenset({"link", "button", "tab", "menuitem", "option",
                              "checkbox", "radio", "switch", "@pointer"})
_ROLES_FOR_TOOL: dict[str, frozenset[str]] = {
    "browser_type": _TYPABLE_ROLES,
    "browser_fill_form": _TYPABLE_ROLES,
    "browser_select_option": frozenset({"combobox", "listbox", "select"}),
    "browser_click": _CLICKABLE_ROLES,
}
_LINE_ROLE_RE = re.compile(r"^\s*-\s*([A-Za-z]+)")
_URL_LINE_RE = re.compile(r"^\s*-\s*/url:\s*(\S+)")
# How much of a fresh snapshot is attached to a failed step's error text for
# the replanner (bounded — snapshots of busy pages run to tens of KB).
_SNAPSHOT_EXCERPT_CHARS = 3500


def _is_unsafe_to_repeat(tool: str, resolved: dict[str, Any]) -> bool:
    if tool in _UNSAFE_TO_REPEAT:
        return True
    # tab management mutates browser state except for a plain listing
    if tool == "browser_tabs":
        return str(resolved.get("action") or "") != "list"
    return False


def _page_state_result(text: str) -> dict[str, Any]:
    """Surface a tool's markdown text as {text, url, title} so bindings have
    stable paths over the otherwise unstructured results."""
    out: dict[str, Any] = {"text": text}
    m = _PAGE_URL_RE.search(text)
    if m:
        out["url"] = m.group(1).strip()
    m = _PAGE_TITLE_RE.search(text)
    if m:
        out["title"] = m.group(1).strip()
    return out


_QUOTED_RE = re.compile(r'"([^"]*)"')


def _role_filter(cands: list[tuple[str, str, str | None]],
                 roles: frozenset[str] | None) -> list[tuple[str, str, str | None]]:
    """Among ambiguous candidates, keep those whose yaml role suits the acting
    tool (typing needs an input-like element; clicking an interactive one —
    "@pointer" in ``roles`` additionally admits [cursor=pointer] lines). Only
    ever NARROWS a tie — when the filter matches nothing, the original
    ambiguity stands."""
    if not roles or len(cands) < 2:
        return cands
    kept = [c for c in cands
            if ((m := _LINE_ROLE_RE.match(c[1])) and m.group(1).lower() in roles)
            or ("@pointer" in roles and "[cursor=pointer]" in c[1])]
    return kept or cands


def _collapse_equivalent(cands: list[tuple[str, str, str | None]]
                         ) -> list[tuple[str, str, str | None]]:
    """N candidates that are the SAME action repeated are one target: keep
    the first. Two provable cases:

    * identical line modulo ref AND the same non-empty /url — one link
      rendered as both tile and row;
    * identical line modulo ref, no /url, but an INTERACTIVE role (button /
      link / tab): per ARIA semantics an identical accessible name on the
      same control role is the same command, duplicated by header/footer/
      mobile nav (observed: three "Attractions" buttons on booking.com kept
      a task ambiguous through every replan).

    Anything that differs in caption, role or destination stays ambiguous."""
    if len(cands) < 2:
        return cands
    sigs = {(_REF_LINE_RE.sub("", line).strip(), url) for _, line, url in cands}
    if len(sigs) != 1:
        return cands
    if cands[0][2]:
        return [cands[0]]
    m = _LINE_ROLE_RE.match(cands[0][1])
    if m and m.group(1).lower() in ("button", "link", "tab"):
        return [cands[0]]
    return cands


def _find_ref(snapshot: str, needle: str,
              roles: frozenset[str] | None = None) -> str:
    """Resolve a ``find:<text>`` target against snapshot text. Two passes,
    both case-insensitive and deterministic — no LLM, no guessing:

    1. EXACT accessible-name match: lines where a quoted name equals the
       needle (`button "Search"` for find:Search). An element's exact caption
       beats lines that merely contain the text (a search box holding the
       typed value, longer captions like "Search flights").
    2. Substring match over whole lines, only when no exact name matched.

    ``roles`` (from the acting tool) breaks remaining ties deterministically:
    a typing tool prefers the textbox line over links that merely mention the
    text. Repeated renderings of the SAME element (identical line + /url)
    collapse to one. Exactly one candidate -> its ref; zero or several ->
    BindingError listing candidates (with each link's /url, so a replanner
    can choose by destination)."""
    want = needle.strip().casefold()
    if not want:
        raise BindingError("find: target has empty text")
    exact: list[tuple[str, str, str | None]] = []
    loose: list[tuple[str, str, str | None]] = []
    lines = snapshot.splitlines()
    for i, line in enumerate(lines):
        m = _REF_LINE_RE.search(line)
        if not m:
            continue
        url = None
        if i + 1 < len(lines):
            mu = _URL_LINE_RE.match(lines[i + 1])
            if mu:
                url = mu.group(1)
        if any(q.strip().casefold() == want for q in _QUOTED_RE.findall(line)):
            exact.append((m.group(1), line.strip(), url))
        elif want in line.casefold():
            loose.append((m.group(1), line.strip(), url))

    def _listed(cands: list[tuple[str, str, str | None]]) -> str:
        return "; ".join(
            f"[{r}] {l[:110]}" + (f" -> {u[:90]}" if u else "")
            for r, l, u in cands[:6])

    exact = _collapse_equivalent(_role_filter(exact, roles))
    if len(exact) == 1:
        return exact[0][0]
    if len(exact) > 1:
        raise BindingError(
            f"find:{needle!r} is AMBIGUOUS — {len(exact)} elements are named "
            f"exactly that: {_listed(exact)} — replan with one of these refs")
    matches = _collapse_equivalent(_role_filter(loose, roles))
    if len(matches) == 1:
        return matches[0][0]
    if not matches:
        raise BindingError(
            f"find:{needle!r} matches NO line of the most recent snapshot — "
            f"the text is not on the page (or not snapshotted yet); replan "
            f"with text visible in the snapshot, or add a browser_snapshot / "
            f"browser_wait_for step first")
    raise BindingError(
        f"find:{needle!r} is AMBIGUOUS — {len(matches)} snapshot lines match: "
        f"{_listed(matches)} — replan with more specific text or one of these "
        f"refs")


def _has_find_target(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("find:")
    if isinstance(value, dict):
        return any(_has_find_target(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_find_target(v) for v in value)
    return False


def _resolve_find_targets(value: Any, snapshot: str | None,
                          roles: frozenset[str] | None = None) -> Any:
    """Recursively resolve every ``find:<text>`` string inside ``value``
    against the given snapshot. No snapshot available -> BindingError."""
    if isinstance(value, str):
        if value.startswith("find:"):
            if not snapshot:
                raise BindingError(
                    "find: target could not be resolved — no page snapshot is "
                    "available (the page may not be open yet); make sure a "
                    "browser_navigate step precedes this action")
            return _find_ref(snapshot, value[len("find:"):], roles)
        return value
    if isinstance(value, dict):
        return {k: _resolve_find_targets(v, snapshot, roles)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_find_targets(v, snapshot, roles) for v in value]
    return value


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


def build_client(settings: Settings, *, allow_tools: list[str] | None = None) -> MCPClient:
    """Construct (not start) an MCPClient wired to the playwright-mcp server,
    with browser artifacts (screenshots, saved snapshots) and the captured
    stderr pointed into the run directory. playwright-mcp has no --read-only
    mode; a restricted session is enforced client-side via ``allow_tools``."""
    settings.ensure_work_dir()
    return MCPClient(
        command=settings.server_command,
        headless=settings.headless,
        isolated=settings.isolated,
        browser=settings.browser,
        user_agent=getattr(settings, "user_agent", None),
        timeout_action=settings.timeout_action,
        output_dir=settings.work_dir / "output",
        log_file=settings.work_dir / "server.log",
        allow_tools=allow_tools,
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
        # Raw text of the most recent successful result that carried element
        # refs (browser_snapshot / browser_find) — the resolution base for
        # "find:" targets and the freshest observation of the page.
        self._last_snapshot: str | None = None
        # Last successfully visited URL — the re-navigation base for the
        # "no open page" self-heal.
        self._last_url: str | None = None

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
                    self._note_session_state(step.tool, se)
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
    def _note_session_state(self, tool: str, se: StepExecution) -> None:
        """Track the session's freshest snapshot and last visited URL."""
        data = se.result_data
        text = data.get("text") if isinstance(data, dict) else None
        if isinstance(text, str) and _REF_LINE_RE.search(text):
            self._last_snapshot = text
        url = data.get("url") if isinstance(data, dict) else None
        if isinstance(url, str) and url and url != "about:blank":
            self._last_url = url

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

        # 1. binding + find:-target resolution (deterministic — no retry).
        # find: targets resolve against a FRESH observation probe taken right
        # now (read-only; the sequence is not altered): refs on dynamic pages
        # go stale in seconds, so the freshest possible snapshot — not the one
        # recorded by an earlier step — is the only reliable resolution base.
        # Probe failure falls back to the last recorded snapshot.
        probe_empty = False
        try:
            resolved = resolve_bindings(step.arguments, outputs)
            if _has_find_target(resolved):
                probe = self._fresh_snapshot_text(client, settle=True)
                if probe:
                    self._last_snapshot = probe
                    self._event(event="find_probe", id=step.id, tool=step.tool)
                else:
                    probe_empty = True
            resolved = _resolve_find_targets(resolved, self._last_snapshot,
                                             _ROLES_FOR_TOOL.get(step.tool))
        except BindingError as e:
            se.status = "binding_error"
            err_text = str(e)
            # A failed find: resolution means the text was not on the page the
            # probe just observed. Attach that fresh snapshot so the replanner
            # picks a target from what the page ACTUALLY shows instead of
            # guessing again (observed: a replan loop burned every attempt
            # re-guessing captions for a widget that wasn't there).
            if _has_find_target(step.arguments) and self._last_snapshot:
                excerpt = self._last_snapshot[:_SNAPSHOT_EXCERPT_CHARS]
                if len(self._last_snapshot) > _SNAPSHOT_EXCERPT_CHARS:
                    excerpt += "\n... [snapshot truncated]"
                err_text += f"\nCURRENT PAGE SNAPSHOT (fresh):\n{excerpt}"
            if probe_empty:
                err_text += (
                    "\nNOTE: the fresh observation probe returned an EMPTY "
                    "page snapshot even after settling — the page renders "
                    "nothing snapshot-able yet (slow load or a bot wall). Add "
                    "browser_wait_for (a short time, or text the page should "
                    "show) after the navigation, then browser_snapshot, "
                    "before this interaction.")
            se.error = err_text
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

        # 3. invoke — retries only for tools that are safe to repeat
        attempts_budget = 1 if _is_unsafe_to_repeat(step.tool, resolved) else self.max_attempts
        self.log.info("[%s] CALL %s args=%s", step.id, step.tool,
                      json.dumps(_args_brief(resolved), default=str))
        healed_page = False  # one "no open page" re-navigation heal per step
        extra = 0            # one stale-ref refresh heal per step (see below)
        retook_empty = False  # one settle-and-retake per empty snapshot step
        retried_status = False  # one settle-and-retry per 4xx/5xx navigation
        n = 0
        while n < attempts_budget + extra:
            n += 1
            t0 = time.time()
            try:
                result = client.call(step.tool, resolved)
                dur = time.time() - t0
                # (fail fast / heal) a navigation that "succeeded" with HTTP
                # >= 400 delivered an error page, not content. 404/410 are
                # deterministic (the URL does not exist) — fail straight into
                # a replan. Throttle/deny codes (401/403/429) and 5xx are
                # often transient token-bucket responses: settle once and
                # re-navigate before failing (observed: huggingface.co starts
                # returning 401 on model pages under repeated batch load).
                if (step.tool == "browser_navigate"
                        and isinstance(result.data, str)):
                    mstat = _HTTP_STATUS_RE.search(result.data)
                    if mstat and int(mstat.group(1)) >= 400:
                        code = int(mstat.group(1))
                        se.attempts.append(Attempt(n, "tool_error",
                                                   f"HTTP {code}", dur))
                        if (code not in (404, 410) and not retried_status
                                and n < attempts_budget):
                            retried_status = True
                            self.log.warning(
                                "[%s] navigate -> HTTP %d; settling and "
                                "retrying once", step.id, code)
                            self._event(event="step_heal_start", id=step.id,
                                        tool=step.tool,
                                        reason="http_status_retry")
                            time.sleep(2 * _SETTLE_SECONDS)
                            continue
                        hint = (
                            "HINT: the URL was likely GUESSED and does not "
                            "exist — never invent content URLs; use the "
                            "site's canonical SEARCH endpoint or a link "
                            "taken from a snapshot."
                            if code in (404, 410) else
                            "HINT: the site refuses this automated session "
                            "(bot protection, auth wall or rate limit) — do "
                            "not retry the same URL; reach the information "
                            "via a different page of the allowed site, or "
                            "snapshot what is available.")
                        se.status = "tool_error"
                        se.error = (f"navigation returned HTTP {code} for "
                                    f"{resolved.get('url')!r} — the page did "
                                    f"not deliver content. | {hint}")
                        self.log.error("[%s] navigate -> HTTP %d; fail fast "
                                       "for replan", step.id, code)
                        self._event(event="step_failed", id=step.id,
                                    kind="tool_error", error=se.error[:400])
                        return se
                # (reconciliation) a planned browser_snapshot that returns an
                # EMPTY accessibility tree: the page has not rendered yet —
                # settle briefly and retake once, so the recorded snapshot
                # (the run's only evidence) carries the actual content.
                if (step.tool == "browser_snapshot" and not retook_empty
                        and n < attempts_budget
                        and isinstance(result.data, str)
                        and _EMPTY_SNAP_RE.search(result.data)):
                    retook_empty = True
                    se.attempts.append(Attempt(n, "empty_snapshot", None, dur))
                    self.log.warning("[%s] snapshot came back EMPTY; settling "
                                     "%ss and retaking once", step.id,
                                     _SETTLE_SECONDS)
                    self._event(event="step_heal_start", id=step.id,
                                tool=step.tool, reason="empty_snapshot_retake")
                    time.sleep(_SETTLE_SECONDS)
                    continue
                se.attempts.append(Attempt(n, "success", None, dur))
                se.status = "success"
                se.result_data = (_page_state_result(result.data)
                                  if isinstance(result.data, str)
                                  else result.data)
                se.result_content = result.content
                if healed_page:
                    self._event(event="step_healed", id=step.id, tool=step.tool,
                                reason="page_reopened")
                self.log.info("[%s] OK (attempt %d, %.2fs)", step.id, n, dur)
                self._event(event="step_success", id=step.id, attempt=n,
                            data_preview=_preview(se.result_data))
                return se
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                dur = time.time() - t0
                msg = getattr(e, "text", None) or str(e)
                low = msg.lower()
                # (idempotency) browser_close on an already-closed browser:
                # the desired end state holds.
                if (isinstance(e, MCPToolError) and step.tool in _IDEMPOTENT_TOOLS
                        and any(k in low for k in _BENIGN_CLOSED_MARKERS)):
                    se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                    se.status = "success"
                    se.result_data = {"text": "(browser already closed)"}
                    self.log.warning("[%s] %s: browser already closed -> success "
                                     "(idempotent)", step.id, step.tool)
                    self._event(event="step_idempotent", id=step.id, tool=step.tool)
                    return se
                # (idempotency) browser_handle_dialog with no dialog open: the
                # desired end state — no blocking dialog — already holds.
                # Planners add dialog steps speculatively; hard-failing one
                # wastes a whole replan cycle (observed: a task burned all 3
                # replans on exactly this error).
                if (isinstance(e, MCPToolError)
                        and step.tool == "browser_handle_dialog"
                        and "can only be used when there is related modal state"
                        in low):
                    se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                    se.status = "success"
                    se.result_data = {"text": "(no dialog was open)"}
                    self.log.warning("[%s] %s: no dialog open -> success "
                                     "(idempotent)", step.id, step.tool)
                    self._event(event="step_idempotent", id=step.id, tool=step.tool)
                    return se
                # (self-heal) observation tool with no page open: re-establish
                # the page by re-navigating to the last visited URL, then retry
                # this step once. Navigation is convergent, so this is safe.
                if (isinstance(e, MCPToolError) and not healed_page
                        and step.tool in _OBSERVATION_TOOLS
                        and self._last_url
                        and ("no open page" in low or "no open tab" in low
                             or "navigate to a page first" in low)):
                    se.attempts.append(Attempt(n, "tool_error", msg, dur))
                    self.log.warning("[%s] %s: no page open; self-healing by "
                                     "re-navigating to %s", step.id, step.tool,
                                     self._last_url)
                    self._event(event="step_heal_start", id=step.id,
                                tool=step.tool, reason="no_open_page")
                    try:
                        client.call("browser_navigate", {"url": self._last_url})
                        healed_page = True
                        continue  # retry the original step
                    except (MCPToolError, MCPProtocolError, MCPTransportError) as e2:
                        msg += f" | self-heal re-navigation also failed: {e2}"
                # (self-heal) stale ref: dynamic pages (hydration, lazy loads)
                # can invalidate a ref between the snapshot and the action —
                # the server rejects the ref BEFORE acting, so the action did
                # not run and a retry cannot double-act. Heal deterministically:
                # take a fresh snapshot, re-resolve the plan's original target
                # against it (a find: needle re-resolves semantically; a
                # literal ref re-syncs the server's ref registry), and retry
                # ONCE. Still failing -> fail fast for replan, with the fresh
                # snapshot attached so the replanner can pick a valid target.
                if isinstance(e, MCPToolError) and _STALE_REF_RE.search(msg):
                    se.attempts.append(Attempt(n, "tool_error", msg, dur))
                    if extra == 0:
                        fresh_full = self._fresh_snapshot_text(client)
                        if fresh_full:
                            self._last_snapshot = fresh_full
                            try:
                                re_res = resolve_bindings(step.arguments, outputs)
                                re_res = _resolve_find_targets(re_res,
                                                               self._last_snapshot)
                            except BindingError as be:
                                msg += (" | fresh-snapshot re-resolution also "
                                        f"failed: {be}")
                            else:
                                if isinstance(re_res, dict):
                                    resolved = re_res
                                    se.arguments_resolved = resolved
                                    extra = 1
                                    self.log.warning(
                                        "[%s] %s: ref went stale (page mutated "
                                        "after the snapshot); re-snapshotted and "
                                        "re-resolved -> retrying once",
                                        step.id, step.tool)
                                    self._event(event="step_heal_start",
                                                id=step.id, tool=step.tool,
                                                reason="stale_ref_refresh")
                                    continue
                    msg += (" | HINT: refs are only valid against the MOST "
                            "RECENT snapshot — replan with a browser_snapshot "
                            "immediately before this action and take the ref "
                            "(or a find:<unique text> target) from it.")
                    if self._last_snapshot:
                        excerpt = self._last_snapshot[:_SNAPSHOT_EXCERPT_CHARS]
                        if len(self._last_snapshot) > _SNAPSHOT_EXCERPT_CHARS:
                            excerpt += "\n... [snapshot truncated]"
                        msg += f"\nCURRENT PAGE SNAPSHOT (fresh):\n{excerpt}"
                    se.status = "tool_error"
                    se.error = msg
                    self.log.error("[%s] stale ref -> fail fast for replan", step.id)
                    self._event(event="step_failed", id=step.id, kind="tool_error",
                                error=msg[:400])
                    return se
                # (fail fast) browser_wait_for on expected text that timed
                # out: the server already waited its FULL timeout — the text
                # is not coming, and safe-tool retries only multiply the stall
                # (observed 13× in one batch, 45s burned per occurrence).
                # Fail with a fresh look at what the page ACTUALLY shows.
                if (isinstance(e, MCPToolError)
                        and step.tool == "browser_wait_for"
                        and ("text" in resolved or "textGone" in resolved)
                        and "timeout" in low):
                    se.attempts.append(Attempt(n, "tool_error", msg, dur))
                    msg += (" | HINT: the expected text did not appear within "
                            "the timeout — it may simply never appear on this "
                            "page. Do not wait for the same text again; "
                            "snapshot the page and base the next step on what "
                            "IS there.")
                    fresh = self._fresh_snapshot_text(client)
                    if fresh:
                        self._last_snapshot = fresh
                        excerpt = fresh[:_SNAPSHOT_EXCERPT_CHARS]
                        if len(fresh) > _SNAPSHOT_EXCERPT_CHARS:
                            excerpt += "\n... [snapshot truncated]"
                        msg += f"\nCURRENT PAGE SNAPSHOT (fresh):\n{excerpt}"
                    se.status = "tool_error"
                    se.error = msg
                    self.log.error("[%s] wait_for text timed out -> fail fast "
                                   "for replan", step.id)
                    self._event(event="step_failed", id=step.id,
                                kind="tool_error", error=msg[:400])
                    return se
                # (replan steering) other well-known failure modes
                if isinstance(e, MCPToolError):
                    if "execution context was destroyed" in low:
                        msg += (" | HINT: the page was navigating while this "
                                "step ran — insert browser_wait_for (expected "
                                "text or a short time) before it, then a fresh "
                                "browser_snapshot.")
                    elif "modal" in low or "dialog" in low:
                        msg += (" | HINT: a browser dialog is blocking the "
                                "page — add a browser_handle_dialog step "
                                "(accept or dismiss) before this action.")
                    elif "is not an <input>" in low:
                        msg += (" | HINT: the resolved element is not a text "
                                "field — typing into it can never succeed. "
                                "Retarget the actual textbox/searchbox: pick "
                                "its line (caption or eNN ref) from the "
                                "snapshot, or navigate via a search URL "
                                "instead.")
                    elif "timeout" in low and "exceeded" in low:
                        if step.tool == "browser_navigate":
                            msg += (" | HINT: the page did not finish loading "
                                    "— the site may be slow, or it may BLOCK "
                                    "automated browsers (bot protection); "
                                    "waiting or retrying the same navigation "
                                    "cannot fix a block. Reach the information "
                                    "via a different page of the allowed site, "
                                    "or snapshot whatever did load.")
                        else:
                            msg += (" | HINT: the content did not appear in "
                                    "time — add browser_wait_for with the "
                                    "expected text, or target a different "
                                    "element that IS in the snapshot.")
                kind = {
                    MCPToolError: "tool_error",
                    MCPProtocolError: "protocol_error",
                    MCPTransportError: "transport_error",
                }[type(e)]
                se.attempts.append(Attempt(n, kind, msg, dur))
                se.status = kind
                se.error = msg
                self.log.warning("[%s] %s on attempt %d/%d: %s",
                                  step.id, kind.upper(), n, attempts_budget, msg[:300])
                self._event(event="step_attempt_failed", id=step.id, attempt=n,
                            kind=kind, error=msg[:400])
                if kind not in _TRANSIENT or n >= attempts_budget:
                    self.log.error("[%s] FAILED after %d attempt(s): %s",
                                   step.id, n, msg[:300])
                    self._event(event="step_failed", id=step.id, kind=kind,
                                error=msg[:400])
                    return se
                time.sleep(min(2 ** (n - 1), 5))  # small backoff
        return se

    def _fresh_snapshot_text(self, client: Any, *,
                             settle: bool = False) -> str | None:
        """The full CURRENT page snapshot — the find:-probe resolution base
        and the replanner's view after a failure. Read-only and best-effort:
        any failure returns None and the original error stands.

        With ``settle`` a snapshot that comes back EMPTY (no refs — the page
        has not rendered yet) is retaken up to twice after short pauses;
        observed on booking.com, where the tree fills in a few seconds after
        navigation."""
        for i in range(3 if settle else 1):
            if i:
                time.sleep(_SETTLE_SECONDS)
            try:
                result = client.call("browser_snapshot", {})
            except Exception:
                return None
            text = result.data if isinstance(result.data, str) else None
            if isinstance(text, str) and text and _REF_LINE_RE.search(text):
                return text
        return None

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


def _args_brief(args: dict[str, Any], limit: int = 120) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + "..."
        else:
            out[k] = v
    return out


def _preview(data: Any, limit: int = 240) -> str:
    s = json.dumps(data, default=str) if not isinstance(data, str) else data
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
