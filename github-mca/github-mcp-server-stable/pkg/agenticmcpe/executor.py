"""Execution agent: replays a plan's tool-call sequence verbatim against the
real Go github-mcp-server (via pkg.mcp_wrapper), with retries and full logging.

Contract (from the workflow spec):
* The sequence is executed STRICTLY in order and is NEVER altered. The executor
  does not add, drop, or reorder steps.
* Each step gets up to 3 attempts on *transient* failures (tool/protocol/
  transport errors). Deterministic failures (binding/schema) fail fast — a retry
  cannot help; that goes straight back to the planner.
* Everything is logged: a human ``execution.log``, a structured ``trace.json``,
  and the server's own JSON-RPC log (``server.log``) + captured stderr.
* On persistent failure the executor returns an ``error_context`` string for the
  planner to revise the plan; it does not try to "fix" the plan itself.
* Idempotent state reconciliation (re-runs / replans): a step whose intended
  effect ALREADY holds on GitHub is adopted as a success instead of failing into
  a replan. Writes that duplicate SILENTLY (issues, comments, gists) are checked
  up-front with a read probe; writes the server rejects ("already exists":
  repos, branches, labels, files, PRs; already-merged PRs; already-deleted
  files; already-attached sub-issues) are healed on-error, which costs nothing
  on fresh runs. Tools GitHub itself treats idempotently (fork_repository,
  star_repository, issue/PR updates, assign_copilot_to_issue) need no handling
  here — repeating them converges to the same state without error.
* Eventual-consistency settle-wait (separate from the idempotency machinery —
  this is about the API lagging, not about work already done): GitHub's Actions
  index can trail a just-committed ``.github/workflows/*.yml`` by several
  seconds, so an immediate ``actions_list`` returns an empty/stale list and
  ``actions_get`` 404s. When such a step follows a workflow-file write in the
  SAME plan, the executor polls with bounded backoff (~30s total) before
  accepting the result; on timeout it accepts the live answer as-is (never
  fabricates).
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
# Statuses for which retrying makes sense (could be a flaky network / rate limit).
_TRANSIENT = {"tool_error", "protocol_error", "transport_error"}
# Tools whose "already exists" error is benign (the resource is already there).
# Includes label_write because `bug` is a default repo label, so re-creating it
# fails with "Name has already been taken" — which is fine, the label exists.
_IDEMPOTENT_TOOLS = {"create_repository", "create_branch", "label_write"}
_BENIGN_EXISTS_MARKERS = ("already exist", "already been taken")

# Single-file write whose "already exists, provide SHA" error is recoverable in
# place (see _heal_existing_file). NOT in _IDEMPOTENT_TOOLS because "exists" here
# means the intended content was NOT written — we must fetch the SHA and either
# accept the file (identical content) or rewrite it. push_files upserts via a git
# tree and does not raise this error, so it stays out of scope.
_FILE_WRITE_TOOLS = {"create_or_update_file"}

# create_pull_request's duplicate error ("A pull request already exists for
# owner:branch"). Recoverable in place: the PR is there, adopt it (see
# _heal_existing_pr) so "$sN.number" bindings keep working.
_PR_EXISTS_MARKER = "pull request already exists"

# Settle-wait for the Actions API (see module docstring): a workflow file
# committed moments ago may not be indexed yet. Escalating sleeps, ~30s total;
# overridable per-agent (settle_sleeps attribute) so tests run instantly.
_SETTLE_SLEEPS: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0, 10.0)
# A workflow file as GitHub Actions defines one (path is repo-root-relative).
_WORKFLOW_PATH_RE = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")


_SHA_RE = re.compile(r"\(SHA: ([0-9a-fA-F]{40})\)")


def _file_sha(content: list[dict[str, Any]] | None) -> str | None:
    """Extract the blob SHA from get_file_contents' status text block."""
    for block in content or []:
        if block.get("type") == "text":
            m = _SHA_RE.search(block.get("text") or "")
            if m:
                return m.group(1)
    return None


def _owner_repo_from_url(url: str) -> tuple[str, str] | None:
    """(owner, repo) from a GitHub html/api repo URL, else None."""
    parts = [p for p in str(url).rstrip("/").split("/") if p]
    if len(parts) >= 2:
        owner, name = parts[-2], parts[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if owner and name:
            return owner, name
    return None


def _single_match_path(msg: str) -> str | None:
    """The single candidate path from a get_file_contents disambiguation
    message ('... matching files: ["a/b.py"]'), else None (zero or several
    matches — the executor must not guess; that stays a replan decision)."""
    m = re.search(r"matching files:\s*(\[[^\]]*\])", msg)
    if not m:
        return None
    try:
        files = json.loads(m.group(1))
    except ValueError:
        return None
    if (isinstance(files, list) and len(files) == 1
            and isinstance(files[0], str) and files[0]):
        return files[0]
    return None


def _is_benign_exists(tool: str, msg: str) -> bool:
    m = msg.lower()
    return tool in _IDEMPOTENT_TOOLS and any(k in m for k in _BENIGN_EXISTS_MARKERS)


def _needs_file_sha(tool: str, msg: str) -> bool:
    """True when a single-file write failed only because the file already exists
    and the server wants its current SHA (the recoverable case for self-heal)."""
    m = msg.lower()
    return tool in _FILE_WRITE_TOOLS and "already exists" in m and "sha" in m


def _benign_exists_result(tool: str, resolved: dict[str, Any]) -> dict[str, Any] | None:
    """Minimal truthful result for a create whose resource already existed:
    echo the identifying arguments. A bare None here used to break EVERY later
    "$sN.<field>" binding into the idempotent step (BindingError -> pointless
    replan); fields we cannot know without another API call are omitted."""
    if tool == "create_repository":
        return {"name": resolved.get("name")}
    if tool == "create_branch":
        branch = resolved.get("branch")
        return {"branch": branch, "ref": f"refs/heads/{branch}"} if branch else None
    if tool == "label_write":
        return {"name": resolved.get("label") or resolved.get("name")}
    return None


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
    """Construct (not start) an MCPClient wired to the bundled binary, with the
    server's own command logging pointed at ``work_dir/server.log``."""
    settings.ensure_work_dir()
    return MCPClient(
        token=settings.tokens.current(),
        binary=settings.binary_path,
        toolsets=["all"],
        read_only=read_only,
        log_file=settings.work_dir / "server.log",
        enable_command_logging=True,
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
        # (owner, repo, number) of issues created by THIS attempt, so the
        # issue_write pre-flight dedup never mistakes an intentional same-title
        # issue created moments ago by this very plan for a leftover of a
        # previous run (see _preflight_existing).
        self._created_issues: set[tuple[Any, Any, int]] = set()
        # (owner, repo, path) of workflow files written by THIS attempt — arms
        # the Actions settle-wait for later actions_list/actions_get steps
        # (see _note_workflow_write / _settle_actions_list).
        self._workflow_writes: set[tuple[Any, Any, str]] = set()
        # repo names forked by THIS attempt — GitHub creates forks
        # asynchronously, so later steps against the new fork can 404 while it
        # materializes; arms the fork settle-wait (see _settle_fork_read).
        self._forked_repos: set[str] = set()
        # Authenticated login, resolved lazily via get_me and cached (see
        # _authenticated_login). Only ever fetched when a result needs an owner
        # the server did not return.
        self._me_login: str | None = None
        self.settle_sleeps: tuple[float, ...] = _SETTLE_SLEEPS

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
                    # Arm the Actions settle-wait when this step wrote a
                    # workflow file (covers healed/idempotent writes too —
                    # the file is on GitHub either way).
                    self._note_workflow_write(step.tool, se.arguments_resolved or {})
                    # Arm the fork settle-wait: the new fork may 404 briefly
                    # while GitHub builds it. Arm both the requested name and
                    # the REAL name from the result (they differ when the
                    # source repo was renamed).
                    if step.tool == "fork_repository":
                        repo = (se.arguments_resolved or {}).get("repo")
                        if isinstance(repo, str) and repo:
                            self._forked_repos.add(repo)
                        real = (se.result_data or {}).get("name") \
                            if isinstance(se.result_data, dict) else None
                        if isinstance(real, str) and real:
                            self._forked_repos.add(real)
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

        # 2.5 pre-flight state reconciliation for duplicate-SILENT writes
        # (issues, comments): GitHub happily accepts a second identical one
        # without any "already exists" error, so the on-error heals below can
        # never fire for them and re-runs/replans used to pile up duplicates.
        # Check current state first; if the intended effect is already present,
        # adopt it and move on to the next step. (File/repo/branch writes stay
        # error-driven on purpose — the server rejects those duplicates, and
        # healing on-error costs zero extra calls on fresh runs.)
        pf = self._preflight_existing(client, step, resolved, se)
        if pf is not None:
            return pf

        # 3. invoke with retries on transient failure
        self.log.info("[%s] CALL %s args=%s", step.id, step.tool, json.dumps(resolved, default=str))
        healed_path = False  # one path-resolution heal per step (see below)
        for n in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                result = client.call(step.tool, resolved)
                # (settle-wait) actions_list right after this plan wrote a
                # workflow file: an empty/stale list is usually the Actions
                # index lagging the commit, not the truth — poll briefly
                # before accepting (no-op unless a workflow write armed it).
                if step.tool == "actions_list":
                    result = self._settle_actions_list(client, step, resolved, result)
                dur = time.time() - t0
                # Result sanity: get_file_contents returns a disambiguation
                # message (a string naming the real path) instead of content
                # when the path is ambiguous/missing. When it names exactly ONE
                # match the server just told us the correct path — retry with
                # it (once). Otherwise treat as a non-retryable failure so the
                # planner can choose the right path on replan.
                if (step.tool == "get_file_contents" and isinstance(result.data, str)
                        and result.data.lstrip().startswith("Resolved potential matches")):
                    fixed = _single_match_path(result.data)
                    if (fixed and not healed_path and fixed != resolved.get("path")
                            and n < self.max_attempts):
                        healed_path = True
                        se.attempts.append(Attempt(n, "tool_error", result.data, dur))
                        self.log.warning("[%s] get_file_contents: path %r not found; "
                                         "server resolved it to %r -> retrying with "
                                         "that path", step.id, resolved.get("path"), fixed)
                        self._event(event="step_heal_start", id=step.id,
                                    tool=step.tool, reason="path_resolved")
                        resolved = {**resolved, "path": fixed}
                        se.arguments_resolved = resolved
                        continue
                    se.attempts.append(Attempt(n, "tool_error", result.data, dur))
                    se.status = "tool_error"
                    se.error = result.data
                    se.result_data = result.data
                    self.log.warning("[%s] get_file_contents returned a disambiguation "
                                     "message (path likely wrong): %s", step.id, result.data[:160])
                    self._event(event="step_failed", id=step.id, kind="tool_error",
                                error=result.data)
                    return se
                se.attempts.append(Attempt(n, "success", None, dur))
                se.status = "success"
                se.result_data = result.data
                # Surface the file SHA (carried only in the status text block,
                # "successfully downloaded ... (SHA: ...)") so a later step can
                # bind "$sN.sha" when updating the file. The body stays at
                # "$sN.content".
                if step.tool == "get_file_contents" and isinstance(result.data, str):
                    se.result_data = {"content": result.data,
                                      "sha": _file_sha(result.content)}
                # search_repositories minimal output (the default) carries only
                # full_name per item; surface owner.login and name from it so
                # plans can bind "$sN.items[i].owner.login" / ".name" no matter
                # the minimal_output flag.
                if (step.tool == "search_repositories"
                        and isinstance(result.data, dict)):
                    raw = result.data.get("items")
                    items = []
                    for it in (raw if isinstance(raw, list) else []):
                        if (isinstance(it, dict)
                                and "/" in str(it.get("full_name") or "")
                                and ("owner" not in it or "name" not in it)):
                            o, _, nm = str(it["full_name"]).partition("/")
                            it = {**it}
                            it.setdefault("name", nm)
                            it.setdefault("owner", {"login": o})
                        items.append(it)
                    # always materialize items (the server omits the key when
                    # a search matches nothing) so a plan binding items[0]
                    # fails as "index 0 out of range" — the clear signal that
                    # the QUERY found nothing — instead of "key not found".
                    se.result_data = {**result.data, "items": items}
                # fork_repository replies may carry only {id, url}; surface the
                # fork's REAL owner/name/full_name from the url — a renamed
                # source repo forks under its CURRENT name, so later steps must
                # bind these instead of assuming the requested name survives.
                if (step.tool == "fork_repository" and isinstance(result.data, dict)
                        and result.data.get("url")
                        and ("name" not in result.data or "owner" not in result.data)):
                    parsed = _owner_repo_from_url(str(result.data["url"]))
                    if parsed:
                        o, nm = parsed
                        d = {**result.data}
                        d.setdefault("name", nm)
                        d.setdefault("full_name", f"{o}/{nm}")
                        d.setdefault("owner", {"login": o})
                        se.result_data = d
                # ...and when GitHub forks ASYNCHRONOUSLY the server answers with
                # the bare text "Fork is in progress" — no url to parse. The
                # planner guardrails require later steps to bind the fork's
                # owner/name from THIS step, so a string result kills the whole
                # plan with a binding_error that no replan can fix. Derive the
                # identity from the call instead: a fork lands under the caller's
                # account (or `organization`) keeping the source name unless
                # `name` overrode it. Best-effort — if the login can't be
                # resolved the raw result is left untouched.
                if step.tool == "fork_repository" and not isinstance(se.result_data, dict):
                    ident = self._fork_identity(client, resolved, se.result_data)
                    if ident is not None:
                        se.result_data = ident
                # issue_write returns only {id, url}; surface the issue NUMBER
                # (the handle every follow-up tool needs) from the url.
                if (step.tool == "issue_write" and isinstance(result.data, dict)
                        and "number" not in result.data):
                    m = re.search(r"/issues/(\d+)$", str(result.data.get("url") or ""))
                    if m:
                        se.result_data = {**result.data, "number": int(m.group(1))}
                # Same treatment for create_pull_request ({id, url} only):
                # surface the PR NUMBER so follow-ups (merge, update, review)
                # can bind "$sN.number" instead of failing into a replan.
                if (step.tool == "create_pull_request" and isinstance(result.data, dict)
                        and "number" not in result.data):
                    m = re.search(r"/pull/(\d+)$", str(result.data.get("url") or ""))
                    if m:
                        se.result_data = {**result.data, "number": int(m.group(1))}
                # Record issues created by THIS attempt for the pre-flight
                # dedup exemption (intentional same-title issues in one plan).
                if (step.tool == "issue_write"
                        and str(resolved.get("method") or "create") == "create"
                        and isinstance(se.result_data, dict)
                        and isinstance(se.result_data.get("number"), int)):
                    self._created_issues.add((resolved.get("owner"),
                                              resolved.get("repo"),
                                              se.result_data["number"]))
                se.result_content = result.content
                if healed_path:
                    self._event(event="step_healed", id=step.id, tool=step.tool,
                                reason="path_resolved")
                self.log.info("[%s] OK (attempt %d, %.2fs)", step.id, n, dur)
                self._event(event="step_success", id=step.id, attempt=n,
                            data_preview=_preview(result.data))
                return se
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                dur = time.time() - t0
                msg = getattr(e, "text", None) or str(e)
                # Idempotency: re-creating an existing repo/branch (e.g. on a
                # replan) is not a real failure. NOT applied to file writes,
                # where "already exists" means the content was not written.
                if isinstance(e, MCPToolError) and _is_benign_exists(step.tool, msg):
                    se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                    se.status = "success"
                    # Echo the identifying args instead of None so later
                    # "$sN...." bindings into this step still resolve.
                    se.result_data = _benign_exists_result(step.tool, resolved)
                    self.log.warning("[%s] %s already exists -> success (idempotent)",
                                      step.id, step.tool)
                    self._event(event="step_idempotent", id=step.id, tool=step.tool)
                    return se
                # File exists and the server wants its SHA. Retrying the bare
                # write can only fail again, and a replan just rediscovers the
                # SHA — so fail fast on the retry loop (B) and heal in place (A):
                # fetch the file, then accept it (identical content) or rewrite
                # it with the SHA.
                if isinstance(e, MCPToolError) and _needs_file_sha(step.tool, msg):
                    se.attempts.append(Attempt(n, "tool_error", msg, dur))
                    self.log.warning("[%s] %s: file exists (server wants SHA); "
                                     "self-healing instead of retrying", step.id, step.tool)
                    self._event(event="step_heal_start", id=step.id, tool=step.tool)
                    return self._heal_existing_file(client, step, resolved, se,
                                                    original_error=msg)
                # (idempotency) create_pull_request: GitHub refuses a second PR
                # for the same head/base, and retrying/replanning can only
                # rediscover the same PR. Adopt the existing one instead (with
                # its real number) so follow-up bindings keep working.
                if (isinstance(e, MCPToolError) and step.tool == "create_pull_request"
                        and _PR_EXISTS_MARKER in msg.lower()):
                    se.attempts.append(Attempt(n, "tool_error", msg, dur))
                    self.log.warning("[%s] %s: a PR for this head already exists; "
                                     "self-healing by adopting it", step.id, step.tool)
                    self._event(event="step_heal_start", id=step.id, tool=step.tool)
                    return self._heal_existing_pr(client, step, resolved, se,
                                                  original_error=msg)
                # (idempotency) merge_pull_request: on a re-run the PR may be
                # ALREADY merged — the desired end state. The error text alone
                # can't distinguish that from a genuinely blocked merge ("Pull
                # Request is not mergeable" covers both), so probe the live PR
                # and decide by its `merged` flag. Not merged -> fall through
                # to the normal transient-retry / fail handling.
                if isinstance(e, MCPToolError) and step.tool == "merge_pull_request":
                    pr = self._already_merged(client, resolved)
                    if pr is not None:
                        se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                        se.status = "success"
                        se.result_data = pr  # live PR object: number, merged=true, ...
                        self.log.warning("[%s] merge_pull_request: PR #%s is already "
                                         "merged -> success (idempotent)",
                                         step.id, pr.get("number"))
                        self._event(event="step_idempotent", id=step.id,
                                    tool=step.tool, reason="already_merged")
                        return se
                # (idempotency) delete_file on an already-deleted path (re-run):
                # absence IS the deletion's desired end state. Verified by a
                # get_file_contents probe rather than by error text (the git
                # tree API's messages vary). Branch-lookup failures are
                # excluded: a missing BRANCH means a wrong plan, not prior
                # success. File still present -> real failure, fall through.
                if (isinstance(e, MCPToolError) and step.tool == "delete_file"
                        and "failed to get branch reference" not in msg.lower()
                        and self._deleted_file_absent(client, resolved)):
                    se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                    se.status = "success"
                    # mirror delete_file's real result shape ({content, commit})
                    se.result_data = {"content": None, "commit": None}
                    self.log.warning("[%s] delete_file: %r is already absent -> "
                                     "success (idempotent)", step.id, resolved.get("path"))
                    self._event(event="step_idempotent", id=step.id, tool=step.tool,
                                reason="already_deleted")
                    return se
                # (idempotency) sub_issue_write(add): re-attaching a sub-issue
                # that is already linked to the parent is rejected by GitHub
                # (422). The exact error wording varies, so instead of matching
                # text, ANY failed `add` triggers a get_sub_issues probe; if the
                # child is already attached, the desired end state holds ->
                # idempotent success. Not attached -> real failure, fall through.
                if (isinstance(e, MCPToolError) and step.tool == "sub_issue_write"
                        and str(resolved.get("method") or "").lower() == "add"):
                    sub = self._sub_issue_already_attached(client, resolved)
                    if sub is not None:
                        se.attempts.append(Attempt(n, "success_idempotent", msg, dur))
                        se.status = "success"
                        se.result_data = sub  # live sub-issue object: id, number, ...
                        self.log.warning("[%s] sub_issue_write: sub-issue %s is "
                                         "already attached to issue #%s -> success "
                                         "(idempotent)", step.id,
                                         resolved.get("sub_issue_id"),
                                         resolved.get("issue_number"))
                        self._event(event="step_idempotent", id=step.id,
                                    tool=step.tool, reason="sub_issue_attached")
                        return se
                # (settle-wait) actions_get(get_workflow) 404 for a workflow
                # file THIS plan just wrote: the Actions index lags the commit,
                # so poll briefly before treating it as a real failure. Only on
                # the first attempt (the settle loop has its own budget; the
                # normal retries must not multiply it).
                if (isinstance(e, MCPToolError) and step.tool == "actions_get"
                        and n == 1
                        and str(resolved.get("method") or "") == "get_workflow"
                        and self._armed_workflow(resolved) is not None
                        and ("404" in msg or "not found" in msg.lower())):
                    settled = self._settle_actions_get(client, step, resolved)
                    if settled is not None:
                        se.attempts.append(Attempt(n, "success", None,
                                                   time.time() - t0))
                        se.status = "success"
                        se.result_data = settled.data
                        se.result_content = settled.content
                        self.log.warning("[%s] actions_get: workflow appeared "
                                         "after settle-wait -> success", step.id)
                        self._event(event="step_settled", id=step.id,
                                    tool=step.tool)
                        return se
                    # budget exhausted -> fall through to normal handling
                # (settle-wait) 404 against a repo THIS plan just forked:
                # GitHub builds forks asynchronously, so the fork (and its
                # contents) can 404 for a while after fork_repository returns.
                # Poll briefly before treating it as a real failure. First
                # attempt only — the settle loop has its own budget.
                if (isinstance(e, MCPToolError) and n == 1
                        and step.tool != "fork_repository"
                        and str(resolved.get("repo") or "") in self._forked_repos
                        and ("404" in msg or "not found" in msg.lower())):
                    settled = self._settle_fork_read(client, step, resolved)
                    if settled is not None:
                        se.attempts.append(Attempt(n, "success", None,
                                                   time.time() - t0))
                        se.status = "success"
                        se.result_data = settled.data
                        se.result_content = settled.content
                        if (step.tool == "get_file_contents"
                                and isinstance(settled.data, str)):
                            se.result_data = {"content": settled.data,
                                              "sha": _file_sha(settled.content)}
                        self.log.warning("[%s] %s: fork became available after "
                                         "settle-wait -> success", step.id, step.tool)
                        self._event(event="step_settled", id=step.id,
                                    tool=step.tool, reason="fork_ready")
                        return se
                    # budget exhausted -> fall through to normal handling
                # Replan steering: translate two well-known GitHub-drift
                # failures (renamed/transferred repos) into actionable
                # guidance carried in the error text the replanner reads.
                if isinstance(e, MCPToolError):
                    low = msg.lower()
                    if (step.tool in ("search_issues", "search_pull_requests")
                            and "cannot be searched" in low):
                        msg += (" | HINT: repo:-scoped issue search fails for "
                                "renamed or nonexistent repos. For a known "
                                "repo call list_issues(owner, repo, state/"
                                "labels) instead — it follows renames and "
                                "returns totalCount — or first resolve the "
                                "CURRENT owner/name with search_repositories "
                                "and bind items[0].owner.login / items[0].name.")
                    elif ("could not resolve to a repository" in low
                          or ("404" in low and "/repos/" in low)):
                        msg += (" | HINT: this owner/name may be stale — repos "
                                "get renamed and transferred. Resolve the "
                                "current owner/name with search_repositories "
                                "(\"<name> in:name\") and bind "
                                "items[0].owner.login / items[0].name.")
                kind = {
                    MCPToolError: "tool_error",
                    MCPProtocolError: "protocol_error",
                    MCPTransportError: "transport_error",
                }[type(e)]
                se.attempts.append(Attempt(n, kind, msg, dur))
                se.status = kind
                se.error = msg
                self.log.warning("[%s] %s on attempt %d/%d: %s",
                                  step.id, kind.upper(), n, self.max_attempts, msg)
                self._event(event="step_attempt_failed", id=step.id, attempt=n,
                            kind=kind, error=msg)
                if kind not in _TRANSIENT or n == self.max_attempts:
                    self.log.error("[%s] FAILED after %d attempt(s): %s", step.id, n, msg)
                    self._event(event="step_failed", id=step.id, kind=kind, error=msg)
                    return se
                time.sleep(min(2 ** (n - 1), 5))  # small backoff
        return se

    def _heal_existing_file(self, client: Any, step: PlanStep,
                            resolved: dict[str, Any], se: StepExecution, *,
                            original_error: str) -> StepExecution:
        """Recover from create_or_update_file's "file already exists, provide
        SHA": fetch the current file, then either accept it as-is (content already
        matches — idempotent no-op) or re-issue the write with the fetched SHA.

        Turns the common "re-run a task whose files already exist" case into a
        zero-replan success. Never raises: if the file can't be fetched or no SHA
        can be extracted, it records a tool_error so the orchestrator can still
        replan (no worse than the pre-heal behaviour)."""
        gf_args: dict[str, Any] = {
            "owner": resolved.get("owner"),
            "repo": resolved.get("repo"),
            "path": resolved.get("path"),
        }
        if resolved.get("branch"):
            gf_args["ref"] = resolved["branch"]
        t0 = time.time()
        try:
            gf = client.call("get_file_contents", gf_args)
        except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
            emsg = getattr(e, "text", None) or str(e)
            se.attempts.append(Attempt(len(se.attempts) + 1, "tool_error", emsg,
                                       time.time() - t0))
            se.status = "tool_error"
            se.error = (f"self-heal could not fetch current SHA ({emsg}); "
                        f"original: {original_error}")
            self.log.error("[%s] self-heal: get_file_contents failed: %s", step.id, emsg)
            self._event(event="step_failed", id=step.id, kind="tool_error", error=se.error)
            return se

        body = gf.data
        current = (body.get("content") if isinstance(body, dict)
                   else body if isinstance(body, str) else None)
        sha = (body.get("sha") if isinstance(body, dict) else None) or _file_sha(gf.content)
        if not sha:
            se.status = "tool_error"
            se.error = f"self-heal could not extract a SHA; original: {original_error}"
            self.log.error("[%s] self-heal: no SHA in get_file_contents result", step.id)
            self._event(event="step_failed", id=step.id, kind="tool_error", error=se.error)
            return se

        # Identical content -> the write is a no-op; accept it idempotently.
        if isinstance(current, str) and current == resolved.get("content"):
            se.attempts.append(Attempt(len(se.attempts) + 1, "success_idempotent",
                                       None, time.time() - t0))
            se.status = "success"
            se.result_data = {"content": current, "sha": sha}
            se.result_content = gf.content
            self.log.warning("[%s] %s: file already has identical content -> success "
                             "(idempotent, no rewrite)", step.id, step.tool)
            self._event(event="step_idempotent", id=step.id, tool=step.tool,
                        reason="content_match")
            return se

        # Content differs -> rewrite with the SHA the server asked for.
        healed = {**resolved, "sha": sha}
        t1 = time.time()
        try:
            result = client.call(step.tool, healed)
        except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
            emsg = getattr(e, "text", None) or str(e)
            se.attempts.append(Attempt(len(se.attempts) + 1, "tool_error", emsg,
                                       time.time() - t1))
            se.status = "tool_error"
            se.error = emsg
            self.log.error("[%s] self-heal rewrite failed: %s", step.id, emsg)
            self._event(event="step_failed", id=step.id, kind="tool_error", error=emsg)
            return se
        se.attempts.append(Attempt(len(se.attempts) + 1, "success", None,
                                   time.time() - t1))
        se.status = "success"
        se.arguments_resolved = healed  # record the SHA we actually sent
        se.result_data = result.data
        se.result_content = result.content
        self.log.warning("[%s] %s: file existed with different content -> healed by "
                         "rewriting with SHA %s", step.id, step.tool, sha)
        self._event(event="step_healed", id=step.id, tool=step.tool, sha=sha)
        return se

    def _preflight_existing(self, client: Any, step: PlanStep,
                            resolved: dict[str, Any],
                            se: StepExecution) -> StepExecution | None:
        """Pre-flight state reconciliation for the duplicate-SILENT write tools
        (issue_write create, add_issue_comment, create_gist): if the step's
        intended effect already exists on GitHub — typically left over from a
        previous run of the same task, or from an earlier attempt of this run —
        adopt the existing resource as this step's result and skip the write
        ("already done, move on"). Best-effort by design: any probe failure
        returns None and the normal call proceeds, so this can never make a run
        worse.

        Trade-off (documented, accepted): dedup keys on exact title/body match,
        so a task that intentionally re-creates an identical open issue across
        DIFFERENT runs would be deduped. Within one attempt intentional
        duplicates are preserved via self._created_issues."""
        owner, repo = resolved.get("owner"), resolved.get("repo")

        # -- issue_write(create): GitHub allows any number of same-title issues,
        # so re-runs used to pile up duplicates (no error to heal from).
        if step.tool == "issue_write":
            if str(resolved.get("method") or "create") != "create":
                return None  # updates are naturally idempotent
            title = resolved.get("title")
            if not isinstance(title, str) or not title:
                return None
            t0 = time.time()
            try:
                data = client.call("list_issues", {"owner": owner, "repo": repo,
                                                   "state": "OPEN",
                                                   "perPage": 100}).data
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                self.log.debug("[%s] pre-flight list_issues probe failed (%s); "
                               "proceeding with the write", step.id, e)
                return None
            issues = data.get("issues") if isinstance(data, dict) else data
            for it in issues if isinstance(issues, list) else []:
                num = it.get("number") if isinstance(it, dict) else None
                if (isinstance(num, int) and it.get("title") == title
                        and (owner, repo, num) not in self._created_issues):
                    se.attempts.append(Attempt(1, "success_idempotent", None,
                                               time.time() - t0))
                    se.status = "success"
                    # Shape-compatible with a real create result (+ the number
                    # the executor surfaces), so "$sN.number"/"$sN.url" bind.
                    se.result_data = {
                        "number": num,
                        "title": title,
                        "state": str(it.get("state") or "open").lower(),
                        "url": f"https://github.com/{owner}/{repo}/issues/{num}",
                    }
                    self.log.warning("[%s] issue titled %r already exists as #%s "
                                     "-> adopting it (pre-flight skip, no duplicate)",
                                     step.id, title, num)
                    self._event(event="step_preflight_skip", id=step.id,
                                tool=step.tool, reason="issue_exists", number=num)
                    return se
            return None

        # -- add_issue_comment: identical duplicates are accepted by GitHub, so
        # a replanned attempt that re-included the step used to double-post.
        if step.tool == "add_issue_comment":
            body, num = resolved.get("body"), resolved.get("issue_number")
            if not isinstance(body, str) or not body or not isinstance(num, int):
                return None
            t0 = time.time()
            try:
                data = client.call("issue_read", {"method": "get_comments",
                                                  "owner": owner, "repo": repo,
                                                  "issue_number": num}).data
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                self.log.debug("[%s] pre-flight get_comments probe failed (%s); "
                               "proceeding with the write", step.id, e)
                return None
            comments = data.get("comments") if isinstance(data, dict) else data
            for c in comments if isinstance(comments, list) else []:
                if isinstance(c, dict) and c.get("body") == body:
                    se.attempts.append(Attempt(1, "success_idempotent", None,
                                               time.time() - t0))
                    se.status = "success"
                    se.result_data = c  # the live comment: id, html_url, body, ...
                    self.log.warning("[%s] identical comment already on issue #%s "
                                     "-> adopting it (pre-flight skip, no duplicate)",
                                     step.id, num)
                    self._event(event="step_preflight_skip", id=step.id,
                                tool=step.tool, reason="comment_exists")
                    return se
            return None

        # -- create_gist: gists have no uniqueness constraint at all, so every
        # re-run/replan used to mint a brand-new duplicate gist (no error to
        # heal from). Candidate identity = description + filename (create_gist
        # is single-file by schema); the candidate is then fetched and adopted
        # ONLY if its live content matches the intended content byte-for-byte —
        # never adopt a gist that does not hold what the plan meant to write.
        if step.tool == "create_gist":
            filename, content = resolved.get("filename"), resolved.get("content")
            if not isinstance(filename, str) or not filename or not isinstance(content, str):
                return None
            desc = resolved.get("description") or ""
            t0 = time.time()
            try:
                data = client.call("list_gists", {"perPage": 100}).data
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                self.log.debug("[%s] pre-flight list_gists probe failed (%s); "
                               "proceeding with the write", step.id, e)
                return None
            for g in data if isinstance(data, list) else []:
                if not isinstance(g, dict) or (g.get("description") or "") != desc:
                    continue
                files = g.get("files")
                gid = g.get("id")
                if not gid or not (isinstance(files, dict) and filename in files):
                    continue
                # list_gists carries file metadata but no content -> fetch the
                # candidate to verify before adopting it. An unverifiable
                # candidate is skipped, not adopted.
                try:
                    full = client.call("get_gist", {"gist_id": gid}).data
                except (MCPToolError, MCPProtocolError, MCPTransportError):
                    continue
                f = (full.get("files") or {}).get(filename) if isinstance(full, dict) else None
                if isinstance(f, dict) and f.get("content") == content:
                    se.attempts.append(Attempt(1, "success_idempotent", None,
                                               time.time() - t0))
                    se.status = "success"
                    # Mirror create_gist's real result shape ({id, url}).
                    se.result_data = {"id": gid,
                                      "url": g.get("html_url") or full.get("html_url")}
                    self.log.warning("[%s] gist %r (%s) already exists as %s with "
                                     "identical content -> adopting it (pre-flight "
                                     "skip, no duplicate)", step.id, desc, filename, gid)
                    self._event(event="step_preflight_skip", id=step.id,
                                tool=step.tool, reason="gist_exists", gist_id=gid)
                    return se
            return None
        return None

    def _heal_existing_pr(self, client: Any, step: PlanStep,
                          resolved: dict[str, Any], se: StepExecution, *,
                          original_error: str) -> StepExecution:
        """Recover from create_pull_request's "a pull request already exists":
        look up the open PR for the same head and adopt it (with its real
        `number`) instead of failing into a replan that could only rediscover
        it. Never raises: if the PR can't be found (e.g. it was closed), the
        original error is recorded so the planner replans exactly as before."""
        owner = resolved.get("owner")
        head = str(resolved.get("head") or "")
        base = resolved.get("base")
        # The list filter needs "user:branch"; a bare branch means same-owner.
        head_filter = head if ":" in head else f"{owner}:{head}"
        head_ref = head.split(":", 1)[1] if ":" in head else head
        t0 = time.time()
        try:
            lp = client.call("list_pull_requests", {
                "owner": owner, "repo": resolved.get("repo"),
                "state": "open", "head": head_filter,
            })
        except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
            emsg = getattr(e, "text", None) or str(e)
            se.attempts.append(Attempt(len(se.attempts) + 1, "tool_error", emsg,
                                       time.time() - t0))
            se.status = "tool_error"
            se.error = (f"self-heal could not list pull requests ({emsg}); "
                        f"original: {original_error}")
            self.log.error("[%s] self-heal: list_pull_requests failed: %s", step.id, emsg)
            self._event(event="step_failed", id=step.id, kind="tool_error", error=se.error)
            return se

        def _ref(v: Any) -> Any:  # head/base may be {"ref": ...} or a bare string
            return v.get("ref") if isinstance(v, dict) else v

        prs = lp.data if isinstance(lp.data, list) else []
        match = next(
            (p for p in prs
             if isinstance(p, dict) and _ref(p.get("head")) == head_ref
             and (not base or _ref(p.get("base")) == base)),
            None,
        )
        if match is None:
            se.status = "tool_error"
            se.error = original_error
            self.log.error("[%s] self-heal: server says a PR exists for head %r but "
                           "none was found open; leaving original error for replan",
                           step.id, head_filter)
            self._event(event="step_failed", id=step.id, kind="tool_error",
                        error=original_error)
            return se
        se.attempts.append(Attempt(len(se.attempts) + 1, "success_idempotent", None,
                                   time.time() - t0))
        se.status = "success"
        se.result_data = match  # live PR object: number, html_url, head, base, ...
        self.log.warning("[%s] %s: PR for head %r already exists -> adopting #%s "
                         "(idempotent)", step.id, step.tool, head_filter,
                         match.get("number"))
        self._event(event="step_idempotent", id=step.id, tool=step.tool,
                    reason="pr_exists", number=match.get("number"))
        return se

    def _already_merged(self, client: Any, resolved: dict[str, Any]) -> dict[str, Any] | None:
        """Live PR lookup for merge_pull_request failures: the PR object when it
        is ALREADY MERGED (idempotent success), else None (real merge blocker —
        normal handling applies). Probe failures also return None: never turn an
        unverified state into a success."""
        pn = resolved.get("pullNumber")
        if not isinstance(pn, int):
            return None
        try:
            d = client.call("pull_request_read", {
                "method": "get", "owner": resolved.get("owner"),
                "repo": resolved.get("repo"), "pullNumber": pn,
            }).data
        except (MCPToolError, MCPProtocolError, MCPTransportError):
            return None
        return d if isinstance(d, dict) and d.get("merged") is True else None

    def _sub_issue_already_attached(self, client: Any,
                                    resolved: dict[str, Any]) -> dict[str, Any] | None:
        """Live probe for sub_issue_write(add) failures: the child issue object
        when `sub_issue_id` is ALREADY among the parent's sub-issues (idempotent
        success — the link the step wanted to create is there), else None (real
        failure — normal handling applies). Probe failures also return None:
        never turn an unverified state into a success."""
        sid = resolved.get("sub_issue_id")
        if not isinstance(sid, int):
            return None
        try:
            d = client.call("issue_read", {
                "method": "get_sub_issues", "owner": resolved.get("owner"),
                "repo": resolved.get("repo"),
                "issue_number": resolved.get("issue_number"),
            }).data
        except (MCPToolError, MCPProtocolError, MCPTransportError):
            return None
        items = d.get("sub_issues") if isinstance(d, dict) else d
        for it in items if isinstance(items, list) else []:
            # sub_issue_id is the child's issue ID (not its number)
            if isinstance(it, dict) and it.get("id") == sid:
                return it
        return None

    def _deleted_file_absent(self, client: Any, resolved: dict[str, Any]) -> bool:
        """True when delete_file's target provably no longer exists at its exact
        path/branch — i.e. the deletion's desired end state already holds. Only
        a clear not-found (or the server's wrong-path disambiguation message)
        counts as absence; transient probe failures must NOT masquerade as a
        successful deletion."""
        args: dict[str, Any] = {"owner": resolved.get("owner"),
                                "repo": resolved.get("repo"),
                                "path": resolved.get("path")}
        if resolved.get("branch"):
            args["ref"] = resolved["branch"]
        try:
            d = client.call("get_file_contents", args).data
        except MCPToolError as e:
            txt = (getattr(e, "text", None) or str(e)).lower()
            return "404" in txt or "not found" in txt or "does not exist" in txt
        except (MCPProtocolError, MCPTransportError):
            return False
        # Exact path missing -> the server returns a disambiguation STRING
        # (same signal _run_step treats as a wrong/absent path) instead of content.
        return isinstance(d, str) and d.lstrip().startswith("Resolved potential matches")

    # -------------------------------------------- Actions settle-wait helpers
    # (eventual consistency, NOT idempotency: the work is done, the API just
    # has not caught up yet — see module docstring)

    def _note_workflow_write(self, tool: str, resolved: dict[str, Any]) -> None:
        """Record workflow files written by a successful step; arms the settle
        wait for later actions_list/actions_get steps of the same plan. Only
        default-branch writes count: GitHub Actions indexes workflows from the
        default branch, so a feature-branch write can never appear in
        list_workflows and waiting on it would just stall the run."""
        if tool not in ("create_or_update_file", "push_files"):
            return
        if resolved.get("branch") not in (None, "main", "master"):
            return
        if tool == "create_or_update_file":
            paths = [resolved.get("path")]
        else:
            paths = [f.get("path") for f in resolved.get("files") or []
                     if isinstance(f, dict)]
        for p in paths:
            if isinstance(p, str) and _WORKFLOW_PATH_RE.match(p):
                self._workflow_writes.add((resolved.get("owner"),
                                           resolved.get("repo"), p))

    def _armed_workflow(self, resolved: dict[str, Any]) -> str | None:
        """The just-written workflow path that this step's ``resource_id``
        refers to (by full path or by file name, the two forms the server
        accepts), else None. Numeric ids never match — an id can only come
        from a listing, which means the workflow is already indexed."""
        rid = str(resolved.get("resource_id") or "")
        if not rid:
            return None
        for (o, r, p) in self._workflow_writes:
            if (o == resolved.get("owner") and r == resolved.get("repo")
                    and rid in (p, p.rsplit("/", 1)[-1])):
                return p
        return None

    def _settle_actions_list(self, client: Any, step: PlanStep,
                             resolved: dict[str, Any], result: Any) -> Any:
        """Bounded poll for actions_list results that look STALE relative to a
        workflow file this plan just committed. Staleness per method:
        * list_workflows — a just-written workflow path is missing from the
          returned list;
        * list_workflow_runs — the listing targets a just-written workflow
          (resource_id by file name or path) and came back empty (its
          push-triggered run has not surfaced yet).
        Returns the freshest result; on timeout or re-poll failure the last
        good result is accepted as-is — a settle-wait must never fail a step
        that already succeeded, and never fabricates data."""
        owner, repo = resolved.get("owner"), resolved.get("repo")
        pending = [p for (o, r, p) in self._workflow_writes
                   if o == owner and r == repo]
        if not pending:
            return result
        method = str(resolved.get("method") or "")
        if method == "list_workflows":
            def stale(d: Any) -> bool:
                if not isinstance(d, dict) or not isinstance(d.get("workflows"), list):
                    return False  # unknown shape -> cannot judge, accept it
                have = {w.get("path") for w in d["workflows"] if isinstance(w, dict)}
                return any(p not in have for p in pending)
        elif method == "list_workflow_runs":
            if self._armed_workflow(resolved) is None:
                return result  # not about a workflow this plan just wrote
            def stale(d: Any) -> bool:
                return (isinstance(d, dict)
                        and isinstance(d.get("workflow_runs"), list)
                        and not d["workflow_runs"])
        else:
            return result
        if not stale(result.data):
            return result
        for i, pause in enumerate(self.settle_sleeps, 1):
            self.log.warning("[%s] actions_list(%s) looks stale (Actions index "
                             "lagging the workflow commit); waiting %.1fs "
                             "(settle poll %d/%d)", step.id, method, pause, i,
                             len(self.settle_sleeps))
            self._event(event="step_settle_wait", id=step.id, tool=step.tool,
                        method=method, wait_s=pause, poll=i)
            time.sleep(pause)
            try:
                fresh = client.call(step.tool, resolved)
            except (MCPToolError, MCPProtocolError, MCPTransportError) as e:
                self.log.warning("[%s] settle re-poll failed (%s); keeping the "
                                 "last good result", step.id, e)
                return result
            result = fresh
            if not stale(result.data):
                self.log.warning("[%s] Actions API settled after %d poll(s)",
                                 step.id, i)
                self._event(event="step_settled", id=step.id, tool=step.tool,
                            polls=i)
                return result
        self.log.warning("[%s] Actions API still stale after ~%.0fs; accepting "
                         "the live result as-is", step.id, sum(self.settle_sleeps))
        self._event(event="step_settle_timeout", id=step.id, tool=step.tool)
        return result

    def _settle_actions_get(self, client: Any, step: PlanStep,
                            resolved: dict[str, Any]) -> Any | None:
        """Bounded re-poll for actions_get(get_workflow) 404s on a workflow
        this plan just wrote: returns the successful result once the Actions
        index catches up, else None (real failure — normal handling applies)."""
        for i, pause in enumerate(self.settle_sleeps, 1):
            self.log.warning("[%s] actions_get 404 on a just-written workflow; "
                             "waiting %.1fs (settle poll %d/%d)", step.id,
                             pause, i, len(self.settle_sleeps))
            self._event(event="step_settle_wait", id=step.id, tool=step.tool,
                        wait_s=pause, poll=i)
            time.sleep(pause)
            try:
                return client.call(step.tool, resolved)
            except MCPToolError as e:
                emsg = (getattr(e, "text", None) or str(e)).lower()
                if "404" in emsg or "not found" in emsg:
                    continue  # still not indexed -> keep waiting
                return None   # different failure -> not a settle problem
            except (MCPProtocolError, MCPTransportError):
                return None
        return None

    def _settle_fork_read(self, client: Any, step: PlanStep,
                          resolved: dict[str, Any]) -> Any | None:
        """Bounded re-poll for 404s against a repo THIS plan just forked:
        GitHub creates forks asynchronously, so the new fork can 404 until it
        materializes. Returns the successful result once the fork is ready,
        else None (real failure — normal handling applies)."""
        for i, pause in enumerate(self.settle_sleeps, 1):
            self.log.warning("[%s] %s: 404 on just-forked repo %r; waiting "
                             "%.1fs (settle poll %d/%d)", step.id, step.tool,
                             resolved.get("repo"), pause, i,
                             len(self.settle_sleeps))
            self._event(event="step_settle_wait", id=step.id, tool=step.tool,
                        reason="fork_not_ready", wait_s=pause, poll=i)
            time.sleep(pause)
            try:
                return client.call(step.tool, resolved)
            except MCPToolError as e:
                emsg = (getattr(e, "text", None) or str(e)).lower()
                if "404" in emsg or "not found" in emsg:
                    continue  # fork still materializing -> keep waiting
                return None   # different failure -> not a settle problem
            except (MCPProtocolError, MCPTransportError):
                return None
        return None

    def _authenticated_login(self, client: Any) -> str | None:
        """The authenticated user's login, fetched once via get_me and cached.
        None when it cannot be resolved — callers must degrade, never guess."""
        if self._me_login is None:
            try:
                d = client.call("get_me", {}).data
            except (MCPToolError, MCPProtocolError, MCPTransportError):
                return None
            login = d.get("login") if isinstance(d, dict) else None
            if not isinstance(login, str) or not login:
                return None
            self._me_login = login
        return self._me_login

    def _fork_identity(self, client: Any, resolved: dict[str, Any],
                       raw: Any) -> dict[str, Any] | None:
        """The fork's {name, full_name, owner:{login}} when fork_repository
        answered with a non-dict (the async "Fork is in progress" text), so
        later "$sN.owner.login" / "$sN.name" bindings still resolve.

        Both parts are knowable from the call itself: the fork lands in
        `organization` when given, else the authenticated account, and keeps the
        SOURCE repo's name unless `name` overrode it. `raw` is preserved under
        `message` so nothing the server said is lost. Returns None when the
        owner cannot be established — better a binding_error than a wrong repo.
        """
        name = resolved.get("name") or resolved.get("repo")
        if not isinstance(name, str) or not name:
            return None
        owner = resolved.get("organization")
        if not (isinstance(owner, str) and owner):
            owner = self._authenticated_login(client)
        if not owner:
            return None
        self.log.info("fork_repository returned no object (%r); resolved the "
                      "fork to %s/%s from the request", raw, owner, name)
        return {"name": name, "full_name": f"{owner}/{name}",
                "owner": {"login": owner}, "message": raw}

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
