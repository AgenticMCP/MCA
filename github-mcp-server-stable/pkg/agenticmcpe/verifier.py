"""Verifier agent: generate + run an execution-based verification script.

The verifier proves the task was carried out correctly. It assembles ONE python
script (``verify.py``) combining three evaluator families, then executes it:

  (1) Format Evaluators  — strict structure/type/required-field checks on each
      step's output. Deterministic, derived from the plan's `expect_fields`.
  (2) Static Evaluators  — value checks: numeric boundaries + fixed-value
      invariants (`invariants`, `bounds`), plus input-schema numeric bounds.
      Deterministic.
  (3) Dynamic Evaluators — real-time / metamorphic checks that re-query GitHub
      live (e.g. "the created issue now exists and is open", "issue count
      increased by 1") and complementary test cases. This section is generated
      by the LLM, grounded on the tool's Go implementation + unit test, then
      compile-checked and sandboxed so a bad snippet can't crash verification.

This hybrid (deterministic core + LLM-only-for-dynamic) is the key reliability
optimization over fully LLM-generated verification scripts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

from .toolsource import load_for_verifier
from .config import LLMClient, Settings, extract_json
from .executor import ExecutionTrace
from .planner import Plan

_DYNAMIC_SYSTEM = """\
You write the BODY of one Python function that VERIFIES a GitHub task was done \
correctly, by re-querying GitHub live (read-only).

You are given (as JSON): the task, available_tools, the plan steps with their \
post_action_properties, the executed steps with arguments_resolved and \
result_data, and the Go impl/test of the tools used.

Write ONLY Python statements (no `def` line, no imports, no markdown) forming \
the body of:  def dynamic_evaluators(gh):

USE ONLY THESE pre-defined helpers for live queries — do NOT call gh.call \
directly (raw tool signatures vary and are error-prone):
  - gh_repo_exists(gh, owner, repo) -> bool
  - gh_branches(gh, owner, repo) -> list[str]            # branch names, ONE PAGE — never use for existence checks
  - gh_branch_exists(gh, owner, repo, branch) -> bool    # pagination-proof branch existence
  - gh_issues(gh, owner, repo, state="all", labels=None, perPage=None) -> list[dict]  # each: number,title,state; pass the SAME labels/perPage filters the verified step used; state values are normalized to lowercase "open"/"closed" — always compare lowercase
  - labels_of(issue) -> list[str]                        # an issue's label names (any shape)
  - gh_labels(gh, owner, repo, number) -> list[str]      # label names on one issue OR PULL REQUEST, read directly
  - gh_label(gh, owner, repo, name) -> dict|None         # a label DEFINITION: name, color (lowercase, no "#"), description; None when it does not exist
  - gh_issue_comments(gh, owner, repo, issue_number) -> list[str]  # comment bodies
  - gh_issue(gh, owner, repo, number) -> dict|None         # ONE issue, WITH has_parent / has_children / sub_issues_summary
  - gh_sub_issues(gh, owner, repo, number) -> list[int]    # numbers of the sub-issues attached to that issue
  - gh_file(gh, owner, repo, path, ref=None) -> str|None   # file content; ref is the branch; None (never a server message) when the path cannot be resolved
  - gh_prs(gh, owner, repo, state="all") -> list[dict]     # ONE PAGE; each PR's `merged` is a normalized bool
  - gh_pr(gh, owner, repo, number) -> dict|None            # ONE PR by number; its ["merged"] is the merge state
  - gh_review_threads(gh, owner, repo, pull_number) -> list[dict]  # review threads; each {"id","is_resolved","comments":[{"id","body","path","line","author"}]}
  - gh_gist(gh, gist_id) -> dict|None                      # {"id","description","public","files": {name: content}}
  - poll(predicate, timeout=120, interval=10) -> (ok, value)  # predicate()->(ok,value); async effects
  - gh_issue_count(gh, owner, repo, state="open", labels=None) -> int|None  # TRUE total (search totalCount), not a page length
  - workflow_run_summary(gh, owner, repo) -> [(name, status, conclusion)]  # did the Action even fire?
  - require_owner() -> str   # the authenticated owner; use this and NEVER re-derive it
  - require_repo() -> str    # the repo this run acted on; NEVER re-derive it either
  - require_target() -> (owner, repo)  # that repo WITH its real owner; use THIS pair for live re-queries
  - step_arg(step_id, key, default=None) / first_arg(tool_or_tools, key, default=None)  # safe argument lookup
  - created_number(step_or_tool) -> int|None  # the number of an issue/PR a step CREATED — it exists ONLY in result_data
Also in scope: check(category, name, passed, detail) — ALWAYS category="dynamic"; \
the TRACE and PLAN dicts; the `time` module. TRACE and PLAN are DICTS — to \
iterate steps use the list aliases `executed_steps` (= TRACE["steps"]) and \
`plan_steps` (= PLAN["steps"]); NEVER iterate TRACE or PLAN directly.

Rules:
  - If `gh is None`: check("dynamic","live_requery",True,"skipped, no token"); return.
  - For the repository this run acted on ALWAYS use `owner, repo = \
require_target()` — the pair from the run's own steps: a repo it created or \
forked lives under the authenticated user, a third-party repo it only read \
keeps its REAL owner (("sveltejs", "svelte") — never the authenticated login \
paired with someone else's repo). require_owner() alone is the authenticated \
login (for "my account" checks); require_repo() alone is just the name. ONLY \
when the task touches a repository: an account-only task (notifications, \
gists, starred repos, projects, global advisories) must NOT call \
require_target()/require_repo(), that would record a failed check. Do NOT \
re-derive these values: get_me carries its login in result_data \
(arguments_resolved is empty), and create_repository names the repo in \
`name`, NOT `repo` — those mistakes produce a None that then fails every later \
check against "None/repo" instead of saying what went wrong. \
The number of an issue or PR the run CREATED is NOT an \
argument of the step that created it — read it with \
created_number("create_pull_request") / created_number("s4"); \
first_arg("create_pull_request","number") always returns None and the check \
then reports "PR None not found" for a run that did create the PR. \
For any OTHER value (branch, path, a number the task itself supplied) read it \
with step_arg("s3","branch") / first_arg("issue_write","title") — NEVER index \
arguments_resolved directly, because a KeyError inside a generator expression \
skips the next(..., None) default and aborts the whole section. Never hardcode \
placeholders.
  - Verify the OBSERVABLE OUTCOMES the task asked for, NOT implementation details. \
Do NOT byte-compare generated workflow YAML or source files.
  - Assert real effects: repo/branches/files exist; a COPIED file's content equals \
its source (gh_file on both; it returns None when a path cannot be resolved, so \
require both sides to be non-None before comparing); issues exist with the \
right labels (use labels_of(issue) — never index an issue's labels yourself).
  - To check the labels on a PULL REQUEST use gh_labels(gh, owner, repo, \
pr_number) (or gh_issue + labels_of). The issue LIST never contains pull \
requests, so gh_issues(..., labels=[...]) and then looking for the PR number \
ALWAYS comes back empty and fails a run that did apply the labels.
  - A LABEL'S OWN DEFINITION (does it exist, what colour/description) comes from \
gh_label(gh, owner, repo, name) — gh_labels answers a different question, which \
labels are ON an issue. Its "color" is already lowercase with no leading "#", so \
compare it to the task's colour the same way: \
str(expected).lstrip("#").lower(). A None means the label does not exist.
  - REVIEW THREADS: a review comment is anchored to a file and a line and NEVER \
appears in gh_issue_comments. Read them with gh_review_threads(gh, owner, repo, \
pr_number). A THREADED reply is a second comment in the SAME thread, so assert \
len(thread["comments"]) > 1 on the thread containing the original — not merely \
that the body exists somewhere. To search all bodies: [c["body"] for t in \
gh_review_threads(...) for c in t["comments"]].
  - GISTS: read one back with gh_gist(gh, gist_id), whose "files" is already \
flattened to {filename: content}. The gist id is not an argument of the step \
that created it — take it from that step's result_data via step_arg/created ids, \
or from the gist_id argument of a later update step. A gist task is \
ACCOUNT-LEVEL: do NOT call require_target() or require_repo() for it.
  - POLL ONLY FOR AUTOMATION. An effect the run performed itself — a comment it \
posted, a label it applied, a file it committed, a PR it merged — is already \
durable by the time you run: assert it directly with ONE query, no poll. Reserve \
poll(..., timeout=300) for an effect produced by something the run only \
TRIGGERED: a GitHub Action posting comments/labels/closing issues, where runner \
queue latency alone routinely exceeds 2 minutes. Polling for an effect that \
never arrives because the run genuinely failed costs five minutes per check and \
tells you nothing a single query would not.
  - FILE CONTENTS ARE ALREADY VERIFIED: a deterministic evaluator re-fetches
every executed file write and compares it byte-for-byte against the step's
`content` argument. Do NOT write your own file-content equality checks, and
NEVER reconstruct expected file text from the task prose (it may be mangled —
lost underscores, added newlines). Use gh_file only for EXISTENCE checks or
cross-file relations (e.g. a copied file equals its live source), at the EXACT
path/branch found in executed_steps[*].arguments_resolved — on BOTH sides,
including the UPSTREAM source path the step actually fetched ("chat/train.py",
not the repo-root name "train.py").
  - List helpers return ONE PAGE and apply ONLY the filters you pass. When the
step under verification queried with labels/state/perPage, pass the SAME
filters before asserting membership; NEVER assert "at most N" or membership
against an unfiltered/unlimited call.
  - COUNTS: to compare against a number the run recorded, use
gh_issue_count(...) — never len(gh_issues(...)). gh_issues returns one page, so
its length is capped by perPage (len == 1 when perPage=1) and comparing that to
a real total fails a correct run. If gh_issue_count returns None, skip the
comparison and check something you can establish instead.
  - SUB-ISSUES: hierarchy lives only on single-issue reads — assert the child's
number is in gh_sub_issues(gh, owner, repo, parent) or gh_issue(gh, owner,
repo, child)["has_parent"] is True. gh_issues() list results carry NO
has_parent / sub_issues_summary; reading them there fails a correct run.
  - MERGE STATE: gh_pr(gh, owner, repo, number)["merged"] (the single-PR
endpoint) or a non-empty "merged_at" from gh_prs. NEVER conclude "not merged"
from a list result's raw `merged` — GitHub's list endpoint does not carry it.
  - ASYNC + poll: a poll predicate must be a plain `def` or lambda returning
(ok, value) — NEVER `async def`; nothing in verify.py is awaited. All poll()
calls in a run SHARE one budget, so use one poll per effect and keep timeouts
tight. When a poll for a GitHub Action's effect
times out, call workflow_run_summary(...) and put the run status/conclusion in
the check detail, so the failure says whether the workflow never fired, is
still queued, or ran and produced the wrong result.
  - Never use bare next(...) or [0] on a possibly-empty lookup — use
next(..., None) / guards, so one miss FAILS its check instead of crashing.
  - `owner` and `repo` are always SEPARATE arguments: gh_repo_exists(gh,
"github", "github-mcp-server") — NEVER a combined "owner/repo" string.
  - To check that a branch exists use gh_branch_exists(...). NEVER assert
membership in gh_branches(...) — it returns one page and misses branches in
large repositories.
  - One check(...) per assertion. READ ONLY.
"""


_REPAIR_NOTE = """

YOUR PREVIOUS ANSWER DID NOT COMPILE.

SyntaxError: {error}

Return the SAME checks, corrected. The usual cause is a non-ASCII character in \
a code position (`→`, `—`, `“`, `≤`) or an unterminated string — use plain ASCII \
operators and close every quote. Output ONLY the corrected function body.

--- your previous answer ---
{code}
"""

# Wall-clock budget for the generated verify.py, and the exit code recorded when
# it runs out. Dynamic checks poll for async GitHub Actions effects
# (poll(..., timeout=300)), so the budget has to be generous — and hitting it is
# a limit of the harness, not evidence that the agent got the task wrong.
_SCRIPT_TIMEOUT = 600
_TIMEOUT_EXIT = 124  # conventional "timed out" status


@dataclass
class VerificationReport:
    task: str
    total: int
    passed: int
    failed: int
    results: list[dict[str, Any]]
    script_path: str
    exit_code: int
    raw_stdout: str = ""
    raw_stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return (not self.timed_out and self.exit_code == 0
                and self.total > 0 and self.failed == 0)


class VerifierAgent:
    def __init__(self, settings: Settings, llm: LLMClient | None = None,
                 tool_names: list[str] | None = None):
        self.settings = settings
        self.llm = llm
        self.tool_names = tool_names or []

    # ------------------------------------------------------------------ public
    def build_script(self, plan: Plan, trace: ExecutionTrace) -> str:
        """Assemble verify.py and write it to work_dir. Returns its path.

        Assembled by concatenation (not one big str.format) so the helper code
        keeps normal braces; only the header has three sentinel substitutions.
        """
        self.settings.ensure_work_dir()
        dynamic_body = self._dynamic_body(plan, trace)
        tools_used = sorted({s.tool for s in plan.steps})
        header = (
            _HEADER
            .replace("__REPO_ROOT__", repr(str(self.settings.repo_root)))
            .replace("__BINARY__", repr(str(self.settings.binary_path)))
            .replace("__TOOLS_USED__", json.dumps(tools_used))
        )
        script = (
            header
            + _HELPERS
            + _FORMAT_STATIC
            + "\ndef dynamic_evaluators(gh):\n"
            + textwrap.indent(dynamic_body, "    ")
            + "\n"
            + _MAIN
        )
        path = self.settings.work_dir / "verify.py"
        path.write_text(script, encoding="utf-8")
        return str(path)

    def run_script(self, script_path: str) -> VerificationReport:
        """Execute verify.py in a subprocess and parse its JSON report.

        A timeout is REPORTED, not raised: the batch drivers record an exception
        out of verify() as a crashed task, which reads exactly like the agent
        having failed the task. It comes back as one explicit failed check
        instead, with ``timed_out`` set.
        """
        token = self.settings.tokens.current() if self.settings.tokens.available else ""
        env = _subprocess_env(token)
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=str(self.settings.repo_root),
                capture_output=True,
                text=True,
                env=env,
                timeout=_SCRIPT_TIMEOUT,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout, stderr = _as_text(e.stdout), _as_text(e.stderr)
            exit_code = _TIMEOUT_EXIT

        report_obj: dict[str, Any] = {}
        # The script prints exactly one (pretty-printed) JSON report object.
        if stdout.strip():
            try:
                report_obj = extract_json(stdout)
            except ValueError:
                report_obj = {}
            if not isinstance(report_obj, dict):
                report_obj = {}
        results = list(report_obj.get("results", []))
        total = report_obj.get("total", 0)
        passed = report_obj.get("passed", 0)
        failed = report_obj.get("failed", 0)
        if timed_out:
            results.append({
                "category": "harness", "name": "verify_timeout", "passed": False,
                "detail": f"verify.py exceeded {_SCRIPT_TIMEOUT}s (dynamic checks "
                          f"poll for async effects); the run itself is unjudged",
            })
            total, failed = len(results), failed + 1
        return VerificationReport(
            task=report_obj.get("task", ""),
            total=total,
            passed=passed,
            failed=failed,
            results=results,
            script_path=script_path,
            exit_code=exit_code,
            raw_stdout=stdout,
            raw_stderr=stderr,
            timed_out=timed_out,
        )

    def verify(self, plan: Plan, trace: ExecutionTrace) -> VerificationReport:
        path = self.build_script(plan, trace)
        report = self.run_script(path)
        work_dir = self.settings.work_dir
        # Keep the subprocess output: when verify.py dies before printing its
        # report there is otherwise nothing on disk to diagnose it from.
        (work_dir / "verify.stdout.txt").write_text(report.raw_stdout, encoding="utf-8")
        (work_dir / "verify.stderr.txt").write_text(report.raw_stderr, encoding="utf-8")
        (work_dir / "verification.json").write_text(
            json.dumps(
                {
                    "ok": report.ok,
                    "total": report.total,
                    "passed": report.passed,
                    "failed": report.failed,
                    "results": report.results,
                    "exit_code": report.exit_code,
                    "timed_out": report.timed_out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return report

    # ------------------------------------------------------------------ internal
    def _dynamic_body(self, plan: Plan, trace: ExecutionTrace) -> str:
        if self.llm is None:
            return ('check("dynamic", "live_requery", True, '
                    '"skipped: no LLM configured for dynamic generation")')
        # Ground the LLM on the Go impl + unit test of the tools used,
        # preferring the prebuilt tool_sources KB (whole definition/test
        # functions) over the legacy first-mention grep, within a fixed
        # character budget split across the tools.
        tools_used = sorted({s.tool for s in plan.steps})
        budget = max(2400, 36000 // max(1, len(tools_used)))
        sources = [
            load_for_verifier(self.settings.repo_root, t, budget=budget)
            for t in tools_used
        ]
        ctx = {
            "task": plan.task,
            "available_tools": self.tool_names,
            "plan_steps": [
                {"id": s.id, "tool": s.tool, "arguments": s.arguments,
                 "post_action_properties": s.post_action_properties}
                for s in plan.steps
            ],
            "executed_steps": [
                {"id": s.id, "tool": s.tool, "arguments_resolved": s.arguments_resolved,
                 "result_data": _truncate(s.result_data)}
                for s in trace.steps
            ],
            "tool_sources": sources,
        }
        user = json.dumps(ctx, default=str, indent=2)
        code = self.llm.chat(_DYNAMIC_SYSTEM, user, json_mode=False)
        body = self._clean_body(code)
        err = _compile_error(body)
        if err is None:
            return body
        # One repair round. The recurring failures are mechanical — a stray
        # `→`/`—` in a code position, an unterminated string — and the model
        # fixes them reliably when shown the SyntaxError. Cheaper than losing
        # the entire dynamic layer, and it costs nothing on the happy path.
        repaired = self._clean_body(self.llm.chat(
            _DYNAMIC_SYSTEM,
            user + _REPAIR_NOTE.format(error=err, code=body),
            json_mode=False))
        if _compile_error(repaired) is None:
            return repaired
        detail = json.dumps(f"LLM produced invalid python (after 1 repair "
                            f"attempt): {err[:110]}")
        return f'check("dynamic", "live_requery", False, {detail})'

    @staticmethod
    def _sanitize_body(code: str) -> str:
        """Clean + compile-check, falling back to a recorded failure.

        Kept for callers that have no LLM to retry with; `_dynamic_body` uses
        `_clean_body` + `_compile_error` so it can attempt one repair first.
        """
        body = VerifierAgent._clean_body(code)
        err = _compile_error(body)
        if err is None:
            return body
        # ONE correctly escaped literal. Interpolating a repr() inside a
        # quoted literal produced nested quotes whenever the SyntaxError text
        # contained one, so verify.py itself failed to compile and the run
        # reported zero checks — the opposite of downgrading a bad snippet to
        # a recorded failure.
        detail = json.dumps(f"LLM produced invalid python: {err[:120]}")
        return f'check("dynamic", "live_requery", False, {detail})'

    @staticmethod
    def _clean_body(code: str) -> str:
        """Strip fences / a `<think>` block / a leading `def` line."""
        code = code.strip()
        if "</think>" in code:
            code = code.rpartition("</think>")[2].strip()
        if code.startswith("```"):
            parts = code.split("```")
            code = parts[1] if len(parts) > 1 else code
            if code.lstrip().lower().startswith("python"):
                code = code.lstrip()[6:]
        lines = code.strip("\n").splitlines()
        # drop a leading "def dynamic_evaluators(...):" if the model included it
        if lines and lines[0].lstrip().startswith("def dynamic_evaluators"):
            lines = lines[1:]
            lines = [textwrap.dedent("\n".join(lines))]
        return "\n".join(lines).strip("\n") or "pass"


def _compile_error(body: str) -> str | None:
    """The SyntaxError from compiling `body` in its target signature, or None."""
    probe = "def dynamic_evaluators(gh):\n" + textwrap.indent(body, "    ")
    try:
        compile(probe, "<dynamic>", "exec")
    except SyntaxError as e:
        return str(e)
    return None


def _truncate(data: Any, limit: int = 2000) -> Any:
    """Cap a step result for the LLM prompt WITHOUT changing its shape.

    The generated checks read the REAL result_data out of trace.json, so a
    truncation that replaced an object with ``{"_truncated", "preview"}`` taught
    the LLM key names that do not exist at run time — the observed
    ``KeyError('preview')`` and several owner derivations that came back None.
    Keep every key; shorten only the values that are too long.
    """
    if len(json.dumps(data, default=str)) <= limit:
        return data
    if isinstance(data, dict):
        share = max(120, limit // max(1, len(data)))
        return {k: _truncate(v, share) for k, v in data.items()}
    if isinstance(data, list):
        share = max(120, limit // max(1, len(data)))
        kept = [_truncate(v, share) for v in data[:10]]
        if len(data) > 10:
            kept.append(f"...(+{len(data) - 10} more)")
        return kept
    if isinstance(data, str):
        return data[:limit] + "...(truncated)"
    return data  # a scalar this long is already its own shortest form


def _as_text(stream: "str | bytes | None") -> str:
    """Subprocess output as text. TimeoutExpired carries whatever was captured
    before the kill, which may be bytes even under text=True."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return str(stream)


def _subprocess_env(token: str) -> dict[str, str]:
    import os
    env = os.environ.copy()
    if token:
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token
    return env


# ---------------------------------------------------------------------------
# The generated verify.py is assembled from these pieces. The deterministic
# Format/Static evaluators and the live-query helpers are fixed; only the
# dynamic body is LLM-authored, and it must use the helpers (not raw gh.call),
# which removes the recurring class of tool-signature/result-shape bugs.
# Three __SENTINEL__ tokens in _HEADER are replaced (not str.format) so the rest
# can use normal braces.
# ---------------------------------------------------------------------------
_HEADER = '''\
#!/usr/bin/env python3
"""Auto-generated execution-based verifier. Do not edit by hand."""
import asyncio, inspect, json, os, re, sys, time
from pathlib import Path

REPO_ROOT = Path(__REPO_ROOT__)
BINARY = __BINARY__
TOOLS_USED = __TOOLS_USED__
sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
TRACE = json.loads((HERE / "trace.json").read_text(encoding="utf-8"))
PLAN = json.loads((HERE / "plan.json").read_text(encoding="utf-8"))

# A dict that ALSO allows attribute access, so an LLM-generated dynamic check
# that writes step.result_data (instead of step["result_data"]) doesn't abort the
# whole dynamic section with AttributeError. Item access is unchanged; this only
# adds tolerance (same spirit as the shape-tolerant helpers below).
class _Step(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

# Aliases matching the names used in the dynamic-generation context, so the LLM
# can reference them directly without a NameError.
executed_steps = [_Step(s) for s in TRACE.get("steps", [])]
plan_steps = [_Step(s) for s in PLAN.get("steps", [])]
task = PLAN.get("task", "")

RESULTS = []
def check(category, name, passed, detail=""):
    RESULTS.append({"category": category, "name": name,
                    "passed": bool(passed), "detail": str(detail)[:400]})

def _step_result(step_id):
    for s in TRACE.get("steps", []):
        if s.get("id") == step_id:
            return _Step(s)
    return None
'''

# Correct, defensive live-query helpers. The dynamic body MUST use these.
_HELPERS = '''
class _OwnerUnresolved(Exception):
    """Raised by require_owner()/require_repo()/require_target() once the failure is recorded."""

def step_arg(step_id, key, default=None):
    """One executed step's resolved argument, without raw indexing.

    `s["arguments_resolved"]["repo"]` raises KeyError inside a generator, where
    a `next(..., None)` default never applies, and takes the whole dynamic
    section down with it. This never raises."""
    for s in TRACE.get("steps", []):
        if s.get("id") == step_id:
            return (s.get("arguments_resolved") or {}).get(key, default)
    return default

def first_arg(tools, key, default=None):
    """The first value of `key` across steps using any of `tools` (a name or a
    list). Use this instead of guessing a step id."""
    names = [tools] if isinstance(tools, str) else list(tools)
    for s in TRACE.get("steps", []):
        if s.get("tool") in names:
            v = (s.get("arguments_resolved") or {}).get(key)
            if v not in (None, ""):
                return v
    return default

def created_number(step_or_tool, default=None):
    """The number of the issue or PR a step CREATED, read from its result.

    A created object's number exists ONLY in `result_data` — the request that
    created it never carried one. `first_arg("create_pull_request","number")`
    and `step_arg("s4","number")` therefore always return None, and the check
    built on them reports "PR None not found" for a run that did create the PR.
    Accepts a step id ("s4") or a tool name ("create_pull_request")."""
    for s in TRACE.get("steps", []):
        if s.get("id") != step_or_tool and s.get("tool") != step_or_tool:
            continue
        d = s.get("result_data")
        if isinstance(d, dict):
            for key in ("number", "issue_number", "pullNumber"):
                v = d.get(key)
                if isinstance(v, bool):
                    continue
                if isinstance(v, int):
                    return v
                if isinstance(v, str) and v.isdigit():
                    return int(v)
    return default

def owner_login():
    """The authenticated owner for this run, read from the trace: get_me's
    login, else the `owner` argument of any step that addressed a repo. None
    when the trace carries neither."""
    for s in TRACE.get("steps", []):
        if s.get("tool") == "get_me":
            d = s.get("result_data")
            if isinstance(d, dict):
                login = d.get("login") or (d.get("details") or {}).get("login")
                if isinstance(login, str) and login:
                    return login
    for s in TRACE.get("steps", []):
        o = (s.get("arguments_resolved") or {}).get("owner")
        if isinstance(o, str) and o:
            return o
    return None

def require_owner():
    """owner_login(), or record ONE failed check and stop the dynamic section.

    Re-deriving the owner by hand is the most common way a generated check goes
    wrong (reading get_me's ARGUMENTS, which are empty, instead of its result).
    A None owner then fails every later assertion against "None/repo" and buries
    the real cause, so this fails once, explicitly, and stops.
    """
    owner = owner_login()
    if not owner:
        check("dynamic", "owner_resolved", False,
              "no get_me login and no owner argument anywhere in the trace")
        raise _OwnerUnresolved()
    return owner

def repo_name():
    """The repo this run acted on: a create_repository step's `name`, else a
    fork's resulting name, else any step's `repo` argument. None if absent.

    `create_repository` names it in `name`, NOT `repo` — reading `repo` there
    is the most common KeyError in generated checks."""
    for s in TRACE.get("steps", []):
        if s.get("tool") == "create_repository":
            v = (s.get("arguments_resolved") or {}).get("name")
            if v:
                return v
    for s in TRACE.get("steps", []):
        if s.get("tool") == "fork_repository":
            d = s.get("result_data")
            if isinstance(d, dict):
                v = d.get("name") or (d.get("full_name") or "").split("/")[-1]
                if v:
                    return v
    # Priority order, NOT trace order: a task that reads an upstream repo and
    # then writes to its own names both. Scanning the trace in step order
    # returned the upstream repo whenever a replan dropped `create_repository`
    # through idempotency self-heal, so a correct run was checked against
    # `pallets/flask` instead of the study repo it had built.
    for group in (["create_or_update_file", "push_files", "create_branch",
                   "issue_write", "sub_issue_write", "label_write",
                   "create_pull_request", "pull_request_write"],
                  ["list_issues", "get_file_contents"]):
        v = first_arg(group, "repo")
        if v:
            return v
    # Read-only audits (list_releases, get_commit, list_discussions ...) name
    # the repo too, just never through those tools; missing them failed a
    # correct run on one repo_resolved check — the fallback the docstring
    # always promised.
    for s in TRACE.get("steps", []):
        v = (s.get("arguments_resolved") or {}).get("repo")
        if v:
            return v
    return None

def require_repo():
    """repo_name(), or record ONE failed check and stop the dynamic section."""
    repo = repo_name()
    if not repo:
        check("dynamic", "repo_resolved", False,
              "no create_repository name, fork result or repo argument in the trace")
        raise _OwnerUnresolved()
    return repo

def target_pair():
    """(owner, repo) of the repository this run acted on. A repo the run
    created or forked lives under the authenticated user; otherwise the pair
    comes from the first step that addressed a repo with BOTH `owner` and
    `repo` arguments (preferring the step that names repo_name()) — so a
    read-only audit of sveltejs/svelte resolves to ("sveltejs", "svelte"), not
    to the authenticated login paired with someone else's repo, which failed
    correct runs on one repo_exists check. (None, None) if nothing resolves."""
    repo = repo_name()
    if not repo:
        return None, None
    steps = TRACE.get("steps", [])
    if any(s.get("tool") in ("create_repository", "fork_repository") for s in steps):
        return owner_login(), repo
    pairs = []
    for s in steps:
        a = s.get("arguments_resolved") or {}
        o, r = a.get("owner"), a.get("repo")
        if isinstance(o, str) and o and isinstance(r, str) and r:
            pairs.append((o, r))
    for o, r in pairs:
        if r == repo:
            return o, r
    if pairs:
        return pairs[0]
    return owner_login(), repo

def require_target():
    """target_pair(), or record ONE failed check and stop the dynamic section."""
    owner, repo = target_pair()
    if not owner or not repo:
        check("dynamic", "target_resolved", False,
              "no (owner, repo) pair could be derived from the trace")
        raise _OwnerUnresolved()
    return owner, repo

def gh_repo_exists(gh, owner, repo):
    try:
        gh.call("list_branches", {"owner": owner, "repo": repo})
        return True
    except Exception:
        return False

def gh_branches(gh, owner, repo):
    try:
        data = gh.call("list_branches", {"owner": owner, "repo": repo}).data
    except Exception:
        return []
    return [b.get("name") for b in data if isinstance(b, dict)] if isinstance(data, list) else []

def gh_branch_exists(gh, owner, repo, branch):
    # Pagination-proof: resolving 1 commit on the ref succeeds iff it exists.
    try:
        gh.call("list_commits", {"owner": owner, "repo": repo, "sha": branch, "perPage": 1})
        return True
    except Exception:
        return False

def gh_issues(gh, owner, repo, state="all", labels=None, perPage=None):
    args = {"owner": owner, "repo": repo, "state": state}
    if labels:
        args["labels"] = list(labels)
    if perPage:
        args["perPage"] = perPage
    try:
        data = gh.call("list_issues", args).data
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("issues") or []
    return _norm_states(data) if isinstance(data, list) else []

def _norm_states(items):
    # GraphQL-backed tools return state OPEN/CLOSED, REST ones open/closed —
    # normalize to lowercase so generated comparisons are stable.
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("state"), str):
            it["state"] = it["state"].lower()
    return items

def gh_issue_count(gh, owner, repo, state="open", labels=None):
    """The TRUE number of matching issues, or None if it cannot be determined.

    `gh_issues` returns ONE page, so `len(gh_issues(..., perPage=1))` is 1 no
    matter how many issues exist — comparing that to a recorded total is the
    single most common way a generated count check fails a correct run. This
    asks the search API for its `totalCount` instead, which is exact and costs
    one call. Returns None rather than a wrong number when the shape is
    unexpected, so a check can skip instead of failing spuriously.
    """
    q = ["repo:%s/%s" % (owner, repo), "is:issue"]
    if state and state != "all":
        q.append("is:" + str(state).lower())
    for name in (labels or []):
        q.append('label:"%s"' % name)
    try:
        data = gh.call("search_issues", {"query": " ".join(q), "perPage": 1}).data
    except Exception:
        return None
    if isinstance(data, dict):
        for key in ("totalCount", "total_count", "total"):
            if isinstance(data.get(key), int):
                return data[key]
    return None

def labels_of(issue):
    """An issue's label names as a list[str], regardless of shape (this server
    returns labels as plain strings; they may also be objects, or absent)."""
    labs = issue.get("labels") if isinstance(issue, dict) else None
    out = []
    for l in (labs or []):
        if isinstance(l, dict):
            out.append(l.get("name", ""))
        elif isinstance(l, str):
            out.append(l)
    return out

def gh_issue_comments(gh, owner, repo, issue_number):
    try:
        data = gh.call("issue_read", {"method": "get_comments", "owner": owner,
                                      "repo": repo, "issue_number": issue_number}).data
    except Exception:
        return []
    items = data.get("comments") if isinstance(data, dict) else data
    items = items if isinstance(items, list) else []
    out = []
    for c in items:
        if isinstance(c, dict):
            out.append(c.get("body", ""))
        elif isinstance(c, str):
            out.append(c)
    return out

def gh_issue(gh, owner, repo, number):
    """ONE issue via issue_read(get) — the only issue response that carries the
    hierarchy flags has_parent / has_children / sub_issues_summary (list
    results never do). Plain dict or None."""
    try:
        number = int(number)
    except (TypeError, ValueError):
        return None
    try:
        data = gh.call("issue_read", {"method": "get", "owner": owner, "repo": repo,
                                      "issue_number": number}).data
    except Exception:
        return None
    return data if isinstance(data, dict) else None

def gh_labels(gh, owner, repo, number):
    """The label names on one issue OR pull request, read directly.

    A pull request IS an issue to the REST API, so issue_read(get) returns its
    labels. The issue LIST never returns pull requests, so filtering
    gh_issues(..., labels=[...]) and looking for the PR number always comes back
    empty and fails a run that did apply the labels."""
    item = gh_issue(gh, owner, repo, number)
    return labels_of(item) if item else []

def gh_label(gh, owner, repo, name):
    """ONE label DEFINITION: {"name", "color", "description"}, or None if the
    repository has no such label.

    `gh_labels` answers "which labels are on this issue"; this answers "does
    this label exist, and what colour/description does it carry" — the question
    a "create these three labels with these colours" task actually asks. There
    was no helper for it, so generated checks called get_label raw, forgot
    `.data`, and compared a ToolResult object to a colour string: three runs
    that had created every label with the right colour were failed by a check
    whose own detail printed that colour (ablation100 tasks 074/075/077).

    `color` is normalised to lowercase with any leading "#" stripped, because
    GitHub stores "d73a4a" while a task prompt may write "#D73A4A"."""
    try:
        data = gh.call("get_label", {"owner": owner, "repo": repo,
                                     "name": str(name)}).data
    except Exception:
        return None  # get_label errors when the label does not exist
    if not isinstance(data, dict) or not data.get("name"):
        return None
    color = data.get("color")
    if isinstance(color, str):
        data["color"] = color.lstrip("#").lower()
    return data

def gh_sub_issues(gh, owner, repo, number):
    """Numbers of the sub-issues attached to issue `number`
    (issue_read get_sub_issues); [] when there are none or on error."""
    try:
        data = gh.call("issue_read", {"method": "get_sub_issues", "owner": owner,
                                      "repo": repo, "issue_number": int(number)}).data
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("sub_issues") or data.get("issues") or data.get("items") or []
    return [i.get("number") for i in data
            if isinstance(i, dict) and i.get("number") is not None] if isinstance(data, list) else []

def _single_match_path(msg):
    """The one candidate path out of a get_file_contents disambiguation message
    ('... matching files: ["chat/train.py"]'), else None. Mirrors the executor's
    heal: with exactly one match the server just told us the real path; with
    zero or several, guessing would be wrong."""
    m = re.search(r'matching files:\\s*(\\[[^\\]]*\\])', msg)
    if not m:
        return None
    try:
        files = json.loads(m.group(1))
    except ValueError:
        return None
    if isinstance(files, list) and len(files) == 1 and isinstance(files[0], str) and files[0]:
        return files[0]
    return None

def _fetch_file(gh, owner, repo, path, ref):
    args = {"owner": owner, "repo": repo, "path": path}
    if ref:
        args["ref"] = ref
    try:
        d = gh.call("get_file_contents", args).data
    except Exception:
        return None
    if d is None:
        return None
    return d if isinstance(d, str) else json.dumps(d)

def gh_file(gh, owner, repo, path, ref=None):
    """File content, or None when the path cannot be resolved.

    When the server answers with its disambiguation message naming exactly ONE
    match, retry once at that path — the same self-heal the executor applies. A
    check written with a nested file's repo-root name ("train.py" for
    "chat/train.py") then compares content instead of an error string. Anything
    still unresolved comes back None, never the server's message text.
    """
    out = _fetch_file(gh, owner, repo, path, ref)
    if not (isinstance(out, str) and out.lstrip().startswith("Resolved potential matches")):
        return out
    fixed = _single_match_path(out)
    if not fixed or fixed == path:
        return None
    out = _fetch_file(gh, owner, repo, fixed, ref)
    if isinstance(out, str) and out.lstrip().startswith("Resolved potential matches"):
        return None
    return out

class _Ref(dict):
    """A PR head/base ref. list_pull_requests returns these as objects
    ({"ref","sha",...}); an LLM-generated check may read pr["head"].get("ref")=="x"
    OR pr["head"]=="x". This dict subclass is indexable like the object AND equals
    its ref string, so either style passes (same shape-tolerance as labels_of)."""
    def __eq__(self, other):
        if isinstance(other, str):
            return self.get("ref") == other
        return dict.__eq__(self, other)
    def __ne__(self, other):
        return not self.__eq__(other)

def _norm_pr(pr):
    """Shape-tolerance for one PR dict, in place: head/base compare as ref
    strings (see _Ref), and `merged` is a real bool. list_pull_requests returns
    the server's MinimalPullRequest, whose `merged` is omitempty and never set
    by GitHub's LIST endpoint (only merged_at is) — so a check reading
    pr.get("merged") from gh_prs used to fail a correctly merged PR."""
    if isinstance(pr, dict):
        for side in ("head", "base"):
            v = pr.get(side)
            if isinstance(v, dict):
                pr[side] = _Ref(v)
            elif isinstance(v, str):
                pr[side] = _Ref({"ref": v})
        pr["merged"] = bool(pr.get("merged")) or bool(pr.get("merged_at"))
    return pr

def gh_prs(gh, owner, repo, state="all"):
    try:
        data = gh.call("list_pull_requests", {"owner": owner, "repo": repo, "state": state}).data
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("pull_requests") or data.get("pullRequests") or data.get("items") or []
    if not isinstance(data, list):
        return []
    for pr in data:
        _norm_pr(pr)
    return _norm_states(data)

def gh_pr(gh, owner, repo, number):
    """ONE pull request via pull_request_read(get) — the single-PR endpoint, the
    only one GitHub answers with a real `merged` flag. Returns a plain dict with
    the same normalisation as gh_prs, or None when it cannot be fetched."""
    try:
        number = int(number)
    except (TypeError, ValueError):
        return None
    try:
        data = gh.call("pull_request_read", {"method": "get", "owner": owner,
                                             "repo": repo, "pullNumber": number}).data
    except Exception:
        return None
    return _norm_pr(data) if isinstance(data, dict) else None

def gh_review_threads(gh, owner, repo, pull_number):
    """The REVIEW threads on a pull request: line comments on the diff, grouped
    into the conversations they belong to.

    Each thread is {"id", "is_resolved", "comments": [...]}, and each comment is
    {"id", "body", "path", "line", "author", "html_url"} — `id` being the numeric
    REST id that `add_reply_to_pull_request_comment` replies to.

    This is NOT gh_issue_comments: a review comment is anchored to a file and a
    line and never appears in the issue conversation. A threaded reply is simply
    a second comment in the SAME thread, which is why this returns the grouping
    rather than a flat list. Four runs that posted a correct threaded reply were
    failed because no helper could read one back (ablation100 tasks
    066/067/071/072)."""
    try:
        pull_number = int(pull_number)
    except (TypeError, ValueError):
        return []
    try:
        data = gh.call("pull_request_read", {"method": "get_review_comments",
                                             "owner": owner, "repo": repo,
                                             "pullNumber": pull_number}).data
    except Exception:
        return []
    threads = data.get("review_threads") if isinstance(data, dict) else data
    out = []
    for t in threads if isinstance(threads, list) else []:
        if not isinstance(t, dict):
            continue
        # Keep the thread whole and only clean `comments`. A helper that
        # narrows a payload to the fields it imagines a check wants just
        # invents a new false failure when a check wants one of the others.
        t["comments"] = [c for c in (t.get("comments") or []) if isinstance(c, dict)]
        t["is_resolved"] = bool(t.get("is_resolved"))
        out.append(t)
    return out

def gh_gist(gh, gist_id):
    """ONE gist: the whole response with "files" flattened to {name: content},
    so "description", "public" and "html_url" are all still there. None when it
    cannot be read.

    The raw response nests each file body under files[name]["content"], and
    every generated check that guessed at that shape — or called get_gist
    without `.data` — read None out of a gist whose description and files were
    exactly right (ablation100 tasks 095-098). Only `files` is rewritten:
    narrowing the dict to the fields a helper imagines a check wants is how the
    first version of this helper broke an issue-body check that needed
    html_url."""
    if not gist_id:
        return None
    try:
        data = gh.call("get_gist", {"gist_id": str(gist_id)}).data
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    files = {}
    for name, meta in (data.get("files") or {}).items():
        files[name] = meta.get("content") if isinstance(meta, dict) else meta
    data["files"] = files
    return data

# Every poll() in a run shares ONE deadline. Several independent
# poll(..., timeout=300) calls used to serialise straight past the script's own
# wall-clock cap, so the run died with zero usable checks instead of reporting
# the ones it had. Clamping to a shared budget means a slow async effect costs
# its own check, not the whole verification.
_POLL_BUDGET_S = 420
_POLL_DEADLINE = time.time() + _POLL_BUDGET_S
POLL_EXHAUSTED = []

def _run_predicate(predicate):
    """Evaluate a poll predicate, running it to completion when it is async.

    Generated checks sometimes write `async def` predicates. Calling one
    returns a coroutine; unpacking that raised TypeError, which the old
    except-clause swallowed — so the poll timed out NO MATTER what GitHub
    said, a guaranteed false failure. Run the coroutine instead."""
    out = predicate()
    if inspect.iscoroutine(out):
        return asyncio.run(out)
    return out

def poll(predicate, timeout=120, interval=10):
    """Call predicate() -> (ok, value) until ok is truthy or the deadline.

    The deadline is min(this call's timeout, the run's shared poll budget).
    A predicate that NEVER evaluates cleanly records one explicit
    `poll_predicate_error` check — a broken predicate must read as a broken
    check, not as "the effect never appeared"."""
    end = min(time.time() + timeout, _POLL_DEADLINE)
    ok, val, clean, last_err = False, None, False, None
    while True:
        try:
            ok, val = _run_predicate(predicate)
            clean = True
        except Exception as e:
            last_err = repr(e)[:200]
            ok, val = False, None
        now = time.time()
        if ok or now >= end:
            if not ok and now >= _POLL_DEADLINE:
                POLL_EXHAUSTED.append(True)
            if not ok and not clean and last_err:
                check("dynamic", "poll_predicate_error", False,
                      "poll predicate never evaluated cleanly (its timeout "
                      "says nothing about GitHub): " + last_err)
            return ok, val
        time.sleep(min(interval, max(0.0, end - now)))

def workflow_run_summary(gh, owner, repo):
    """Recent Actions runs as (name, status, conclusion) — for telling
    'the workflow never fired' apart from 'it ran and did the wrong thing'.
    A poll that times out cannot distinguish those on its own."""
    try:
        data = gh.call("actions_list", {"method": "list_workflow_runs",
                                        "owner": owner, "repo": repo,
                                        "per_page": 10}).data
    except Exception:
        return []
    runs = data.get("workflow_runs") if isinstance(data, dict) else data
    out = []
    for r in runs if isinstance(runs, list) else []:
        if isinstance(r, dict):
            out.append((r.get("name") or r.get("display_title") or "?",
                        r.get("status"), r.get("conclusion")))
    return out
'''

_FORMAT_STATIC = '''
# ---------------- (1) Format Evaluators ----------------
# Only what is reliably knowable from the trace: did the step run and succeed,
# is there a result. Output-content correctness is the Dynamic layer's job.
def format_evaluators():
    for pstep in PLAN.get("steps", []):
        sid = pstep.get("id")
        tr = _step_result(sid)
        if tr is None:
            check("format", f"{sid}:executed", False, "step missing from trace")
            continue
        ok = tr.get("status") == "success"
        check("format", f"{sid}:status", ok,
              "success" if ok else f"status={tr.get('status')} error={tr.get('error')}")
        if ok:
            data = tr.get("result_data")
            # A benign idempotent success ("already exists") legitimately
            # carries no result payload — the resource pre-existed.
            idem = any(a.get("status") == "success_idempotent"
                       for a in tr.get("attempts") or [])
            present = idem or (data is not None and data != "")
            check("format", f"{sid}:result_present", present,
                  "idempotent (resource already existed)" if idem
                  else f"type={type(data).__name__}")

# -------------- (1b) Deterministic live file checks --------------
# Every executed file write is re-fetched and compared BYTE-FOR-BYTE against
# the exact `content` argument that was sent. No LLM judgment involved — this
# is the authoritative file-content check (the dynamic section is told not to
# re-derive expected file contents from task prose).
def file_write_evaluators(gh):
    if gh is None:
        return
    last_write = {}  # (owner, repo, branch, path) -> (sid, content)
    for tr in TRACE.get("steps", []):
        if tr.get("status") != "success":
            continue
        args = tr.get("arguments_resolved") or {}
        tool = tr.get("tool")
        if tool == "create_or_update_file":
            key = (args.get("owner"), args.get("repo"), args.get("branch"), args.get("path"))
            last_write[key] = (tr.get("id"), args.get("content"))
        elif tool == "push_files":
            for f in args.get("files") or []:
                if isinstance(f, dict):
                    key = (args.get("owner"), args.get("repo"), args.get("branch"), f.get("path"))
                    last_write[key] = (tr.get("id"), f.get("content"))
        elif tool == "delete_file":
            # A later delete means the earlier write was meant to be undone
            # (a rename/move writes the new path, then removes the old one),
            # so the path SHOULD be gone — don't assert it persisted.
            last_write.pop((args.get("owner"), args.get("repo"),
                            args.get("branch"), args.get("path")), None)
    for (owner, repo, branch, path), (sid, content) in last_write.items():
        if not all(isinstance(x, str) and x for x in (owner, repo, path)) or not isinstance(content, str):
            continue
        live = gh_file(gh, owner, repo, path, ref=branch)
        ok = live == content
        detail = f"{path}@{branch} persisted byte-exact" if ok else (
            f"{path}@{branch} live content differs from {sid}'s content argument "
            f"(live head: {repr((live or '')[:80])})")
        check("dynamic", f"{sid}:file_persisted:{path}", ok, detail)

# ---------------- (2) Static Evaluators ----------------
# Value checks on the INPUTS actually sent: id/number args positive; key string
# args non-empty.
def static_evaluators():
    for pstep in PLAN.get("steps", []):
        sid = pstep.get("id")
        tr = _step_result(sid)
        if tr is None or tr.get("status") != "success":
            continue
        for k, v in (tr.get("arguments_resolved") or {}).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, int) and ("number" in k or k.endswith("_id") or k == "id"):
                check("static", f"{sid}:argpos[{k}]", v > 0, f"{k}={v} must be positive")
            if isinstance(v, str) and k in ("owner", "repo", "title", "branch", "path", "head", "base"):
                check("static", f"{sid}:argnonempty[{k}]", v.strip() != "", f"{k} must be non-empty")

# ---------------- (3) Dynamic Evaluators (LLM-generated, uses helpers) ----------------
'''

_MAIN = '''
def _open_gh():
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return None
    try:
        from pkg.mcp_wrapper import MCPClient
        gh = MCPClient(token=token, binary=BINARY, toolsets=["all"], read_only=True)
        gh.start()
        return gh
    except Exception as e:
        check("dynamic", "client_init", False, f"could not start MCP client: {e}")
        return None

def main():
    format_evaluators()
    static_evaluators()
    gh = _open_gh()
    try:
        file_write_evaluators(gh)
    except Exception as e:
        check("dynamic", "file_write_exception", False, repr(e))
    try:
        dynamic_evaluators(gh)
    except _OwnerUnresolved:
        pass  # require_owner() already recorded the one check that explains it
    except Exception as e:  # a bad dynamic check must not crash verification
        check("dynamic", "dynamic_exception", False, repr(e))
    finally:
        if gh is not None:
            try:
                gh.close()
            except Exception:
                pass
    passed = sum(1 for r in RESULTS if r["passed"])
    report = {
        "task": PLAN.get("task", ""),
        "total": len(RESULTS),
        "passed": passed,
        "failed": len(RESULTS) - passed,
        "results": RESULTS,
    }
    print(json.dumps(report, indent=2))
    sys.exit(0 if RESULTS and passed == len(RESULTS) else 1)

if __name__ == "__main__":
    main()
'''


__all__ = ["VerifierAgent", "VerificationReport"]
