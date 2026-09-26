"""Offline harness for the executor idempotency changes (no network, no LLM).

Run it after touching the executor's idempotency / settle-wait / binding paths:

    pkg/venv/bin/python pkg/agenticmcpe/idem_selftest.py

Drives ExecutionAgent with a scripted
FakeClient and asserts each new behaviour: issue/comment/gist pre-flight dedup,
PR-exists heal, already-merged heal, already-deleted heal, already-attached
sub-issue heal, benign-exists arg echo (+bindings), PR number surfacing, the
Actions eventual-consistency settle-wait, and the pre-existing file-SHA heal
as a regression check.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pkg.agenticmcpe.executor import ExecutionAgent  # noqa: E402
from pkg.agenticmcpe.planner import Plan, PlanStep  # noqa: E402
from pkg.mcp_wrapper import MCPToolError  # noqa: E402


class FakeClient:
    """Scripted MCP client: handlers[tool] -> data | Exception | callable."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []          # [(tool, args), ...]
        self.stderr_output = ""

    def get_tool(self, name):
        return SimpleNamespace(input_schema={"type": "object"})

    def call(self, tool, args):
        self.calls.append((tool, dict(args)))
        h = self.handlers[tool]
        out = h(args) if callable(h) else h
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(data=out, content=[])


def agent():
    wd = Path(tempfile.mkdtemp(prefix="idem-"))
    settings = SimpleNamespace(work_dir=wd, run_id=wd.name,
                               ensure_work_dir=lambda: None)
    return ExecutionAgent(settings, client=FakeClient({}), log_to_console=False)


def run(handlers, steps, settle_sleeps=None):
    ag = agent()
    if settle_sleeps is not None:  # instant Actions settle polls in tests
        ag.settle_sleeps = settle_sleeps
    fake = FakeClient(handlers)
    ag._client = fake
    plan = Plan(task="t", summary="", steps=[PlanStep(**s) for s in steps])
    trace = ag.run(plan)
    return trace, fake


def seq(*payloads):
    """Handler that returns/raises each payload in turn (last one repeats)."""
    state = {"n": 0}

    def h(args):
        out = payloads[min(state["n"], len(payloads) - 1)]
        state["n"] += 1
        return out

    return h


def err(tool, text):
    return MCPToolError(tool, text, {})


PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("ok   " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


# ---------------------------------------------------------------- S1 issue dedup
tr, fk = run(
    {"list_issues": {"issues": [{"number": 7, "title": "X", "state": "OPEN"}],
                     "totalCount": 1}},
    [{"id": "s0", "tool": "issue_write",
      "arguments": {"method": "create", "owner": "o", "repo": "r",
                    "title": "X", "body": "b"}}],
)
s = tr.steps[0]
check("S1 issue dedup: success", tr.success and s.status == "success")
check("S1 issue dedup: adopted #7", isinstance(s.result_data, dict)
      and s.result_data.get("number") == 7
      and s.result_data.get("url", "").endswith("/issues/7"))
check("S1 issue dedup: no create call",
      [c[0] for c in fk.calls] == ["list_issues"])
check("S1 issue dedup: idempotent attempt",
      s.attempts and s.attempts[0].status == "success_idempotent")

# ------------------------------------------- S2 same-run duplicate titles kept
state = {"issues": []}


def li(args):
    return {"issues": list(state["issues"]), "totalCount": len(state["issues"])}


def iw(args):
    n = len(state["issues"]) + 1
    state["issues"].append({"number": n, "title": args["title"], "state": "OPEN"})
    return {"id": str(n), "url": f"https://github.com/o/r/issues/{n}"}


tr, fk = run(
    {"list_issues": li, "issue_write": iw},
    [{"id": "s0", "tool": "issue_write",
      "arguments": {"method": "create", "owner": "o", "repo": "r", "title": "X"}},
     {"id": "s1", "tool": "issue_write",
      "arguments": {"method": "create", "owner": "o", "repo": "r", "title": "X"}}],
)
nums = [st.result_data.get("number") for st in tr.steps]
check("S2 same-run duplicates: both created", tr.success and nums == [1, 2],
      f"nums={nums}")
check("S2 same-run duplicates: 2 create calls",
      [c[0] for c in fk.calls].count("issue_write") == 2)

# rerun of the same task on the now-populated repo: dedup, no 3rd issue
tr, fk = run(
    {"list_issues": li, "issue_write": iw},
    [{"id": "s0", "tool": "issue_write",
      "arguments": {"method": "create", "owner": "o", "repo": "r", "title": "X"}}],
)
check("S2 rerun: adopted existing, no create",
      tr.success and tr.steps[0].result_data.get("number") == 1
      and "issue_write" not in [c[0] for c in fk.calls])

# ---------------------------------------------------------- S3 comment dedup
tr, fk = run(
    {"issue_read": [{"id": 9, "body": "B", "html_url": "u"}]},
    [{"id": "s0", "tool": "add_issue_comment",
      "arguments": {"owner": "o", "repo": "r", "issue_number": 7, "body": "B"}}],
)
check("S3 comment dedup: success, no post",
      tr.success and tr.steps[0].status == "success"
      and [c[0] for c in fk.calls] == ["issue_read"])
check("S3 comment dedup: adopted live comment",
      tr.steps[0].result_data.get("id") == 9)

tr, fk = run(
    {"issue_read": [{"id": 9, "body": "B"}],
     "add_issue_comment": {"id": "10", "url": "u"}},
    [{"id": "s0", "tool": "add_issue_comment",
      "arguments": {"owner": "o", "repo": "r", "issue_number": 7, "body": "C"}}],
)
check("S3 different body: posts normally",
      tr.success and [c[0] for c in fk.calls] == ["issue_read", "add_issue_comment"])

# ------------------------------------------------- S4 PR-exists heal + binding
tr, fk = run(
    {"create_pull_request": err("create_pull_request",
                                "failed to create pull request: 422 Validation "
                                "Failed [A pull request already exists for o:feat.]"),
     "list_pull_requests": [{"number": 3, "title": "T", "state": "open",
                             "head": {"ref": "feat"}, "base": {"ref": "main"},
                             "merged": False}],
     "merge_pull_request": {"merged": True, "sha": "abc"}},
    [{"id": "s0", "tool": "create_pull_request",
      "arguments": {"owner": "o", "repo": "r", "title": "T",
                    "head": "feat", "base": "main"}},
     {"id": "s1", "tool": "merge_pull_request",
      "arguments": {"owner": "o", "repo": "r", "pullNumber": "$s0.number"}}],
)
check("S4 PR heal: adopted #3 and merged via binding",
      tr.success and tr.steps[0].result_data.get("number") == 3)
merge_args = next(a for t, a in fk.calls if t == "merge_pull_request")
check("S4 PR heal: binding resolved to int 3",
      merge_args["pullNumber"] == 3 and isinstance(merge_args["pullNumber"], int))

# PR-exists but lookup finds nothing open -> keep original error for replan
tr, fk = run(
    {"create_pull_request": err("create_pull_request",
                                "A pull request already exists for o:feat."),
     "list_pull_requests": []},
    [{"id": "s0", "tool": "create_pull_request",
      "arguments": {"owner": "o", "repo": "r", "title": "T",
                    "head": "feat", "base": "main"}}],
)
check("S4b PR heal miss: original error kept",
      not tr.success and tr.steps[0].status == "tool_error"
      and "already exists" in (tr.steps[0].error or ""))

# ------------------------------------------------- S5 merge: already merged
tr, fk = run(
    {"merge_pull_request": err("merge_pull_request",
                               "405 Pull Request is not mergeable"),
     "pull_request_read": {"number": 3, "merged": True, "state": "closed"}},
    [{"id": "s0", "tool": "merge_pull_request",
      "arguments": {"owner": "o", "repo": "r", "pullNumber": 3}}],
)
check("S5 already merged: idempotent success",
      tr.success and tr.steps[0].result_data.get("merged") is True)

# ------------------------------------------------- S6 merge: genuinely blocked
tr, fk = run(
    {"merge_pull_request": lambda a: err("merge_pull_request",
                                         "405 Pull Request is not mergeable"),
     "pull_request_read": {"number": 3, "merged": False, "state": "open"}},
    [{"id": "s0", "tool": "merge_pull_request",
      "arguments": {"owner": "o", "repo": "r", "pullNumber": 3}}],
)
check("S6 blocked merge: still fails after retries",
      not tr.success and tr.steps[0].status == "tool_error"
      and len([a for a in tr.steps[0].attempts if a.status == "tool_error"]) == 3)

# ------------------------------------------------- S7 delete: already deleted
tr, fk = run(
    {"delete_file": err("delete_file",
                        "failed to create tree: 422 path does not exist"),
     "get_file_contents": err("get_file_contents",
                              "failed to get file contents: 404 Not Found")},
    [{"id": "s0", "tool": "delete_file",
      "arguments": {"owner": "o", "repo": "r", "path": "a.txt",
                    "message": "m", "branch": "main"}}],
)
check("S7 already deleted: idempotent success",
      tr.success and tr.steps[0].result_data == {"content": None, "commit": None})

# ------------------------------------------------- S8 delete: file still there
tr, fk = run(
    {"delete_file": lambda a: err("delete_file", "500 something went wrong"),
     "get_file_contents": "hello world"},
    [{"id": "s0", "tool": "delete_file",
      "arguments": {"owner": "o", "repo": "r", "path": "a.txt",
                    "message": "m", "branch": "main"}}],
)
check("S8 delete real failure: not masked",
      not tr.success and tr.steps[0].status == "tool_error")

# ------------------------------- S9 benign-exists arg echo keeps bindings alive
tr, fk = run(
    {"create_repository": err("create_repository",
                              "failed to create repository: name already exists "
                              "on this account"),
     "create_or_update_file": {"content": {"path": "R.md"}, "commit": {}}},
    [{"id": "s0", "tool": "create_repository",
      "arguments": {"name": "demo", "autoInit": False}},
     {"id": "s1", "tool": "create_or_update_file",
      "arguments": {"owner": "o", "repo": "$s0.name", "path": "R.md",
                    "content": "hi", "message": "m", "branch": "main"}}],
)
write_args = next(a for t, a in fk.calls if t == "create_or_update_file")
check("S9 benign-exists echo: repo binding resolved",
      tr.success and tr.steps[0].result_data == {"name": "demo"}
      and write_args["repo"] == "demo")

# ------------------------------------------- S10 file-SHA heal regression (A+B)
SHA = "a" * 40


def couf(args):
    if "sha" not in args:
        return err("create_or_update_file", "file already exists; provide sha")
    return {"content": {"path": args["path"]}, "commit": {}}


class ShaFake(FakeClient):
    def call(self, tool, args):
        self.calls.append((tool, dict(args)))
        if tool == "create_or_update_file":
            out = couf(args)
            if isinstance(out, Exception):
                raise out
            return SimpleNamespace(data=out, content=[])
        if tool == "get_file_contents":
            return SimpleNamespace(
                data="hello",
                content=[{"type": "text",
                          "text": f"successfully downloaded (SHA: {SHA})"}])
        raise AssertionError(tool)


ag = agent()
fake = ShaFake({})
ag._client = fake
plan = Plan(task="t", summary="", steps=[PlanStep(
    id="s0", tool="create_or_update_file",
    arguments={"owner": "o", "repo": "r", "path": "R.md", "content": "hello",
               "message": "m", "branch": "main"})])
tr = ag.run(plan)
check("S10 file-SHA heal regression: identical content accepted",
      tr.success and tr.steps[0].result_data == {"content": "hello", "sha": SHA}
      and [c[0] for c in fake.calls] == ["create_or_update_file", "get_file_contents"])

# --------------------------- S11 create_pull_request surfaces number on success
tr, fk = run(
    {"create_pull_request": {"id": "77", "url": "https://github.com/o/r/pull/12"}},
    [{"id": "s0", "tool": "create_pull_request",
      "arguments": {"owner": "o", "repo": "r", "title": "T",
                    "head": "feat", "base": "main"}}],
)
check("S11 PR number surfaced from url",
      tr.success and tr.steps[0].result_data.get("number") == 12)

# ------------------------------------------------------------- S12 gist dedup
GIST = {"id": "g1", "description": "notes",
        "files": {"a.md": {"filename": "a.md"}},
        "html_url": "https://gist.github.com/u/g1"}
tr, fk = run(
    {"list_gists": [GIST],
     "get_gist": {"id": "g1", "html_url": "https://gist.github.com/u/g1",
                  "files": {"a.md": {"filename": "a.md", "content": "hello"}}}},
    [{"id": "s0", "tool": "create_gist",
      "arguments": {"description": "notes", "filename": "a.md",
                    "content": "hello", "public": False}}],
)
check("S12 gist dedup: adopted, no create",
      tr.success and tr.steps[0].result_data == {
          "id": "g1", "url": "https://gist.github.com/u/g1"}
      and [c[0] for c in fk.calls] == ["list_gists", "get_gist"])
check("S12 gist dedup: idempotent attempt",
      tr.steps[0].attempts and tr.steps[0].attempts[0].status == "success_idempotent")

# ---------------------- S13 gist content differs / probe fails -> real create
tr, fk = run(
    {"list_gists": [GIST],
     "get_gist": {"id": "g1",
                  "files": {"a.md": {"filename": "a.md", "content": "OLD"}}},
     "create_gist": {"id": "g2", "url": "https://gist.github.com/u/g2"}},
    [{"id": "s0", "tool": "create_gist",
      "arguments": {"description": "notes", "filename": "a.md",
                    "content": "hello"}}],
)
check("S13 gist content differs: creates normally",
      tr.success and tr.steps[0].result_data.get("id") == "g2"
      and [c[0] for c in fk.calls] == ["list_gists", "get_gist", "create_gist"])

tr, fk = run(
    {"list_gists": err("list_gists", "500 boom"),
     "create_gist": {"id": "g3", "url": "u3"}},
    [{"id": "s0", "tool": "create_gist",
      "arguments": {"filename": "a.md", "content": "hello"}}],
)
check("S13 gist probe failure: creates normally",
      tr.success and tr.steps[0].result_data.get("id") == "g3")

# ------------------------------------- S14 sub-issue add: already attached
tr, fk = run(
    {"sub_issue_write": lambda a: err("sub_issue_write",
                                      "422 Validation Failed"),
     "issue_read": [{"id": 555, "number": 2, "title": "child"}]},
    [{"id": "s0", "tool": "sub_issue_write",
      "arguments": {"method": "add", "owner": "o", "repo": "r",
                    "issue_number": 1, "sub_issue_id": 555}}],
)
probe = next((a for t, a in fk.calls if t == "issue_read"), None)
check("S14 sub-issue already attached: idempotent success",
      tr.success and tr.steps[0].result_data.get("id") == 555
      and probe is not None and probe.get("method") == "get_sub_issues")

# --------------------------------- S15 sub-issue add: real failure not masked
tr, fk = run(
    {"sub_issue_write": lambda a: err("sub_issue_write",
                                      "422 Validation Failed"),
     "issue_read": [{"id": 999, "number": 3, "title": "other"}]},
    [{"id": "s0", "tool": "sub_issue_write",
      "arguments": {"method": "add", "owner": "o", "repo": "r",
                    "issue_number": 1, "sub_issue_id": 555}}],
)
check("S15 sub-issue real failure: not masked",
      not tr.success and tr.steps[0].status == "tool_error"
      and len([a for a in tr.steps[0].attempts if a.status == "tool_error"]) == 3)

# ----------------------------------- S16 Actions settle: list_workflows lags
WFPATH = ".github/workflows/ci.yml"
WFWRITE = {"id": "s0", "tool": "create_or_update_file",
           "arguments": {"owner": "o", "repo": "r", "path": WFPATH,
                         "content": "on: push", "message": "m",
                         "branch": "main"}}
WF = {"id": 9, "name": "CI", "path": WFPATH, "state": "active"}
tr, fk = run(
    {"create_or_update_file": {"content": {"path": WFPATH}, "commit": {}},
     "actions_list": seq({"total_count": 0, "workflows": []},
                         {"total_count": 1, "workflows": [WF]})},
    [WFWRITE,
     {"id": "s1", "tool": "actions_list",
      "arguments": {"method": "list_workflows", "owner": "o", "repo": "r"}}],
    settle_sleeps=(0.0, 0.0, 0.0),
)
check("S16 settle: stale list re-polled until workflow appears",
      tr.success and tr.steps[1].result_data == {"total_count": 1, "workflows": [WF]}
      and [c[0] for c in fk.calls].count("actions_list") == 2)

# ------------------------------ S17 Actions settle: budget out, stale accepted
tr, fk = run(
    {"create_or_update_file": {"content": {"path": WFPATH}, "commit": {}},
     "actions_list": {"total_count": 0, "workflows": []}},
    [WFWRITE,
     {"id": "s1", "tool": "actions_list",
      "arguments": {"method": "list_workflows", "owner": "o", "repo": "r"}}],
    settle_sleeps=(0.0, 0.0),
)
check("S17 settle timeout: live (stale) result accepted, step still succeeds",
      tr.success and tr.steps[1].result_data == {"total_count": 0, "workflows": []}
      and [c[0] for c in fk.calls].count("actions_list") == 3)  # initial + 2 polls

# ------------------- S18 no settle when nothing armed / non-default branch
tr, fk = run(
    {"actions_list": {"total_count": 0, "workflows": []}},
    [{"id": "s0", "tool": "actions_list",
      "arguments": {"method": "list_workflows", "owner": "o", "repo": "r"}}],
    settle_sleeps=(0.0,),
)
check("S18 no workflow write: empty list accepted with no polling",
      tr.success and [c[0] for c in fk.calls] == ["actions_list"])

tr, fk = run(
    {"create_or_update_file": {"content": {"path": WFPATH}, "commit": {}},
     "actions_list": {"total_count": 0, "workflows": []}},
    [{**WFWRITE, "arguments": {**WFWRITE["arguments"], "branch": "feat"}},
     {"id": "s1", "tool": "actions_list",
      "arguments": {"method": "list_workflows", "owner": "o", "repo": "r"}}],
    settle_sleeps=(0.0,),
)
check("S18 feature-branch workflow write does not arm the settle-wait",
      tr.success and [c[0] for c in fk.calls].count("actions_list") == 1)

# --------------------------- S19 Actions settle: actions_get 404 then appears
tr, fk = run(
    {"create_or_update_file": {"content": {"path": WFPATH}, "commit": {}},
     "actions_get": seq(err("actions_get",
                            "failed to get workflow: 404 Not Found"),
                        {"id": 17, "path": WFPATH, "state": "active"})},
    [WFWRITE,
     {"id": "s1", "tool": "actions_get",
      "arguments": {"method": "get_workflow", "owner": "o", "repo": "r",
                    "resource_id": "ci.yml"}}],
    settle_sleeps=(0.0, 0.0),
)
check("S19 actions_get 404 settles into success",
      tr.success and tr.steps[1].result_data.get("id") == 17)

# persistent 404 -> settle budget out -> real failure surfaces (not masked)
tr, fk = run(
    {"create_or_update_file": {"content": {"path": WFPATH}, "commit": {}},
     "actions_get": lambda a: err("actions_get",
                                  "failed to get workflow: 404 Not Found")},
    [WFWRITE,
     {"id": "s1", "tool": "actions_get",
      "arguments": {"method": "get_workflow", "owner": "o", "repo": "r",
                    "resource_id": "ci.yml"}}],
    settle_sleeps=(0.0,),
)
check("S19 persistent 404: still fails after settle budget",
      not tr.success and tr.steps[1].status == "tool_error")

# ----------- S20 push_files arms it too; list_workflow_runs settles by name
tr, fk = run(
    {"push_files": {"ref": "refs/heads/main"},
     "actions_list": seq({"total_count": 0, "workflow_runs": []},
                         {"total_count": 1,
                          "workflow_runs": [{"id": 5, "status": "queued"}]})},
    [{"id": "s0", "tool": "push_files",
      "arguments": {"owner": "o", "repo": "r", "branch": "main", "message": "m",
                    "files": [{"path": WFPATH, "content": "on: push"}]}},
     {"id": "s1", "tool": "actions_list",
      "arguments": {"method": "list_workflow_runs", "owner": "o", "repo": "r",
                    "resource_id": "ci.yml"}}],
    settle_sleeps=(0.0, 0.0),
)
check("S20 push_files-armed run listing settles once a run appears",
      tr.success and tr.steps[1].result_data["workflow_runs"] == [{"id": 5, "status": "queued"}]
      and [c[0] for c in fk.calls].count("actions_list") == 2)

# ------------------- S21 embedded interpolation builds queries/report content
tr, fk = run(
    {"search_repositories": {"items": [{"name": "diffusers",
                                        "full_name": "huggingface/diffusers",
                                        "owner": {"login": "huggingface"}}]},
     "list_issues": {"issues": [], "totalCount": 100},
     "create_or_update_file": {"commit": {"sha": "c1"}}},
    [{"id": "s0", "tool": "search_repositories",
      "arguments": {"query": "diffusers in:name"}},
     {"id": "s1", "tool": "list_issues",
      "arguments": {"owner": "$s0.items[0].owner.login",
                    "repo": "$s0.items[0].name",
                    "labels": ["bug"], "state": "OPEN", "perPage": 1}},
     {"id": "s2", "tool": "create_or_update_file",
      "arguments": {"owner": "o", "repo": "report", "path": "r.csv",
                    "message": "m", "branch": "main",
                    "content": "repository_name,open_bug_count\n"
                               "$s0.items[0].full_name,$s1.totalCount\n"}}],
)
_w = next(a for t, a in fk.calls if t == "create_or_update_file")
check("S21 embedded interpolation: composed CSV content",
      tr.success and _w["content"]
      == "repository_name,open_bug_count\nhuggingface/diffusers,100\n",
      repr(_w.get("content")))
check("S21 whole-value binding beside embedded stays typed",
      next(a for t, a in fk.calls if t == "list_issues")["owner"] == "huggingface")

# ------------------- S22 unresolvable embedded binding fails fast, no bad write
tr, fk = run(
    {"search_repositories": {"items": [{"name": "diffusers"}]},
     "create_or_update_file": {"ok": True}},
    [{"id": "s0", "tool": "search_repositories", "arguments": {"query": "q"}},
     {"id": "s1", "tool": "create_or_update_file",
      "arguments": {"owner": "o", "repo": "r", "path": "p", "message": "m",
                    "content": "x $s0.items[0].nope y"}}],
)
check("S22 bad embedded path -> binding_error, nothing sent",
      not tr.success and tr.steps[1].status == "binding_error"
      and not any(t == "create_or_update_file" for t, _ in fk.calls))

# ------------------- S23 fork settle-wait: 404s until the fork materializes
tr, fk = run(
    {"fork_repository": {"id": 1, "url": "u"},
     "get_file_contents": seq(err("get_file_contents", "404 Not Found"),
                              err("get_file_contents", "404 Not Found"),
                              err("get_file_contents", "404 Not Found"),
                              err("get_file_contents", "404 Not Found"),
                              "file body")},
    [{"id": "s0", "tool": "fork_repository",
      "arguments": {"owner": "up", "repo": "proj"}},
     {"id": "s1", "tool": "get_file_contents",
      "arguments": {"owner": "me", "repo": "proj", "path": "README.md",
                    "ref": "main"}}],
    settle_sleeps=(0.0, 0.0, 0.0, 0.0),
)
check("S23 fork settle: read succeeds once the fork is ready",
      tr.success and tr.steps[1].status == "success"
      and tr.steps[1].result_data == {"content": "file body", "sha": None}
      and [t for t, _ in fk.calls].count("get_file_contents") == 5)

tr, fk = run(  # 404 on a repo this plan did NOT fork -> plain retries, no settle
    {"get_file_contents": err("get_file_contents", "404 Not Found")},
    [{"id": "s0", "tool": "get_file_contents",
      "arguments": {"owner": "me", "repo": "other", "path": "x", "ref": "main"}}],
    settle_sleeps=(0.0,),
)
check("S23b unforked 404: fails normally (no settle polls)",
      not tr.success and [t for t, _ in fk.calls].count("get_file_contents") == 3)

# ------------------- S24 get_file_contents path heal from the server's hint
tr, fk = run(
    {"get_file_contents": seq(
        'Resolved potential matches in the repository tree (resolved refs: '
        '{"Ref":"refs/heads/main"}, matching files: ["videollama2/trainer.py"]).',
        "print(1)\n")},
    [{"id": "s0", "tool": "get_file_contents",
      "arguments": {"owner": "x", "repo": "y", "path": "trainer.py",
                    "ref": "main"}}],
)
check("S24 path heal: retried with the server-resolved path",
      tr.success and tr.steps[0].status == "success"
      and fk.calls[1][1]["path"] == "videollama2/trainer.py"
      and tr.steps[0].result_data == {"content": "print(1)\n", "sha": None})

tr, fk = run(  # several matches -> the executor must not guess; replan decides
    {"get_file_contents":
        'Resolved potential matches in the repository tree '
        '(matching files: ["a.py", "b/a.py"]).'},
    [{"id": "s0", "tool": "get_file_contents",
      "arguments": {"owner": "x", "repo": "y", "path": "a.py", "ref": "main"}}],
)
check("S24b multi-match: fails for replan (no guessing)",
      not tr.success and tr.steps[0].status == "tool_error"
      and len(fk.calls) == 1)

# ------------- S25 minimal search_repositories items gain owner.login / name
tr, fk = run(
    {"search_repositories": {"total_count": 1, "items": [
        {"full_name": "google-gemini/deprecated-generative-ai-python",
         "html_url": "h", "default_branch": "main"}]},
     "list_issues": {"issues": [], "totalCount": 3}},
    [{"id": "s0", "tool": "search_repositories",
      "arguments": {"query": "generative-ai in:name"}},
     {"id": "s1", "tool": "list_issues",
      "arguments": {"owner": "$s0.items[0].owner.login",
                    "repo": "$s0.items[0].name", "state": "OPEN"}}],
)
_li = next(a for t, a in fk.calls if t == "list_issues")
check("S25 minimal search items enriched: owner/name bindings resolve",
      tr.success and _li["owner"] == "google-gemini"
      and _li["repo"] == "deprecated-generative-ai-python")

# -------- S26 fork of a RENAMED source: real name surfaced from url + settled
tr, fk = run(
    {"fork_repository": {"id": 9, "url": "https://github.com/me/NewName"},
     "get_file_contents": seq(err("get_file_contents", "404 Not Found"),
                              err("get_file_contents", "404 Not Found"),
                              err("get_file_contents", "404 Not Found"),
                              err("get_file_contents", "404 Not Found"),
                              "readme body")},
    [{"id": "s0", "tool": "fork_repository",
      "arguments": {"owner": "up", "repo": "OldName"}},
     {"id": "s1", "tool": "get_file_contents",
      "arguments": {"owner": "me", "repo": "$s0.name", "path": "README.md",
                    "ref": "main"}}],
    settle_sleeps=(0.0, 0.0, 0.0, 0.0),
)
check("S26 fork url surfacing: $s0.name binds to the REAL fork name",
      tr.success and tr.steps[0].result_data.get("name") == "NewName"
      and tr.steps[0].result_data.get("owner", {}).get("login") == "me"
      and fk.calls[1][1]["repo"] == "NewName")
check("S26 renamed fork armed for settle: read succeeds after polls",
      tr.success and tr.steps[1].result_data == {"content": "readme body",
                                                 "sha": None})

# -------- S27 search_issues drift 422 carries an actionable replan hint
tr, fk = run(
    {"search_issues": err(
        "search_issues",
        "failed to search issues: 422 Validation Failed "
        "[{Message:The listed users and repositories cannot be searched ...}]")},
    [{"id": "s0", "tool": "search_issues",
      "arguments": {"query": "repo:facebook/react is:issue is:open"}}],
)
check("S27 drift 422: error text steers replan to list_issues",
      not tr.success and "list_issues" in (tr.steps[0].error or "")
      and "search_repositories" in (tr.steps[0].error or ""))

# -------- S28 zero-result search: items key materialized, clear binding error
tr, fk = run(
    {"search_repositories": {"total_count": 0, "incomplete_results": False},
     "list_issues": {"issues": [], "totalCount": 0}},
    [{"id": "s0", "tool": "search_repositories", "arguments": {"query": "q"}},
     {"id": "s1", "tool": "list_issues",
      "arguments": {"owner": "$s0.items[0].owner.login",
                    "repo": "$s0.items[0].name", "state": "OPEN"}}],
)
check("S28 zero-result search: binding fails as index-out-of-range",
      not tr.success and tr.steps[1].status == "binding_error"
      and "out of range" in (tr.steps[1].error or ""))

# ------- S29 async fork ("Fork is in progress" text): identity from the request
tr, fk = run(
    {"get_me": {"login": "me"},
     "fork_repository": "Fork is in progress",
     "get_file_contents": "readme body"},
    [{"id": "s0", "tool": "get_me", "arguments": {}},
     {"id": "s1", "tool": "fork_repository",
      "arguments": {"owner": "up", "repo": "Kimi-VL"}},
     {"id": "s2", "tool": "get_file_contents",
      "arguments": {"owner": "$s1.owner.login", "repo": "$s1.name",
                    "path": "README.md"}}],
)
check("S29 async fork: owner/name bindings resolve from the request",
      tr.success and tr.steps[1].result_data.get("name") == "Kimi-VL"
      and tr.steps[1].result_data.get("owner", {}).get("login") == "me"
      and fk.calls[-1][1] == {"owner": "me", "repo": "Kimi-VL",
                              "path": "README.md"})
check("S29 async fork: the server's text is preserved, not discarded",
      tr.steps[1].result_data.get("message") == "Fork is in progress")

# ------- S30 async fork into an org / renamed target uses the explicit args
tr, fk = run(
    {"fork_repository": "Fork is in progress"},
    [{"id": "s0", "tool": "fork_repository",
      "arguments": {"owner": "up", "repo": "src", "name": "renamed",
                    "organization": "acme"}}],
)
check("S30 async fork: organization + name override the defaults",
      tr.success and tr.steps[0].result_data.get("full_name") == "acme/renamed"
      and "get_me" not in [c[0] for c in fk.calls])

# ------- S31 async fork with no resolvable owner: raw result left untouched
tr, fk = run(
    {"get_me": err("get_me", "401 Bad credentials"),
     "fork_repository": "Fork is in progress"},
    [{"id": "s0", "tool": "fork_repository",
      "arguments": {"owner": "up", "repo": "src"}}],
)
check("S31 async fork: unresolvable owner leaves the result as-is (no guess)",
      tr.success and tr.steps[0].result_data == "Fork is in progress")

# ---------------------------------------------------------------------- summary
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
