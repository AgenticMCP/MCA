"""Offline harness for the verifier's false-failure fixes (no network, no LLM).

Run it after touching verifier.py:

    pkg/venv/bin/python pkg/agenticmcpe/verifier_selftest.py

Every case here is a defect that made the verifier fail a task the agent had
actually completed (found while analysing the 2026-08-20 RAG ablation):

  1. a non-compiling LLM snippet produced a verify.py that itself did not
     compile, so the run reported ZERO checks instead of one failed check;
  2. gh_file returned the server's path-disambiguation message as if it were
     file content, so "copied file equals its source" compared error text;
  3. hand-rolled owner derivation returned None and every later check asserted
     against "None/repo";
  4. verify.py hitting its wall-clock cap raised out of verify() and the batch
     recorded a crashed task, indistinguishable from an agent failure;
  5. truncating a step result to {"_truncated", "preview"} taught the LLM key
     names that do not exist in the trace.json the checks actually read.

It execs the GENERATED verify.py (with a fake MCP client) rather than a copy of
the helpers, so the assertions cover the code that really ships.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pkg.agenticmcpe import verifier as V  # noqa: E402

FAILURES: list[str] = []


def ok(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


class FakeGH:
    """Scripted MCP client: get_file_contents answers per path."""

    def __init__(self, by_path):
        self.by_path = by_path
        self.paths = []          # every path asked for, in order

    def call(self, tool, args):
        assert tool == "get_file_contents", tool
        self.paths.append(args["path"])
        if args["path"] not in self.by_path:
            raise RuntimeError("404")
        return SimpleNamespace(data=self.by_path[args["path"]], content=[])


DISAMBIG_ONE = ('Resolved potential matches for "train.py". '
                'matching files: ["chat/train.py"]')
DISAMBIG_MANY = ('Resolved potential matches for "train.py". '
                 'matching files: ["a/train.py", "b/train.py"]')


def settings(work_dir: Path, *, token: str = ""):
    return SimpleNamespace(
        repo_root=REPO,
        binary_path=REPO / "pkg" / "mcp_wrapper" / "bin" / "github-mcp-server",
        work_dir=work_dir,
        run_id=work_dir.name,
        ensure_work_dir=lambda: work_dir.mkdir(parents=True, exist_ok=True),
        tokens=SimpleNamespace(available=bool(token), current=lambda: token),
    )


def build_generated(work_dir: Path, trace: dict, plan: dict) -> dict:
    """Generate verify.py for this trace/plan and exec it WITHOUT running main().
    Returns the module namespace so the helpers can be called directly."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "trace.json").write_text(json.dumps(trace), "utf-8")
    (work_dir / "plan.json").write_text(json.dumps(plan), "utf-8")
    agent = V.VerifierAgent(settings(work_dir), llm=None)  # llm=None -> stub body
    path = Path(agent.build_script(
        SimpleNamespace(task=plan["task"],
                        steps=[SimpleNamespace(tool=s["tool"]) for s in plan["steps"]]),
        SimpleNamespace(steps=[])))
    src = path.read_text("utf-8")
    ns = {"__name__": "verify_under_test", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)
    return ns


# --------------------------------------------------------------------- cases
def case_invalid_snippet_still_compiles():
    """(1) A snippet whose SyntaxError text contains quotes must still yield a
    verify.py that compiles and records one failed check."""
    print("\n[1] non-compiling LLM snippet -> one failed check, not a broken script")
    # An em-dash makes compile() raise with a message containing single quotes,
    # exactly the shape that used to produce nested quotes in the output.
    body = V.VerifierAgent._sanitize_body("x = 1 — 2")
    probe = "def dynamic_evaluators(gh):\n" + textwrap.indent(body, "    ")
    try:
        compile(probe, "<probe>", "exec")
        compiles = True
    except SyntaxError as e:
        compiles, detail = False, str(e)
    ok(compiles, "fallback body compiles", "" if compiles else detail)
    ok("LLM produced invalid python" in body, "fallback records the reason")

    # ... and the same through the full script, which is what actually ran.
    wd = Path(tempfile.mkdtemp(prefix="vst-syntax-"))
    trace = {"steps": []}
    plan = {"task": "t", "steps": []}
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "trace.json").write_text(json.dumps(trace), "utf-8")
    (wd / "plan.json").write_text(json.dumps(plan), "utf-8")
    agent = V.VerifierAgent(settings(wd), llm=None)
    agent._dynamic_body = lambda p, t: body  # type: ignore[method-assign]
    script = Path(agent.build_script(SimpleNamespace(task="t", steps=[]),
                                     SimpleNamespace(steps=[])))
    try:
        compile(script.read_text("utf-8"), str(script), "exec")
        whole_ok, why = True, ""
    except SyntaxError as e:
        whole_ok, why = False, str(e)
    ok(whole_ok, "generated verify.py compiles", why)


def case_gh_file_disambiguation():
    """(2) gh_file follows a single-match disambiguation; never returns the
    server's message as content."""
    print("\n[2] gh_file resolves the server's single-match disambiguation")
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-file-")),
                         {"steps": []}, {"task": "t", "steps": []})
    gh_file = ns["gh_file"]

    gh = FakeGH({"train.py": DISAMBIG_ONE, "chat/train.py": "REAL CONTENT"})
    ok(gh_file(gh, "o", "r", "train.py") == "REAL CONTENT",
       "single match -> content from the resolved path")
    ok(gh.paths == ["train.py", "chat/train.py"], "retried exactly once",
       str(gh.paths))

    gh = FakeGH({"train.py": DISAMBIG_MANY})
    ok(gh_file(gh, "o", "r", "train.py") is None,
       "several matches -> None (never the message text)")

    gh = FakeGH({"a.py": "PLAIN"})
    ok(gh_file(gh, "o", "r", "a.py") == "PLAIN", "exact path is untouched")
    ok(gh_file(gh, "o", "r", "missing.py") is None, "missing path -> None")

    # The real-world regression: the check asked for the repo-root name of a
    # nested file and used to compare the error string against real content.
    gh = FakeGH({"generation.py": DISAMBIG_ONE.replace("chat/train.py",
                                                       "llama/generation.py"),
                 "llama/generation.py": "SRC"})
    ok(gh_file(gh, "meta-llama", "codellama", "generation.py") == "SRC",
       "root-name lookup of a nested file resolves")


def case_owner_helper():
    """(3) require_owner() derives the login from get_me's RESULT, and fails
    once (not everywhere) when it cannot."""
    print("\n[3] require_owner() replaces hand-rolled owner derivation")
    trace = {"steps": [{"id": "s0", "tool": "get_me", "arguments_resolved": {},
                        "result_data": {"login": "AgenticMCP", "id": 1}}]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-owner-")),
                         trace, {"task": "t", "steps": [{"tool": "get_me"}]})
    ok(ns["require_owner"]() == "AgenticMCP", "login read from get_me result_data")

    # get_me present but shapeless -> falls back to a step's owner argument
    trace2 = {"steps": [{"id": "s0", "tool": "get_me", "arguments_resolved": {},
                         "result_data": "unparseable"},
                        {"id": "s1", "tool": "create_repository",
                         "arguments_resolved": {"owner": "fallback-owner"}}]}
    ns2 = build_generated(Path(tempfile.mkdtemp(prefix="vst-owner2-")),
                          trace2, {"task": "t", "steps": [{"tool": "get_me"}]})
    ok(ns2["require_owner"]() == "fallback-owner", "falls back to an owner argument")

    # nothing to derive from -> ONE recorded check, and the abort is not
    # double-reported as dynamic_exception
    ns3 = build_generated(Path(tempfile.mkdtemp(prefix="vst-owner3-")),
                          {"steps": []}, {"task": "t", "steps": []})
    raised = False
    try:
        ns3["require_owner"]()
    except ns3["_OwnerUnresolved"]:
        raised = True
    ok(raised, "unresolvable owner aborts the dynamic section")
    results = ns3["RESULTS"]
    ok(len(results) == 1 and results[0]["name"] == "owner_resolved"
       and not results[0]["passed"], "exactly one explanatory check",
       json.dumps(results))


def case_timeout_is_reported():
    """(4) A verify.py that outruns the cap is reported, not raised."""
    print("\n[4] verify.py timeout is a reportable check, not a crash")
    wd = Path(tempfile.mkdtemp(prefix="vst-timeout-"))
    wd.mkdir(parents=True, exist_ok=True)
    script = wd / "slow.py"
    script.write_text("import time\nprint('starting')\ntime.sleep(30)\n", "utf-8")
    agent = V.VerifierAgent(settings(wd), llm=None)
    original = V._SCRIPT_TIMEOUT
    V._SCRIPT_TIMEOUT = 1
    try:
        report = agent.run_script(str(script))
    except Exception as e:  # the old behaviour
        V._SCRIPT_TIMEOUT = original
        ok(False, "run_script does not raise", repr(e))
        return
    V._SCRIPT_TIMEOUT = original
    ok(report.timed_out, "report.timed_out set")
    ok(report.exit_code == V._TIMEOUT_EXIT, "exit code 124", str(report.exit_code))
    ok(not report.ok, "report is not ok")
    ok([r["name"] for r in report.results] == ["verify_timeout"],
       "one verify_timeout check", json.dumps(report.results))
    ok(report.total == 1 and report.failed == 1, "totals count it",
       f"total={report.total} failed={report.failed}")


def case_output_persisted():
    """(4b) stdout/stderr land on disk next to verification.json."""
    print("\n[4b] subprocess output is persisted for debugging")
    wd = Path(tempfile.mkdtemp(prefix="vst-persist-"))
    trace = {"steps": [{"id": "s0", "tool": "get_me", "status": "success",
                        "arguments_resolved": {}, "result_data": {"login": "x"}}]}
    plan = {"task": "t", "steps": [{"id": "s0", "tool": "get_me"}]}
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "trace.json").write_text(json.dumps(trace), "utf-8")
    (wd / "plan.json").write_text(json.dumps(plan), "utf-8")
    agent = V.VerifierAgent(settings(wd), llm=None)
    report = agent.verify(
        SimpleNamespace(task="t", steps=[SimpleNamespace(tool="get_me")]),
        SimpleNamespace(steps=[]))
    ok((wd / "verify.stdout.txt").is_file(), "verify.stdout.txt written")
    ok((wd / "verify.stderr.txt").is_file(), "verify.stderr.txt written")
    vj = json.loads((wd / "verification.json").read_text("utf-8"))
    ok("timed_out" in vj, "verification.json records timed_out")
    ok(vj["timed_out"] is False, "not timed out on a fast script")
    ok(report.total > 0, "deterministic checks still ran", str(report.total))


def case_truncate_keeps_shape():
    """(5) Truncation must not invent key names the trace does not have."""
    print("\n[5] _truncate preserves shape (no fabricated 'preview' key)")
    big = {"login": "AgenticMCP", "id": 1, "details": {"bio": "x" * 5000}}
    out = V._truncate(big, limit=200)
    ok(isinstance(out, dict), "still a dict", type(out).__name__)
    ok(set(out) == set(big), "top-level keys preserved", json.dumps(list(out)))
    ok(out["login"] == "AgenticMCP", "short values untouched")
    ok("preview" not in json.dumps(out), "no fabricated 'preview' key")
    ok(len(json.dumps(out["details"]["bio"])) < 5000, "long values shortened")

    small = {"a": 1}
    ok(V._truncate(small, limit=200) == small, "small payloads pass through")

    lst = [{"n": i, "pad": "y" * 400} for i in range(30)]
    out = V._truncate(lst, limit=200)
    ok(isinstance(out, list) and len(out) == 11, "long lists capped with a marker",
       str(len(out) if isinstance(out, list) else out))


def case_repair_round():
    """(6) A non-compiling snippet gets ONE repair round before being given up on."""
    print("\n[6] a non-compiling snippet is repaired, not discarded")
    calls = []

    class StubLLM:
        """First answer has a stray arrow; the second is clean."""
        settings = SimpleNamespace(provider="stub", model="stub")

        def chat(self, system, user, json_mode=False):
            calls.append(user)
            if len(calls) == 1:
                return 'n = 5 → 3\ncheck("dynamic", "t", True, "ok")'
            return 'check("dynamic", "t", True, "ok")'

    wd = Path(tempfile.mkdtemp(prefix="vst-repair-"))
    wd.mkdir(parents=True, exist_ok=True)
    agent = V.VerifierAgent(settings(wd), llm=StubLLM())
    plan = SimpleNamespace(task="t", steps=[SimpleNamespace(
        id="s0", tool="get_me", arguments={}, post_action_properties={})])
    trace = SimpleNamespace(steps=[])
    body = agent._dynamic_body(plan, trace)
    ok(len(calls) == 2, "a repair round was attempted", f"{len(calls)} call(s)")
    ok("SyntaxError" in calls[1] if len(calls) > 1 else False,
       "the repair prompt carries the SyntaxError")
    ok(V._compile_error(body) is None, "the repaired body compiles")
    ok("live_requery" not in body, "no give-up stub after a successful repair")

    # ...and when the repair also fails, it degrades to one recorded check.
    class AlwaysBad(StubLLM):
        def chat(self, system, user, json_mode=False):
            calls.append(user)
            return "n = 5 → 3"
    calls.clear()
    agent2 = V.VerifierAgent(settings(Path(tempfile.mkdtemp(prefix="vst-repair2-"))),
                             llm=AlwaysBad())
    body2 = agent2._dynamic_body(plan, trace)
    ok(len(calls) == 2, "gives up after exactly one repair", f"{len(calls)} call(s)")
    ok(V._compile_error(body2) is None, "the give-up stub still compiles")
    ok("after 1 repair attempt" in body2, "the stub says a repair was tried")


def case_new_helpers():
    """(7) require_repo / step_arg / gh_issue_count / shared poll budget."""
    print("\n[7] the helpers that replace hand-rolled derivation and counting")
    trace = {"steps": [
        {"id": "s0", "tool": "get_me", "arguments_resolved": {},
         "result_data": {"login": "AgenticMCP"}},
        # create_repository names the repo in `name`, not `repo`
        {"id": "s1", "tool": "create_repository",
         "arguments_resolved": {"name": "my-repo", "autoInit": False},
         "result_data": {"name": "my-repo"}},
        {"id": "s2", "tool": "create_or_update_file",
         "arguments_resolved": {"owner": "AgenticMCP", "repo": "my-repo",
                                "branch": "dev", "path": "a.txt"},
         "result_data": {}},
    ]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-helpers-")), trace,
                         {"task": "t", "steps": [{"tool": "create_repository"}]})
    ok(ns["require_repo"]() == "my-repo", "require_repo reads create_repository's `name`")
    ok(ns["step_arg"]("s2", "branch") == "dev", "step_arg reads a resolved argument")
    ok(ns["step_arg"]("s1", "repo") is None, "step_arg on a missing key returns None, not KeyError")
    ok(ns["first_arg"]("create_or_update_file", "path") == "a.txt", "first_arg finds by tool")
    ok(ns["step_arg"]("nope", "x", "dflt") == "dflt", "step_arg honours its default")

    # gh_issue_count uses search totalCount rather than a page length
    class CountGH:
        def __init__(self): self.q = None
        def call(self, tool, args):
            assert tool == "search_issues", tool
            self.q = args["query"]
            return SimpleNamespace(data={"totalCount": 329, "items": [{}]}, content=[])
    gh = CountGH()
    n = ns["gh_issue_count"](gh, "huggingface", "diffusers", state="open", labels=["bug"])
    ok(n == 329, "gh_issue_count returns the true total", str(n))
    ok("repo:huggingface/diffusers" in gh.q and "is:open" in gh.q
       and 'label:"bug"' in gh.q, "the search query carries repo/state/label", gh.q)

    class BadGH:
        def call(self, tool, args): raise RuntimeError("boom")
    ok(ns["gh_issue_count"](BadGH(), "o", "r") is None,
       "gh_issue_count returns None rather than a wrong number")

    # every poll shares one deadline, so several polls cannot outrun the script
    ns["_POLL_DEADLINE"] = __import__("time").time() + 0.4
    t0 = __import__("time").time()
    okv, _ = ns["poll"](lambda: (False, None), timeout=300, interval=0.1)
    elapsed = __import__("time").time() - t0
    ok(not okv and elapsed < 5,
       "poll is clamped by the shared budget, not its own timeout",
       f"{elapsed:.1f}s")


def case_poll_predicates():
    """(8) poll() runs async predicates and surfaces broken ones.

    The task_29 false failure: the LLM wrote `async def` predicates, poll()
    got a coroutine back, unpacking raised TypeError, and the old except
    swallowed it — so the poll timed out regardless of live GitHub state."""
    print("\n[8] poll() with async and broken predicates")
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-poll-")),
                         {"steps": []}, {"task": "t", "steps": []})

    # exactly the task_29 shape: an async predicate that would succeed
    src = textwrap.dedent("""
        async def has_reply():
            return (True, 42)
        RESULT = poll(has_reply, timeout=5, interval=0.1)
    """)
    exec(compile(src, "<async-pred>", "exec"), ns)
    ok(ns["RESULT"] == (True, 42),
       "async predicate is awaited, not timed out", str(ns["RESULT"]))

    # a predicate that NEVER evaluates cleanly -> (False, None) fast, plus one
    # explicit poll_predicate_error check instead of a silent timeout
    before = len(ns["RESULTS"])
    def broken():
        raise NameError("nope")
    out = ns["poll"](broken, timeout=0.3, interval=0.1)
    added = ns["RESULTS"][before:]
    ok(out == (False, None), "broken predicate still returns (False, None)")
    ok(len(added) == 1 and added[0]["name"] == "poll_predicate_error"
       and "NameError" in added[0]["detail"],
       "one poll_predicate_error check names the exception", json.dumps(added))

    # raising once then succeeding is a clean poll -> no extra check
    before = len(ns["RESULTS"])
    state = {"n": 0}
    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("transient")
        return (True, "done")
    out = ns["poll"](flaky, timeout=5, interval=0.05)
    ok(out == (True, "done"), "predicate may raise transiently then succeed")
    ok(len(ns["RESULTS"]) == before, "no error check after a clean success")


def case_pr_merged():
    """(9) A merged PR that the LIST endpoint reports without `merged`.

    The pr_merged false failure: generated checks read p.get("merged") from
    gh_prs(...), but list_pull_requests returns MinimalPullRequest whose
    `merged` is omitempty and never set by GitHub's list endpoint (only
    merged_at is). 67 of 227 hard write tasks in the July rejected runs were
    rejected on a PR the trace's merge_pull_request result shows as merged."""
    print("\n[9] merged PRs: list results carry merged_at only; gh_pr reads the single PR")
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-pr-")),
                         {"steps": []}, {"task": "t", "steps": []})

    class PRGH:
        """list_pull_requests as the server really answers (no `merged` key);
        pull_request_read(get) with the real flag."""
        def __init__(self): self.calls = []
        def call(self, tool, args):
            self.calls.append((tool, dict(args)))
            if tool == "list_pull_requests":
                return SimpleNamespace(data=[
                    {"number": 1, "title": "merged one", "state": "closed",
                     "merged_at": "2026-07-25T05:20:00Z",
                     "head": {"ref": "feature/x"}, "base": {"ref": "main"}},
                    {"number": 2, "title": "still open", "state": "open",
                     "head": {"ref": "feature/y"}, "base": {"ref": "main"}},
                ], content=[])
            if tool == "pull_request_read":
                assert args["method"] == "get", args
                if args["pullNumber"] == 1:
                    return SimpleNamespace(data={"number": 1, "state": "closed",
                                                 "merged": True,
                                                 "merged_at": "2026-07-25T05:20:00Z",
                                                 "head": "feature/x", "base": "main"},
                                           content=[])
                if args["pullNumber"] == 2:
                    return SimpleNamespace(data={"number": 2, "state": "open",
                                                 "head": "feature/y", "base": "main"},
                                           content=[])
                raise RuntimeError("404")
            raise AssertionError(tool)

    gh = PRGH()
    prs = ns["gh_prs"](gh, "o", "r", state="all")
    p1 = next((p for p in prs if p.get("number") == 1), None)
    p2 = next((p for p in prs if p.get("number") == 2), None)
    ok(p1 is not None and p1.get("merged") is True,
       "list result: merged_at implies merged is True (the failing pattern)", json.dumps(p1))
    ok(p2 is not None and p2.get("merged") is False,
       "list result: an open PR reads merged=False, never None", json.dumps(p2))
    ok(p1["head"] == "feature/x" and p1["head"].get("ref") == "feature/x",
       "head/base tolerance unchanged")

    pr = ns["gh_pr"](gh, "o", "r", 1)
    ok(isinstance(pr, dict) and pr["merged"] is True, "gh_pr(get) reports merged=True",
       json.dumps(pr))
    ok(gh.calls[-1] == ("pull_request_read", {"method": "get", "owner": "o",
                                              "repo": "r", "pullNumber": 1}),
       "gh_pr calls pull_request_read(get) with pullNumber", str(gh.calls[-1]))
    ok(ns["gh_pr"](gh, "o", "r", "2")["merged"] is False,
       "string number accepted; open PR reads merged=False")
    ok(pr["head"] == "feature/x", "gh_pr: string head compares as a ref")
    ok(ns["gh_pr"](gh, "o", "r", 99) is None, "unknown PR -> None, not an exception")
    ok(ns["gh_pr"](gh, "o", "r", "abc") is None, "non-numeric number -> None")
    ok("gh_pr(gh, owner, repo, number)" in V._DYNAMIC_SYSTEM
       and "MERGE STATE" in V._DYNAMIC_SYSTEM, "prompt documents gh_pr and the rule")


def case_repo_fallback():
    """(10) require_repo() falls back to ANY step's repo argument.

    Read-only audits (list_releases / get_commit / list_discussions on a
    third-party repo) never call the six tools repo_name() used to inspect, so
    require_repo() recorded a failed repo_resolved check and the whole dynamic
    section aborted — one harness check failing an otherwise 26/26 run."""
    print("\n[10] require_repo() on a read-only audit that names the repo elsewhere")
    trace = {"steps": [
        {"id": "s0", "tool": "get_me", "arguments_resolved": {},
         "result_data": {"login": "AgenticMCP"}},
        {"id": "s1", "tool": "list_releases",
         "arguments_resolved": {"owner": "sveltejs", "repo": "svelte", "perPage": 3},
         "result_data": []},
        {"id": "s2", "tool": "get_commit",
         "arguments_resolved": {"owner": "sveltejs", "repo": "svelte", "sha": "abc"},
         "result_data": {}},
    ]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-repofb-")), trace,
                         {"task": "t", "steps": [{"tool": "list_releases"}]})
    ok(ns["require_repo"]() == "svelte", "repo taken from the first step naming one")
    ok(not any(r["name"] == "repo_resolved" for r in ns["RESULTS"]),
       "no repo_resolved failure recorded")
    # the six-tool path still wins when present
    trace2 = {"steps": [
        {"id": "s0", "tool": "list_releases",
         "arguments_resolved": {"owner": "o", "repo": "other"}, "result_data": []},
        {"id": "s1", "tool": "list_issues",
         "arguments_resolved": {"owner": "o", "repo": "primary"}, "result_data": []},
    ]}
    ns2 = build_generated(Path(tempfile.mkdtemp(prefix="vst-repofb2-")), trace2,
                          {"task": "t", "steps": [{"tool": "list_issues"}]})
    ok(ns2["require_repo"]() == "primary", "known write/list tools still take precedence")
    # ...and the prompt no longer demands require_repo() on repo-less tasks
    # (account surveys: notifications, gists, stars), which failed by design.
    ok("ONLY when the task touches a repository" in V._DYNAMIC_SYSTEM,
       "prompt makes require_repo() conditional on a repo being involved")


def case_sub_issues():
    """(11) Sub-issue hierarchy is only visible on single-issue reads.

    Generated checks read has_parent / sub_issues_summary from gh_issues()
    (list_issues), which never carries them, so a correctly linked sub-issue
    failed `sub_issue_relationship` with detail ({}, None)."""
    print("\n[11] sub-issue hierarchy via gh_issue / gh_sub_issues, not list results")
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-subissue-")),
                         {"steps": []}, {"task": "t", "steps": []})

    class IssueGH:
        def call(self, tool, args):
            assert tool == "issue_read", tool
            if args["method"] == "get" and args["issue_number"] == 2:
                return SimpleNamespace(data={"id": 5468351384, "number": 2, "state": "open",
                                             "has_parent": True}, content=[])
            if args["method"] == "get_sub_issues" and args["issue_number"] == 1:
                return SimpleNamespace(data=[{"id": 5468351384, "number": 2, "title": "child"}],
                                       content=[])
            if args["method"] == "get_sub_issues":
                return SimpleNamespace(data=[], content=[])
            raise RuntimeError("404")

    gh = IssueGH()
    child = ns["gh_issue"](gh, "o", "r", 2)
    ok(isinstance(child, dict) and child.get("has_parent") is True,
       "gh_issue(get) carries has_parent", json.dumps(child))
    ok(ns["gh_sub_issues"](gh, "o", "r", 1) == [2], "gh_sub_issues lists the child's number")
    ok(ns["gh_sub_issues"](gh, "o", "r", 2) == [], "an issue without children -> []")
    ok(ns["gh_issue"](gh, "o", "r", 99) is None, "unknown issue -> None, not an exception")
    ok(ns["gh_issue"](gh, "o", "r", "x") is None, "non-numeric number -> None")
    ok("gh_sub_issues(gh, owner, repo, number)" in V._DYNAMIC_SYSTEM
       and "SUB-ISSUES" in V._DYNAMIC_SYSTEM, "prompt documents the helpers and the rule")


def case_target_pair():
    """(12) require_target() pairs the repo with its REAL owner.

    Generated checks did `owner = require_owner(); repo = require_repo()` and
    re-queried f"{owner}/{repo}" — for a read-only audit of sveltejs/svelte
    that is "AgenticMCP/svelte", so repo_exists failed a correct run in every
    ablation arm (runs/ablation_vh/*/task_19)."""
    print("\n[12] require_target(): third-party repo keeps its owner, own repo keeps the login")
    me = {"id": "s0", "tool": "get_me", "arguments_resolved": {}, "result_data": {"login": "AgenticMCP"}}
    # read-only audit of someone else's repository
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-target-")),
                         {"steps": [me,
                                    {"id": "s1", "tool": "list_tags", "arguments_resolved": {"owner": "sveltejs", "repo": "svelte", "perPage": 5}, "result_data": []},
                                    {"id": "s2", "tool": "list_commits", "arguments_resolved": {"owner": "sveltejs", "repo": "svelte", "sha": "main"}, "result_data": []}]},
                         {"task": "t", "steps": [{"tool": "list_tags"}]})
    ok(ns["require_target"]() == ("sveltejs", "svelte"), "third-party audit -> (sveltejs, svelte)", str(ns["target_pair"]()))
    ok(ns["require_owner"]() == "AgenticMCP", "require_owner() still the authenticated login")
    ok(not any(r["name"] == "target_resolved" for r in ns["RESULTS"]), "no target_resolved failure recorded")
    # a repo the run created lives under the authenticated user
    ns2 = build_generated(Path(tempfile.mkdtemp(prefix="vst-target2-")),
                          {"steps": [me,
                                     {"id": "s1", "tool": "create_repository", "arguments_resolved": {"name": "my-repo", "autoInit": True}, "result_data": {"name": "my-repo"}},
                                     {"id": "s2", "tool": "create_or_update_file", "arguments_resolved": {"owner": "AgenticMCP", "repo": "my-repo", "path": "a.txt"}, "result_data": {}}]},
                          {"task": "t", "steps": [{"tool": "create_repository"}]})
    ok(ns2["require_target"]() == ("AgenticMCP", "my-repo"), "created repo -> (login, name)", str(ns2["target_pair"]()))
    # a fork: upstream args on the fork step, but the fork lives under the login
    ns3 = build_generated(Path(tempfile.mkdtemp(prefix="vst-target3-")),
                          {"steps": [me,
                                     {"id": "s1", "tool": "fork_repository", "status": "success", "arguments_resolved": {"owner": "QwenLM", "repo": "Qwen3-VL"}, "result_data": {"name": "Qwen3-VL", "full_name": "AgenticMCP/Qwen3-VL"}},
                                     {"id": "s2", "tool": "get_file_contents", "arguments_resolved": {"owner": "AgenticMCP", "repo": "Qwen3-VL", "path": "README.md"}, "result_data": "x"}]},
                          {"task": "t", "steps": [{"tool": "fork_repository"}]})
    ok(ns3["require_target"]() == ("AgenticMCP", "Qwen3-VL"), "fork -> (login, fork name)", str(ns3["target_pair"]()))
    # nothing repo-like at all -> ONE explanatory check, then the section stops
    ns4 = build_generated(Path(tempfile.mkdtemp(prefix="vst-target4-")), {"steps": [me]}, {"task": "t", "steps": [{"tool": "get_me"}]})
    raised = False
    try:
        ns4["require_target"]()
    except ns4["_OwnerUnresolved"]:
        raised = True
    ok(raised and [r["name"] for r in ns4["RESULTS"]] == ["target_resolved"],
       "no repo anywhere -> one target_resolved failure and abort", json.dumps(ns4["RESULTS"]))
    ok("require_target() -> (owner, repo)" in V._DYNAMIC_SYSTEM and "owner, repo = " in V._DYNAMIC_SYSTEM,
       "prompt documents require_target() and tells the model to use the pair")


def case_created_number():
    """The number of an object the run CREATED lives only in result_data.

    Generated checks read it with first_arg("create_pull_request","number") /
    step_arg("s4","number"), which look in the REQUEST arguments, so pr_number
    was always None and `pr_exists` reported "PR None not found" for a run that
    had created the PR and commented on it twice (ablation_vh arm1/arm3
    task_07)."""
    print("\n[13] created_number(): a created PR/issue number comes from the result")
    me = {"id": "s0", "tool": "get_me", "arguments_resolved": {}, "result_data": {"login": "AgenticMCP"}}
    trace = {"steps": [me,
                       {"id": "s1", "tool": "create_repository", "arguments_resolved": {"name": "bench"}, "result_data": {"id": "1", "url": "u"}},
                       {"id": "s4", "tool": "create_pull_request",
                        "arguments_resolved": {"owner": "AgenticMCP", "repo": "bench", "base": "main",
                                               "head": "feature", "title": "T"},
                        "result_data": {"number": 7, "title": "T"}},
                       {"id": "s5", "tool": "issue_write",
                        "arguments_resolved": {"method": "create", "owner": "AgenticMCP", "repo": "bench", "title": "I"},
                        "result_data": {"number": "12"}}]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-created-")), trace,
                         {"task": "t", "steps": [{"tool": "create_pull_request"}]})
    ok(ns["first_arg"]("create_pull_request", "number") is None,
       "first_arg on the request arguments still finds nothing (the old bug)")
    ok(ns["created_number"]("create_pull_request") == 7, "by tool name -> 7", str(ns["created_number"]("create_pull_request")))
    ok(ns["created_number"]("s4") == 7, "by step id -> 7", str(ns["created_number"]("s4")))
    ok(ns["created_number"]("issue_write") == 12, "string number coerced to int", str(ns["created_number"]("issue_write")))
    ok(ns["created_number"]("create_repository") is None, "a step whose result carries no number -> None")
    ok(ns["created_number"]("fork_repository", -1) == -1, "absent step -> the caller's default")
    ok("created_number(" in V._DYNAMIC_SYSTEM and "ONLY in result_data" in V._DYNAMIC_SYSTEM,
       "prompt documents created_number() and says where the number lives")


def case_repo_name_priority():
    """repo_name()'s fallback must rank tools, not trace position.

    A read-then-write task names two repos. When a replan dropped
    create_repository through idempotency self-heal, the fallback scanned steps
    in TRACE order over a list that reads like a priority order, so it returned
    the upstream repo being read: a correct run that built
    agenticmcpe-bench-license-study was checked against `pallets/flask`
    (ablation_vh arm1/arm2 task_04, arm1 task_15)."""
    print("\n[14] repo_name(): write steps outrank the upstream repo being read")
    me = {"id": "s0", "tool": "get_me", "arguments_resolved": {}, "result_data": {"login": "AgenticMCP"}}
    trace = {"steps": [me,
                       {"id": "s1", "tool": "get_repository_tree", "arguments_resolved": {"owner": "pallets", "repo": "flask"}, "result_data": []},
                       {"id": "s2", "tool": "get_file_contents", "arguments_resolved": {"owner": "pallets", "repo": "flask", "path": "LICENSE"}, "result_data": "x"},
                       {"id": "s3", "tool": "create_branch", "arguments_resolved": {"owner": "AgenticMCP", "repo": "agenticmcpe-bench-license-study", "branch": "study/flask"}, "result_data": {}},
                       {"id": "s4", "tool": "create_or_update_file", "arguments_resolved": {"owner": "AgenticMCP", "repo": "agenticmcpe-bench-license-study", "path": "LICENSE"}, "result_data": {}}]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-reponame-")), trace,
                         {"task": "t", "steps": [{"tool": "create_branch"}]})
    ok(ns["repo_name"]() == "agenticmcpe-bench-license-study",
       "the repo written to wins over the repo read from", str(ns["repo_name"]()))
    ok(ns["require_target"]() == ("AgenticMCP", "agenticmcpe-bench-license-study"),
       "require_target() pairs it with the authenticated login", str(ns["target_pair"]()))
    # a purely read-only audit is unaffected
    ro = {"steps": [me,
                    {"id": "s1", "tool": "get_file_contents", "arguments_resolved": {"owner": "cli", "repo": "cli", "path": "go.mod"}, "result_data": "x"},
                    {"id": "s2", "tool": "list_issues", "arguments_resolved": {"owner": "cli", "repo": "cli"}, "result_data": []}]}
    ns2 = build_generated(Path(tempfile.mkdtemp(prefix="vst-reponame2-")), ro,
                          {"task": "t", "steps": [{"tool": "list_issues"}]})
    ok(ns2["require_target"]() == ("cli", "cli"), "read-only audit still resolves to its real owner", str(ns2["target_pair"]()))


def case_pr_labels_and_poll_rule():
    """Labels on a PR must be read off the PR, not hunted in the issue list.

    A generated check did gh_issues(..., labels=["reference","checklist"]) and
    looked for the PR number in the result. The issue list never contains pull
    requests, so it came back empty and failed a run that had applied both
    labels (ablation_vh2 arm1 task_05)."""
    print("\n[15] gh_labels(): a pull request's labels come off the PR itself")
    me = {"id": "s0", "tool": "get_me", "arguments_resolved": {}, "result_data": {"login": "AgenticMCP"}}
    trace = {"steps": [me,
                       {"id": "s1", "tool": "create_repository", "arguments_resolved": {"name": "bench"}, "result_data": {}},
                       {"id": "s2", "tool": "create_pull_request",
                        "arguments_resolved": {"owner": "AgenticMCP", "repo": "bench", "title": "T"},
                        "result_data": {"number": 1}},
                       {"id": "s3", "tool": "issue_write",
                        "arguments_resolved": {"method": "update", "owner": "AgenticMCP", "repo": "bench",
                                               "issue_number": 1, "labels": ["reference", "checklist"]},
                        "result_data": {}}]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-prlabels-")), trace,
                         {"task": "t", "steps": [{"tool": "create_pull_request"}]})
    ok(callable(ns.get("gh_labels")), "gh_labels() is in the generated namespace")

    class FakeGH:
        def __init__(self): self.calls = []
        def call(self, tool, args):
            self.calls.append((tool, args))
            class R: data = {"number": 1, "labels": [{"name": "reference"}, {"name": "checklist"}]}
            return R()
    gh = FakeGH()
    ok(sorted(ns["gh_labels"](gh, "AgenticMCP", "bench", 1)) == ["checklist", "reference"],
       "reads both labels off the PR", str(ns["gh_labels"](gh, "AgenticMCP", "bench", 1)))
    ok(gh.calls and gh.calls[0][0] == "issue_read" and gh.calls[0][1].get("method") == "get",
       "goes through issue_read(get), not the issue list", str(gh.calls[:1]))
    ok(ns["gh_labels"](gh, "AgenticMCP", "bench", None) == [],
       "a missing number yields [] rather than raising")
    ok("gh_labels(gh, owner, repo, number)" in V._DYNAMIC_SYSTEM
       and "issue LIST never contains pull requests" in V._DYNAMIC_SYSTEM,
       "prompt documents the helper and says why the issue list will not do")
    ok("POLL ONLY FOR AUTOMATION" in V._DYNAMIC_SYSTEM
       and "assert it directly with ONE query, no poll" in V._DYNAMIC_SYSTEM,
       "prompt reserves poll() for effects the run only triggered")


class ScriptedGH:
    """Answers one tool from a callable; records every call."""

    def __init__(self, tool, answer):
        self.tool, self.answer, self.calls = tool, answer, []

    def call(self, tool, args):
        self.calls.append((tool, args))
        if tool != self.tool:
            raise RuntimeError("unexpected tool " + tool)
        out = self.answer(args)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(data=out, content=[])


def case_gh_label():
    """A label's DEFINITION must be readable. With no helper for it, generated
    checks called get_label raw, dropped `.data`, and compared a ToolResult to a
    colour string — failing three runs that had created every label correctly
    (ablation100 tasks 074/075/077)."""
    print("\n[16] gh_label(): a label's own colour and description")
    trace = {"steps": [{"id": "s0", "tool": "get_me", "arguments_resolved": {},
                        "result_data": {"login": "AgenticMCP"}},
                       {"id": "s1", "tool": "create_repository",
                        "arguments_resolved": {"name": "bench"}, "result_data": {}}]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-label-")), trace,
                         {"task": "t", "steps": [{"tool": "label_write"}]})
    ok(callable(ns.get("gh_label")), "gh_label() is in the generated namespace")

    gh = ScriptedGH("get_label", lambda a: {"id": "L_1", "name": a["name"],
                                            "color": "D73A4A", "description": "Bug"})
    got = ns["gh_label"](gh, "AgenticMCP", "bench", "bug")
    ok(got and got["name"] == "bug", "returns the label dict", str(got))
    ok(got and got["color"] == "d73a4a",
       "normalises colour to lowercase hex with no '#'", str(got))
    ok(gh.calls and gh.calls[0][1] == {"owner": "AgenticMCP", "repo": "bench", "name": "bug"},
       "calls get_label with owner/repo/name separately", str(gh.calls[:1]))

    missing = ScriptedGH("get_label", lambda a: RuntimeError("label 'nope' not found"))
    ok(ns["gh_label"](missing, "AgenticMCP", "bench", "nope") is None,
       "a label that does not exist is None, not an exception")
    junk = ScriptedGH("get_label", lambda a: "some server message")
    ok(ns["gh_label"](junk, "AgenticMCP", "bench", "bug") is None,
       "a non-dict payload is None rather than a string compared to a colour")
    ok("gh_label(gh, owner, repo, name)" in V._DYNAMIC_SYSTEM
       and 'lstrip("#").lower()' in V._DYNAMIC_SYSTEM,
       "prompt documents the helper and how to compare the colour")


def case_gh_review_threads():
    """Review comments live on the diff, not in the issue conversation, and
    nothing could read them back. Four runs posted a correct threaded reply and
    were failed anyway (ablation100 tasks 066/067/071/072)."""
    print("\n[17] gh_review_threads(): threaded replies on a pull request")
    trace = {"steps": [{"id": "s0", "tool": "get_me", "arguments_resolved": {},
                        "result_data": {"login": "AgenticMCP"}},
                       {"id": "s1", "tool": "create_repository",
                        "arguments_resolved": {"name": "bench"}, "result_data": {}}]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-review-")), trace,
                         {"task": "t", "steps": [{"tool": "add_reply_to_pull_request_comment"}]})
    ok(callable(ns.get("gh_review_threads")),
       "gh_review_threads() is in the generated namespace")

    payload = {"review_threads": [
        {"id": "PRRT_1", "is_resolved": False, "total_count": 2, "comments": [
            {"id": 4037321228, "body": "please rename this", "path": "a.py", "line": 3},
            {"id": 4037321299, "body": "done, renamed", "path": "a.py", "line": 3}]},
        {"id": "PRRT_2", "is_resolved": True, "total_count": 1, "comments": [
            {"id": 4037321300, "body": "nit", "path": "b.py", "line": 9}]}],
        "totalCount": 2}
    gh = ScriptedGH("pull_request_read", lambda a: payload)
    threads = ns["gh_review_threads"](gh, "AgenticMCP", "bench", 7)
    ok(len(threads) == 2, "returns both threads", str(len(threads)))
    ok(threads and len(threads[0]["comments"]) == 2,
       "keeps the grouping, so a reply is a 2nd comment in the SAME thread")
    ok(threads[0]["comments"][1]["id"] == 4037321299,
       "carries the numeric comment id add_reply_to_pull_request_comment uses")
    ok(threads[1]["is_resolved"] is True, "surfaces is_resolved as a bool")
    ok(threads[0].get("total_count") == 2,
       "keeps the thread's other fields instead of narrowing it")
    ok(gh.calls and gh.calls[0][1].get("method") == "get_review_comments"
       and gh.calls[0][1].get("pullNumber") == 7,
       "calls pull_request_read(get_review_comments) with pullNumber", str(gh.calls[:1]))
    ok(ns["gh_review_threads"](gh, "AgenticMCP", "bench", None) == [],
       "a missing PR number yields [] rather than raising")
    boom = ScriptedGH("pull_request_read", lambda a: RuntimeError("404"))
    ok(ns["gh_review_threads"](boom, "AgenticMCP", "bench", 7) == [],
       "an unreadable PR yields [] rather than raising")
    ok("gh_review_threads(gh, owner, repo, pull_number)" in V._DYNAMIC_SYSTEM
       and "NEVER appears in gh_issue_comments" in V._DYNAMIC_SYSTEM,
       "prompt documents the helper and that the issue conversation will not do")


def case_gh_gist():
    """get_gist has no helper, so checks called it raw and read None out of a
    gist whose description and files were right (ablation100 tasks 095-098)."""
    print("\n[18] gh_gist(): a gist's description and file contents")
    trace = {"steps": [{"id": "s0", "tool": "get_me", "arguments_resolved": {},
                        "result_data": {"login": "AgenticMCP"}},
                       {"id": "s1", "tool": "create_gist",
                        "arguments_resolved": {"description": "Debounce helper"},
                        "result_data": {"id": "abc123"}}]}
    ns = build_generated(Path(tempfile.mkdtemp(prefix="vst-gist-")), trace,
                         {"task": "t", "steps": [{"tool": "create_gist"}]})
    ok(callable(ns.get("gh_gist")), "gh_gist() is in the generated namespace")

    gh = ScriptedGH("get_gist", lambda a: {
        "id": a["gist_id"], "description": "Debounce helper (reviewed)", "public": True,
        "html_url": "https://gist.github.com/AgenticMCP/abc123",
        "files": {"debounce.js": {"filename": "debounce.js", "size": 12,
                                  "content": "export const x = 1\n"}}})
    got = ns["gh_gist"](gh, "abc123")
    ok(got and got["description"] == "Debounce helper (reviewed)",
       "reads the updated description", str(got))
    ok(got and got["files"] == {"debounce.js": "export const x = 1\n"},
       "flattens files to {filename: content}", str(got and got["files"]))
    # The first version of this helper rebuilt a 4-key dict and dropped
    # html_url, which failed a task whose issue body IS the gist URL.
    ok(got and got.get("html_url") == "https://gist.github.com/AgenticMCP/abc123",
       "keeps html_url and every other field instead of narrowing the gist")
    ok(gh.calls and gh.calls[0][1] == {"gist_id": "abc123"},
       "calls get_gist with the id as a string", str(gh.calls[:1]))
    ok(ns["gh_gist"](gh, None) is None, "a missing gist id is None, not a call")
    boom = ScriptedGH("get_gist", lambda a: RuntimeError("404"))
    ok(ns["gh_gist"](boom, "abc123") is None, "an unreadable gist is None")
    ok("gh_gist(gh, gist_id)" in V._DYNAMIC_SYSTEM
       and "ACCOUNT-LEVEL: do NOT call require_target()" in V._DYNAMIC_SYSTEM,
       "prompt documents the helper and that a gist task has no repo")


def main() -> int:
    # Guarantee "no network": the generated script only opens an MCP client when
    # it finds a token in the environment.
    for k in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN", "GITHUB_TOKENS"):
        os.environ.pop(k, None)
    for case in (case_invalid_snippet_still_compiles,
                 case_gh_file_disambiguation,
                 case_owner_helper,
                 case_timeout_is_reported,
                 case_output_persisted,
                 case_truncate_keeps_shape,
                 case_repair_round,
                 case_new_helpers,
                 case_poll_predicates,
                 case_pr_merged,
                 case_repo_fallback,
                 case_sub_issues,
                 case_target_pair,
                 case_created_number,
                 case_repo_name_priority,
                 case_pr_labels_and_poll_rule,
                 case_gh_label,
                 case_gh_review_threads,
                 case_gh_gist):
        case()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all verifier selftests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
