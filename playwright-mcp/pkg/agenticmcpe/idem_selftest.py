"""Offline harness for the executor's Playwright reconciliation paths (no
network, no LLM, no browser).

Run it after touching the executor's find-resolution / retry / heal paths:

    python3 pkg/agenticmcpe/idem_selftest.py

Drives ExecutionAgent with a scripted FakeClient and asserts each behaviour:
find:-target resolution (unique / ambiguous / zero / no-snapshot), the
unsafe-to-repeat single-attempt rule vs 3 retries for safe tools, the
"no open page" re-navigation self-heal, stale-ref fail-fast with a fresh
snapshot excerpt in the error, the page-state result shim + $sN.url bindings,
browser_close idempotency, and the error->HINT rewrites.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pkg.agenticmcpe.executor as executor_mod  # noqa: E402
from pkg.agenticmcpe.executor import ExecutionAgent  # noqa: E402
from pkg.agenticmcpe.planner import Plan, PlanStep  # noqa: E402
from pkg.mcp_wrapper import MCPToolError  # noqa: E402

executor_mod._SETTLE_SECONDS = 0  # no real sleeps in the offline harness


class FakeClient:
    """Scripted MCP client: handlers[tool] -> text | Exception | callable."""

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


def run(handlers, steps):
    ag = agent()
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


NAV_TEXT = ("### Ran Playwright code\n```js\nawait page.goto('https://example.com');\n```\n"
            "### Page\n- Page URL: https://example.com/\n- Page Title: Example Domain\n")
SNAP_TEXT = ("### Page\n- Page URL: https://example.com/\n- Page Title: Example Domain\n"
             "### Snapshot\n```yaml\n"
             "- generic [ref=e2]:\n"
             "  - heading \"Example Domain\" [level=1] [ref=e3]\n"
             "  - paragraph [ref=e4]: Some text mentioning results\n"
             "  - link \"Learn more\" [ref=e6] [cursor=pointer]:\n"
             "    - /url: https://iana.org/domains/example\n"
             "  - button \"Search\" [ref=e7]\n"
             "  - button \"Search flights\" [ref=e8]\n"
             "```\n")
CLICK_TEXT = ("### Page\n- Page URL: https://iana.org/domains/example\n"
              "- Page Title: Example Domains\n")

NAV = {"id": "s0", "tool": "browser_navigate",
       "arguments": {"url": "https://example.com"}}
SNAP = {"id": "s1", "tool": "browser_snapshot", "arguments": {}}


# ------------------------------------------------- P1 page-state result shim
tr, fk = run({"browser_navigate": NAV_TEXT}, [NAV])
d = tr.steps[0].result_data
check("P1 shim: result surfaced as {text,url,title}",
      tr.success and isinstance(d, dict)
      and d.get("url") == "https://example.com/"
      and d.get("title") == "Example Domain"
      and d.get("text", "").startswith("### Ran Playwright"))

# -------------------------------------- P2 $sN.url binding over the shim dict
tr, fk = run(
    {"browser_navigate": seq(NAV_TEXT, CLICK_TEXT)},
    [NAV,
     {"id": "s1", "tool": "browser_navigate", "arguments": {"url": "$s0.url"}}],
)
check("P2 binding: $s0.url resolves through the shim",
      tr.success and fk.calls[1][1]["url"] == "https://example.com/")

def _call_args(fk, tool):
    return next(a for t, a in fk.calls if t == tool)


# ---------------- P3 find: unique resolution (via automatic fresh probe)
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_click": CLICK_TEXT},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "Learn more link", "target": "find:Learn more"}}],
)
check("P3 find: unique text resolves to its ref",
      tr.success and _call_args(fk, "browser_click")["target"] == "e6")
check("P3 find: resolved args recorded in trace",
      tr.steps[2].arguments_resolved.get("target") == "e6")
check("P3 find: fresh probe snapshot taken before resolving",
      [t for t, _ in fk.calls].count("browser_snapshot") == 2)

# -------------------- P4 find: exact accessible-name beats substring matches
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_click": CLICK_TEXT},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "search", "target": "find:Search"}}],
)
check("P4 find exact-name priority: 'Search' resolves to e7 not ambiguous",
      tr.success and _call_args(fk, "browser_click")["target"] == "e7")

# substring-only collision (exact name of neither) is still ambiguous
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_click": CLICK_TEXT},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "search", "target": "find:Sear"}}],
)
check("P4b find substring ambiguity: binding_error lists candidates",
      not tr.success and tr.steps[2].status == "binding_error"
      and not any(t == "browser_click" for t, _ in fk.calls)
      and "AMBIGUOUS" in (tr.steps[2].error or "")
      and "e7" in tr.steps[2].error and "e8" in tr.steps[2].error)

# two IDENTICAL url-less buttons: one duplicated control -> first instance
SNAP_DUP = SNAP_TEXT.replace("button \"Search flights\" [ref=e8]",
                             "button \"Search\" [ref=e8]")
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_DUP,
     "browser_click": CLICK_TEXT},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "search", "target": "find:Search"}}],
)
check("P4c identical duplicate buttons: collapsed to the first instance",
      tr.success and _call_args(fk, "browser_click")["target"] == "e7")
# ... but identical NON-interactive duplicates stay ambiguous
SNAP_DUP_P = ("### Page\n- Page URL: https://example.com/\n"
              "### Snapshot\n```yaml\n"
              "- paragraph [ref=e41]: Terms apply\n"
              "- paragraph [ref=e42]: Terms apply\n"
              "```\n")
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_DUP_P,
     "browser_click": CLICK_TEXT},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "terms", "target": "find:Terms apply"}}],
)
check("P4d identical duplicate paragraphs: still ambiguous",
      not tr.success and tr.steps[2].status == "binding_error"
      and "AMBIGUOUS" in (tr.steps[2].error or ""))

# ------------------------------------------------- P5 find: zero matches
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_click": CLICK_TEXT},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "ghost", "target": "find:Nonexistent caption"}}],
)
check("P5 find zero: binding_error names the miss",
      not tr.success and tr.steps[2].status == "binding_error"
      and "matches NO line" in (tr.steps[2].error or ""))
check("P5b find zero: fresh snapshot excerpt attached for the replanner",
      "CURRENT PAGE SNAPSHOT" in (tr.steps[2].error or "")
      and "Learn more" in (tr.steps[2].error or ""))

# ------- P6 find: with no page/probe available -> clear binding_error
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_click": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "x", "target": "find:Learn more"}}],
)
check("P6 find with no snapshot available: binding_error names the cause",
      not tr.success and tr.steps[1].status == "binding_error"
      and "no page snapshot is available" in (tr.steps[1].error or ""))

# ------- P6b find: needs NO explicit snapshot step when the probe works
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_click": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "Learn more link", "target": "find:Learn more"}}],
)
check("P6b find without a planned snapshot: probe resolves it",
      tr.success and _call_args(fk, "browser_click")["target"] == "e6")

# ---------------------- P7 unsafe-to-repeat: click fails once, never retried
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_click": lambda a: err("browser_click",
                                    "TimeoutError: Timeout 5000ms exceeded")},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "x", "target": "e6"}}],
)
check("P7 unsafe click: exactly 1 attempt, no auto-retry",
      not tr.success and len(tr.steps[2].attempts) == 1
      and [t for t, _ in fk.calls].count("browser_click") == 1)
check("P7 timeout HINT attached",
      "browser_wait_for" in (tr.steps[2].error or ""))

# --------------------------- P8 safe tool: transient error retried 3 times
tr, fk = run(
    {"browser_navigate": lambda a: err("browser_navigate",
                                       "net::ERR_NAME_NOT_RESOLVED")},
    [NAV],
)
check("P8 safe navigate: 3 attempts on transient error",
      not tr.success and len(tr.steps[0].attempts) == 3
      and [t for t, _ in fk.calls].count("browser_navigate") == 3)

# ---- P9 stale ref: one refresh-heal (re-snapshot + retry), then fail fast
tr, fk = run(
    {"browser_navigate": NAV_TEXT,
     "browser_snapshot": SNAP_TEXT,
     "browser_click": lambda a: err("browser_click",
                                    "Error: Ref e99 not found in the current "
                                    "page snapshot. Try capturing new snapshot.")},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "x", "target": "e99"}}],
)
e = tr.steps[1].error or ""
check("P9 stale ref: healed once (2 attempts), then fail",
      not tr.success and len(tr.steps[1].attempts) == 2
      and [t for t, _ in fk.calls] == ["browser_navigate", "browser_click",
                                       "browser_snapshot", "browser_click"])
check("P9 stale ref: HINT + fresh snapshot excerpt attached",
      "MOST RECENT snapshot" in e and "CURRENT PAGE SNAPSHOT" in e
      and "Learn more" in e)

# ---- P9b stale find: ref heals by re-resolving against the fresh snapshot
SNAP2 = SNAP_TEXT.replace("[ref=e6]", "[ref=e26]")
tr, fk = run(
    {"browser_navigate": NAV_TEXT,
     "browser_snapshot": seq(SNAP_TEXT, SNAP2),
     "browser_click": seq(err("browser_click",
                              "Error: Ref e6 not found in the current page "
                              "snapshot. Try capturing new snapshot."),
                          CLICK_TEXT)},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "Learn more link", "target": "find:Learn more"}}],
)
check("P9b stale find: re-resolved to the fresh ref and succeeded",
      tr.success and tr.steps[2].arguments_resolved.get("target") == "e26"
      and [a.status for a in tr.steps[2].attempts] == ["tool_error", "success"])

# --------------- P10 "no open page" on observation tool: re-navigate heal
tr, fk = run(
    {"browser_navigate": NAV_TEXT,
     "browser_snapshot": seq(err("browser_snapshot",
                                 "No open pages available. Use the "
                                 "browser_navigate tool to navigate to a "
                                 "page first."),
                             SNAP_TEXT)},
    [NAV, SNAP],
)
check("P10 no-page heal: re-navigated and retried to success",
      tr.success
      and [t for t, _ in fk.calls] == ["browser_navigate", "browser_snapshot",
                                       "browser_navigate", "browser_snapshot"]
      and fk.calls[2][1] == {"url": "https://example.com/"})

# no prior navigation -> nothing to re-open; the failure stands
tr, fk = run(
    {"browser_snapshot": err("browser_snapshot", "No open pages available.")},
    [SNAP],
)
check("P10b no-page without a prior URL: fails (no guessed navigation)",
      not tr.success
      and [t for t, _ in fk.calls].count("browser_navigate") == 0)

# ---------------------------------------- P11 browser_close idempotency
tr, fk = run(
    {"browser_close": err("browser_close", "Browser is already closed.")},
    [{"id": "s0", "tool": "browser_close", "arguments": {}}],
)
check("P11 close idempotent: adopted as success",
      tr.success and tr.steps[0].attempts[0].status == "success_idempotent"
      and tr.steps[0].result_data == {"text": "(browser already closed)"})

# ---------------------------------------- P12 dialog-blocked error HINT
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_click": err("browser_click",
                          'Tool "browser_click" cannot be used while a modal '
                          'dialog is open')},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_click",
      "arguments": {"element": "x", "target": "e6"}}],
)
check("P12 dialog HINT steers to browser_handle_dialog",
      not tr.success and "browser_handle_dialog" in (tr.steps[2].error or ""))

# ----------------- P13 nested find: resolution (browser_fill_form fields)
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_fill_form": CLICK_TEXT},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_fill_form",
      "arguments": {"fields": [{"element": "search box",
                                "target": "find:Search flights",
                                "name": "q", "type": "textbox",
                                "value": "Beijing"}]}}],
)
check("P13 nested find: resolved inside fill_form fields",
      tr.success and _call_args(fk, "browser_fill_form")["fields"][0]["target"] == "e8")

# ----------------- P14 browser_tabs: list is safe (retried), new is not
tr, fk = run(
    {"browser_tabs": lambda a: err("browser_tabs", "boom")},
    [{"id": "s0", "tool": "browser_tabs", "arguments": {"action": "list"}}],
)
check("P14 tabs list: retried as a safe tool",
      not tr.success and len(tr.steps[0].attempts) == 3)
tr, fk = run(
    {"browser_tabs": lambda a: err("browser_tabs", "boom")},
    [{"id": "s0", "tool": "browser_tabs", "arguments": {"action": "new"}}],
)
check("P14 tabs new: single attempt (mutates browser state)",
      not tr.success and len(tr.steps[0].attempts) == 1)

# --------------- P15 handle_dialog with no dialog open: idempotent no-op
tr, fk = run(
    {"browser_navigate": NAV_TEXT,
     "browser_handle_dialog": err(
         "browser_handle_dialog",
         '### Error\nError: The tool "browser_handle_dialog" can only be used '
         'when there is related modal state present.')},
    [NAV,
     {"id": "s1", "tool": "browser_handle_dialog",
      "arguments": {"accept": True}}],
)
check("P15 handle_dialog with no dialog: adopted as idempotent success",
      tr.success and tr.steps[1].attempts[0].status == "success_idempotent"
      and tr.steps[1].result_data == {"text": "(no dialog was open)"})

# --------------- P16 planner close-repair: task asks to close -> appended
from pkg.agenticmcpe.planner import PlannerAgent  # noqa: E402

CATALOG = [
    {"tool": "browser_navigate", "doc": "",
     "params": [{"name": "url", "required": True, "type": "string"}]},
    {"tool": "browser_snapshot", "doc": "", "params": []},
    {"tool": "browser_close", "doc": "", "params": []},
]


class FakeLLM:
    settings = SimpleNamespace(provider="fake", model="fake")

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def chat(self, system, user, json_mode=False):
        self.calls += 1
        return self.reply


PLAN_NO_CLOSE = ('{"summary": "nav+read", "steps": ['
                 '{"id": "s0", "tool": "browser_navigate", '
                 '"arguments": {"url": "https://example.com"}},'
                 '{"id": "s1", "tool": "browser_snapshot", "arguments": {}}]}')
pl = PlannerAgent(FakeLLM(PLAN_NO_CLOSE), CATALOG).plan(
    "Read the heading of example.com. Remember to close the browser when "
    "you finish the task.")
check("P16 close-repair: browser_close appended when the task asks",
      [s.tool for s in pl.steps] == ["browser_navigate", "browser_snapshot",
                                     "browser_close"]
      and pl.steps[-1].id == "s2"
      and any("repair" in w for w in pl.warnings))
pl = PlannerAgent(FakeLLM(PLAN_NO_CLOSE), CATALOG).plan(
    "Read the heading of example.com.")
check("P16b close-repair: NOT appended when the task does not ask",
      [s.tool for s in pl.steps] == ["browser_navigate", "browser_snapshot"])

# --------------- P17 verifier codegen: compile-retry with error feedback
from pkg.agenticmcpe.verifier import (  # noqa: E402
    VerifierAgent, generate_dynamic_body)


class SeqLLM:
    settings = SimpleNamespace(provider="fake", model="fake")

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, system, user, json_mode=False):
        self.prompts.append(user)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


BAD_CODE = 'check("dynamic", "kickoff", True, 07:30)'   # leading-zero literal
GOOD_CODE = 'check("dynamic", "kickoff", True, "07:30")'
llm = SeqLLM(BAD_CODE, GOOD_CODE)
body = generate_dynamic_body(llm, "CTX")
check("P17 codegen retry: 2nd attempt used after compile error",
      body == GOOD_CODE and len(llm.prompts) == 2
      and "FAILED TO COMPILE OR LINT" in llm.prompts[1])
llm = SeqLLM(BAD_CODE)
body = generate_dynamic_body(llm, "CTX")
check("P17b codegen retries exhausted: failing stub, 3 attempts made",
      "LLM produced invalid python" in body and len(llm.prompts) == 3
      and compile("def dynamic_evaluators(pw):\n    " + body, "<t>", "exec"))

# --------------- P18 run_script: a crashing verify.py -> failing check, not 0/0
wd = Path(tempfile.mkdtemp(prefix="idem-verify-"))
crash = wd / "verify.py"
crash.write_text("import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n",
                 encoding="utf-8")
va = VerifierAgent(SimpleNamespace(repo_root=REPO, work_dir=wd))
rep = va.run_script(str(crash))
check("P18 crashed verify.py: synthesized failing check with stderr tail",
      rep.total == 1 and rep.failed == 1 and not rep.ok
      and "produced no JSON report" in rep.results[0]["detail"]
      and "boom" in rep.results[0]["detail"])

# ----------- P19 wait_for expected-text timeout: fail fast with page excerpt
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_wait_for": err("browser_wait_for",
                             "TimeoutError: Timeout 15000ms exceeded.\n"
                             "waiting for getByText('Beijing')")},
    [NAV,
     {"id": "s1", "tool": "browser_wait_for", "arguments": {"text": "Beijing"}}],
)
check("P19 wait_for text timeout: single attempt (no safe-tool retries)",
      not tr.success and len(tr.steps[1].attempts) == 1
      and [t for t, _ in fk.calls].count("browser_wait_for") == 1)
check("P19 wait_for text timeout: HINT + fresh snapshot excerpt attached",
      "never appear" in (tr.steps[1].error or "")
      and "CURRENT PAGE SNAPSHOT" in (tr.steps[1].error or ""))
# a plain time-based wait keeps the normal safe-tool retry budget
tr, fk = run(
    {"browser_navigate": NAV_TEXT,
     "browser_wait_for": err("browser_wait_for",
                             "TimeoutError: Timeout 15000ms exceeded.")},
    [NAV, {"id": "s1", "tool": "browser_wait_for", "arguments": {"time": 2}}],
)
check("P19b wait_for by time: still retried as a safe tool",
      not tr.success and len(tr.steps[1].attempts) == 3)

# ----------- P20 role-aware find: typing prefers the input-like candidate
SNAP_ROLES = ("### Page\n- Page URL: https://example.com/\n"
              "- Page Title: Example Domain\n"
              "### Snapshot\n```yaml\n"
              "- textbox \"Search models, datasets, users...\" [ref=e11]\n"
              "- link \"Search the docs\" [ref=e12] [cursor=pointer]\n"
              "- link \"Advanced Search tips\" [ref=e13] [cursor=pointer]\n"
              "```\n")
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_ROLES,
     "browser_type": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_type",
      "arguments": {"element": "search box", "target": "find:Search",
                    "text": "bert", "submit": True}}],
)
check("P20 typing find: ambiguity resolved to the textbox line",
      tr.success and _call_args(fk, "browser_type")["target"] == "e11")
# clicking has no role preference: the same needle stays ambiguous
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_ROLES,
     "browser_click": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "search", "target": "find:Search"}}],
)
check("P20b clicking the same needle: still ambiguous (no role preference)",
      not tr.success and tr.steps[1].status == "binding_error"
      and "AMBIGUOUS" in (tr.steps[1].error or ""))

# ----------- P21 empty snapshots: settle-and-retake, and honest find: errors
EMPTY_SNAP = ("### Page\n- Page URL: https://example.com/\n"
              "### Snapshot\n```yaml\n\n```")
tr, fk = run(
    {"browser_navigate": NAV_TEXT,
     "browser_snapshot": seq(EMPTY_SNAP, SNAP_TEXT)},
    [NAV, SNAP],
)
check("P21 planned snapshot came back empty: retaken once with content",
      tr.success
      and [a.status for a in tr.steps[1].attempts] == ["empty_snapshot",
                                                       "success"]
      and "Learn more" in tr.steps[1].result_data["text"])
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": EMPTY_SNAP,
     "browser_click": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "x", "target": "find:Learn more"}}],
)
check("P21b find: on a persistently empty page: NOTE names the cause",
      not tr.success and tr.steps[1].status == "binding_error"
      and "EMPTY page snapshot" in (tr.steps[1].error or "")
      and [t for t, _ in fk.calls].count("browser_snapshot") == 3)

# ----------- P22 verifier lint: bad check() arity / undefined names retried
LINT_BAD_ARITY = 'check("dynamic", "pts", True, "detail", "extra")'
llm = SeqLLM(LINT_BAD_ARITY, GOOD_CODE)
body = generate_dynamic_body(llm, "CTX")
check("P22 lint: 5-arg check() caught and retried",
      body == GOOD_CODE and len(llm.prompts) == 2
      and "check() takes 3-4 arguments" in llm.prompts[1])
LINT_BAD_NAME = 'check("dynamic", "url", bool(resto_url), "ok")'
llm = SeqLLM(LINT_BAD_NAME, GOOD_CODE)
body = generate_dynamic_body(llm, "CTX")
check("P22b lint: undefined name caught and retried",
      body == GOOD_CODE and len(llm.prompts) == 2
      and "never defined" in llm.prompts[1])
LINT_OK = ('for s in executed_steps:\n'
           '    t = step_text(s["id"])\n'
           '    check("dynamic", "step_" + s["id"], has(t, "x") or True, t[:40])')
llm = SeqLLM(LINT_OK)
body = generate_dynamic_body(llm, "CTX")
check("P22c lint: legitimate helper/alias usage passes first try",
      body == LINT_OK and len(llm.prompts) == 1)

# ----------- P23 same-name links: /url in the ambiguity listing; equal-/url
# duplicates collapse to one target
SNAP_DUP_LINKS = ("### Page\n- Page URL: https://example.com/\n"
                  "### Snapshot\n```yaml\n"
                  "- link \"Devstral\" [ref=e21] [cursor=pointer]:\n"
                  "  - /url: /mistralai/Devstral\n"
                  "- link \"Devstral\" [ref=e22] [cursor=pointer]:\n"
                  "  - /url: /unsloth/Devstral\n"
                  "```\n")
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_DUP_LINKS,
     "browser_click": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "model link", "target": "find:Devstral"}}],
)
check("P23 same-name different-url links: ambiguous, urls listed",
      not tr.success and tr.steps[1].status == "binding_error"
      and "/mistralai/Devstral" in (tr.steps[1].error or "")
      and "/unsloth/Devstral" in (tr.steps[1].error or ""))
SNAP_SAME_URL = SNAP_DUP_LINKS.replace("/unsloth/Devstral", "/mistralai/Devstral")
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_SAME_URL,
     "browser_click": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "model link", "target": "find:Devstral"}}],
)
check("P23b same-name SAME-url links: collapsed to the first (equivalent)",
      tr.success and _call_args(fk, "browser_click")["target"] == "e21")

# ----------- P24 clicking prefers interactive lines over plain text
SNAP_CLICK_PREF = ("### Page\n- Page URL: https://example.com/\n"
                   "### Snapshot\n```yaml\n"
                   "- paragraph [ref=e31]: Registration deadlines and "
                   "Registration desk hours\n"
                   "- link \"Registration info\" [ref=e32] [cursor=pointer]:\n"
                   "  - /url: /registration\n"
                   "```\n")
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_CLICK_PREF,
     "browser_click": CLICK_TEXT},
    [NAV,
     {"id": "s1", "tool": "browser_click",
      "arguments": {"element": "registration link",
                    "target": "find:Registration"}}],
)
check("P24 click find: paragraph-vs-link tie resolved to the link",
      tr.success and _call_args(fk, "browser_click")["target"] == "e32")

# ----------- P25 typing into a non-input match: steering HINT attached
tr, fk = run(
    {"browser_navigate": NAV_TEXT, "browser_snapshot": SNAP_TEXT,
     "browser_type": err("browser_type",
                         "Error: Element is not an <input>, <textarea>, "
                         "<select> or [contenteditable] and does not have a "
                         "role allowing [aria-readonly]")},
    [NAV, SNAP,
     {"id": "s2", "tool": "browser_type",
      "arguments": {"element": "x", "target": "e6", "text": "q"}}],
)
check("P25 type into non-input: HINT retargets to a text field",
      not tr.success and "not a text field" in (tr.steps[2].error or "")
      and "textbox/searchbox" in (tr.steps[2].error or ""))

# ----------- P26 lint: live pw_* helpers are banned in dynamic bodies
LIVE_BODY = 'check("dynamic", "live", bool(pw_find(pw, "x")), "live check")'
llm = SeqLLM(LIVE_BODY, GOOD_CODE)
body = generate_dynamic_body(llm, "CTX")
check("P26 lint: live helper use caught and retried into trace-only code",
      body == GOOD_CODE and len(llm.prompts) == 2
      and "live-browser facility" in llm.prompts[1])

# ----------- P28 navigate that lands on an HTTP error page: fail fast
NAV_404 = ("### Page\n- Page URL: https://example.com/nope\n"
           "- Page Title: 404 Not Found\n- HTTP status: 404\n")
tr, fk = run(
    {"browser_navigate": NAV_404},
    [{"id": "s0", "tool": "browser_navigate",
      "arguments": {"url": "https://example.com/nope"}}],
)
check("P28 navigate 404: single attempt, guessed-URL hint",
      not tr.success and len(tr.steps[0].attempts) == 1
      and "HTTP 404" in (tr.steps[0].error or "")
      and "SEARCH endpoint" in (tr.steps[0].error or ""))
NAV_403 = NAV_404.replace("404 Not Found", "Access Denied") \
                 .replace("HTTP status: 404", "HTTP status: 403")
tr, fk = run(
    {"browser_navigate": NAV_403},
    [{"id": "s0", "tool": "browser_navigate",
      "arguments": {"url": "https://example.com/nope"}}],
)
check("P28b navigate 403: one settle-retry, then fail with honest hint",
      not tr.success and len(tr.steps[0].attempts) == 2
      and "HTTP 403" in (tr.steps[0].error or "")
      and "bot protection" in (tr.steps[0].error or ""))
# a transient throttle (429 then OK) heals within the step
NAV_429 = NAV_404.replace("HTTP status: 404", "HTTP status: 429")
tr, fk = run(
    {"browser_navigate": seq(NAV_429, NAV_TEXT)},
    [NAV],
)
check("P28d navigate 429 then OK: healed by the in-step retry",
      tr.success
      and [a.status for a in tr.steps[0].attempts] == ["tool_error", "success"])
# an OK navigation (no status line) is untouched
tr, fk = run({"browser_navigate": NAV_TEXT}, [NAV])
check("P28c navigate without an HTTP-status stamp: success as before",
      tr.success and tr.steps[0].status == "success")

# ----------- P27 headless sessions default to a standard-Chrome user agent
import os  # noqa: E402

from pkg.agenticmcpe.config import Settings  # noqa: E402

os.environ.pop("AGENTICMCPE_PW_USER_AGENT", None)
s = Settings.load(run_id="ua-selftest")
check("P27 headless default UA: standard Chrome, no 'Headless' marker",
      s.headless and isinstance(s.user_agent, str)
      and "Chrome/" in s.user_agent and "Headless" not in s.user_agent)
os.environ["AGENTICMCPE_PW_USER_AGENT"] = ""
s = Settings.load(run_id="ua-selftest2")
check("P27b empty AGENTICMCPE_PW_USER_AGENT disables the override",
      s.user_agent is None)
os.environ.pop("AGENTICMCPE_PW_USER_AGENT", None)

# ---------------------------------------------------------------------- summary
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
