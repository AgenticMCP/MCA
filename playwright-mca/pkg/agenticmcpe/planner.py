"""Planner agent: task prompt -> grounded tool-call sequence (s0..sn) + summary.

The planner is *grounded*: it only ever sees the real playwright-mcp tool
catalog (names, one-line docs, parameter schemas), so it cannot invent tools or
parameters. After the LLM proposes a plan we validate every step against the
catalog (tool exists, required params present) before writing it out — cheap
local rejection beats a failed server round-trip.

Output artifacts (written under ``settings.work_dir``):

* ``plan.json``  — machine-readable sequence the executor replays verbatim.
* ``plan.md``    — human summary of the sequence with inputs/expected outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import LLMClient, Settings, extract_json
from .rag import (KBRetriever, RetrievedExample, catalog_enum_index,
                  task_grounds_entry)

# A binding is a whole-value string "$<step_id>" optionally followed by a path.
_BINDING_REF_RE = re.compile(r"^\$(?P<step>[A-Za-z_][A-Za-z0-9_-]*)(?:[.\[].*)?$")

# Does the task text ask for the browser/page/tab to be closed at the end?
# Used by the deterministic close-repair (see PlannerAgent._ensure_close_step).
_CLOSE_RE = re.compile(
    r"clos(?:e|es|ing)\s+(?:the\s+)?(?:browser|page|tab|window)", re.IGNORECASE)

# RAG (knowledge-base) retrieval thresholds, all tunable:
# - TOP_K        how many solved examples to retrieve.
# - STRONG       cosine >= this AND every literal grounded in the task => REUSE
#                the stored sequence directly, with no LLM planning call.
# - EXAMPLE_FLOOR below this a hit is noise and is not even shown to the LLM.
#   Raised 0.18 -> 0.35 after a same-day A/B (runs_v7 13/35 with RAG vs
#   runs_v7_norag 17/35 without): matches in the 0.22-0.29 band injected
#   browse-style interactive precedent that nudged plans off the URL-first
#   discipline. Weak precedent is worse than none.
RAG_TOP_K = 3
RAG_STRONG_THRESHOLD = 0.86
RAG_EXAMPLE_FLOOR = 0.35

# JSON-Schema scalar type -> accepted python types, for local plan validation.
_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _binding_refs(value: Any) -> set[str]:
    """Collect the step ids referenced by binding templates inside ``value``."""
    refs: set[str] = set()
    if isinstance(value, str):
        m = _BINDING_REF_RE.match(value)
        if m:
            refs.add(m.group("step"))
    elif isinstance(value, dict):
        for v in value.values():
            refs |= _binding_refs(v)
    elif isinstance(value, list):
        for v in value:
            refs |= _binding_refs(v)
    return refs

_PLANNER_SYSTEM = """\
You are the PLANNER in an agentic workflow that replaces a human driving a web \
browser through the Playwright MCP server. You translate a natural-language \
task into a STRICT, ordered tool-call sequence using ONLY the provided tool \
catalog.

Rules:
- Use ONLY tools that appear in the catalog. Never invent tools or parameters.
- Only use parameter names that appear in a tool's `params`. Provide every \
required parameter.
- Steps are ordered and identified s0, s1, ... A later step may reference an \
earlier step's output with a binding string "$<id>.<json.path>" \
(e.g. "$s2.url"). A binding that is the WHOLE value of a parameter keeps its \
native type. A binding may ALSO be EMBEDDED inside a longer string; every \
"$sN.path" segment is substituted with the value rendered as text at \
execution time. NEVER hand-copy or guess a value that a prior step returns.
- Prefer the minimum number of steps that accomplishes the task. Do not add \
speculative steps unless the task requires them.
- For each step include `post_action_properties`: a JSON object of checkable \
expectations about the result of that step, that a verifier can assert later. \
Use these keys when relevant:
    - "expect_fields": [list of field names the result object must contain]
    - "invariants": { "field": expected_fixed_value, ... }  (values that must equal the input/known constant)
    - "bounds": { "field": {"min": n, "max": m} }  (numeric boundaries)
    - "metamorphic": "short natural-language relation to re-check live, e.g. 'the page now shows search results for Beijing'"

Playwright browser-task guardrails (follow exactly):
- ALWAYS make `browser_navigate` to the task's starting URL the first step \
(s0). Never assume a page is already open — every session starts with a blank \
browser.
- OBSERVE before you ACT: `browser_click`/`browser_type`/`browser_hover`/\
`browser_select_option`/`browser_drag`/`browser_fill_form` need a `target` \
element reference that ONLY a `browser_snapshot` (or `browser_find`) step of \
THIS session provides. Plan a `browser_snapshot` immediately after every \
navigation and after any action that changes the page, BEFORE the next \
interaction.
- `target` accepts two forms, and you must use exactly these:
  1. "eNN" — a ref that appeared as [ref=eNN] in the MOST RECENT snapshot. \
Refs go STALE after any navigation or DOM change (like a blob SHA after a \
write); a stale ref fails with "Ref not found". Use refs when replanning from \
a snapshot shown in the execution report.
  2. "find:<visible text>" — resolved by the executor against a FRESH \
automatic observation of the page at execution time (you do NOT need a \
snapshot step before an action that uses find:), deterministically: an \
element whose quoted accessible NAME equals the text exactly wins \
(find:Search picks the button named "Search" even when other lines contain \
the word); otherwise the text must be a substring of EXACTLY ONE snapshot \
line. Prefer an element's exact caption. If resolution is ambiguous or empty \
the step fails and the error lists the candidates — replan with better text \
or a concrete ref. find: is the DEFAULT targeting form; explicit \
browser_snapshot steps are still how you READ page content and where refs \
come from.
  NEVER invent a bare ref you did not see in a snapshot, and never use CSS \
selectors — they are unreliable on this server.
- Result shapes (authoritative — never guess others): EVERY tool returns \
markdown TEXT, which the executor surfaces as an object {text, url, title} -> \
bind "$sN.text" (whole result text), "$sN.url" (current page URL), \
"$sN.title" (page title). url/title exist only when the result carries page \
state (`browser_snapshot`, `browser_navigate`, most actions). There are NO \
other fields — never bind "$sN.items", "$sN.ref" or similar.
- The full page snapshot is INLINE only in `browser_snapshot` and \
`browser_find` results. Navigation/action results do NOT carry the page \
content — to read or verify page content, plan an explicit `browser_snapshot`.
- When the page loads content asynchronously (search results, maps, prices), \
plan `browser_wait_for` BEFORE the snapshot that reads it. `text` may ONLY be \
(a) literal text YOU typed/submitted in an earlier step of THIS plan, or (b) \
text an earlier snapshot of THIS plan already showed. NEVER wait for \
PREDICTED content — a price, a duration like "min", a heading or city name \
you guessed the page will use: if the guess is wrong the step burns its whole \
timeout and fails. When you cannot point to the text's origin, use a short \
`time` wait (2-3 seconds) instead, then `browser_snapshot` and read what is \
actually there.
- If a `browser_snapshot` comes back EMPTY (no elements), the page had not \
rendered yet: add `browser_wait_for` with `time` 2-3 after the navigation, \
then snapshot again.
- `browser_find` text must be PLAIN literal words — no parentheses, brackets, \
slashes or other special characters (the server treats the text as a pattern \
and rejects invalid ones).
- Do NOT guess CONTENT URLs (blog-post paths, article slugs, listing pages \
you have not seen). But DO use a site's canonical SEARCH endpoint when you \
know it — e.g. "https://huggingface.co/search/full-text?q=<terms>", \
"https://arxiv.org/abs/<id>" when the task gives the id, \
"https://www.google.com/maps/dir/<origin>/<destination>" — navigating to a \
search URL is MORE reliable than driving a dynamic search widget. After a \
search-results navigation, `browser_wait_for` a term you searched for (results \
load asynchronously), then snapshot.
- On google.com/maps specifically, ALWAYS prefer the URL forms \
".../maps/search/<query>" and ".../maps/dir/<origin>/<destination>" over \
typing into the maps search widget: headless sessions often land on a consent \
interstitial where the widget does not exist. If a snapshot shows a consent \
page instead of the map, dismiss it first.
- Sites may serve a LOCALIZED UI based on the machine's region (booking.com \
often renders in Chinese here — pin the locale by URL when known: \
"https://www.booking.com/flights/index.en-us.html"). When a snapshot shows \
captions in another language, target the captions AS SHOWN in the snapshot — \
never their English translation.
- Some sites refuse automated browsers outright: the page shows "Access \
Denied", an HTTP 403, or a bot check, or navigation times out on every \
attempt. Retrying or waiting CANNOT fix a block. Reach the information via a \
different page of the allowed site if one exists; otherwise snapshot the \
denial page as the outcome — NEVER invent content the browser could not see.
- Some heavy pages (large SPA homepages) re-render continuously, so refs go \
stale between snapshot and action no matter how fresh the snapshot is. If an \
interaction on such a page keeps failing, route around it: reach the content \
by URL (search endpoint) or via a simpler page instead of retrying the widget.
- When the task supplies literal text to type or search for, reproduce it \
VERBATIM in `browser_type`'s `text` — preserve spelling and case.
- Type into a search box with `browser_type` (set `submit: true` to press \
Enter) instead of clicking through suggestion dropdowns when possible — it is \
one deterministic step.
- Consent/cookie dialogs: public sites often show one on first load. When one \
is visible in the snapshot, dismiss it (click its decline/reject button if \
present, else accept) before other interactions. If a NATIVE browser dialog \
(alert/confirm) blocks the page, use `browser_handle_dialog` — but ONLY as a \
REACTION to a dialog you can see in a snapshot or that an error reports as \
blocking. Never add it speculatively: with no dialog open it does nothing.
- Use `browser_tabs` (action "select"/"list") when a click opens a new tab; \
otherwise stay on the current tab.
- Do NOT use `browser_evaluate` or `browser_run_code_unsafe` unless the task \
explicitly requires running JavaScript.
- TASK COVERAGE — a plan is complete only when EVERY requirement of the task \
maps to a step:
  - A task that asks to READ / EXTRACT / CHECK information must RECORD it: \
end with a `browser_snapshot` (or `browser_find`) taken ON THE PAGE THAT \
SHOWS that information, before any `browser_close`. The recorded snapshot is \
the only evidence the run produces — information that was never snapshotted \
is LOST. Never emit a navigate-only plan for such a task.
  - When the task names MULTIPLE entities (two players, both teams, several \
papers or arXiv IDs), the plan must RECORD EACH entity's information — a \
snapshot on each entity's page, or one snapshot that visibly contains them \
all. A verifier will look for EVERY named entity in the recorded snapshots; \
answering about one entity while never visiting the other fails.
  - When the task says to close the browser, the FINAL step MUST be \
`browser_close`.
  - TASK-NAMED GESTURES are part of the requirement: when the task itself \
asks for a browser gesture, use the corresponding tool instead of routing \
around it — "press <key>" as its own action -> `browser_press_key` (type \
first with submit false), "hover over X" -> `browser_hover`, "go back" / \
"use the back button" -> `browser_navigate_back`, "in a new/second tab" -> \
`browser_tabs`, "fill in the form" -> `browser_fill_form`, "check the \
network requests" -> `browser_network_requests` (one specific request -> \
`browser_network_request`), "take/grab a screenshot" -> \
`browser_take_screenshot`, "phone-sized/resize the window" -> \
`browser_resize`. For tasks that do NOT name a gesture, keep the default \
route (URL navigation, type with submit) — it is more deterministic.
- Bindings are LOOKUPS ONLY — there are NO expressions, comparisons or \
arithmetic. When an action depends on COMPARING runtime values, include the \
read steps plus the action with your best-guess literal; if the guess is \
wrong it fails, and on replan the per-step outputs show the real values.
- A binding "$sN..." may only reference a step that appears EARLIER in this \
same plan. Never bind to a step id you did not include.
- Output a SINGLE JSON object, no prose, with this exact shape:
{
  "summary": "<one paragraph describing what the sequence does>",
  "steps": [
    {
      "id": "s0",
      "tool": "<tool name from catalog>",
      "arguments": { ... },
      "post_action_properties": { ... },
      "description": "<why this step>"
    }
  ]
}
"""

_REPLAN_NOTE = """\

The previous plan FAILED during execution. Below is the prior plan with the \
per-step outcome and, for observation steps, what they returned.

Produce a FRESH plan (renumber from s0) that completes the ORIGINAL task. \
Requirements:
- Browser state does NOT survive between attempts: the new plan starts from a \
BLANK browser. Re-include `browser_navigate` and every interaction needed to \
reach the point of failure — on a fresh page these are NOT duplicates. \
(The exception is a DURABLE external effect the report shows already \
happened, e.g. a form actually submitted to the site — do not repeat that; \
navigate directly to the resulting page when its URL is in the report.)
- Element refs shown in the report come from the OLD session's snapshots. \
After the SAME navigation path the page usually renders the same, so a ref \
read from a report snapshot is a reasonable target — but when the report \
shows the ref failed or the page varies, prefer "find:<unique visible text>" \
taken from the report's snapshot text.
- Fix what failed using the outputs below: if a step failed with "matches \
several/no snapshot lines" or "Ref not found", the report includes the \
CURRENT snapshot lines — pick the exact ref, or COPY the visible text \
VERBATIM from one of those snapshot lines (never re-guess a caption from \
memory).
- If a snapshot shows a consent/cookie dialog, add a step to dismiss it \
before the interaction that failed.
- If content had not loaded yet (empty results, missing text), add \
`browser_wait_for` BY TIME (2-3 seconds) before the snapshot. Text seen in \
the OLD session's snapshots does NOT qualify as wait_for text — the new \
session is a fresh browser and the page may render differently; wait_for \
`text` in the new plan may only be text the NEW plan itself types or submits.
- NEVER resubmit the failed plan unchanged — materially change the failing \
step (different target, an added wait/snapshot/dismiss step) based on the \
error text.
- NEVER resubmit a find: text that just failed as AMBIGUOUS or NO-match: use \
one of the refs LISTED in the error, or a DIFFERENT caption copied verbatim \
from the snapshot excerpt (for ambiguous links the error shows each \
candidate's /url — pick by destination).

EXECUTION REPORT:
{feedback}
"""


@dataclass
class PlanStep:
    id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    post_action_properties: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expect_output: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: dict[str, Any], default_id: str) -> "PlanStep":
        raw_args = dict(obj.get("arguments") or obj.get("args") or {})
        # A step without a tool is malformed output, not a crash: raise the
        # same error every other malformation raises so the orchestrator's
        # self-correction retry can feed the reason back to the planner.
        if not obj.get("tool"):
            raise PlannerError(
                f"step {obj.get('id') or default_id!r} has no 'tool' field: {obj!r}")
        return cls(
            id=str(obj.get("id") or default_id),
            tool=obj["tool"],
            # Models emit `null` for optional params they want to skip; the
            # schema rejects nulls, so omitting is the only correct encoding.
            arguments={k: v for k, v in raw_args.items() if v is not None},
            post_action_properties=dict(obj.get("post_action_properties") or {}),
            description=obj.get("description", ""),
            expect_output=obj.get("expect_output"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "arguments": self.arguments,
            "post_action_properties": self.post_action_properties,
            "description": self.description,
            **({"expect_output": self.expect_output} if self.expect_output else {}),
        }


@dataclass
class Plan:
    task: str
    summary: str
    steps: list[PlanStep]
    provider: str = ""
    model: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "provider": self.provider,
            "model": self.model,
            "summary": self.summary,
            "warnings": self.warnings,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Plan":
        steps = [PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(obj.get("steps", []))]
        return cls(
            task=obj.get("task", ""),
            summary=obj.get("summary", ""),
            steps=steps,
            provider=obj.get("provider", ""),
            model=obj.get("model", ""),
            warnings=list(obj.get("warnings", [])),
        )


class PlannerError(RuntimeError):
    pass


class PlannerAgent:
    def __init__(self, llm: LLMClient, condensed_catalog: list[dict[str, Any]],
                 retriever: "KBRetriever | None" = None):
        self.llm = llm
        self.catalog = condensed_catalog
        self._by_name = {c["tool"]: c for c in condensed_catalog}
        # Enum/method values from the schema, exempted by the reuse gate below.
        self._enum_index = catalog_enum_index(condensed_catalog)
        # Optional RAG corpus of verified task->sequence mappings. When None the
        # planner behaves exactly as before (pure LLM planning). The taskgen
        # pipeline deliberately leaves this None so it keeps rediscovering
        # sequences independently — the basis of its ground truth.
        self.retriever = retriever

    # ------------------------------------------------------------------ public
    def plan(self, task: str, *, feedback: str | None = None) -> Plan:
        # --- RAG: consult the knowledge base BEFORE the LLM. Only on the initial
        # plan (a state-aware replan must reason about live execution state, not
        # a precedent), and only when a retriever is wired and non-empty. ---
        examples: list[RetrievedExample] = []
        if self.retriever is not None and not feedback and not self.retriever.is_empty:
            examples = self.retriever.retrieve(task, k=RAG_TOP_K)
            reused = self._maybe_reuse(task, examples)
            if reused is not None:
                return reused  # sequence "referable" -> no LLM call needed

        system = _PLANNER_SYSTEM
        catalog_json = json.dumps(self.catalog, separators=(",", ":"))
        user = f"TOOL CATALOG (authoritative):\n{catalog_json}\n\nTASK:\n{task}\n"
        user += self._format_examples(examples)
        if feedback:
            user += _REPLAN_NOTE.format(feedback=feedback)

        reply = self.llm.chat(system, user, json_mode=True)
        try:
            obj = extract_json(reply)
        except ValueError as e:
            raise PlannerError(f"planner produced unparseable output: {e}") from e
        if not isinstance(obj, dict) or "steps" not in obj:
            raise PlannerError(f"planner output missing 'steps': {obj!r}")

        steps = [
            PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(obj.get("steps", []))
        ]
        plan = Plan(
            task=task,
            summary=str(obj.get("summary", "")),
            steps=steps,
            provider=self.llm.settings.provider,
            model=self.llm.settings.model,
        )
        plan.warnings = self._validate(plan)
        # Record RAG participation so runs are analyzable post-hoc: full reuse
        # is already logged by _maybe_reuse; note in-context precedent here.
        shown = sum(1 for h in examples if h.score >= RAG_EXAMPLE_FLOOR)
        if shown:
            plan.warnings.append(
                f"RAG: {shown} verified KB example(s) shown to the planner as "
                f"precedent (top similarity={examples[0].score:.2f})")
        self._ensure_close_step(plan)
        return plan

    def _ensure_close_step(self, plan: Plan) -> None:
        """Deterministic repair: a task that says to close the browser must end
        with browser_close. Models under-comply (batch evidence: several plans
        omitted it and verifiers checked for it); browser_close is idempotent,
        so appending is always safe. The repair is recorded as a warning so
        plan.json/plan.md show it transparently."""
        if not _CLOSE_RE.search(plan.task):
            return
        if any(s.tool == "browser_close" for s in plan.steps):
            return
        if "browser_close" not in self._by_name or not plan.steps:
            return
        ids = {s.id for s in plan.steps}
        n = len(plan.steps)
        new_id = f"s{n}"
        while new_id in ids:
            n += 1
            new_id = f"s{n}"
        plan.steps.append(PlanStep(
            id=new_id, tool="browser_close",
            description="auto-appended: the task asks to close the browser"))
        plan.warnings.append(
            f"repair: appended {new_id} browser_close — the task asks to "
            "close the browser and the model's plan omitted it")

    # ----------------------------------------------------------------- RAG path
    def _maybe_reuse(self, task: str, hits: list[RetrievedExample]) -> Plan | None:
        """If the top KB hit is a near-identical task whose literal arguments are
        all grounded in THIS task, reuse its verified sequence verbatim — that is
        what "the tool-call sequence can be referred" means. Returns the reused
        Plan, or None to fall through to LLM planning. The reused plan is still
        re-validated against the live catalog, so a stale entry (a tool/param
        that no longer exists) safely falls back instead of being trusted."""
        if not hits:
            return None
        top = hits[0]
        if top.score < RAG_STRONG_THRESHOLD or not task_grounds_entry(
                top.entry, task, self._enum_index):
            return None
        steps_raw = top.entry.get("steps") or []
        if not steps_raw:
            return None
        steps = [PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(steps_raw)]
        plan = Plan(
            task=task,
            summary=str(top.entry.get("task_summary", "")),
            steps=steps,
            provider=self.llm.settings.provider,
            model=self.llm.settings.model,
        )
        try:
            plan.warnings = self._validate(plan)
        except PlannerError:
            return None  # stored sequence no longer valid -> let the LLM replan
        plan.warnings.insert(
            0, f"RAG: reused verified KB entry {top.entry.get('id')} "
               f"(similarity={top.score:.2f}); no LLM planning call was made")
        return plan

    def _format_examples(self, hits: list[RetrievedExample]) -> str:
        """Render retrieved solved tasks as in-context precedent for the LLM.
        Empty string when there is nothing worth showing."""
        useful = [h for h in hits if h.score >= RAG_EXAMPLE_FLOOR]
        if not useful:
            return ""
        items = [
            {
                "similar_task": h.entry.get("task_prompt", ""),
                "tool_sequence": h.entry.get("tool_sequence", []),
                "steps": _compact_steps(h.entry.get("steps") or []),
            }
            for h in useful
        ]
        return (
            "\n\nSOLVED EXAMPLES retrieved from the verified knowledge base (most "
            "similar first). Treat them as PRECEDENT, not the answer: follow the "
            "tool choice and ORDER when the task shape matches, but RE-DERIVE every "
            "argument from THIS task and from your own earlier steps via $bindings "
            "— never copy another task's literal URLs, search terms or element "
            "targets:\n"
            + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        )

    def write(self, plan: Plan, settings: Settings) -> tuple[str, str]:
        """Persist plan.json + plan.md. Returns their paths."""
        settings.ensure_work_dir()
        json_path = settings.work_dir / "plan.json"
        md_path = settings.work_dir / "plan.md"
        json_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self._render_md(plan), encoding="utf-8")
        return str(json_path), str(md_path)

    # ------------------------------------------------------------------ internal
    def _validate(self, plan: Plan) -> list[str]:
        """Reject hallucinated tools/params and dangling bindings before
        execution. Hard errors raise; soft issues are returned as warnings."""
        warnings: list[str] = []
        seen_ids: set[str] = set()
        for st in plan.steps:
            if st.id in seen_ids:
                raise PlannerError(f"duplicate step id {st.id!r}")
            # binding references must point to an EARLIER step in this plan
            for ref in _binding_refs(st.arguments):
                if ref not in seen_ids:
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): binding '${ref}...' references "
                        f"step {ref!r}, which is not an earlier step in this plan "
                        f"(earlier ids: {sorted(seen_ids)})"
                    )
            seen_ids.add(st.id)
            spec = self._by_name.get(st.tool)
            if spec is None:
                raise PlannerError(
                    f"step {st.id}: tool {st.tool!r} is not in the catalog"
                )
            known = {p["name"] for p in spec["params"]}
            required = {p["name"] for p in spec["params"] if p["required"]}
            present = set(st.arguments)
            missing = required - present
            if missing:
                raise PlannerError(
                    f"step {st.id} ({st.tool}): missing required params {sorted(missing)}"
                )
            unknown = present - known
            if unknown:
                warnings.append(
                    f"step {st.id} ({st.tool}): params not in schema {sorted(unknown)} "
                    "(server may reject)"
                )
            # enum/type check literal scalar args (bindings resolve later) —
            # catching these locally turns an execution-time validation
            # failure into a free planner self-correction.
            by_name = {p["name"]: p for p in spec["params"]}
            for k, v in st.arguments.items():
                p = by_name.get(k)
                if p is None or (isinstance(v, str) and _BINDING_REF_RE.match(v)):
                    continue
                if p.get("enum") and isinstance(v, (str, int)) and v not in p["enum"]:
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} not in enum "
                        f"{p['enum']}")
                expected = _TYPE_MAP.get(p.get("type"))
                if expected and not isinstance(v, expected):
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} must be of "
                        f"type {p['type']!r}")
                if p.get("type") == "integer" and isinstance(v, bool):
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} must be an integer")
        return warnings

    @staticmethod
    def _render_md(plan: Plan) -> str:
        lines = [
            f"# Plan for task\n",
            f"> {plan.task}\n",
            f"**Provider/model:** {plan.provider} / {plan.model}\n",
            f"## Summary\n\n{plan.summary}\n",
        ]
        if plan.warnings:
            lines.append("## Warnings\n")
            lines += [f"- {w}" for w in plan.warnings]
            lines.append("")
        lines.append("## Tool-call sequence\n")
        for st in plan.steps:
            lines.append(f"### {st.id} — `{st.tool}`")
            if st.description:
                lines.append(f"{st.description}\n")
            lines.append("```json")
            lines.append(json.dumps(st.arguments, indent=2))
            lines.append("```")
            if st.post_action_properties:
                lines.append("**Post-action properties:**")
                lines.append("```json")
                lines.append(json.dumps(st.post_action_properties, indent=2))
                lines.append("```")
            lines.append("")
        return "\n".join(lines)


def _compact_steps(steps: list[dict[str, Any]], max_val: int = 120) -> list[dict[str, Any]]:
    """Trim a KB entry's steps for the prompt: keep tool + arguments, truncating
    long string values so examples stay token-cheap."""
    out: list[dict[str, Any]] = []
    for s in steps:
        args: dict[str, Any] = {}
        for k, v in (s.get("arguments") or {}).items():
            if isinstance(v, str) and len(v) > max_val:
                args[k] = v[:max_val] + "…<truncated>"
            else:
                args[k] = v
        out.append({"tool": s.get("tool"), "arguments": args})
    return out


__all__ = ["PlanStep", "Plan", "PlannerAgent", "PlannerError"]
