"""Planner agent: natural-language task -> grounded MCP tool-call sequence.

A clean-room port of github-mcp-server's PlannerAgent, adapted for the
yfinance server's read-only surface. Key differences from the github
planner:

* No `get_me` bootstrap (the yfinance server has no notion of an
  authenticated user; tickers come from the task text).
* No write-side guardrails (every tool is read-only — there is no
  possibility of accidentally clobbering remote state).
* The catalog is condensed for the prompt: one line of description plus
  parameter types/required markers. Local validation rejects
  hallucinated tool names and missing required params before the plan
  reaches the executor.
* `post_action_properties` are emitted per step so the verifier can
  assert on numeric bounds, expected fields, and metamorphic relations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pkg.finance_mcp_wrapper.types import Tool
# _EMBEDDED_RE is the resolver's own grammar for embedded "$sN.path" segments.
# It is imported rather than restated so plan-time reference checking and
# execution-time resolution can never disagree about what counts as a binding.
from pkg.finance_mcp_wrapper.sequence import _EMBEDDED_RE, validate_against_schema

from .llm import LLMClient, Message
from .rag import KBRetriever, RetrievedExample, task_grounds_entry, catalog_enum_index

# A binding is a whole-value string "$<step_id>" optionally followed by a path.
_BINDING_REF_RE = re.compile(r"^\$(?P<step>[A-Za-z_][A-Za-z0-9_-]*)(?:[.\[].*)?$")

# RAG retrieval thresholds. Same shape as the github planner so we can
# share review notes / dashboards across the two implementations.
RAG_TOP_K = 3
RAG_STRONG_THRESHOLD = 0.86
RAG_EXAMPLE_FLOOR = 0.18

# JSON-Schema scalar type -> accepted python types, for local plan validation.
_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _binding_refs(value: Any) -> set[str]:
    """Collect the step ids referenced by binding templates inside ``value``.

    Covers BOTH binding forms the resolver understands: a whole-value
    ``"$s1.close"`` and any ``$sN.path`` segments embedded in a longer
    string (report lines, composed queries). Missing the embedded form
    meant a reference to a later or non-existent step survived local
    validation and only surfaced as a runtime ``BindingError``, costing a
    replan round for something rejectable for free.
    """
    refs: set[str] = set()
    if isinstance(value, str):
        m = _BINDING_REF_RE.match(value)
        if m:
            refs.add(m.group("step"))
        else:
            # Not a whole-value template; the resolver would interpolate
            # embedded segments, so check those the same way it will.
            refs.update(em.group("step") for em in _EMBEDDED_RE.finditer(value))
    elif isinstance(value, dict):
        for v in value.values():
            refs |= _binding_refs(v)
    elif isinstance(value, list):
        for v in value:
            refs |= _binding_refs(v)
    return refs


# ---------------------------------------------------------------------------
# Catalog condensation
# ---------------------------------------------------------------------------


# The server's tool signatures type these parameters as plain `str`, so
# FastMCP publishes NO enum in the generated schema even though
# servers/yahoo_finance/server.py validates each against a real Enum (or an
# explicit membership test). PORTING seam 3 calls for a manual table when a
# server does not publish its own. Values are taken from that source file.
#
# Consumed by the planner prompt (so the choices are visible as data rather
# than only as prose) and by `rag.catalog_enum_index`, whose groundedness
# exemption for schema-enumerated values was otherwise dead — every stored
# `financial_type`/`holder_type`/... blocked verbatim KB reuse.
_ENUM_OVERRIDES: dict[tuple[str, str], list[str]] = {
    ("get_financial_statement", "financial_type"): [
        "income_stmt", "quarterly_income_stmt", "balance_sheet",
        "quarterly_balance_sheet", "cashflow", "quarterly_cashflow",
    ],
    ("get_holder_info", "holder_type"): [
        "major_holders", "institutional_holders", "mutualfund_holders",
        "insider_transactions", "insider_purchases", "insider_roster_holders",
    ],
    ("get_recommendations", "recommendation_type"): [
        "recommendations", "upgrades_downgrades",
    ],
    ("get_option_chain", "option_type"): ["calls", "puts"],
    ("get_historical_stock_prices", "interval"): [
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
        "1d", "5d", "1wk", "1mo", "3mo",
    ],
}


def _condensed_catalog(tools: Iterable[Tool]) -> list[dict[str, Any]]:
    """Compress the catalog for the planner prompt: name, one-line
    description, parameter list (name, type, required)."""
    out: list[dict[str, Any]] = []
    for t in tools:
        props = (t.input_schema or {}).get("properties") or {}
        required = set((t.input_schema or {}).get("required") or [])
        params: list[dict[str, Any]] = []
        for pname, pschema in props.items():
            params.append(
                {
                    "name": pname,
                    "type": pschema.get("type", "string"),
                    "required": pname in required,
                    "enum": pschema.get("enum") or _ENUM_OVERRIDES.get((t.name, pname)),
                    "description": (pschema.get("description") or "").split("\n")[0],
                }
            )
        out.append(
            {
                "tool": t.name,
                "description": (t.description or "").split("\n")[0],
                "params": params,
            }
        )
    return out


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are the PLANNER in an agentic workflow that replaces a human driving \
the yahoo_finance MCP server. You translate a natural-language task \
into a STRICT, ordered tool-call sequence using ONLY the provided tool \
catalog.

Rules:
- Use ONLY tools that appear in the catalog. Never invent tools or parameters.
- Only use parameter names that appear in a tool's `params`. Provide every \
required parameter.
- Steps are ordered and identified s0, s1, ... A later step may reference an \
earlier step's output with a binding string "$<id>.<json.path>" \
(e.g. "$s0.ticker", "$s1.items[0].close"). A binding that is the WHOLE \
value of a parameter keeps its native type (numbers stay numbers). A \
binding may ALSO be EMBEDDED inside a longer string: every "$sN.path" \
segment is substituted with the value rendered as text at execution time — \
e.g. a report line "$s1.items[0].ticker gained $s2.items[0].close". Build \
queries, report lines and quoted text from prior outputs this way; NEVER \
hand-copy or guess a value that a prior step returns.
- Prefer the minimum number of steps that accomplishes the task. Do not add \
speculative or cleanup steps unless the task requires them.
- For each step include `post_action_properties`: a JSON object of checkable \
expectations about the result of that step, that a verifier can assert later. \
Use these keys when relevant:
    - "expect_fields": [field names the result must contain, spelled EXACTLY \
as the result-shape table below spells them — for an ARRAY result these are \
the keys of an element, not of the array]
    - "invariants": { "field": expected_fixed_value, ... }  (a field whose \
value must EQUAL a known constant; equality only — do not write comparison \
strings like ">= 1")
    - "bounds": { "field": {"min": n, "max": m} }  (numeric boundaries)
    - "min_length" / "exact_length": integer cardinality of an ARRAY result
    - "metamorphic": "short natural-language relation to re-check live, e.g. 'a price for this ticker now exists'"

yfinance read-task guardrails (follow exactly):
- Tickers come from the TASK TEXT or from prior step outputs — never \
invent a ticker you weren't given or didn't derive. A task mentioning \
multiple companies requires a call per company unless the tool genuinely \
takes a list.
- Dates: pass ISO strings ("YYYY-MM-DD").
- CRITICAL — NEVER bind a date OUT of one result INTO a date parameter. \
Bindings are lookups: the engine cannot truncate, reformat or convert. The \
two date fields this server returns are both unusable as-is:
  * a price bar's "Date" is a full timestamp, "2025-01-02T05:00:00.000Z". \
Feeding it to start_date/end_date fails with "unconverted data remains: \
T05:00:00.000Z".
  * get_holder_info's "Date Reported" is EPOCH MILLISECONDS (an int, e.g. \
1782777600000). Feeding it to a date parameter fails schema validation \
with "expected type 'string', got int".
  So `"start_date": "$s0[0].Date"` and `"start_date": "$s0[0].Date \
Reported"` are BOTH errors. When a later step needs a date, write it as a \
LITERAL "YYYY-MM-DD" taken from the task, and if the task defines the date \
only in terms of an earlier result, fetch a window wide enough to contain \
it and let the report step name the date in its text.
- CRITICAL — `end_date` is EXCLUSIVE. `get_historical_stock_prices` returns \
bars for start_date <= Date < end_date. To include the bar FOR a target \
date you MUST pass end_date = target + 1 day. Consequences you must \
respect:
  * A single day's bar needs start_date=D and end_date=D+1. Passing \
start_date == end_date returns an EMPTY array — never do it.
  * "as of market close on D" / "held until D" means the bar FOR D, so \
end_date = D + 1 day.
  * Sizing a window by TRADING days: roughly 21 per month and 252 per year, \
so a calendar window yields about 0.69x its days as bars, fewer around \
holidays. When a task needs N trading days of history (an N-day moving \
average, "the last N sessions"), request at least N * 1.5 calendar days \
before the target date — under-fetching makes the task unanswerable.
  * Do NOT declare a cardinality you are only estimating. "exact_length" \
and "min_length" above 1 are treated as advisory precisely because holiday \
calendars make them unreliable; use "min_length": 1 to say "this must not \
be empty", which is the claim that actually matters.
  * Only assert an "invariants" value that the TASK states (a ticker or \
date the user gave). Never predict what the data will contain — the \
largest holder, the sector, this quarter's direction — because the real \
answer may differ and that is not a failure of the run.
  * Markets are closed on weekends and US holidays. If your target date \
may be a non-trading day, widen the window by a few days and select the \
bar you need by its "Date" field rather than assuming an index.
- "as of market close on YYYY-MM-DD" uses that date's daily bar; do not \
invent intraday times.
- NEVER bind into an array index that may be empty (a search that \
returned nothing, a list of size 1 with index 0 from a multi-ticker \
query). If you must index into a list, FIRST take a step whose \
post_action_properties guarantees non-emptiness ("min_length": 1).
- Computations (percent returns, moving averages, derived ratios) MUST \
be expressed as the literal math in the report step's quoted text — the \
planner/executor do NOT evaluate expressions. Do NOT write bindings like \
"$s1.close - $s0.open"; instead include both reads and let the report \
compose the values as text.
- Indexing rules (the binding engine enforces these):
  * A COMPOSED string (a report line, a sentence, any value holding more \
than one binding) MUST NOT START with "$". A value beginning with "$" is \
parsed as a single whole-value binding and the entire remainder of the \
string is swallowed as its path, which is a HARD ERROR. Put a word first: \
write "Close was $s0[0].Close vs $s1.currentPrice", NEVER \
"$s0[0].Close vs $s1.currentPrice". A bare "$s0[0].Close" alone as the \
whole parameter value is fine — the rule applies only when text or a \
second binding follows.
  * A path segment is either ".key" or "[integer]". The JS-style \
'["key"]' form is a HARD ERROR. Keys containing spaces still use the dot \
form: write "$s0[0].Date Reported", never '$s0[0]["Date Reported"]'.
  * Array indices must be NON-NEGATIVE integer literals. "$sN[-1]" is a \
hard error. To read the LAST bar of a series, request a narrow date range \
so the bar you want sits at a known index (e.g. a 1-trading-day window \
gives you "$sN[0]").
  * There is NO ".length" / ".size" accessor. To assert a series is \
non-empty use post_action_properties {"min_length": 1}, not an invariant.
  * An index into an empty array is a hard error, so never index a series \
whose date range may contain no trading day (weekends, holidays, future \
dates). Widen the window by a few days instead.

- Result shapes for the yfinance server (authoritative, verified against \
the live server — never guess, and mind the CAPITALISATION):
  * `get_historical_stock_prices` -> ARRAY of daily bars, oldest first. \
Each bar has exactly: "Date" (ISO-8601 with timezone, e.g. \
"2023-01-09T05:00:00.000Z"), "Open", "High", "Low", "Close", "Volume", \
"Dividends", "Stock Splits". There is NO "Adj Close" and NO lowercase \
alias. Bind e.g. "$s0[0].Close".
  * `get_stock_info` -> flat OBJECT of quote/company fields, e.g. \
"longName", "sector", "industry", "currentPrice", "marketCap", \
"trailingPE", "dividendYield", "fiftyTwoWeekHigh". Bind e.g. \
"$s0.currentPrice".
  * `get_stock_actions` -> ARRAY of {"Date", "Dividends", "Stock Splits"}.
  * `get_financial_statement` -> ARRAY of period objects, newest first. \
Each has "date" (lowercase, "YYYY-MM-DD") plus one key per line item, \
named exactly as Yahoo reports it with spaces and capitals \
("Total Revenue", "Net Income", "EBITDA", ...).
  * `get_holder_info` -> ARRAY whose element shape depends on holder_type: \
major_holders -> {"metric", "Value"}; institutional_holders and \
mutualfund_holders -> {"Date Reported", "Holder", "pctHeld", "Shares", \
"Value", "pctChange"}; the insider_* types have their own columns.
  * `get_option_expiration_dates` -> ARRAY of plain date STRINGS \
("YYYY-MM-DD"); bind an element as "$s0[0]", not "$s0[0].date".
  * `get_option_chain` -> ARRAY of contract objects ("contractSymbol", \
"strike", "lastPrice", "bid", "ask", "volume", "openInterest", \
"impliedVolatility").
  * `get_recommendations` -> ARRAY; recommendation_type "recommendations" \
gives {"period", "strongBuy", "buy", "hold", "sell", "strongSell"}, \
"upgrades_downgrades" gives {"GradeDate", "Firm", "ToGrade", "FromGrade", \
"Action"}.
  * `get_yahoo_finance_news` returns a formatted PLAIN TEXT block — the \
data is a string, not a JSON array; bind the whole value as "$sN", never \
with a path that assumes JSON.

- This server has NO calculator tool. When a task asks for a computed \
answer (percent return, final portfolio value, a moving average, a \
signal), your job is to fetch every input the arithmetic needs and state \
the arithmetic in the final step's `description` — do NOT invent a tool \
to do the maths and do NOT fabricate the numeric answer.
- Bindings may ONLY reference a step that appears EARLIER in this same \
plan. Never bind to a step id you did not include.
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
per-step outcome and, for read steps, what they returned.

Produce a FRESH plan (renumber from s0) that completes the ORIGINAL task FROM \
THE CURRENT STATE. Requirements:
- SUCCESS steps already happened; their data is already available. Do NOT \
include steps that re-fetch the same primary data unless a later step \
binds a path that requires it.
- You MAY (and usually must) RE-INCLUDE read steps, because later steps \
bind to their outputs. Anything you bind to with "$sN" MUST be a step you \
include in this plan.
- Fix what failed using the read-step outputs below — e.g. if a date range \
returned an empty list, broaden the dates; if the ticker was wrong, swap \
to the corrected one.
- NEVER resubmit the failed plan unchanged — materially change the failing \
step (different arguments, a corrected binding, or an extra lookup step) \
based on the error text.

EXECUTION REPORT:
{feedback}
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


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
        return cls(
            id=str(obj.get("id") or default_id),
            tool=obj["tool"],
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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlannerAgent:
    """Task -> grounded Plan (s0..sn) + summary, validated locally."""

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        retriever: KBRetriever | None = None,
    ):
        self.llm = llm
        self.condensed = _condensed_catalog(tools)
        self.tools = {t.name: t for t in tools}
        self._enum_index = catalog_enum_index(self.condensed)
        # Optional RAG corpus of verified task->sequence mappings. When None
        # the planner behaves as a pure LLM planner (the taskgen pipeline
        # deliberately leaves it None so it keeps rediscovering sequences).
        self.retriever = retriever

    # ----------------------------------------------------------------- public

    def plan(self, task: str, *, feedback: str | None = None) -> Plan:
        """Plan a single sequence. ``feedback`` is the executor's
        error_context on a replan — see `_REPLAN_NOTE` for the contract."""
        examples: list[RetrievedExample] = []
        if self.retriever is not None and not feedback and not self.retriever.is_empty:
            examples = self.retriever.retrieve(task, k=RAG_TOP_K)
            reused = self._maybe_reuse(task, examples)
            if reused is not None:
                return reused

        system = _PLANNER_SYSTEM
        catalog_json = json.dumps(self.condensed, separators=(",", ":"))
        user = f"TOOL CATALOG (authoritative):\n{catalog_json}\n\nTASK:\n{task}\n"
        user += self._format_examples(examples)
        if feedback:
            user += _REPLAN_NOTE.format(feedback=feedback)

        reply = self.llm.complete(
            system=system,
            messages=[Message("user", user)],
            json_mode=True,
        )
        try:
            obj = _extract_json(reply)
        except ValueError as e:
            raise PlannerError(f"planner produced unparseable output: {e}") from e
        if not isinstance(obj, dict) or "steps" not in obj:
            raise PlannerError(f"planner output missing 'steps': {obj!r}")

        steps = [
            PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(obj.get("steps", []))
        ]
        settings = getattr(self.llm, "settings", None)
        plan = Plan(
            task=task,
            summary=str(obj.get("summary", "")),
            steps=steps,
            provider=getattr(settings, "provider", "") or type(self.llm).__name__,
            model=getattr(settings, "model", ""),
        )
        plan.warnings = self._validate(plan)
        return plan

    # ----------------------------------------------------------------- RAG path

    def _maybe_reuse(self, task: str, hits: list[RetrievedExample]) -> Plan | None:
        """Reuse a verified KB entry verbatim when the top hit is near-
        identical AND every literal argument is grounded in THIS task."""
        if not hits:
            return None
        top = hits[0]
        if top.score < RAG_STRONG_THRESHOLD or not task_grounds_entry(
            top.entry, task, self._enum_index
        ):
            return None
        steps_raw = top.entry.get("steps") or []
        if not steps_raw:
            return None
        steps = [PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(steps_raw)]
        plan = Plan(
            task=task,
            summary=str(top.entry.get("task_summary", "")),
            steps=steps,
        )
        try:
            plan.warnings = self._validate(plan)
        except PlannerError:
            return None
        plan.warnings.insert(
            0,
            f"RAG: reused verified KB entry {top.entry.get('id')} "
            f"(similarity={top.score:.2f}); no LLM planning call was made",
        )
        return plan

    def _format_examples(self, hits: list[RetrievedExample]) -> str:
        """Render retrieved solved tasks as in-context precedent for the LLM."""
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
            "argument from THIS task and from your own earlier steps via $bindings — "
            "never copy another task's literal tickers, dates or amounts:\n"
            + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        )

    # ----------------------------------------------------------------- validation

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
            tool = self.tools.get(st.tool)
            if tool is None:
                raise PlannerError(
                    f"step {st.id}: tool {st.tool!r} is not in the catalog"
                )
            # Validate against the live inputSchema. We treat required
            # params missing and unknown params as a hard reject; the
            # server will reject them anyway.
            props = (tool.input_schema or {}).get("properties") or {}
            required = set((tool.input_schema or {}).get("required") or [])
            present = set(st.arguments)
            missing = required - present
            if missing:
                raise PlannerError(
                    f"step {st.id} ({st.tool}): missing required params {sorted(missing)}"
                )
            unknown = present - set(props.keys())
            if unknown:
                warnings.append(
                    f"step {st.id} ({st.tool}): params not in schema {sorted(unknown)} "
                    "(server may reject)"
                )
            # Type/enum checks for literal scalar args.
            for k, v in st.arguments.items():
                pschema = props.get(k)
                if pschema is None or (isinstance(v, str) and _BINDING_REF_RE.match(v)):
                    continue
                errs = validate_against_schema(v, pschema, path=k)
                if errs:
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} failed "
                        f"schema check: {errs[0]}"
                    )
        return warnings


# ---------------------------------------------------------------------------
# JSON extraction + KB utilities
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Any:
    """Parse the first complete JSON object or array out of ``text``.

    Tolerant of leading/trailing prose, ```json fences, and single-line
    comments. Falls back to ``json.loads`` on the whole string. Raises
    ``ValueError`` on failure.
    """
    s = text.strip()
    # Strip code fences if present.
    if s.startswith("```"):
        # drop opening fence (optionally with language tag) and trailing fence
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Find the first { or [ and the matching closer.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if esc:
                esc = False
                continue
            if in_str:
                if ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(s[start : i + 1])
        # If we didn't find a balanced closer, fall through to whole-string parse.
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse JSON from planner reply: {e}") from e


def _compact_steps(steps: list[dict[str, Any]], max_val: int = 120) -> list[dict[str, Any]]:
    """Trim a KB entry's steps for the prompt: keep tool + arguments, truncating
    long string values (e.g. embedded report lines)."""
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


__all__ = [
    "Plan",
    "PlanStep",
    "PlannerAgent",
    "PlannerError",
    "_condensed_catalog",
    "_extract_json",
]