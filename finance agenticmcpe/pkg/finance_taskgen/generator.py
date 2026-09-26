"""Task-generation agent: realistic finance-analysis tasks grounded in the
yahoo_finance catalog.

The generator does NOT invent tasks at random. It is forced through a
human-thinking scaffold (persona/motive -> concrete goal -> mental
walkthrough -> natural prompt) seeded with a scenario archetype, so the
resulting prompt reads like something a real user would ask and its
tool-call sequence is recoverable by an independent planner. The
generator's ``expected_tools`` is a hypothesis used for
coverage/diversity metadata only — ground truth is always established
downstream by execution + verification (see pipeline.py).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from pkg.finance_agenticmcpe.config import AgenticConfig
from pkg.finance_agenticmcpe.llm import LLMClient, Message
from pkg.finance_agenticmcpe.planner import _extract_json

# Generation wants VARIETY, so it runs the LLM hotter than the
# planner/verifier (which need temperature 0 for determinism).
GENERATION_TEMPERATURE = 0.85

# Personas: who is asking and why.
PERSONAS: list[str] = [
    "a retail investor considering a long-term position",
    "a day trader looking for momentum signals",
    "a financial planner building a client portfolio report",
    "a curious investor comparing companies in a sector",
    "a student learning about stock markets",
    "a treasury analyst doing competitor benchmarking",
    "a passive index investor checking their holdings",
    "an accountant preparing a year-end valuation summary",
    "a financial journalist writing a sector roundup",
    "an options trader evaluating volatility and expiry",
]

# Phrasing styles — vary voice/register/length per task.
PHRASING_STYLES: list[str] = [
    "terse and imperative — one direct sentence, no pleasantries",
    "conversational and polite — a couple of natural sentences, as if chatting",
    "context-first — a clause of background or motivation, then the concrete ask",
    "detailed and precise — names the exact ticker, dates and amounts up front",
    "slightly informal — natural wording, maybe a contraction or aside, still clear",
]

# Scenario archetypes. Every one is read-only by definition: the
# yfinance server exposes no write tools. The `tools_hint` is what the
# archetype naturally exercises; the planner decides the actual
# sequence at execution time.
ARCHETYPES: list[dict[str, Any]] = [
    {
        "name": "historical-return-calc",
        "intent": "Calculate the total return (or percentage change) for a "
                  "stock over a defined historical window.",
        "tools_hint": ["get_historical_stock_prices"],
    },
    {
        "name": "company-snapshot",
        "intent": "Pull a single company's snapshot: sector, market cap, "
                  "dividend yield, trailing PE, etc.",
        "tools_hint": ["get_stock_info"],
    },
    {
        "name": "sector-comparison",
        "intent": "Compare several companies' basic profile data to surface "
                  "differences in valuation, growth, or risk.",
        "tools_hint": ["get_stock_info"],
    },
    {
        "name": "news-monitoring",
        "intent": "Read recent news for one or more tickers and summarize "
                  "what was reported.",
        "tools_hint": ["get_yahoo_finance_news"],
    },
    {
        "name": "corporate-actions-survey",
        "intent": "Audit corporate actions (splits and dividends) for a "
                  "ticker over a window of years.",
        "tools_hint": ["get_stock_actions"],
    },
    {
        "name": "financial-statement-deep-dive",
        "intent": "Inspect a company's income statement, balance sheet, or "
                  "cash flow over multiple years.",
        "tools_hint": ["get_financial_statement"],
    },
    {
        "name": "ownership-and-holders",
        "intent": "Look at who owns a company (institutional / insider / "
                  "major holders) and the breakdown.",
        "tools_hint": ["get_holder_info"],
    },
    {
        "name": "analyst-recommendations",
        "intent": "Pull analyst recommendations and recent rating changes "
                  "for a ticker.",
        "tools_hint": ["get_recommendations"],
    },
    {
        "name": "options-chain-check",
        "intent": "Inspect the available options expirations and the chain "
                  "for a chosen expiry and right.",
        "tools_hint": ["get_option_expiration_dates", "get_option_chain"],
    },
    {
        "name": "multi-ticker-portfolio-snapshot",
        "intent": "Pull the same data field (e.g. sector or latest close) "
                  "for several tickers at once and report back.",
        "tools_hint": ["get_stock_info", "get_historical_stock_prices"],
    },
    {
        "name": "dividend-history",
        "intent": "Inspect dividend payments over years for an income-"
                  "focused position.",
        "tools_hint": ["get_stock_actions"],
    },
    {
        "name": "volatility-vol",
        "intent": "Look at the available options chain around a few "
                  "expiries to gauge volatility.",
        "tools_hint": ["get_option_expiration_dates", "get_option_chain"],
    },
    # --- cross-family archetypes -------------------------------------------
    # The archetypes above each stay inside one tool family, which is why the
    # corpus saturated on short homogeneous signatures (info->info->info,
    # statement->statement, ...). These deliberately span 3+ families so the
    # walkthrough has to interleave different data sources, producing the
    # longer, genuinely mixed sequences the KB was missing.
    {
        "name": "fundamentals-vs-price",
        "intent": "Set a company's reported fundamentals against how its "
                  "share price actually moved over the same window, to see "
                  "whether the market followed the numbers.",
        "tools_hint": ["get_financial_statement", "get_historical_stock_prices",
                       "get_stock_info"],
    },
    {
        "name": "earnings-reaction-study",
        "intent": "Line up the most recent reporting periods with the price "
                  "action around them and the news coverage, to judge how the "
                  "market reacted to each result.",
        "tools_hint": ["get_financial_statement", "get_historical_stock_prices",
                       "get_yahoo_finance_news"],
    },
    {
        "name": "dividend-safety-check",
        "intent": "Judge whether a company's dividend looks sustainable by "
                  "reading its payout history alongside earnings and cash "
                  "flow.",
        "tools_hint": ["get_stock_actions", "get_financial_statement",
                       "get_stock_info"],
    },
    {
        "name": "institutional-conviction-vs-performance",
        "intent": "Compare who holds a company against how the shares have "
                  "performed and what analysts currently say.",
        "tools_hint": ["get_holder_info", "get_historical_stock_prices",
                       "get_recommendations"],
    },
    {
        "name": "analyst-vs-market-divergence",
        "intent": "Check whether analyst ratings and recent rating changes "
                  "were borne out by the subsequent price move and the news.",
        "tools_hint": ["get_recommendations", "get_historical_stock_prices",
                       "get_yahoo_finance_news"],
    },
    {
        "name": "options-positioning-vs-fundamentals",
        "intent": "Read options positioning for a chosen expiry next to the "
                  "company's valuation and reported results.",
        "tools_hint": ["get_option_expiration_dates", "get_option_chain",
                       "get_stock_info", "get_financial_statement"],
    },
    {
        "name": "peer-benchmark-deep",
        "intent": "Benchmark two or three peers against each other on both "
                  "profile data and reported financials, not just price.",
        "tools_hint": ["get_stock_info", "get_financial_statement",
                       "get_historical_stock_prices"],
    },
    {
        "name": "balance-sheet-strength-screen",
        "intent": "Assess balance-sheet strength by reading more than one "
                  "statement type for the same company and relating it to the "
                  "current valuation.",
        "tools_hint": ["get_financial_statement", "get_stock_info"],
    },
    {
        "name": "split-adjusted-performance",
        "intent": "Work out true performance across a period that contains "
                  "corporate actions, using the action history alongside the "
                  "price series.",
        "tools_hint": ["get_stock_actions", "get_historical_stock_prices",
                       "get_stock_info"],
    },
    {
        "name": "insider-vs-institutional",
        "intent": "Contrast insider activity with institutional ownership for "
                  "the same company, and relate both to recent performance.",
        "tools_hint": ["get_holder_info", "get_historical_stock_prices"],
    },
    {
        "name": "full-due-diligence",
        "intent": "Assemble a complete pre-investment dossier on one company: "
                  "profile, reported financials, ownership, analyst view, "
                  "recent news and price history.",
        "tools_hint": ["get_stock_info", "get_financial_statement",
                       "get_holder_info", "get_recommendations",
                       "get_yahoo_finance_news", "get_historical_stock_prices"],
    },
    {
        "name": "sector-rotation-scan",
        "intent": "Scan several companies from different sectors on profile, "
                  "price performance and analyst sentiment to see where "
                  "momentum sits.",
        "tools_hint": ["get_stock_info", "get_historical_stock_prices",
                       "get_recommendations"],
    },
]

# Difficulty -> (min_steps, max_steps) for the mental walkthrough.
# "expert" exists so the generator can reach for the long interleaved
# walkthroughs the cross-family archetypes describe; without a tier above
# 6 steps it kept collapsing them back into short single-family chains.
DIFFICULTY_STEPS: dict[str, tuple[int, int]] = {
    "easy": (1, 2),
    "medium": (2, 4),
    "hard": (4, 7),
    "expert": (6, 9),
}


_GENERATOR_SYSTEM = """\
You are a TASK GENERATION agent. You invent realistic finance-analysis \
tasks that a human would ask an AI assistant to perform using a \
yahoo_finance MCP server. Each task is later given to an INDEPENDENT \
planner that must rediscover the right tool-call sequence, execute it \
against the live server, and verify the outcome; tasks that survive \
become benchmark ground truth.

Think like a real user, in this exact order (the required human-
thinking structure):

1. PERSONA & MOTIVE — who is asking and why (retail investor, day \
   trader, financial planner, curious learner, ...).
2. CONCRETE GOAL — the specific outcome they want, with concrete \
   tickers, dates, amounts and bounds.
3. MENTAL WALKTHROUGH — how the human would do it by hand. Each \
   mental step must correspond to exactly ONE tool from the catalog, \
   and each step's inputs must be obtainable from the task text or \
   an earlier step.
4. TASK PROMPT — the request the persona would actually type, in \
   natural language. It must be self-contained and unambiguous enough \
   that the walkthrough can be reconstructed from the prompt alone.

Hard rules:
- expected_tools lists the catalog tool name of each mental step, in \
  order. Use ONLY tool names from the provided catalog.
- The task prompt must NEVER contain tool names or the words "MCP"/"API"; \
  it describes WHAT to achieve, not WHICH function to call.
- All tasks are READ-ONLY — the server exposes no write tools. Never \
  describe creating, modifying, or deleting anything.
- Real public tickers only. Use well-known US equities (AAPL, MSFT, \
  GOOG, AMZN, NVDA, TSLA, META, JPM, JNJ, V, KO, ...). Do NOT invent \
  tickers or refer to delisted symbols.
- Dates must be ISO strings ("YYYY-MM-DD"). Date ranges must cover \
  enough trading days for the request to make sense.
- NEVER invent an options expiration date. Listed expiries are a live, \
  changing set, so a hardcoded one is almost always rejected by the \
  server. Ask for the expiry by DESCRIPTION instead — "the nearest listed \
  expiry", "the furthest-out expiration available", "the first expiry \
  after the next earnings" — so the walkthrough has to look the dates up \
  first and pick from what actually exists.
- Currency amounts use plain numbers (no $ in the task prompt; it can \
  appear in quoted text if you need to).
- Percentages in the prompt are goals ("return"), not computations. \
  Never ask the planner to compute them — that goes into the report \
  step's quoted text instead, written as inline math.
- The executor can only PASS values between steps verbatim — it \
  cannot compute, aggregate, reformat or combine data. Never ask for \
  derived content (counts written into a report, summaries of fetched \
  data, "a file listing the results"). Report content must be either \
  literal text given in the task or the verbatim content of exactly \
  one tool result.
- Keep result sizes bounded: ask for "up to N" items with N <= 10.
- The outcome must be verifiable afterwards by re-querying yfinance \
  (counts, titles, prices, sectors, dates — not vague impressions).
- Do not duplicate any of the EXISTING TASKS provided.

Naturalness (make it read like a real person, not a generated spec):
- Write the TASK PROMPT in the assigned persona's voice and phrasing \
  style. Vary sentence length and register from task to task; never \
  settle into one template.
- A real user says WHAT they want and maybe why — not an enumerated \
  procedure. Never number the steps in the prompt, and never hint at \
  how many calls or which tools it takes. The MENTAL WALKTHROUGH \
  carries that structure, not the prompt.

Output a SINGLE JSON object, no prose:
{
  "category": "<archetype name>",
  "thinking": ["<step 1 of the human walkthrough>", "..."],
  "task_prompt": "<the natural-language request>",
  "expected_tools": ["<tool>", "..."]
}
"""


class TaskGenError(RuntimeError):
    pass


@dataclass
class TaskSpec:
    category: str
    difficulty: str
    mode: str  # always "readonly" for finance
    thinking: list[str]
    task_prompt: str
    expected_tools: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "difficulty": self.difficulty,
            "mode": self.mode,
            "thinking": self.thinking,
            "task_prompt": self.task_prompt,
            "expected_tools": self.expected_tools,
            "warnings": self.warnings,
        }


class TaskGenerator:
    def __init__(self, llm: LLMClient, condensed_catalog: list[dict[str, Any]]):
        self.llm = llm
        self.catalog = condensed_catalog
        self._by_name = {c["tool"]: c for c in condensed_catalog}
        # token-cheap view for the generation prompt (no parameter schemas)
        self._gen_catalog = [
            {"tool": c["tool"], "summary": c["description"]}
            for c in condensed_catalog
        ]

    # ------------------------------------------------------------------ public

    def generate(
        self,
        archetype: dict[str, Any],
        *,
        difficulty: str = "medium",
        undercovered: list[str] | None = None,
        avoid: list[str] | None = None,
        attempts: int = 2,
    ) -> TaskSpec:
        """Generate one validated TaskSpec; retries once on a rejected spec."""
        mode = "readonly"
        persona_pool = random.sample(PERSONAS, k=min(3, len(PERSONAS)))
        style = random.choice(PHRASING_STYLES)
        feedback = ""
        last: TaskGenError | None = None
        for _ in range(attempts):
            user = self._user_prompt(
                archetype,
                mode,
                difficulty,
                undercovered or [],
                avoid or [],
                feedback,
                persona_pool,
                style,
            )
            reply = self.llm.complete(
                system=_GENERATOR_SYSTEM,
                messages=[Message("user", user)],
                temperature=GENERATION_TEMPERATURE,
                # The prompt demands a single JSON object. Without this the
                # hot generation temperature makes reasoning models return an
                # empty content block often enough to abort a whole run.
                json_mode=True,
            )
            try:
                obj = _extract_json(reply)
                return self._validate(obj, archetype, mode, difficulty, avoid or [])
            except (ValueError, TaskGenError) as e:
                last = e if isinstance(e, TaskGenError) else TaskGenError(str(e))
                feedback = (
                    f"\n\nYour previous attempt was REJECTED for this "
                    f"reason:\n{last}\nGenerate a corrected task."
                )
        assert last is not None
        raise last

    # ---------------------------------------------------------------- internal

    def _user_prompt(
        self,
        archetype: dict[str, Any],
        mode: str,
        difficulty: str,
        undercovered: list[str],
        avoid: list[str],
        feedback: str,
        persona_pool: list[str],
        style: str,
    ) -> str:
        lo, hi = DIFFICULTY_STEPS[difficulty]
        parts = [
            "TOOL CATALOG (authoritative):",
            json.dumps(self._gen_catalog, separators=(",", ":")),
            "",
            f"SCENARIO ARCHETYPE: {archetype['name']} — {archetype['intent']}",
            f"MODE: {mode}",
            f"DIFFICULTY: {difficulty} — the walkthrough should need {lo}-{hi} tool calls.",
            f"PERSONA — adopt whichever best fits this archetype: {persona_pool}",
            f"PHRASING STYLE — shape the prompt's register and length like this: {style}",
        ]
        if undercovered:
            parts.append(
                "UNDER-COVERED TOOLS (prefer exercising 1-2 of these when it fits "
                f"the archetype naturally; never force an unnatural fit): {undercovered[:15]}"
            )
        if avoid:
            parts.append("EXISTING TASKS (do not duplicate):")
            parts.append(json.dumps(avoid[-20:], ensure_ascii=False))
        parts.append("\nGenerate ONE task now.")
        return "\n".join(parts) + feedback

    def _validate(
        self,
        obj: Any,
        archetype: dict[str, Any],
        mode: str,
        difficulty: str,
        avoid: list[str],
    ) -> TaskSpec:
        if not isinstance(obj, dict):
            raise TaskGenError(f"generator output is not an object: {obj!r}")
        prompt = str(obj.get("task_prompt") or "").strip()
        tools = obj.get("expected_tools") or []
        if len(prompt) < 40:
            raise TaskGenError("task_prompt is missing or too short")
        if not isinstance(tools, list) or not tools:
            raise TaskGenError("expected_tools is missing or empty")
        unknown = [t for t in tools if t not in self._by_name]
        if unknown:
            raise TaskGenError(f"expected_tools not in catalog: {unknown}")
        # Cross-check the prompt text for forbidden phrases.
        low = prompt.lower()
        forbidden = ("mcp", "tool_call", "api call", "endpoint", "via the ")
        for needle in forbidden:
            if needle in low:
                raise TaskGenError(
                    f"task_prompt contains a forbidden phrase {needle!r}; "
                    "describe WHAT to achieve, not HOW"
                )
        # Difficulty step count check.
        lo, hi = DIFFICULTY_STEPS[difficulty]
        if not (lo <= len(tools) <= hi):
            # Soft warning — not a rejection. Some prompts legitimately
            # need a tighter range.
            warnings = [
                f"expected_tools has {len(tools)} step(s); difficulty {difficulty} "
                f"asks for {lo}-{hi} — adjust the walkthrough if needed."
            ]
        else:
            warnings = []
        # Duplicate avoidance against the existing KB (cheap: exact match).
        norm = prompt.casefold()
        if any(a.casefold() == norm for a in avoid):
            raise TaskGenError("task_prompt duplicates an existing task")
        return TaskSpec(
            category=archetype["name"],
            difficulty=difficulty,
            mode=mode,
            thinking=list(obj.get("thinking") or []),
            task_prompt=prompt,
            expected_tools=list(tools),
            warnings=warnings,
        )


__all__ = [
    "ARCHETYPES",
    "DIFFICULTY_STEPS",
    "PERSONAS",
    "PHRASING_STYLES",
    "GENERATION_TEMPERATURE",
    "TaskSpec",
    "TaskGenerator",
    "TaskGenError",
]