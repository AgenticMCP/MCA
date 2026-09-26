"""Task-generation agent: realistic browser tasks grounded in the live catalog.

The generator does NOT invent tasks at random. It is forced through a
human-thinking scaffold (persona/motive -> concrete goal -> mental walkthrough
-> natural prompt) seeded with a scenario archetype, so the resulting prompt
reads like something a real user would ask, and its tool-call sequence is
recoverable by an independent planner. The generator's `expected_tools` is a
hypothesis used for coverage/diversity metadata only — ground truth is always
established downstream by execution + verification (see pipeline.py).

Browser benchmark note (divergence from the GitHub original, by design): every
archetype here is "browse" work — search, filter, read and report on public
sites. Clicking and typing are fine (they are how a browser is driven), but no
task may cause a DURABLE external effect: no logins, no account creation, no
purchases/checkout, no posting comments or submitting personal data. The
`mode` field is kept as "readonly" for KB-schema compatibility.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from pkg.agenticmcpe.config import LLMClient, extract_json

# Generation wants VARIETY, so it runs the LLM hotter than the planner/verifier
# (which need temperature 0 for determinism). Used by the CLI when it builds
# the generator's LLM client.
GENERATION_TEMPERATURE = 0.85

# Personas and phrasing styles are sampled per task and injected into the prompt
# so generated tasks vary in voice, register and length instead of converging on
# one "assistant task" template. They steer wording only — never the tool choice.
PERSONAS: list[str] = [
    "a traveller comparing options before booking anything",
    "a student gathering sources for a term paper",
    "an ML engineer scouting models and datasets for a project",
    "a sports fan who wants tonight's fixtures without the fluff",
    "a commuter planning tomorrow's route",
    "a developer reading docs before adopting a library",
    "a researcher checking what a well-known lab just published",
    "a journalist fact-checking a claim against a primary source",
    "a bargain hunter who only cares about the bottom-line price",
    "a project manager who just wants a quick, concrete answer",
]

PHRASING_STYLES: list[str] = [
    "terse and imperative — one direct sentence, no pleasantries",
    "conversational and polite — a couple of natural sentences, as if chatting",
    "context-first — a clause of background or motivation, then the concrete ask",
    "detailed and precise — names the exact site, filters and limits up front",
    "slightly informal — natural wording, maybe a contraction or aside, still clear",
]

# Stable, long-lived public sites generated tasks may target. The pipeline's
# plan gate enforces this allowlist on every literal navigation URL, so a
# generated task can never send the executor to an arbitrary host.
ALLOWED_SITES: dict[str, str] = {
    "www.booking.com": "flight/hotel search (search + filter only, never book)",
    "www.google.com": "google maps directions and place lookup (/maps)",
    "huggingface.co": "model/dataset/space search and cards",
    "arxiv.org": "paper search, abstracts, listings",
    "www.espn.com": "scores, schedules, standings",
    "www.premierleague.com": "Premier League fixtures, results, standings",
    "www.fifa.com": "FIFA tournament pages, match results",
    "iclr.cc": "ICLR conference site (accepted papers, schedules)",
    "en.wikipedia.org": "encyclopedia articles",
    "developer.mozilla.org": "web documentation",
    "github.com": "public repository pages (read-only browsing)",
    "example.com": "smoke-test page",
}

# Scenario archetypes: the human intents tasks are sampled from. All are
# browse-mode (`write: False`) — see the module docstring.
ARCHETYPES: list[dict[str, Any]] = [
    {"name": "flight-price-lookup", "write": False,
     "intent": "Find the price of a flight on www.booking.com: set origin, "
               "destination and date, apply filters (direct only, cabin "
               "class, airport preference), and read the cheapest price off "
               "the results. Search and read ONLY — never select a fare or "
               "start a booking."},
    {"name": "maps-route", "write": False,
     "intent": "Plan a route with Google Maps (www.google.com/maps): "
               "directions between two named places, a travel mode, and read "
               "off the duration/distance of the best route, or look up a "
               "place's details (address, hours, rating)."},
    {"name": "model-hub-lookup", "write": False,
     "intent": "Scout huggingface.co: search models or datasets by keyword, "
               "sort or filter (downloads, task, license), open one result's "
               "card and read a concrete fact (parameter count, license, "
               "last-updated date, download count)."},
    {"name": "paper-lookup", "write": False,
     "intent": "Find a paper on arxiv.org: search by title/author/topic, "
               "open the right result, and read concrete facts off the "
               "abstract page (authors, submission date, categories, "
               "abstract statements)."},
    {"name": "sports-schedule", "write": False,
     "intent": "Check www.espn.com for a team's or league's schedule, "
               "scores or standings, and read off concrete entries (who "
               "plays whom, when, current standings positions)."},
    {"name": "docs-lookup", "write": False,
     "intent": "Answer a factual question from en.wikipedia.org or "
               "developer.mozilla.org: search the site, open the right "
               "article, and read the specific fact the task asks for."},
    {"name": "repo-web-lookup", "write": False,
     "intent": "Browse a well-known public repository on github.com (the "
               "website, not an API): open the repo page, navigate to a "
               "file, releases or issues TAB, and read a concrete fact "
               "(star count magnitude, latest release name, a file's "
               "presence)."},
]

# Tools generated tasks must never require: arbitrary JS execution and
# file/drag interactions are unsafe or unverifiable for benchmark browsing.
UNSUPPORTED_TOOLS = {
    "browser_run_code_unsafe",
    "browser_evaluate",
    "browser_file_upload",
    "browser_drag",
    "browser_drop",
    "browser_handle_dialog",  # tasks must not be designed around native dialogs
}

# difficulty -> (min_steps, max_steps) for the mental walkthrough
DIFFICULTY_STEPS: dict[str, tuple[int, int]] = {
    "easy": (2, 4),
    "medium": (5, 8),
    "hard": (9, 14),
}

_GENERATOR_SYSTEM = """\
You are a TASK GENERATION agent. You invent realistic web-browsing tasks that a
human would ask an AI assistant to perform in a browser. Each task is later
given to an INDEPENDENT planner that must rediscover the right tool-call
sequence, execute it in a real headless browser, and verify the outcome; tasks
that survive become benchmark ground truth.

Think like a real person at a browser, in this exact order (the required
human-thinking structure):
1. PERSONA & MOTIVE — who is asking and why (a traveller pricing a flight, a
   student hunting a paper, a fan checking fixtures, ...).
2. CONCRETE GOAL — the specific outcome they want, with concrete names, dates
   and limits (which route, which model, which team, what exactly to read off).
3. MENTAL WALKTHROUGH — how the human would do it by hand in the browser, as
   ordered steps ("open the site, dismiss the cookie banner if it appears,
   type X into the search box, wait for results, read Y"). Each mental step
   must correspond to exactly ONE tool from the catalog, and each step's
   inputs must be obtainable from the task text or an earlier step's page.
4. TASK PROMPT — the request the persona would actually type, in natural
   language. It must be self-contained and unambiguous enough that the
   walkthrough can be reconstructed from the prompt alone.

Hard rules:
- expected_tools lists the catalog tool name of each mental step, in order.
  Use ONLY tool names from the provided catalog.
- The task prompt must NEVER contain tool names or the words "MCP"/"API"; it
  describes WHAT to achieve, not WHICH function to call.
- Tasks target ONLY the allowed sites provided, and the prompt names the site
  (or its well-known name) so the planner knows where to start.
- OBSERVATION ONLY, NO DURABLE EFFECTS: searching, filtering, clicking through
  results and reading pages is fine; the task must NEVER log in, create an
  account, buy/book/checkout, post/submit content, upload files, or enter
  personal data. For travel sites say explicitly that only the price/option is
  to be read, not booked.
- No task may depend on being logged in, on CAPTCHAs being solved, or on
  location permissions.
- Dates must be self-contained: express them relative to today ("5 days from
  now") or as concrete future dates, never as past dates.
- The outcome must be VERIFIABLE from the page afterwards: a concrete fact
  (a price appears, a duration, an author list, a fixture) that a snapshot of
  the final page shows. Never ask for vague impressions.
- The executor can only PASS values between steps verbatim — it cannot
  compute, aggregate or reformat data. Ask to "find and report" facts the
  page itself shows (the cheapest price shown, the top result's name), not
  derived calculations.
- Volatile data (prices, scores, availability) is fine to READ, but the task
  must not assert a specific value in advance — ask for "the price shown",
  not "confirm it costs $123".
- Keep result sizes bounded: ask for "up to N" items with N <= 10.
- End the task by asking to close the browser.
- Do not duplicate any of the EXISTING TASKS provided.

Naturalness (make it read like a real person, not a generated spec):
- Write the TASK PROMPT in the assigned persona's voice and phrasing style. Vary
  sentence length and register from task to task; never settle into one template.
- A real user says WHAT they want and maybe why — not an enumerated procedure.
  Never number the steps in the prompt, and never hint at how many calls or which
  tools it takes. The MENTAL WALKTHROUGH carries that structure, not the prompt.

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
    mode: str  # always "readonly" for the browser benchmark
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
            {"tool": c["tool"], "ro": c["read_only"], "summary": c["summary"]}
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
        style_exemplars: list[str] | None = None,
        attempts: int = 2,
    ) -> TaskSpec:
        """Generate one validated TaskSpec; retries once on a rejected spec.

        ``style_exemplars`` are REFERENCE task prompts (e.g. the benchmark's
        own): the generated task copies their register, shape and level of
        concreteness — but never their entities — so the resulting KB entries
        live near the reference distribution and retrieval similarity against
        it is meaningful."""
        mode = "readonly"
        # Sample the voice once per task (kept stable across the retry so a
        # rejection only fixes content, not flavour); varies across tasks.
        persona_pool = random.sample(PERSONAS, k=min(3, len(PERSONAS)))
        style = random.choice(PHRASING_STYLES)
        feedback = ""
        last: TaskGenError | None = None
        for _ in range(attempts):
            user = self._user_prompt(archetype, mode, difficulty,
                                     undercovered or [], avoid or [], feedback,
                                     persona_pool, style,
                                     style_exemplars or [])
            reply = self.llm.chat(_GENERATOR_SYSTEM, user, json_mode=True)
            try:
                obj = extract_json(reply)
                return self._validate(obj, archetype, mode, difficulty, avoid or [])
            except (ValueError, TaskGenError) as e:
                last = e if isinstance(e, TaskGenError) else TaskGenError(str(e))
                feedback = (f"\n\nYour previous attempt was REJECTED for this "
                            f"reason:\n{last}\nGenerate a corrected task.")
        assert last is not None
        raise last

    # ---------------------------------------------------------------- internal
    def _user_prompt(self, archetype: dict[str, Any], mode: str, difficulty: str,
                     undercovered: list[str], avoid: list[str], feedback: str,
                     persona_pool: list[str], style: str,
                     style_exemplars: list[str]) -> str:
        lo, hi = DIFFICULTY_STEPS[difficulty]
        parts = [
            "TOOL CATALOG (authoritative; 'ro' = read-only):",
            json.dumps(self._gen_catalog, separators=(",", ":")),
            "",
            "ALLOWED SITES (tasks may target ONLY these):",
            json.dumps(ALLOWED_SITES, indent=1),
            "",
            f"SCENARIO ARCHETYPE: {archetype['name']} — {archetype['intent']}",
            f"MODE: {mode} (observation only, no durable effects)",
            f"DIFFICULTY: {difficulty} — the walkthrough should need {lo}-{hi} tool calls.",
            f"PERSONA — adopt whichever best fits this archetype: {persona_pool}",
            f"PHRASING STYLE — shape the prompt's register and length like this: {style}",
        ]
        if style_exemplars:
            parts.append(
                "REFERENCE TASKS — your task must read as if it came from the "
                "SAME batch as these: copy their register, structure, level of "
                "concreteness, the way they name the site, and their closing "
                "instruction. But use ENTIRELY DIFFERENT concrete entities "
                "(other cities/routes/teams/players/models/papers/dates) and "
                "never copy a reference sentence verbatim:\n"
                + json.dumps(style_exemplars, ensure_ascii=False, indent=1))
        if undercovered:
            parts.append(
                "UNDER-COVERED TOOLS — REQUIRED: pick 1-2 of these and design "
                "the task so that completing it STRICTLY REQUIRES them, the "
                "way 'check how the page looks on a phone-sized window' "
                "requires resizing, 'grab me a screenshot of it' requires "
                "screenshotting, 'does the page log any console errors' "
                "requires reading the console, 'use the site's own search "
                "box' requires typing, 'go back to the results and open the "
                "next one' requires the back button, or 'fill in the "
                "advanced-search form' requires form filling. Phrase the "
                "prompt in USER language (never tool names), but make the "
                "requirement unavoidable — a task the planner can solve by "
                f"URL navigation alone does not count: {undercovered[:15]}"
            )
        if avoid:
            parts.append("EXISTING TASKS (do not duplicate):")
            parts.append(json.dumps(avoid[-20:], ensure_ascii=False))
        parts.append("\nGenerate ONE task now.")
        return "\n".join(parts) + feedback

    def _validate(self, obj: Any, archetype: dict[str, Any], mode: str,
                  difficulty: str, avoid: list[str]) -> TaskSpec:
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
        unsupported = [t for t in tools if t in UNSUPPORTED_TOOLS]
        if unsupported:
            raise TaskGenError(
                f"expected_tools use benchmark-unsupported tools: {unsupported}")
        leaked = [t for t in self._by_name if t in prompt and "_" in t]
        if leaked:
            raise TaskGenError(f"task_prompt leaks literal tool names: {leaked}")
        low = prompt.casefold()
        forbidden = ("log in", "login", "sign in", "sign up", "register",
                     "checkout", "buy ", "purchase", "book the", "credit card",
                     "password")
        hits = [w for w in forbidden if w in low]
        if hits:
            raise TaskGenError(
                f"task_prompt asks for a durable/authenticated action: {hits}")
        norm = prompt.casefold()
        if any(norm == a.casefold() for a in avoid):
            raise TaskGenError("task_prompt duplicates an existing task")

        warnings: list[str] = []
        lo, hi = DIFFICULTY_STEPS[difficulty]
        if not lo <= len(tools) <= hi:
            warnings.append(
                f"expected {lo}-{hi} steps for {difficulty}, got {len(tools)}")
        return TaskSpec(
            category=str(obj.get("category") or archetype["name"]),
            difficulty=difficulty,
            mode=mode,
            thinking=[str(t) for t in (obj.get("thinking") or [])],
            task_prompt=prompt,
            expected_tools=[str(t) for t in tools],
            warnings=warnings,
        )


__all__ = ["ALLOWED_SITES", "ARCHETYPES", "DIFFICULTY_STEPS", "TaskGenError",
           "TaskGenerator", "TaskSpec", "UNSUPPORTED_TOOLS",
           "GENERATION_TEMPERATURE", "PERSONAS", "PHRASING_STYLES"]
