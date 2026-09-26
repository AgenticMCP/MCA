"""Task-coverage gate: did the run actually DO the task, or just pass checks?

A long replan budget puts the planner under search pressure to satisfy the
verifier, and the verifier's dynamic checks are generated FROM the executed
trace. Those two facts combine badly: a plan that does less produces weaker
checks, so "do almost nothing" is a local optimum that passes. Observed live at
budget 15 (never at the base pipeline's budget 2) — e.g. a Booking.com flight
search accepted 22/22 for navigating to the empty search form and closing,
including a check the verifier named ``search_was_NOT_executed_to_results``.

This gate runs AFTER verification passes and BEFORE the KB write. It keys on
pathologies in the VERIFIER'S OUTPUT, not on the shape of the plan: counting
interaction steps was tried first and over-fires badly, because reaching a
GitHub Issues tab by its canonical URL instead of clicking it is a legitimate
route that still captures the data.

Four pathologies, each seen in a real accepted entry:

  inverted      a check that PASSED while its own detail reports the evidence
                was absent ("LHR not found in recorded snapshots")
  escape_hatch  an unfalsifiable check whose OR accepts the empty case
                ("results snapshot has a price OR the no-results message")
  work_skipped  a check asserting the task's own action did not happen
                ("search_was_NOT_executed_to_results")
  uncovered     the deliverable the task asks for (a price, a duration, a star
                count) is named by no check at all — the run verified the
                search FORM instead of the result
"""

from __future__ import annotations

import re
from typing import Any

from pkg.agenticmcpe.verifier import VerificationReport

# A passed check whose detail admits the evidence was not there.
_INVERTED = re.compile(
    r"\b(not found|not present|no .{0,24} found|not located|absent from|"
    r"missing from|could not find|not in the (recorded )?snapshots?)\b", re.I)

# An OR that makes the check true either way. The "no results"/"empty" branch is
# the tell: a genuine check does not offer itself an exit.
_ESCAPE_HATCH = re.compile(
    r"(\bor\b|/)[^.]{0,40}\b(no[-_ ]?results?|no[-_ ]?result state|empty|"
    r"nothing found|not shown|no[-_ ]?prices?|absent)\b", re.I)

# A check asserting the task's own work was never performed.
_WORK_SKIPPED = re.compile(
    r"(was_not_|_not_executed|_not_performed|_not_submitted|never_|did_not_|"
    r"_not_run\b|not_reached)", re.I)

# Deliverables a browser task asks the agent to come back WITH. If the prompt
# names one and no check mentions it, the run verified something else.
_DELIVERABLES: dict[str, tuple[str, ...]] = {
    "price":        ("price", "fare", "cost", "cheapest", "$", "usd", "eur", "gbp"),
    "duration":     ("duration", "how long", "travel time", "minutes", "hours"),
    "distance":     ("distance", "miles", "km", "kilometre", "kilometer"),
    "star count":   ("star count", "stars", "stargazers"),
    "release":      ("release", "latest version", "tag name"),
    "issue count":  ("open issues", "issue count", "issues count"),
    # Both sides of the match use this family, so a prompt saying "date of
    # birth" and a check saying "birth date" must share a member.
    "date":         ("date of birth", "birth date", "birthdate", "born",
                     "first appeared", "release date", "last updated",
                     "kick-off", "kickoff", "date"),
    "rating":       ("star rating", "rating", "reviews"),
    "parameters":   ("parameter count", "parameters", "model size"),
    "license":      ("license",),
    "score":        ("score", "standings", "record", "wins", "losses"),
}


def _text(check: dict[str, Any]) -> str:
    return f"{check.get('name', '')} {check.get('detail', '')}"


def required_deliverables(task_prompt: str) -> list[str]:
    """Deliverables the task prompt asks the agent to report back."""
    low = task_prompt.casefold()
    return [name for name, words in _DELIVERABLES.items()
            if any(w in low for w in words)]


def degeneracy_reasons(task_prompt: str, report: VerificationReport) -> list[str]:
    """Why this passing report does not prove the task was done. Empty = fine."""
    passed = [c for c in report.results if c.get("passed")]
    dynamic = [c for c in passed if c.get("category") == "dynamic"]
    reasons: list[str] = []

    for c in dynamic:
        detail = str(c.get("detail", ""))
        name = str(c.get("name", ""))
        if _INVERTED.search(detail):
            reasons.append(
                f"inverted check {name!r} PASSED while reporting the evidence "
                f"was absent: {detail[:120]!r}")
        if _ESCAPE_HATCH.search(_text(c)):
            reasons.append(
                f"unfalsifiable check {name!r} accepts the empty case: "
                f"{detail[:120]!r}")
        if _WORK_SKIPPED.search(name):
            reasons.append(
                f"check {name!r} asserts the task's own action did not happen")

    # Deliverable coverage: only meaningful when the dynamic layer produced
    # checks at all (the format/static layers never mention task content).
    if dynamic:
        blob = " ".join(_text(c) for c in dynamic).casefold()
        for want in required_deliverables(task_prompt):
            if not any(w in blob for w in _DELIVERABLES[want]):
                reasons.append(
                    f"the task asks for a {want}, but no check mentions one — "
                    f"the run verified something other than the deliverable")

    # De-duplicate while keeping order; one pathology often trips several checks.
    seen: set[str] = set()
    out: list[str] = []
    for r in reasons:
        key = r.split("'")[0] + r[-40:]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def coverage_feedback(task_prompt: str, reasons: list[str]) -> str:
    """Replan feedback for a run that passed verification without doing the task."""
    return (
        "Your plan EXECUTED and PASSED verification, but it did NOT ACTUALLY DO "
        "THE TASK — it satisfied the checks without producing what was asked "
        "for. This is rejected:\n"
        + "\n".join(f"- {r}" for r in reasons)
        + "\n\nThe checks are generated from YOUR trace, so a plan that does "
          "less is graded more leniently. That is not a way to pass.\n"
          "- Carry the task through to the RESULT: if it says search, the plan "
          "must reach the results page; if it says apply a filter, the filtered "
          "results must be on screen and snapshotted.\n"
          "- Verifying the URL you just typed proves nothing about the page. "
          "Capture the PAGE CONTENT that contains the answer.\n"
          "- Every value the task asks you to report back must be visible in "
          "some recorded snapshot. If the site blocks you from reaching it, the "
          "correct outcome is a failed run, not a plan that avoids looking.\n"
          "- Do not verify that the search FORM exists. The form is not the "
          "answer."
    )


__all__ = ["coverage_feedback", "degeneracy_reasons", "required_deliverables"]
