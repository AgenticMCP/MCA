"""Select rejected tasks worth rectifying.

``rejects.jsonl`` records why a task was dropped but not the plan it produced;
the plan lives in the run directory the reject points at
(``runs_taskgen/<run_id>/plan.json``). Joining the two gives the one thing this
package selects on: how LONG the attempted sequence was. Long tasks are the
target because a 3-step lookup that failed is usually a bad task, while a
12-step sequence that failed is usually a good task with a fixable plan.

Selection is by task PROMPT, not by reject record: the same prompt can appear
in several rejects (different rounds). The longest plan any of its attempts
produced is what the prompt is ranked by.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pkg.agenticmcpe.config import DEFAULT_RUNS_DIR

from pkg.taskgen.generator import TaskSpec
from pkg.taskgen.kb import KnowledgeBase
from pkg.taskgen.pipeline import DEFAULT_REJECTS_PATH

DEFAULT_TASKGEN_RUNS_DIR = DEFAULT_RUNS_DIR.parent / "runs_taskgen"

# Reject reasons this package can actually do something about. A tool-signature
# rejection is not a failure (the run passed; the KB just had enough clones),
# and a duplicate prompt is already represented — re-running either is waste.
RECTIFIABLE_PREFIXES = (
    "execution failed",
    "verification failed",
    "plan gate",
    "replan gate",
    "planner error",
    "replanner error",
    "pipeline error",
)


def reason_bucket(reason: str) -> str:
    for p in (*RECTIFIABLE_PREFIXES, "tool signature", "duplicate"):
        if reason.startswith(p):
            return p
    return reason[:40]


@dataclass
class RectifySpec(TaskSpec):
    """A TaskSpec carrying where the task was rejected from.

    A rejected task has no generator hypothesis (no ``thinking``, no
    ``expected_tools``) — it has a history instead, and that history is what
    the accepted entry records as provenance.
    """
    source_reject_reason: str = ""
    source_run_id: str = ""
    original_tool_sequence: list[str] = field(default_factory=list)


@dataclass
class RejectRecord:
    task_prompt: str
    category: str
    mode: str
    steps: int                      # longest plan across this prompt's attempts
    reason: str                     # most recent rejection reason
    run_id: str                     # run dir of the attempt `steps` came from
    tool_sequence: list[str] = field(default_factory=list)
    attempts: int = 1               # how many times this prompt was rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_prompt": self.task_prompt,
            "category": self.category,
            "mode": self.mode,
            "steps": self.steps,
            "reason": self.reason,
            "reason_bucket": reason_bucket(self.reason),
            "run_id": self.run_id,
            "tool_sequence": self.tool_sequence,
            "attempts": self.attempts,
        }

    def to_spec(self) -> RectifySpec:
        return RectifySpec(
            category=self.category,
            # rejects.jsonl never recorded the generator's difficulty label;
            # inventing one would be a fabricated field in the KB.
            difficulty="unknown",
            mode=self.mode,
            thinking=[],
            task_prompt=self.task_prompt,
            expected_tools=[],
            source_reject_reason=self.reason,
            source_run_id=self.run_id,
            original_tool_sequence=list(self.tool_sequence),
        )


def load_pool(
    *,
    rejects_path: Path | None = None,
    runs_dir: Path | None = None,
    kb_paths: Iterable[Path] = (),
    min_steps: int = 7,
    categories: Iterable[str] | None = None,
    exclude_categories: Iterable[str] | None = None,
    reasons: Iterable[str] | None = None,
) -> list[RejectRecord]:
    """Rejected prompts eligible for rectification, longest plan first.

    ``kb_paths`` are knowledge bases whose prompts are already solved; any
    prompt present in one of them is dropped.
    """
    rejects_path = rejects_path or DEFAULT_REJECTS_PATH
    runs_dir = runs_dir or DEFAULT_TASKGEN_RUNS_DIR

    solved: set[str] = set()
    for p in kb_paths:
        if Path(p).is_file():
            for prompt in KnowledgeBase(p).prompts():
                solved.add(prompt.strip().casefold())

    by_prompt: dict[str, RejectRecord] = {}
    for line in rejects_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = str(rec.get("task_prompt", "")).strip()
        reason = str(rec.get("reason", ""))
        if not prompt or not reason.startswith(RECTIFIABLE_PREFIXES):
            continue
        if prompt.casefold() in solved:
            continue

        tools = _plan_tools(runs_dir, str(rec.get("run_id", "")))
        prior = by_prompt.get(prompt)
        attempts = (prior.attempts + 1) if prior else 1
        # Keep the LONGEST attempt's plan, but always the LATEST reason.
        if prior is None or len(tools) > prior.steps:
            by_prompt[prompt] = RejectRecord(
                task_prompt=prompt,
                category=str(rec.get("category", "?")),
                mode=str(rec.get("mode", "readonly")),
                steps=len(tools),
                reason=reason,
                run_id=str(rec.get("run_id", "")),
                tool_sequence=tools,
                attempts=attempts,
            )
        else:
            prior.attempts = attempts
            prior.reason = reason

    pool = [r for r in by_prompt.values() if r.steps >= min_steps]
    if categories:
        wanted = set(categories)
        pool = [r for r in pool if r.category in wanted]
    if exclude_categories:
        unwanted = set(exclude_categories)
        pool = [r for r in pool if r.category not in unwanted]
    if reasons:
        wanted = set(reasons)
        pool = [r for r in pool if reason_bucket(r.reason) in wanted]
    pool.sort(key=lambda r: (-r.steps, r.task_prompt))
    return pool


def _plan_tools(runs_dir: Path, run_id: str) -> list[str]:
    if not run_id:
        return []
    path = runs_dir / run_id / "plan.json"
    if not path.is_file():
        return []
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [str(s.get("tool", "?")) for s in plan.get("steps", [])]


__all__ = ["DEFAULT_TASKGEN_RUNS_DIR", "RectifySpec", "RejectRecord", "load_pool",
           "reason_bucket"]
