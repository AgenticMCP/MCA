"""Rejected taskgen runs, loaded as re-runnable tasks.

The task set is the curated ``pkg/agenticmcpe/runs/rejected_runs/`` tree —
``execution failure/`` and ``verification failed/``, one agenticmcpe run
directory each, exactly as taskgen left them. ``pkg/taskgen/rejects.jsonl``
supplies what a run directory lacks (scenario category, mode, rejection
reason); a directory with no matching record is still loaded, with its mode
inferred from the plan's tools and its category recorded as ``unknown``.

"Hard" follows taskgen's own difficulty buckets (``DIFFICULTY_STEPS``): the
rejected plan had at least 7 steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pkg.taskgen.generator import DIFFICULTY_STEPS, TaskSpec
from pkg.taskgen.pipeline import DEFAULT_REJECTS_PATH as TASKGEN_REJECTS_PATH

DEFAULT_REJECTED_RUNS = (Path(__file__).resolve().parents[1]
                         / "agenticmcpe" / "runs" / "rejected_runs")
BUCKETS = ("execution failure", "verification failed")
HARD_MIN_STEPS = DIFFICULTY_STEPS["hard"][0]


@dataclass
class RejectedTask:
    run_id: str
    bucket: str            # which rejected_runs/ subfolder it came from
    run_dir: Path
    task_prompt: str
    tools: list[str]       # the rejected plan's tool sequence
    category: str = "unknown"
    mode: str = "readonly"
    reason: str = ""
    failed_step: str | None = None
    failed_tool: str | None = None
    failed_status: str | None = None
    failed_error: str = ""
    failed_checks: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.tools)

    @property
    def difficulty(self) -> str:
        for name, (_lo, hi) in DIFFICULTY_STEPS.items():
            if self.n_steps <= hi:
                return name
        return "hard"

    def to_spec(self) -> TaskSpec:
        # The rejected plan's tools are the hypothesis the rectified sequence
        # is compared against (expected_match) — the generator's guess is gone.
        return TaskSpec(category=self.category, difficulty=self.difficulty,
                        mode=self.mode, thinking=[], task_prompt=self.task_prompt,
                        expected_tools=list(self.tools))

    def source(self) -> dict[str, Any]:
        """Provenance block for the KB entry / run summary."""
        return {
            "run_id": self.run_id, "bucket": self.bucket, "reason": self.reason,
            "tools": list(self.tools),
            "failed_step": self.failed_step, "failed_tool": self.failed_tool,
            "failed_status": self.failed_status, "failed_error": self.failed_error,
            "failed_checks": list(self.failed_checks),
        }

    def brief(self) -> str:
        if self.failed_step:
            fail = f"{self.failed_step}:{self.failed_tool} [{self.failed_status}]"
        elif self.failed_checks:
            fail = "checks failed: " + ", ".join(self.failed_checks[:4])
        else:
            fail = "-"
        return (f"{self.run_id} | {self.bucket} | {self.mode} | {self.category} | "
                f"{self.n_steps} steps | {fail}")


def load_rejected_tasks(runs_root: Path = DEFAULT_REJECTED_RUNS,
                        rejects_path: Path = TASKGEN_REJECTS_PATH,
                        *, run_ids: set[str] | None = None) -> list[RejectedTask]:
    """Every rejected run that has a plan, in run-id order (execution
    failures first), one per distinct prompt."""
    records = _rejects_by_prompt(rejects_path)
    read_only: dict[str, bool] = {}
    tasks: list[RejectedTask] = []
    for bucket in BUCKETS:
        bdir = runs_root / bucket
        if not bdir.is_dir():
            continue
        for d in sorted(p for p in bdir.iterdir() if p.is_dir()):
            if run_ids is not None and d.name not in run_ids:
                continue
            plan = _read_json(d / "plan.json")
            steps = (plan or {}).get("steps") or []
            prompt = str((plan or {}).get("task") or "").strip()
            if not steps or not prompt:
                continue
            if not read_only:
                read_only = _read_only_map(d / "catalog.json")
            task = RejectedTask(run_id=d.name, bucket=bucket, run_dir=d,
                                task_prompt=prompt,
                                tools=[str(s.get("tool", "")) for s in steps])
            rec = records.get(prompt.casefold())
            if rec:
                task.category = rec.get("category") or task.category
                task.mode = rec.get("mode") or task.mode
                task.reason = rec.get("reason") or bucket
            else:
                task.mode = _infer_mode(task.tools, read_only)
                task.reason = bucket
            _attach_failure(task)
            tasks.append(task)
    seen: set[str] = set()
    unique: list[RejectedTask] = []
    for t in tasks:
        key = t.task_prompt.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def _rejects_by_prompt(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out.setdefault(str(rec.get("task_prompt", "")).strip().casefold(), rec)
    return out


def _read_only_map(catalog_path: Path) -> dict[str, bool]:
    raw = _read_json(catalog_path)
    if not isinstance(raw, list):
        return {}
    return {t.get("name", ""): bool((t.get("annotations") or {}).get("readOnlyHint"))
            for t in raw if isinstance(t, dict)}


def _infer_mode(tools: list[str], read_only: dict[str, bool]) -> str:
    if read_only and any(not read_only.get(t, False) for t in tools):
        return "write"
    return "readonly"


def _attach_failure(task: RejectedTask) -> None:
    trace = _read_json(task.run_dir / "trace.json")
    if isinstance(trace, dict) and trace.get("failed_step"):
        task.failed_step = trace["failed_step"]
        step = next((s for s in trace.get("steps", [])
                     if s.get("id") == task.failed_step), None)
        if step:
            task.failed_tool = step.get("tool")
            task.failed_status = step.get("status")
            task.failed_error = str(step.get("error") or "")[:300]
    verification = _read_json(task.run_dir / "verification.json")
    if isinstance(verification, dict):
        task.failed_checks = [str(r.get("name")) for r in verification.get("results", [])
                              if not r.get("passed")]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__ = ["BUCKETS", "DEFAULT_REJECTED_RUNS", "HARD_MIN_STEPS", "RejectedTask",
           "load_rejected_tasks"]
