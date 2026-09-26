#!/usr/bin/env python3
"""Batch driver: run every browser_automation task prompt through the full
plan -> execute -> verify pipeline. This IS the regression harness — there is
no Python unit suite beyond idem_selftest.py; correctness is checked end-to-end
by running real tasks and inspecting verifier output.

    python3 pkg/agenticmcpe/run/run_batch.py                 # all tasks
    python3 pkg/agenticmcpe/run/run_batch.py booking         # every booking task
    python3 pkg/agenticmcpe/run/run_batch.py booking_task_0001 paper_task_0002

Task prompts live in browser_automation/*.txt at the repo root (long,
quote-heavy — read from disk to avoid shell escaping), one file per task:

    Task Number: 1
    Task Name: playwright_booking_task_0001

    PROMPT:
    <the natural-language task>

Each task gets runs/<task-name>/, plus a consolidated runs/batch_summary.json.
RESUMABLE: the summary is merged by task id, so an interrupt keeps prior
results and a re-run only updates the tasks it runs.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pkg.agenticmcpe.config import Settings  # noqa: E402
from pkg.agenticmcpe.orchestrator import Orchestrator  # noqa: E402

TASKS_DIR = REPO / "browser_automation"
SUMMARY = None  # resolved after Settings gives us the runs dir

# LLM-endpoint failures, as opposed to task failures. Once the quota is
# exhausted every remaining task "fails" in seconds and overwrites good records
# with crash records, so the batch stops instead of burning the task list.
_API_DEAD = ("insufficient_balance", "Error code: 402", "用量上限",
             "rate_limit_error", "Error code: 429", "速率限制")


def load_tasks(tasks_dir: Path | None = None) -> list[tuple[str, str]]:
    """[(task_id, prompt)] from <tasks_dir>/playwright_*_task_*.txt."""
    out: list[tuple[str, str]] = []
    for p in sorted((tasks_dir or TASKS_DIR).glob("playwright_*task_*.txt")):
        text = p.read_text(encoding="utf-8")
        _, sep, prompt = text.partition("PROMPT:")
        if not sep:
            print(f"[batch] skipping {p.name}: no PROMPT: section")
            continue
        out.append((p.stem, prompt.strip()))
    return out


def main(argv: list[str]) -> int:
    # Optional --summary=<name> gives this batch its own summary file, so
    # several batches (e.g. one per task category) can run in parallel
    # without racing on one file. Remaining args are task-id substrings.
    summary_name = "batch_summary.json"
    tasks_dir = TASKS_DIR
    selectors: list[str] = []
    for a in argv:
        if a.startswith("--summary="):
            summary_name = a.split("=", 1)[1]
        elif a.startswith("--tasks-dir="):
            # Point the harness at an alternative benchmark (e.g.
            # browser_automation_v2). Defaults to browser_automation, so every
            # existing invocation behaves exactly as before.
            tasks_dir = Path(a.split("=", 1)[1])
            if not tasks_dir.is_absolute():
                tasks_dir = REPO / tasks_dir
        else:
            selectors.append(a)
    tasks = load_tasks(tasks_dir)
    if selectors:
        tasks = [(tid, prompt) for tid, prompt in tasks
                 if any(sel in tid for sel in selectors)]
    if not tasks:
        print(f"[batch] no tasks matched in {TASKS_DIR}")
        return 1

    # Resolve the runs dir once (also validates config early).
    probe = Settings.load(run_id="batch-probe")
    runs_dir = probe.runs_dir
    summary_path = runs_dir / summary_name
    summary: dict = {"tasks": {}}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("tasks", {})

    print(f"[batch] {len(tasks)} task(s); summary -> {summary_path}")
    for i, (tid, prompt) in enumerate(tasks, 1):
        print(f"\n=== [{i}/{len(tasks)}] {tid} ===")
        started = time.time()
        record: dict = {"task_id": tid, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            settings = Settings.load(run_id=tid)
            orch = Orchestrator(settings)
            result = orch.run(prompt, max_replans=3)
            record.update({
                "ok": result.ok,
                "execution_success": result.trace.success,
                "failed_step": result.trace.failed_step,
                "replans": result.replans,
                "steps": len(result.plan.steps),
                "tools": [s.tool for s in result.plan.steps],
                "verification": (
                    {"ok": result.report.ok, "passed": result.report.passed,
                     "total": result.report.total}
                    if result.report else None),
                "tokens": getattr(orch, "last_usage", None),
                "run_dir": str(settings.work_dir),
            })
        except Exception as e:  # keep the batch alive; record why
            record.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]})
            print(f"[batch] {tid} crashed: {e}")
            if any(m in str(e) for m in _API_DEAD):
                print(f"[batch] LLM endpoint is refusing calls — stopping here "
                      f"rather than failing the remaining {len(tasks) - i} "
                      f"task(s) in seconds. {tid} was NOT recorded; restore "
                      f"quota and re-run.")
                break
        record["duration_s"] = round(time.time() - started, 1)
        # Re-read before merging so a concurrent batch's writes are kept
        # (each write is then last-writer-wins over a fresh read).
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary.setdefault("tasks", {})
            except (json.JSONDecodeError, OSError):
                pass
        summary["tasks"][tid] = record
        summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        done = [r for r in summary["tasks"].values()]
        summary["totals"] = {
            "tasks": len(done),
            "ok": sum(1 for r in done if r.get("ok")),
            "failed": sum(1 for r in done if not r.get("ok")),
        }
        runs_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        print(f"[batch] {tid}: {'OK' if record.get('ok') else 'FAIL'} "
              f"({record['duration_s']}s)")

    t = summary["totals"]
    print(f"\n[batch] done: {t['ok']}/{t['tasks']} ok -> {summary_path}")
    return 0 if t["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
