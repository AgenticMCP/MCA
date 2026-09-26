"""CLI for the rectify agent.

    python -m pkg.taskgen_rectify list                    # hard rejected tasks: counts
    python -m pkg.taskgen_rectify list --verbose          # ... one line per task
    python -m pkg.taskgen_rectify run --run-id bench-20260725-153221
    python -m pkg.taskgen_rectify run --mode readonly --bucket exec --limit 5 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from pkg.agenticmcpe.config import DEFAULT_RUNS_DIR, ConfigError
from .kb import SharedKnowledgeBase

from .pipeline import (DEFAULT_KB_PATH, DEFAULT_REJECTS_PATH, RectifyPipeline,
                       find_promotable)
from .sources import (DEFAULT_REJECTED_RUNS, HARD_MIN_STEPS, RejectedTask,
                      load_rejected_tasks)

_BUCKET = {"exec": "execution failure", "verify": "verification failed"}


def _select(args: argparse.Namespace) -> list[RejectedTask]:
    run_ids = set(args.run_id) if args.run_id else None
    tasks = load_rejected_tasks(Path(args.runs_root), run_ids=run_ids)
    out: list[RejectedTask] = []
    for t in tasks:
        # An explicitly named run is wanted whatever its length.
        if run_ids is None and t.n_steps < args.min_steps:
            continue
        if args.mode != "all" and t.mode != args.mode:
            continue
        if args.bucket != "all" and t.bucket != _BUCKET[args.bucket]:
            continue
        if args.category and t.category != args.category:
            continue
        if args.exclude_tool and any(x in t.tools for x in args.exclude_tool):
            continue
        out.append(t)
    return out


def cmd_list(args: argparse.Namespace) -> int:
    tasks = _select(args)
    print(f"{len(tasks)} rejected task(s) selected "
          f"(min_steps={args.min_steps}, mode={args.mode}, bucket={args.bucket})")
    for (bucket, mode), n in sorted(Counter((t.bucket, t.mode) for t in tasks).items()):
        print(f"  {n:5d}  {bucket} | {mode}")
    print("\nby failure (top 20):")
    failures = Counter(
        f"{t.failed_tool} [{t.failed_status}]" if t.failed_step
        else "checks: " + ", ".join(t.failed_checks[:2]) for t in tasks)
    for k, n in failures.most_common(20):
        print(f"  {n:5d}  {k}")
    if args.verbose:
        print()
        for t in tasks:
            print(t.brief())
            print(f"    {t.task_prompt[:160]}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    tasks = _select(args)
    kb = SharedKnowledgeBase(args.kb)
    rejects_path = Path(args.rejects) if args.rejects else DEFAULT_REJECTS_PATH
    if args.resume:
        done = {p.casefold() for p in kb.prompts()} | _logged_prompts(rejects_path)
        tasks = [t for t in tasks if t.task_prompt.casefold() not in done]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        print("[rectify] nothing to run")
        return 1
    pipeline = RectifyPipeline(
        kb, provider=args.provider, max_replans=args.max_replans,
        stall_limit=args.stall_limit, min_replans=args.min_replans, cleanup=args.cleanup,
        log_to_console=not args.quiet, rejects_path=rejects_path)

    accepted = rejected = 0
    for i, t in enumerate(tasks, 1):
        if i > 1 and args.task_interval > 0:
            time.sleep(args.task_interval)  # pace repo creation under GitHub's secondary limit
        print(f"\n=== task {i}/{len(tasks)} [{t.brief()}] ===")
        print(f"[rectify] prompt: {t.task_prompt}")
        print(f"[rectify] rejected plan: {' -> '.join(t.tools)}")
        result = pipeline.run_task(t)
        if result.accepted:
            accepted += 1
            e = result.entry
            print(f"[rectify] ACCEPTED as {e['id']} (replans={e['replans']}, "
                  f"match={e['expected_match']}, run={result.run_id})")
        else:
            rejected += 1
            print(f"[rectify] run={result.run_id}")
    print(f"\n[rectify] done: {accepted} accepted, {rejected} rejected; "
          f"KB now has {len(kb.entries)} entries at {kb.path}")
    return 0 if accepted > 0 else 1


def cmd_promote(args: argparse.Namespace) -> int:
    """Finish gate-rejected runs with >= --min-replans replans from their
    archived attempts (consolidate, re-execute, verify, store)."""
    runs_dir = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_DIR
    cands = find_promotable(runs_dir, args.min_replans)
    if args.limit:
        cands = cands[:args.limit]
    if not cands:
        print("[rectify] nothing to promote")
        return 1
    tasks = {t.run_id: t for t in load_rejected_tasks(
        Path(args.runs_root), run_ids={c.source_run_id for c in cands})}
    kb = SharedKnowledgeBase(args.kb)
    pipeline = RectifyPipeline(
        kb, provider=args.provider, min_replans=args.min_replans, cleanup=args.cleanup,
        log_to_console=not args.quiet,
        rejects_path=Path(args.rejects) if args.rejects else DEFAULT_REJECTS_PATH)
    accepted = rejected = 0
    for i, c in enumerate(cands, 1):
        task = tasks.get(c.source_run_id)
        if task is None:
            print(f"[rectify] skip {c.run_dir.name}: source {c.source_run_id} not found")
            rejected += 1
            continue
        if i > 1 and args.task_interval > 0:
            time.sleep(args.task_interval)
        print(f"\n=== promote {i}/{len(cands)} {c.run_dir.name} ({c.replans} replans) "
              f"[{task.brief()}] ===")
        result = pipeline.promote_run(c.run_dir, task, args.min_replans)
        if result.accepted:
            accepted += 1
            e = result.entry
            print(f"[rectify] ACCEPTED as {e['id']} (replans={e['replans']}, "
                  f"match={e['expected_match']}, run={result.run_id})")
        else:
            rejected += 1
            print(f"[rectify] not promoted: {result.reason} (run={result.run_id})")
    print(f"\n[rectify] promote done: {accepted} accepted, {rejected} rejected; "
          f"KB now has {len(kb.entries)} entries at {kb.path}")
    return 0 if accepted > 0 else 1


def _logged_prompts(path: Path) -> set[str]:
    out: set[str] = set()
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    out.add(str(json.loads(line).get("task_prompt", "")).casefold())
                except ValueError:
                    continue
    return out


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--runs-root", default=str(DEFAULT_REJECTED_RUNS),
                   help="rejected_runs/ tree (default: pkg/agenticmcpe/runs/rejected_runs).")
    p.add_argument("--min-steps", type=int, default=HARD_MIN_STEPS,
                   help=f"Minimum steps of the rejected plan (default {HARD_MIN_STEPS} "
                        "= taskgen's 'hard').")
    p.add_argument("--mode", choices=["readonly", "write", "all"], default="all")
    p.add_argument("--bucket", choices=["exec", "verify", "all"], default="all",
                   help="exec = 'execution failure', verify = 'verification failed'.")
    p.add_argument("--category", help="Only this scenario archetype.")
    p.add_argument("--run-id", action="append",
                   help="Only this rejected run id (repeatable; ignores --min-steps).")
    p.add_argument("--exclude-tool", action="append", metavar="TOOL",
                   help="Skip tasks whose rejected plan used this tool (repeatable) — "
                        "e.g. classes known to be impossible right now.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pkg.taskgen_rectify",
        description="Re-run rejected taskgen tasks with a wide replan budget, "
                    "consolidate the attempts into one from-scratch sequence, "
                    "re-execute and verify it, and persist the proven ones.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="Show the selected rejected tasks.")
    _add_filters(pl)
    pl.add_argument("--verbose", action="store_true", help="One line per task.")
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="Rectify the selected rejected tasks.")
    _add_filters(pr)
    pr.add_argument("--limit", type=int, default=0, help="Stop after N tasks.")
    pr.add_argument("--max-replans", type=int, default=10)
    pr.add_argument("--stall-limit", type=int, default=3,
                    help="Give up after N consecutive identical failures (default 3).")
    pr.add_argument("--min-replans", type=int, default=3,
                    help="Store a run only if it needed at least N replans (default 3): "
                         "the KB is for rectified tasks, not first-time passes.")
    pr.add_argument("--task-interval", type=float, default=0.0, metavar="SECONDS",
                    help="Pause between tasks. Write batches creating a repo per task "
                         "trip GitHub's secondary rate limit at ~3 tasks/min; 120 keeps "
                         "clear of it.")
    pr.add_argument("--cleanup", action="store_true",
                    help="Delete agenticmcpe-bench-* repos created by write tasks — "
                         "also before the consolidated re-execution, so it starts "
                         "from scratch (token needs delete_repo scope).")
    pr.add_argument("--kb", default=str(DEFAULT_KB_PATH),
                    help="Output KB (default: pkg/taskgen_rectify/knowledge_base_rectified.json).")
    pr.add_argument("--rejects", help="Rejection log (default: rejects_rectify.jsonl).")
    pr.add_argument("--provider", help="LLM provider override (see agenticmcpe).")
    pr.add_argument("--resume", action="store_true",
                    help="Skip prompts already in the KB or the rejection log.")
    pr.add_argument("--quiet", action="store_true")
    pr.set_defaults(func=cmd_run)
    pp = sub.add_parser("promote",
                        help="Finish runs the min-replans gate rejected, from their "
                             "archived attempts, at a lower threshold.")
    pp.add_argument("--min-replans", type=int, default=2,
                    help="Promote runs that needed at least N replans (default 2).")
    pp.add_argument("--runs-dir", help="agenticmcpe runs dir holding rectify-* runs "
                                       "(default: pkg/agenticmcpe/runs).")
    pp.add_argument("--runs-root", default=str(DEFAULT_REJECTED_RUNS),
                    help="rejected_runs/ tree the sources came from.")
    pp.add_argument("--limit", type=int, default=0)
    pp.add_argument("--task-interval", type=float, default=0.0, metavar="SECONDS")
    pp.add_argument("--cleanup", action="store_true",
                    help="Delete the bench repos the re-execution creates.")
    pp.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    pp.add_argument("--rejects")
    pp.add_argument("--provider")
    pp.add_argument("--quiet", action="store_true")
    pp.set_defaults(func=cmd_promote)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
