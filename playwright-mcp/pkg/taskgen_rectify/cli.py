"""CLI for the rectification taskgen.

    python -m pkg.taskgen_rectify pool --min-steps 7            # what's eligible
    python -m pkg.taskgen_rectify rectify --limit 2             # small test run
    python -m pkg.taskgen_rectify rectify --limit 50 --resume   # scale run
    python -m pkg.taskgen_rectify stats                         # rectified KB

``rectify`` is resumable: every finished task appends to a progress JSONL, and
``--resume`` skips prompts already in it. Kill a scale run and restart it with
the same ``--progress`` file to carry on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from pkg.agenticmcpe.config import ConfigError

from pkg.taskgen.kb import DEFAULT_KB_PATH as TASKGEN_KB_PATH
from pkg.taskgen.kb import KnowledgeBase

from .pipeline import DEFAULT_KB_PATH, MIN_REPLANS, RectifyPipeline
from .selector import DEFAULT_TASKGEN_RUNS_DIR, load_pool, reason_bucket

DEFAULT_PROGRESS_PATH = Path(__file__).resolve().parent / "progress.jsonl"

# Failures of the LLM endpoint, not of the task. A run that hits one has not
# been attempted: it must not be recorded in the progress log (which would make
# --resume skip it forever), and continuing is pointless — an exhausted quota
# turns the rest of the pool into instant rejects in seconds. Observed
# live: a quota cutoff burned 141 of 189 tasks into the progress files in under
# 20 minutes before this guard existed.
_FATAL_API = ("insufficient_balance", "402", "用量上限")
_THROTTLE_API = ("rate_limit", "429", "速率限制", "APIConnectionError",
                 "APITimeoutError")
_MAX_CONSECUTIVE_THROTTLE = 3


def _api_failure(reason: str) -> str | None:
    """'fatal' (stop now), 'throttle' (stop after a few), or None."""
    if not reason.startswith("pipeline error"):
        return None
    if any(m in reason for m in _FATAL_API):
        return "fatal"
    if any(m in reason for m in _THROTTLE_API):
        return "throttle"
    return None


# ------------------------------------------------------------------ selection
def _pool(args: argparse.Namespace):
    # Always consult the merged rectified KB, not just this arm's --kb: a
    # parallel arm writes to its own empty shard, so without this every arm
    # would happily redo tasks another arm (or an earlier run) already
    # rectified.
    kb_paths = [Path(args.kb or DEFAULT_KB_PATH), Path(DEFAULT_KB_PATH)]
    if not args.ignore_taskgen_kb:
        kb_paths.append(TASKGEN_KB_PATH)
    pool = load_pool(
        runs_dir=Path(args.runs_dir) if args.runs_dir else DEFAULT_TASKGEN_RUNS_DIR,
        kb_paths=kb_paths,
        min_steps=args.min_steps,
        categories=args.category or None,
        exclude_categories=args.exclude_category or None,
        reasons=args.reason or None,
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(pool)
    if getattr(args, "shard", None):
        i, n = _parse_shard(args.shard)
        # Stride, not block: neighbouring pool entries have similar plan lengths
        # (the pool is sorted by length), so every shard gets the same mix of
        # short and long tasks and the arms finish at about the same time.
        pool = pool[i::n]
    return pool


def _parse_shard(spec: str) -> tuple[int, int]:
    try:
        i, n = (int(x) for x in spec.split("/", 1))
    except ValueError:
        raise SystemExit(f"--shard expects I/N, got {spec!r}")
    if not 0 <= i < n:
        raise SystemExit(f"--shard index out of range: {spec!r}")
    return i, n


def cmd_pool(args: argparse.Namespace) -> int:
    pool = _pool(args)
    if args.json:
        print(json.dumps([r.to_dict() for r in pool], indent=2, ensure_ascii=False))
        return 0
    print(f"eligible rejected tasks (>= {args.min_steps} steps): {len(pool)}")
    print("\nby reason:")
    for k, v in Counter(reason_bucket(r.reason) for r in pool).most_common():
        print(f"  {v:4d}  {k}")
    print("\nby category:")
    for k, v in Counter(r.category for r in pool).most_common():
        print(f"  {v:4d}  {k}")
    print("\nby plan length:")
    for k, v in sorted(Counter(r.steps for r in pool).items()):
        print(f"  {k:4d} steps: {v}")
    if args.list:
        print("\ntasks:")
        for i, r in enumerate(pool[: args.list]):
            print(f"\n[{i}] {r.steps} steps | {r.category} | "
                  f"{reason_bucket(r.reason)} | {r.run_id}")
            print(f"    {r.task_prompt[:240]}")
    return 0


# ---------------------------------------------------------------- rectify run
def cmd_rectify(args: argparse.Namespace) -> int:
    pool = _pool(args)

    progress_path = Path(args.progress)
    done: set[str] = set()
    if args.resume and progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["task_prompt"].strip().casefold())
                except (json.JSONDecodeError, KeyError):
                    continue
        pool = [r for r in pool if r.task_prompt.strip().casefold() not in done]
        print(f"[rectify] resume: {len(done)} task(s) already attempted, "
              f"{len(pool)} remain")

    if args.offset:
        pool = pool[args.offset:]
    if args.limit:
        pool = pool[: args.limit]
    if not pool:
        print("[rectify] nothing to do")
        return 1

    kb = KnowledgeBase(args.kb or DEFAULT_KB_PATH)
    pipeline = RectifyPipeline(
        kb,
        provider=args.provider,
        max_replans=args.max_replans,
        min_replans=args.min_replans,
        max_per_signature=args.max_per_signature,
        log_to_console=not args.quiet,
        task_timeout_s=args.task_timeout,
    )

    print(f"[rectify] {len(pool)} task(s) | max_replans={args.max_replans} "
          f"| accept window {args.min_replans}..{args.max_replans} replans "
          f"| KB {kb.path} ({len(kb.entries)} entries)")

    accepted = solved = failed = throttled = 0
    t0 = time.time()
    for i, rec in enumerate(pool, 1):
        print(f"\n=== [{i}/{len(pool)}] {rec.category} | {rec.steps}-step reject "
              f"| {reason_bucket(rec.reason)} ===")
        print(f"[rectify] task: {rec.task_prompt[:300]}")
        started = time.time()
        result = pipeline.run_spec(rec.to_spec())
        elapsed = round(time.time() - started, 1)

        # --- the LLM endpoint died: this is not a task result ---
        kind = None if result.accepted else _api_failure(result.reason)
        if kind:
            throttled += 1
            print(f"[rectify] LLM API failure ({kind}), task NOT recorded so it "
                  f"is retried later: {result.reason[:200]}")
            if kind == "fatal":
                print("[rectify] token plan exhausted — stopping this arm. "
                      "Restore quota, then re-run with --resume.")
                break
            if throttled >= _MAX_CONSECUTIVE_THROTTLE:
                print(f"[rectify] {throttled} consecutive throttled calls — "
                      "stopping this arm rather than burning the pool.")
                break
            time.sleep(30)
            continue
        throttled = 0

        if result.accepted:
            accepted += 1
            solved += 1
            print(f"[rectify] ACCEPTED as {result.entry['id']} after "
                  f"{result.replans} replan(s) in {elapsed}s")
        else:
            failed += 1
            if result.solved:
                solved += 1

        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task_prompt": rec.task_prompt,
                "category": rec.category,
                "source_run_id": rec.run_id,
                "source_reason": rec.reason,
                "source_steps": rec.steps,
                "run_id": result.run_id,
                "accepted": result.accepted,
                "solved": result.solved,
                "replans": result.replans,
                "reason": result.reason,
                "entry_id": (result.entry or {}).get("id"),
                "seconds": elapsed,
                "rounds": result.rounds,
            }, ensure_ascii=False) + "\n")

    mins = (time.time() - t0) / 60
    print(f"\n[rectify] done in {mins:.1f} min: {accepted} accepted, "
          f"{solved} solved (incl. below-floor), {failed} not accepted; "
          f"KB now has {len(kb.entries)} entries at {kb.path}")
    return 0 if accepted else 1


def cmd_merge(args: argparse.Namespace) -> int:
    """Fold parallel shard KBs into one file, renumbering ids.

    Each shard runs its own KnowledgeBase, so ids restart at kb-0001 per shard
    and a shared file would lose writes to the last-writer. Merging afterwards
    is the only safe way to run the arms in parallel.
    """
    from .coverage import degeneracy_reasons

    target = KnowledgeBase(args.into)
    seen = {p.strip().casefold() for p in target.prompts()}
    runs = [Path(p) for p in (args.runs_dirs or [])]
    rejected = KnowledgeBase(args.degenerate_into) if runs else None
    added = skipped = gated = 0
    for src in args.shards:
        path = Path(src)
        if not path.is_file():
            print(f"[merge] missing shard {path}")
            continue
        for entry in KnowledgeBase(path).entries:
            prompt = str(entry.get("task_prompt", "")).strip().casefold()
            if prompt in seen:
                skipped += 1
                continue
            entry = dict(entry)
            entry["shard"] = path.name
            # Gate here as well as in the pipeline: shards are a raw record and
            # keep entries written before the gate existed, so a plain merge
            # would quietly re-import them on the next run.
            reasons = []
            if rejected is not None:
                report = _load_report(runs, str(entry.get("run_id", "")))
                if report is not None:
                    reasons = degeneracy_reasons(str(entry.get("task_prompt", "")),
                                                 report)
            if reasons:
                entry["id"] = rejected.next_id()
                entry["quarantine_reasons"] = reasons
                rejected.entries.append(entry)
                gated += 1
            else:
                entry["id"] = target.next_id()
                target.entries.append(entry)
                added += 1
            seen.add(prompt)
    target.save()
    if rejected is not None:
        rejected.save()
    print(f"[merge] {added} added, {skipped} duplicate(s) skipped, "
          f"{gated} turned away by the coverage gate; "
          f"{target.path} now has {len(target.entries)}")
    return 0


def cmd_prune_progress(args: argparse.Namespace) -> int:
    """Drop rows that record an LLM-endpoint failure rather than a task result.

    Those tasks were never really attempted, but a progress row makes --resume
    skip them permanently. Needed for logs written before the API guard existed.
    """
    total = removed = 0
    for p in args.progress:
        path = Path(p)
        if not path.is_file():
            continue
        kept, dropped = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if not rec.get("accepted") and _api_failure(str(rec.get("reason", ""))):
                dropped += 1
            else:
                kept.append(line)
        removed += dropped
        if args.dry_run:
            print(f"[prune] {path.name}: would drop {dropped}, keep {len(kept)}")
            continue
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"[prune] {path.name}: dropped {dropped}, kept {len(kept)}")
    print(f"[prune] {removed}/{total} row(s) were API failures, not attempts"
          + (" (dry run)" if args.dry_run else "; those tasks are eligible again"))
    return 0


def cmd_quarantine(args: argparse.Namespace) -> int:
    """Move entries that passed verification without doing the task out of the KB.

    Re-applies the coverage gate to already-accepted entries (they were written
    before the gate existed). Flagged entries go to a separate file rather than
    being deleted: they are the evidence that a long replan budget degrades a
    trace-derived verifier. Their rows are also dropped from the progress logs,
    so a `--resume` run re-attempts them under the gate.
    """
    from pkg.agenticmcpe.verifier import VerificationReport

    from .coverage import degeneracy_reasons

    kb = KnowledgeBase(args.kb or DEFAULT_KB_PATH)
    runs = [Path(p) for p in args.runs_dirs]
    keep, moved = [], []
    for e in kb.entries:
        report = _load_report(runs, str(e.get("run_id", "")))
        if report is None:
            print(f"[quarantine] {e['id']}: no verification.json found, keeping")
            keep.append(e)
            continue
        reasons = degeneracy_reasons(str(e.get("task_prompt", "")), report)
        if reasons:
            e = dict(e)
            e["quarantine_reasons"] = reasons
            moved.append(e)
        else:
            keep.append(e)

    if args.dry_run:
        print(f"[quarantine] would move {len(moved)}, keep {len(keep)}")
        for e in moved:
            print(f"  {e['id']} {e['category']}: {e['quarantine_reasons'][0][:100]}")
        return 0

    out = KnowledgeBase(args.into)
    for e in moved:
        e["id"] = out.next_id()
        out.entries.append(e)
    out.save()
    kb.entries = keep
    kb.save()

    # Drop the quarantined prompts from the progress logs so --resume retries
    # them; a task that is still in a progress file is considered done.
    prompts = {str(e.get("task_prompt", "")).strip().casefold() for e in moved}
    cleared = 0
    for p in args.progress:
        path = Path(p)
        if not path.is_file():
            continue
        rows = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        kept_rows = []
        for line in rows:
            try:
                done = json.loads(line)["task_prompt"].strip().casefold()
            except (json.JSONDecodeError, KeyError):
                kept_rows.append(line)
                continue
            if done in prompts:
                cleared += 1
            else:
                kept_rows.append(line)
        path.write_text("\n".join(kept_rows) + ("\n" if kept_rows else ""),
                        encoding="utf-8")
    print(f"[quarantine] moved {len(moved)} to {out.path} ({len(keep)} left in "
          f"{kb.path}); cleared {cleared} progress row(s) so they re-run")
    return 0


def _load_report(runs: list[Path], run_id: str):
    from pkg.agenticmcpe.verifier import VerificationReport
    if not run_id:
        return None
    for root in runs:
        for cand in list(root.glob(f"*/{run_id}/verification.json")) + \
                    list(root.glob(f"{run_id}/verification.json")):
            d = json.loads(cand.read_text(encoding="utf-8"))
            return VerificationReport(
                task="", total=d["total"], passed=d["passed"], failed=d["failed"],
                results=d["results"], script_path="", exit_code=d["exit_code"])
    return None


def cmd_stats(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(args.kb or DEFAULT_KB_PATH)
    stats = kb.stats()
    replans = [e.get("replans", 0) for e in kb.entries]
    stats["by_replans"] = dict(sorted(Counter(replans).items()))
    stats["by_origin"] = dict(Counter(e.get("origin", "generated") for e in kb.entries))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------- parser
def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-steps", type=int, default=7,
                   help="Only rejects whose attempted plan had at least this "
                        "many tool calls (default: 7).")
    p.add_argument("--category", action="append",
                   help="Restrict to an archetype (repeatable).")
    p.add_argument("--exclude-category", action="append",
                   help="Skip an archetype (repeatable). Use for categories "
                        "whose sites block the data outright — spending 15 "
                        "replans there cannot succeed honestly.")
    p.add_argument("--reason", action="append",
                   help="Restrict to a reject reason bucket, e.g. "
                        "'execution failed' / 'verification failed' (repeatable).")
    p.add_argument("--runs-dir", help="Where the taskgen run dirs live "
                                      "(default: pkg/agenticmcpe/runs_taskgen).")
    p.add_argument("--kb", help=f"Rectified KB path (default: {DEFAULT_KB_PATH}). "
                                "Point at pkg/taskgen/knowledge_base.json to "
                                "write straight into the main KB.")
    p.add_argument("--ignore-taskgen-kb", action="store_true",
                   help="Do not treat prompts in the main taskgen KB as solved.")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle the pool (default order is longest plan first).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", metavar="I/N",
                   help="Take every Nth task starting at I, for running N arms "
                        "in parallel. Give each arm its own --kb and --progress, "
                        "then fold them together with the `merge` command.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pkg.taskgen_rectify",
        description="Re-run REJECTED long tasks with a large replan budget and "
                    "add the ones that get rectified to a knowledge base.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pool", help="Show which rejects are eligible.")
    _add_selection_args(pp)
    pp.add_argument("--list", type=int, default=0, metavar="N",
                    help="Also print the first N task prompts.")
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=cmd_pool)

    pr = sub.add_parser("rectify", help="Run the rectification pipeline.")
    _add_selection_args(pr)
    pr.add_argument("--limit", type=int, default=0, help="Max tasks to attempt.")
    pr.add_argument("--offset", type=int, default=0)
    pr.add_argument("--max-replans", type=int, default=15)
    pr.add_argument("--min-replans", type=int, default=MIN_REPLANS,
                    help="Replans a run must have needed to be KB-worthy "
                         f"(default: {MIN_REPLANS}).")
    pr.add_argument("--max-per-signature", type=int, default=3)
    pr.add_argument("--task-timeout", type=float, default=3600.0,
                    help="Wall-clock cap per task, checked between rounds.")
    pr.add_argument("--provider", help="LLM provider override.")
    pr.add_argument("--progress", default=str(DEFAULT_PROGRESS_PATH))
    pr.add_argument("--resume", action="store_true",
                    help="Skip prompts already recorded in the progress file.")
    pr.add_argument("--quiet", action="store_true")
    pr.set_defaults(func=cmd_rectify)

    pm = sub.add_parser("merge", help="Fold parallel shard KBs into one file.")
    pm.add_argument("shards", nargs="+", help="Shard KB JSON files.")
    pm.add_argument("--into", default=str(DEFAULT_KB_PATH),
                    help=f"Destination KB (default: {DEFAULT_KB_PATH}).")
    pm.add_argument("--runs-dirs", nargs="*",
                    help="Roots holding run dirs with verification.json. Given "
                         "these, the coverage gate is applied while merging and "
                         "entries that pass checks without doing the task are "
                         "diverted instead of imported.")
    pm.add_argument("--degenerate-into",
                    default=str(DEFAULT_KB_PATH.with_name("knowledge_base_degenerate.json")))
    pm.set_defaults(func=cmd_merge)

    pp2 = sub.add_parser("prune-progress",
                         help="Drop progress rows that record an LLM API "
                              "failure, so those tasks are retried.")
    pp2.add_argument("progress", nargs="+", help="Progress JSONL files.")
    pp2.add_argument("--dry-run", action="store_true")
    pp2.set_defaults(func=cmd_prune_progress)

    pq = sub.add_parser("quarantine",
                        help="Move entries that passed verification without "
                             "doing the task into a separate file.")
    pq.add_argument("--kb", help=f"KB to clean (default: {DEFAULT_KB_PATH}).")
    pq.add_argument("--into",
                    default=str(DEFAULT_KB_PATH.with_name("knowledge_base_degenerate.json")))
    pq.add_argument("--runs-dirs", nargs="+", required=True,
                    help="Roots holding the run dirs with verification.json.")
    pq.add_argument("--progress", nargs="*", default=[],
                    help="Progress logs to clear the quarantined tasks from.")
    pq.add_argument("--dry-run", action="store_true")
    pq.set_defaults(func=cmd_quarantine)

    ps = sub.add_parser("stats", help="Rectified KB size, coverage and replan mix.")
    ps.add_argument("--kb")
    ps.set_defaults(func=cmd_stats)
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
