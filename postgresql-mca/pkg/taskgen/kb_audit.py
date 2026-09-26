"""Audit every knowledge_base.json entry against the run artifacts that back it.

The KB is only worth as much as its ground-truth policy: an entry must be
provably backed by a run that EXECUTED end-to-end and PASSED verification. This
script re-checks that claim from the artifacts on disk rather than trusting the
metadata the pipeline wrote, and reports (or with --prune, removes) entries that
cannot be substantiated.

    python3 pkg/taskgen/kb_audit.py              # report only
    python3 pkg/taskgen/kb_audit.py --verbose    # list every flagged entry
    python3 pkg/taskgen/kb_audit.py --prune      # drop unsubstantiated entries

Integrity checks (any failure = unsubstantiated):
  * the run dir exists and holds plan.json + trace.json + verification.json
    (a dir with only catalog.json means the pipeline died before planning);
  * trace.success is true AND every step's status is "success";
  * verification has total > 0, failed == 0, exit_code == 0;
  * the entry's tool_sequence matches the tools the trace actually executed;
  * the entry's steps match plan.json's steps (the RAG reuse payload IS the
    sequence that was proven);
  * the entry's verification counts match verification.json;
  * replans == 0 (a replanned plan is not a from-scratch sequence);
  * plan.json's task matches the stored task_prompt.

Evidence-quality warnings (reported, never pruned — judgement calls):
  * fewer than 5 checks, or no dynamic (live re-query) checks at all;
  * a dynamic layer that was skipped/stubbed instead of really authored;
  * steps that succeeded only by idempotent adoption (the effect pre-existed,
    so this run did not prove the sequence performs it);
  * a task_prompt duplicated by another entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "pkg" / "agenticmcpe" / "runs"
DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def audit_entry(entry: dict, runs_dir: Path) -> tuple[list[str], list[str]]:
    """(hard failures, quality warnings) for one KB entry."""
    bad: list[str] = []
    warn: list[str] = []
    run_id = str(entry.get("run_id") or "")
    d = runs_dir / run_id
    if not run_id or not d.is_dir():
        return [f"run dir missing: {run_id!r}"], warn

    plan, trace, ver = (_load(d / n) for n in
                        ("plan.json", "trace.json", "verification.json"))
    for name, obj in (("plan.json", plan), ("trace.json", trace),
                      ("verification.json", ver)):
        if obj is None:
            bad.append(f"{name} missing/unreadable")
    if bad:
        return bad, warn

    steps = trace.get("steps") or []
    if not trace.get("success"):
        bad.append("trace.success is false")
    statuses = {s.get("status") for s in steps}
    if not steps or statuses != {"success"}:
        bad.append(f"step statuses {sorted(statuses)}")
    if ver.get("total", 0) <= 0:
        bad.append("verification recorded no checks")
    if ver.get("failed", 0):
        bad.append(f"verification failed={ver['failed']}")
    if ver.get("exit_code") not in (0, None):
        bad.append(f"verifier exit_code={ver['exit_code']}")
    if entry.get("verification") != {"passed": ver.get("passed"),
                                     "total": ver.get("total")}:
        bad.append("stored verification != verification.json")
    if entry.get("tool_sequence") != [s.get("tool") for s in steps]:
        bad.append("tool_sequence != executed trace")
    if ([s.get("tool") for s in entry.get("steps") or []]
            != [s.get("tool") for s in plan.get("steps") or []]):
        bad.append("stored steps != plan.json steps")
    if entry.get("replans", 0):
        bad.append("replans > 0 (not a from-scratch sequence)")
    if (plan.get("task") or "").strip() != (entry.get("task_prompt") or "").strip():
        bad.append("plan.json task != task_prompt")

    results = ver.get("results") or []
    dynamic = [r for r in results if r.get("category") == "dynamic"]
    if ver.get("total", 0) < 5:
        warn.append(f"thin verification ({ver.get('total')} checks)")
    if not dynamic:
        warn.append("no dynamic (live re-query) checks")
    if any("skipped" in str(r.get("detail", "")).lower() for r in dynamic):
        warn.append("dynamic layer skipped/stubbed")
    adopted = [s.get("id") for s in steps
               if any(a.get("status") == "success_idempotent"
                      for a in s.get("attempts") or [])]
    if adopted:
        warn.append(f"{len(adopted)} step(s) succeeded by idempotent adoption")
    return bad, warn


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 pkg/taskgen/kb_audit.py",
        description="Verify every KB entry against the run artifacts backing it.")
    p.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--verbose", action="store_true",
                   help="List every flagged entry, not just the summary.")
    p.add_argument("--prune", action="store_true",
                   help="Remove unsubstantiated entries from the KB (a .bak "
                        "copy is written first).")
    args = p.parse_args(argv)

    kb_path, runs_dir = Path(args.kb), Path(args.runs_dir)
    kb = _load(kb_path)
    if not kb:
        print(f"cannot read KB at {kb_path}", file=sys.stderr)
        return 2
    entries = kb.get("entries") or []

    keep, drop = [], []
    fail_reasons, warn_reasons = Counter(), Counter()
    prompts = Counter(str(e.get("task_prompt", "")).strip().casefold()
                      for e in entries)
    for e in entries:
        bad, warn = audit_entry(e, runs_dir)
        if prompts[str(e.get("task_prompt", "")).strip().casefold()] > 1:
            warn.append("duplicate task_prompt")
        fail_reasons.update(bad)
        warn_reasons.update(warn)
        (drop if bad else keep).append(e)
        if args.verbose and (bad or warn):
            print(f"{e.get('id')} [{e.get('run_id')}]"
                  + (f"\n  FAIL: {'; '.join(bad)}" if bad else "")
                  + (f"\n  warn: {'; '.join(warn)}" if warn else ""))

    print(f"\naudited {len(entries)} entr(ies): "
          f"{len(keep)} substantiated, {len(drop)} unsubstantiated")
    if fail_reasons:
        print("\nhard failures:")
        for k, v in fail_reasons.most_common():
            print(f"  {v:4d}  {k}")
    if warn_reasons:
        print("\nquality warnings (not pruned):")
        for k, v in warn_reasons.most_common():
            print(f"  {v:4d}  {k}")

    if drop and args.prune:
        backup = kb_path.with_suffix(kb_path.suffix + ".bak")
        backup.write_text(json.dumps(kb, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        kb["entries"] = keep
        kb_path.write_text(json.dumps(kb, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print(f"\npruned {len(drop)} entr(ies); previous KB saved to {backup.name}")
    elif drop:
        print("\nre-run with --prune to remove them")
    return 1 if drop else 0


if __name__ == "__main__":
    sys.exit(main())
