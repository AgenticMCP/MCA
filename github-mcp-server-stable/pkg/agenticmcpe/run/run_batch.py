"""Batch-drive the full agenticmcpe workflow over every task_NN.txt prompt in
this directory (reads files to avoid shell-escaping the long, quote-heavy
prompts). Each task gets its own run dir runs/task_NN/ (plan/trace/verify/etc.,
written by the orchestrator) plus a consolidated runs/batch_summary.{json,md}.

Resumable: an existing batch_summary.json is loaded and merged by task id, so a
crash/interrupt keeps prior results and a re-run only updates the tasks it runs.
Modifies nothing in the package.

    pkg/venv/bin/python pkg/agenticmcpe/run/run_batch.py            # all tasks
    pkg/venv/bin/python pkg/agenticmcpe/run/run_batch.py 01 12 30   # a subset

Teardown of the repos the batch created (owner = authenticated user; the 6
preserved repos are never touched, and only repos that appear in a
create_repository step of the run traces are eligible):

    pkg/venv/bin/python pkg/agenticmcpe/run/run_batch.py --delete-repos   # run, then delete
    pkg/venv/bin/python pkg/agenticmcpe/run/run_batch.py --cleanup-only   # delete only, no run
    ... add --dry-run to either to print the targets without deleting.
"""
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# repo root = .../pkg/agenticmcpe/run/run_batch.py -> up 3
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

from pkg.agenticmcpe.config import Settings
from pkg.agenticmcpe.orchestrator import Orchestrator
from token_meter import TokenMeter  # noqa: E402  (sibling script)

# Task prompts live next to this script by default; AGENTICMCPE_TASK_DIR points a
# batch at another set (e.g. run/tasks_veryhard) without touching the 28 here.
RUN_DIR = Path(os.environ.get("AGENTICMCPE_TASK_DIR") or Path(__file__).resolve().parent)

# Patched onto the LLM SDK, so it counts the planner AND the verifier's
# generation call — every token this workflow spends on a task.
METER = TokenMeter().install()
TASK_RE = re.compile(r"^task_(\d{2,3})$")


def discover_tasks() -> list[Path]:
    # two- or three-digit ids; a 105-task suite needs three.
    return sorted(set(RUN_DIR.glob("task_[0-9][0-9].txt")) | set(RUN_DIR.glob("task_[0-9][0-9][0-9].txt")))


def select(files: list[Path], argv: list[str]) -> list[Path]:
    if not argv:
        return files
    # accept "7", "07", "007" or "task_007" for any id width present on disk
    wanted = {a.removeprefix("task_").lstrip("0") or "0" for a in argv}
    return [f for f in files
            if (TASK_RE.match(f.stem).group(1).lstrip("0") or "0") in wanted]


def load_summary(path: Path) -> dict[str, dict]:
    if path.is_file():
        try:
            return {r["task_id"]: r for r in json.loads(path.read_text("utf-8"))["tasks"]}
        except Exception:
            pass
    return {}


def write_summary(path: Path, rows: dict[str, dict]) -> None:
    ordered = [rows[k] for k in sorted(rows)]
    passed = sum(1 for r in ordered if r.get("ok"))
    exec_ok = sum(1 for r in ordered if r.get("execution_success"))
    tok = {k: sum((r.get("tokens") or {}).get(k, 0) for r in ordered)
           for k in ("calls", "calls_without_usage", "prompt", "completion", "total")}
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": {"tasks": len(ordered), "overall_ok": passed,
                   "execution_ok": exec_ok, "tokens": tok},
        "tasks": ordered,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    _write_md(path.with_suffix(".md"), payload)


def _write_md(path: Path, payload: dict) -> None:
    t = payload["totals"]
    lines = [
        "# agenticmcpe batch results",
        "",
        f"_generated {payload['generated_at']}_",
        "",
        f"**{t['overall_ok']}/{t['tasks']}** tasks fully OK "
        f"(execution+verification); **{t['execution_ok']}/{t['tasks']}** "
        f"executed successfully.",
        "",
        f"LLM tokens for the round: **{t.get('tokens', {}).get('total', 0):,}** "
        f"({t.get('tokens', {}).get('prompt', 0):,} in / "
        f"{t.get('tokens', {}).get('completion', 0):,} out over "
        f"{t.get('tokens', {}).get('calls', 0)} call(s)).",
        "",
        "| task | overall | execution | replans | verify (pass/total) | secs | tokens | note |",
        "|------|---------|-----------|---------|---------------------|------|--------|------|",
    ]
    for r in payload["tasks"]:
        v = r.get("verification")
        vtxt = f"{v['passed']}/{v['total']}" if v else "—"
        ov = "✅" if r.get("ok") else "❌"
        ex = "ok" if r.get("execution_success") else (r.get("failed_step") or "fail")
        note = r.get("error", "") or ""
        if note:
            note = note.replace("\n", " ")[:60]
        lines.append(
            f"| {r['task_id']} | {ov} | {ex} | {r.get('replans', '—')} | "
            f"{vtxt} | {r.get('seconds', '—')} | "
            f"{(r.get('tokens') or {}).get('total', '—')} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n", "utf-8")


def run_one(task_id: str, prompt: str) -> dict:
    s = Settings.load(run_id=task_id)
    print(f"\n{'#' * 70}\n### {task_id}  ->  {s.work_dir}\n{'#' * 70}")
    row: dict = {"task_id": task_id, "work_dir": str(s.work_dir)}
    tokens_before = METER.snapshot()
    t0 = time.time()
    try:
        result = Orchestrator(s, log_to_console=True).run(
            prompt, max_replans=2, verify=True)
        row.update(
            ok=result.ok,
            replans=result.replans,
            execution_success=result.trace.success,
            failed_step=result.trace.failed_step,
            steps=[{"id": st.id, "tool": st.tool, "status": st.status}
                   for st in result.trace.steps],
            verification=(
                {"ok": result.report.ok, "passed": result.report.passed,
                 "total": result.report.total}
                if result.report is not None else None),
        )
    except Exception as e:  # keep the batch alive; record the failure
        row.update(ok=False, execution_success=False, error=repr(e))
        print(f"\n!!! {task_id} RAISED: {e!r}")
        traceback.print_exc()
    row["seconds"] = round(time.time() - t0, 1)
    row["tokens"] = METER.delta(tokens_before)
    print(f"[tokens] {task_id}: {row['tokens']['total']} "
          f"({row['tokens']['prompt']} in / {row['tokens']['completion']} out, "
          f"{row['tokens']['calls']} call(s))")
    return row


# Repos owned by the test account that are NOT created by any task and must
# NEVER be deleted by the teardown. This is a belt-and-suspenders guard: the
# teardown already only targets repo names that appear in a create_repository
# step of this batch's own traces, so it cannot reach an unrelated repo.
PRESERVE_REPOS = {
    "build-your-own-x", "claude-code", "EasyR1", "harmony",
    "mcpmark-cicd", "missing-semester",
}

_KNOWN_FLAGS = {"--delete-repos", "--cleanup-only", "--dry-run"}


def _gh_api(method: str, url: str, token: str):
    req = urllib.request.Request(
        url, method=method,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    return urllib.request.urlopen(req, timeout=30)


def _created_repo_names(runs_dir: Path, task_ids: list[str]) -> set[str]:
    """Repo names this batch actually created, read from each task's trace.json
    (plus any archived attempt*-trace.json) create_repository steps that
    succeeded. Reading the traces — not the task prompts — guarantees we only
    ever target repos the runs really made."""
    names: set[str] = set()
    for tid in task_ids:
        d = runs_dir / tid
        for tf in [d / "trace.json", *sorted(d.glob("attempt*-trace.json"))]:
            if not tf.is_file():
                continue
            try:
                steps = json.loads(tf.read_text("utf-8")).get("steps", [])
            except Exception:
                continue
            for st in steps:
                if (st.get("tool") == "create_repository"
                        and str(st.get("status", "")).startswith("success")):
                    name = (st.get("arguments_resolved") or {}).get("name")
                    if name:
                        names.add(name)
    return names


def delete_created_repos(runs_dir: Path, task_ids: list[str], *,
                         dry_run: bool = False) -> int:
    """Delete every repo this batch created (owner = authenticated user),
    skipping the PRESERVE_REPOS guard list. Returns the count deleted. Uses the
    REST API (the server exposes no repo-delete tool); needs a token with the
    delete_repo scope — a 403/404 per repo is reported and skipped."""
    s = Settings.load(run_id="_")
    if not s.tokens.available:
        print("[cleanup] no GitHub token; cannot delete")
        return 0
    token = s.tokens.current()
    try:
        login = json.load(_gh_api("GET", "https://api.github.com/user", token))["login"]
    except Exception as e:
        print(f"[cleanup] could not resolve authenticated user: {e!r}")
        return 0
    names = _created_repo_names(runs_dir, task_ids)
    targets = sorted(n for n in names if n not in PRESERVE_REPOS)
    preserved = sorted(n for n in names if n in PRESERVE_REPOS)
    print(f"[cleanup] owner={login}; {len(targets)} repo(s) to delete"
          + (f"; PRESERVED {preserved}" if preserved else "")
          + (" (DRY RUN)" if dry_run else ""))
    deleted = 0
    for name in targets:
        if dry_run:
            print(f"[cleanup]   would delete {login}/{name}")
            continue
        try:
            _gh_api("DELETE", f"https://api.github.com/repos/{login}/{name}", token)
            print(f"[cleanup]   deleted {login}/{name}")
            deleted += 1
        except urllib.error.HTTPError as e:
            print(f"[cleanup]   skip {login}/{name}: HTTP {e.code} {e.reason}")
        except Exception as e:
            print(f"[cleanup]   skip {login}/{name}: {e!r}")
    if not dry_run:
        print(f"[cleanup] done: {deleted}/{len(targets)} deleted")
    return deleted


def main(argv: list[str]) -> int:
    for a in argv:
        if a.startswith("--") and a not in _KNOWN_FLAGS:
            print(f"unknown flag {a!r}; known: {sorted(_KNOWN_FLAGS)}")
            return 2
    flags = {a for a in argv if a.startswith("--")}
    task_args = [a for a in argv if not a.startswith("--")]
    dry_run = "--dry-run" in flags
    cleanup_only = "--cleanup-only" in flags
    delete_repos = "--delete-repos" in flags
    files = select(discover_tasks(), task_args)
    if not files:
        print("no matching task_NN.txt files")
        return 2
    runs_dir = Settings.load(run_id="_").runs_dir
    if cleanup_only:
        delete_created_repos(runs_dir, [f.stem for f in files], dry_run=dry_run)
        return 0
    summary_path = runs_dir / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_summary(summary_path)

    print(f"[batch] {len(files)} task(s): "
          f"{', '.join(TASK_RE.match(f.stem).group(1) for f in files)}")
    for i, f in enumerate(files, 1):
        task_id = f.stem
        print(f"\n[batch] ({i}/{len(files)}) {task_id}")
        prompt = f.read_text("utf-8").strip()
        rows[task_id] = run_one(task_id, prompt)
        write_summary(summary_path, rows)  # checkpoint after every task

    ordered = [rows[k] for k in sorted(rows)]
    ok = sum(1 for r in ordered if r.get("ok"))
    print(f"\n{'=' * 70}\n[batch] DONE. overall OK: {ok}/{len(ordered)}  "
          f"-> {summary_path}\n{'=' * 70}")
    for r in ordered:
        v = r.get("verification")
        vtxt = f"{v['passed']}/{v['total']}" if v else "—"
        flag = "OK " if r.get("ok") else "FAIL"
        print(f"  {r['task_id']}  {flag}  exec="
              f"{'ok' if r.get('execution_success') else r.get('failed_step') or 'fail'}"
              f"  replans={r.get('replans', '—')}  verify={vtxt}  "
              f"{r.get('seconds', '—')}s"
              + (f"  ERR {r['error'][:50]}" if r.get("error") else ""))
    if delete_repos:
        print(f"\n{'=' * 70}\n[cleanup] tearing down repos this batch created\n"
              f"{'=' * 70}")
        delete_created_repos(runs_dir, [f.stem for f in files], dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
