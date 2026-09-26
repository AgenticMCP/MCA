"""Batch-drive the full agenticmcpe workflow over every task_NN.txt prompt in
this directory (reads files to avoid shell-escaping the long, quote-heavy
prompts). Each task gets its own run dir runs/task_NN/ (plan/trace/verify/etc.,
written by the orchestrator) plus a consolidated runs/batch_summary.{json,md}.

Resumable: an existing batch_summary.json is loaded and merged by task id, so a
crash/interrupt keeps prior results and a re-run only updates the tasks it runs.
Modifies nothing in the package.

    python3 pkg/agenticmcpe/run/run_batch.py            # all tasks
    python3 pkg/agenticmcpe/run/run_batch.py 01 03      # a subset

Teardown of the benchmark schemas the batch created (only schemas named
agenticmcpe_bench_* that appear in the run traces' successful execute_sql
steps are eligible — nothing pre-existing can legally carry that prefix):

    python3 pkg/agenticmcpe/run/run_batch.py --drop-schemas   # run, then drop
    python3 pkg/agenticmcpe/run/run_batch.py --cleanup-only   # drop only, no run
    ... add --dry-run to either to print the targets without dropping.
"""
import json
import re
import sys
import time
import traceback
from pathlib import Path

# repo root = .../pkg/agenticmcpe/run/run_batch.py -> up 3
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pkg.agenticmcpe.config import Settings
from pkg.agenticmcpe.orchestrator import Orchestrator
from pkg.mcp_wrapper import MCPClient

RUN_DIR = Path(__file__).resolve().parent
TASK_RE = re.compile(r"^task_(\d{2})$")

# The benchmark namespace: only schemas with this prefix are ever dropped.
BENCH_SCHEMA_PREFIX = "agenticmcpe_bench_"
_BENCH_SCHEMA_RE = re.compile(rf"\b({re.escape(BENCH_SCHEMA_PREFIX)}[a-z0-9_]*)",
                              re.IGNORECASE)


def discover_tasks() -> list[Path]:
    return sorted(RUN_DIR.glob("task_[0-9][0-9].txt"))


def select(files: list[Path], argv: list[str]) -> list[Path]:
    if not argv:
        return files
    wanted = {a.lstrip("task_").zfill(2) for a in argv}
    return [f for f in files if TASK_RE.match(f.stem).group(1) in wanted]


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
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": {"tasks": len(ordered), "overall_ok": passed,
                   "execution_ok": exec_ok},
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
        "| task | overall | execution | replans | verify (pass/total) | secs | note |",
        "|------|---------|-----------|---------|---------------------|------|------|",
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
            f"{vtxt} | {r.get('seconds', '—')} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n", "utf-8")


def run_one(task_id: str, prompt: str) -> dict:
    s = Settings.load(run_id=task_id)
    print(f"\n{'#' * 70}\n### {task_id}  ->  {s.work_dir}\n{'#' * 70}")
    row: dict = {"task_id": task_id, "work_dir": str(s.work_dir)}
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
    return row


_KNOWN_FLAGS = {"--drop-schemas", "--cleanup-only", "--dry-run"}


def _created_bench_schemas(runs_dir: Path, task_ids: list[str]) -> set[str]:
    """Benchmark schemas this batch actually touched, read from each task's
    trace.json (plus any archived attempt*-trace.json) successful execute_sql
    steps. Reading the traces — not the task prompts — guarantees we only ever
    target schemas the runs really used, and the prefix requirement guarantees
    nothing pre-existing is reachable."""
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
                if (st.get("tool") == "execute_sql"
                        and str(st.get("status", "")).startswith("success")):
                    sql = (st.get("arguments_resolved") or {}).get("sql") or ""
                    for m in _BENCH_SCHEMA_RE.finditer(str(sql)):
                        names.add(m.group(1).lower())
    return names


def drop_bench_schemas(runs_dir: Path, task_ids: list[str], *,
                       dry_run: bool = False) -> int:
    """DROP SCHEMA ... CASCADE every benchmark schema this batch touched.
    Returns the count dropped. Strictly prefix-guarded — see
    _created_bench_schemas."""
    s = Settings.load(run_id="_")
    if not s.database.available:
        print("[cleanup] no DATABASE_URI; cannot drop schemas")
        return 0
    targets = sorted(_created_bench_schemas(runs_dir, task_ids))
    print(f"[cleanup] {len(targets)} schema(s) to drop"
          + (" (DRY RUN)" if dry_run else ""))
    if not targets:
        return 0
    dropped = 0
    if dry_run:
        for name in targets:
            print(f"[cleanup]   would drop {name}")
        return 0
    with MCPClient(database_uri=s.database.current(),
                   server_cmd=s.server_cmd or None,
                   access_mode="unrestricted") as pg:
        for name in targets:
            if not name.startswith(BENCH_SCHEMA_PREFIX):
                continue  # paranoia: never drop anything unprefixed
            if not re.fullmatch(r"[a-z0-9_]+", name):
                continue
            try:
                # either a dedicated schema or an advisor-task table in
                # public — IF EXISTS no-ops the non-applicable form
                pg.call("execute_sql",
                        {"sql": f"DROP SCHEMA IF EXISTS {name} CASCADE"})
                pg.call("execute_sql",
                        {"sql": f"DROP TABLE IF EXISTS public.{name} CASCADE"})
                print(f"[cleanup]   dropped {name}")
                dropped += 1
            except Exception as e:
                print(f"[cleanup]   skip {name}: {e!r}")
    print(f"[cleanup] done: {dropped}/{len(targets)} dropped")
    return dropped


def main(argv: list[str]) -> int:
    for a in argv:
        if a.startswith("--") and a not in _KNOWN_FLAGS:
            print(f"unknown flag {a!r}; known: {sorted(_KNOWN_FLAGS)}")
            return 2
    flags = {a for a in argv if a.startswith("--")}
    task_args = [a for a in argv if not a.startswith("--")]
    dry_run = "--dry-run" in flags
    cleanup_only = "--cleanup-only" in flags
    drop_schemas = "--drop-schemas" in flags
    files = select(discover_tasks(), task_args)
    if not files:
        print("no matching task_NN.txt files")
        return 2
    runs_dir = Settings.load(run_id="_").runs_dir
    if cleanup_only:
        drop_bench_schemas(runs_dir, [f.stem for f in files], dry_run=dry_run)
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
    if drop_schemas:
        print(f"\n{'=' * 70}\n[cleanup] tearing down schemas this batch created\n"
              f"{'=' * 70}")
        drop_bench_schemas(runs_dir, [f.stem for f in files], dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
