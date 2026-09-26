"""Batch-drive the full finance_agenticmcpe workflow over every task_NN.txt
prompt in this directory (reads files to avoid shell-escaping the long,
quote-heavy prompts). Each task gets its own run dir under
``runs/<run_id>/`` (plan/trace/verify/etc., written by the orchestrator)
plus a consolidated ``runs/batch_summary.{json,md}``.

Resumable: an existing ``batch_summary.json`` is loaded and merged by
task id, so a crash or interrupt keeps prior results and a re-run only
updates the tasks it runs. Modifies nothing in the package.

Usage::

    python pkg/finance_agenticmcpe/run/run_batch.py            # all tasks
    python pkg/finance_agenticmcpe/run/run_batch.py 01 12 30   # a subset

The finance server is read-only — no teardown is needed; nothing to
clean up. (The github batch has a repo-delete step the finance batch
deliberately omits.)
"""
from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path

# repo root = .../pkg/finance_agenticmcpe/run/run_batch.py -> up 3
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pkg.finance_agenticmcpe.config import AgenticConfig, default_config
from pkg.finance_agenticmcpe.orchestrator import Orchestrator
from pkg.finance_agenticmcpe import token_meter

RUN_DIR = Path(__file__).resolve().parent
TASK_RE = re.compile(r"^task_(\d{2})$")
# Rebindable via --runs-dir. Every arm previously wrote to the same
# runs/task_NN and runs/batch_summary.json, so two concurrent batches
# silently interleaved results from different code revisions into one
# summary — that corrupted two measurements before this existed.
RUNS_DIR = ROOT / "runs"


def discover_tasks() -> list[Path]:
    return sorted(RUN_DIR.glob("task_[0-9][0-9].txt"))


def parse_args(argv: list[str]) -> tuple[list[str], Path]:
    """Split argv into task ids and the output directory.

    ``--runs-dir DIR`` (or ``--runs-dir=DIR``) redirects every artifact this
    run writes; everything else is a task id.
    """
    ids: list[str] = []
    runs_dir = RUNS_DIR
    it = iter(range(len(argv)))
    skip = -1
    for i, a in enumerate(argv):
        if i == skip:
            continue
        if a == "--runs-dir":
            if i + 1 >= len(argv):
                raise SystemExit("--runs-dir needs a directory")
            runs_dir = Path(argv[i + 1]).expanduser().resolve()
            skip = i + 1
        elif a.startswith("--runs-dir="):
            runs_dir = Path(a.split("=", 1)[1]).expanduser().resolve()
        else:
            ids.append(a)
    return ids, runs_dir


def select(files: list[Path], argv: list[str]) -> list[Path]:
    if not argv:
        return files
    wanted = {a.lstrip("task_").zfill(2) for a in argv}
    return [
        f for f in files
        if TASK_RE.match(f.stem) and TASK_RE.match(f.stem).group(1) in wanted
    ]


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
           for k in ("calls", "prompt_tokens", "completion_tokens",
                     "total_tokens", "reasoning_tokens")}
    # Record which model produced these numbers. Only react_summary.json used
    # to carry it, so identifying a pipeline arm's model meant inferring it
    # from the sibling ReAct file plus whatever the shared .env happened to
    # hold later. Worse, provider model strings are often floating aliases --
    # the same string means different models on different dates, and the run
    # date was the only way to tell them apart.
    try:
        # Absolute, like every other import here: this file runs as a script,
        # so __package__ is empty and a relative import raises.
        from pkg.finance_agenticmcpe.config import LLMSettings

        _s = LLMSettings.from_env()
        _llm = {"provider": _s.provider, "model": _s.model, "base_url": _s.base_url,
                "temperature": _s.temperature, "max_tokens": _s.max_tokens,
                "extra_body": _s.extra_body}
    except Exception:  # noqa: BLE001 — provenance is metadata; never fail the run.
        _llm = {}
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm": _llm,
        "totals": {
            "tasks": len(ordered),
            "overall_ok": passed,
            "execution_ok": exec_ok,
        },
        "tokens": tok,
        "tasks": ordered,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    _write_md(path.with_suffix(".md"), payload)


def _write_md(path: Path, payload: dict) -> None:
    t = payload["totals"]
    lines = [
        "# finance_agenticmcpe batch results",
        "",
        f"_generated {payload['generated_at']}_",
        "",
        f"**{t['overall_ok']}/{t['tasks']}** tasks fully OK "
        f"(execution+verification); **{t['execution_ok']}/{t['tasks']}** "
        f"executed successfully.",
        "",
        f"LLM usage: **{payload.get('tokens', {}).get('total_tokens', 0):,} tokens** "
        f"over {payload.get('tokens', {}).get('calls', 0)} calls "
        f"({payload.get('tokens', {}).get('prompt_tokens', 0):,} prompt / "
        f"{payload.get('tokens', {}).get('completion_tokens', 0):,} completion).",
        "",
        "| task | overall | execution | attempts | verify (pass/total) | tokens | secs | note |",
        "|------|---------|-----------|----------|---------------------|--------|------|------|",
    ]
    for r in payload["tasks"]:
        v = r.get("verification")
        vtxt = f"{v['passed']}/{v['total']}" if v else "—"
        ov = "OK" if r.get("ok") else "FAIL"
        ex = "ok" if r.get("execution_success") else (r.get("failed_step") or "fail")
        note = r.get("error", "") or ""
        if note:
            note = note.replace("\n", " ")[:60]
        lines.append(
            f"| {r['task_id']} | {ov} | {ex} | "
            f"{r.get('attempts', '—')} | {vtxt} | "
            f"{(r.get('tokens') or {}).get('total_tokens', 0):,} | "
            f"{r.get('seconds', '—')} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n", "utf-8")


def run_one(task_id: str, prompt: str) -> dict:
    cfg = default_config()
    work_dir = str(RUNS_DIR / task_id)
    token_meter.reset()
    print(f"\n{'#' * 70}\n### {task_id}  ->  {work_dir}\n{'#' * 70}")
    row: dict = {"task_id": task_id, "work_dir": work_dir}
    t0 = time.time()
    try:
        orch = Orchestrator(cfg)
        result = orch.run(prompt, run_id=task_id, work_dir=work_dir, stream=True)
        n_ver_pass = (
            sum(1 for c in result.final_verification.checks if c.passed)
            if result.final_verification is not None
            else None
        )
        n_ver_total = (
            len(result.final_verification.checks)
            if result.final_verification is not None
            else None
        )
        row.update(
            ok=result.success,
            attempts=len(result.attempts),
            execution_success=(
                result.final_trace.success if result.final_trace is not None else False
            ),
            failed_step=(
                result.final_trace.failed_step if result.final_trace is not None else None
            ),
            steps=[
                {"id": st.id, "tool": st.tool, "status": st.status}
                for st in (result.final_trace.steps if result.final_trace else [])
            ],
            verification=(
                {"ok": result.final_verification.passed,
                 "passed": n_ver_pass, "total": n_ver_total}
                if result.final_verification is not None else None
            ),
        )
    except Exception as e:
        row.update(ok=False, execution_success=False, error=repr(e))
        print(f"\n!!! {task_id} RAISED: {e!r}")
        traceback.print_exc()
    row["seconds"] = round(time.time() - t0, 1)
    # Every LLM call this task made, metered at the provider SDK so the
    # figure is comparable with the ReAct baseline's.
    row["tokens"] = token_meter.snapshot()
    return row


def main(argv: list[str]) -> int:
    global RUNS_DIR
    ids, RUNS_DIR = parse_args(argv)
    token_meter.install()
    files = select(discover_tasks(), ids)
    if not files:
        print("no matching task_NN.txt files")
        return 2
    summary_path = RUNS_DIR / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_summary(summary_path)

    print(f"[batch] {len(files)} task(s): "
          f"{', '.join(TASK_RE.match(f.stem).group(1) for f in files)}")
    for i, f in enumerate(files, 1):
        task_id = f.stem
        print(f"\n[batch] ({i}/{len(files)}) {task_id}")
        prompt = f.read_text(encoding="utf-8").strip()
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
        print(
            f"  {r['task_id']}  {flag}  exec="
            f"{'ok' if r.get('execution_success') else r.get('failed_step') or 'fail'}"
            f"  attempts={r.get('attempts', '—')}  verify={vtxt}  "
            f"{r.get('seconds', '—')}s"
            + (f"  ERR {r['error'][:50]}" if r.get("error") else "")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))