"""Drive the full finance_agenticmcpe workflow on a single task read from a
file (avoids shell-escaping the long, quote-heavy prompt). Modifies
nothing in the package.

Usage::

    python pkg/finance_agenticmcpe/run/run_task.py                       # task_prompt.txt
    python pkg/finance_agenticmcpe/run/run_task.py task_05               # task_05.txt
    python pkg/finance_agenticmcpe/run/run_task.py task_05 my_run_id
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# repo root = .../pkg/finance_agenticmcpe/run/run_task.py -> up 3
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pkg.finance_agenticmcpe.config import AgenticConfig, default_config
from pkg.finance_agenticmcpe.orchestrator import Orchestrator

RUN_DIR = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    task_arg = argv[0] if argv else "task_prompt"
    run_id = argv[1] if len(argv) > 1 else f"run_{int(time.time())}"

    prompt_path = RUN_DIR / (task_arg if task_arg.endswith(".txt") else task_arg + ".txt")
    if not prompt_path.is_file():
        print(f"task file not found: {prompt_path}", file=sys.stderr)
        return 2
    task = prompt_path.read_text(encoding="utf-8").strip()

    cfg = default_config()
    print(f"[cfg] model={cfg.model} anthropic_key={'yes' if cfg.effective_anthropic_api_key() else 'NO'}")
    print(f"[cfg] server={cfg.server_command} {cfg.server_args}")
    print(f"[cfg] work_dir=runs/{run_id}\n")

    work_dir = str(RUN_DIR.parent.parent.parent / "runs" / run_id)

    orch = Orchestrator(cfg)
    try:
        result = orch.run(task, run_id=run_id, work_dir=work_dir, stream=True)
    except Exception as e:
        print("\n!!! RUN RAISED:", repr(e))
        traceback.print_exc()
        return 3

    print("\n================ RESULT ================")
    print("overall ok :", result.success)
    print(f"attempts   : {len(result.attempts)}")
    if result.final_trace is not None:
        print("execution  :", "SUCCESS" if result.final_trace.success
              else f"FAILED at {result.final_trace.failed_step}")
        print("\nstep statuses:")
        for st in result.final_trace.steps:
            err = f"  ERR: {st.error[:80]}" if st.error else ""
            print(f"  {st.id:>3} {st.tool:<28} -> {st.status:<16} attempts={len(st.attempts)}{err}")
    if result.final_verification is not None:
        n_pass = sum(1 for c in result.final_verification.checks if c.passed)
        n_total = len(result.final_verification.checks)
        print(f"\nverification: {n_pass}/{n_total} "
              f"-> {'OK' if result.final_verification.passed else 'FAIL'}")
        for c in result.final_verification.checks:
            if not c.passed:
                print(f"  FAIL [{c.layer}] {c.name}: {c.detail[:120]}")
    print(f"\nartifacts in: {work_dir}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))