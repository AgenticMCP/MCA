"""Drive the full agenticmcpe workflow on a task read from a file (avoids shell
escaping of the long, quote-heavy prompt). Modifies nothing in the package."""
import sys
from pathlib import Path

# repo root = .../pkg/agenticmcpe/run/run_task.py -> up 3
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pkg.agenticmcpe.config import Settings
from pkg.agenticmcpe.orchestrator import Orchestrator

task = (Path(__file__).resolve().parent / "task_prompt.txt").read_text(encoding="utf-8").strip()
run_id = sys.argv[1] if len(sys.argv) > 1 else "test_run"
s = Settings.load(run_id=run_id)
print(f"[cfg] provider={s.llm.provider} model={s.llm.model} "
      f"tokens={len(s.tokens.tokens)} key={'yes' if s.llm.api_key else 'NO'}")
print(f"[cfg] work_dir={s.work_dir}\n")

orch = Orchestrator(s, log_to_console=True)
try:
    result = orch.run(task, max_replans=2, verify=True)
except Exception as e:  # surface a clean message, keep artifacts
    import traceback
    print("\n!!! RUN RAISED:", repr(e))
    traceback.print_exc()
    sys.exit(3)

print("\n================ RESULT ================")
print("overall ok :", result.ok)
print("replans    :", result.replans)
print("execution  :", "SUCCESS" if result.trace.success else f"FAILED at {result.trace.failed_step}")
print("\nstep statuses:")
for st in result.trace.steps:
    print(f"  {st.id:>3} {st.tool:<22} -> {st.status:<16} attempts={len(st.attempts)}"
          + (f"  ERR: {st.error[:80]}" if st.error else ""))
if result.report is not None:
    print(f"\nverification: {result.report.passed}/{result.report.total} "
          f"-> {'OK' if result.report.ok else 'FAIL'}")
    for r in result.report.results:
        if not r["passed"]:
            print(f"  FAIL [{r['category']}] {r['name']}: {r['detail'][:120]}")
print(f"\nartifacts in: {s.work_dir}")
sys.exit(0 if result.ok else 1)
