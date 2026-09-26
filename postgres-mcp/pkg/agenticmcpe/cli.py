"""CLI for the agenticmcpe workflow (PostgreSQL edition).

    python -m pkg.agenticmcpe config
    python -m pkg.agenticmcpe plan   --task "Create a table inventory.items with 3 seeded rows"
    python -m pkg.agenticmcpe exec   --plan runs/<id>/plan.json
    python -m pkg.agenticmcpe verify --plan runs/<id>/plan.json --trace runs/<id>/trace.json
    python -m pkg.agenticmcpe run    --task "..."        # full pipeline
    python -m pkg.agenticmcpe selfcheck                  # live read-only end-to-end
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .catalog import condense_catalog, load_catalog
from .config import ConfigError, LLMClient, Settings
from .executor import ExecutionAgent, load_plan
from .orchestrator import SELFCHECK_TASK, Orchestrator, selfcheck
from .planner import PlannerAgent, PlannerError
from .rag import KBRetriever
from .verifier import VerifierAgent


def _retriever(args: argparse.Namespace) -> "KBRetriever | None":
    """Build the KB retriever for RAG planning, unless --no-rag / the env switch
    is set or the KB is empty/missing."""
    if getattr(args, "no_rag", False) or os.environ.get("AGENTICMCPE_DISABLE_RAG"):
        return None
    r = KBRetriever()
    return r if not r.is_empty else None


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", help="LLM provider (deepseek|openai|gemini|anthropic|"
                                       "minimax|qwen|glm|kimi|grok|xiaomi|custom). "
                                       "Default: deepseek or $AGENTICMCPE_LLM_PROVIDER.")
    p.add_argument("--run-id", help="Reuse/name a run directory under runs/.")
    p.add_argument("--quiet", action="store_true", help="Suppress progress logging.")


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.load(provider=getattr(args, "provider", None),
                         run_id=getattr(args, "run_id", None))


def cmd_config(args: argparse.Namespace) -> int:
    s = _settings(args)
    print(json.dumps(s.summary(), indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    s = _settings(args)
    catalog = load_catalog(s)
    retriever = _retriever(args)
    if retriever is not None and not args.quiet:
        print(f"[plan] RAG enabled: {len(retriever)} verified KB example(s)")
    planner = PlannerAgent(LLMClient(s.llm), condense_catalog(catalog),
                           retriever=retriever)
    plan = planner.plan(args.task)
    json_path, md_path = planner.write(plan, s)
    print(f"[plan] {len(plan.steps)} step(s) -> {json_path}")
    for st in plan.steps:
        print(f"  {st.id:>3} {st.tool}({', '.join(st.arguments)})")
    for w in plan.warnings:
        print(f"  [warn] {w}")
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    s = _settings(args)
    plan = load_plan(args.plan)
    agent = ExecutionAgent(s, log_to_console=not args.quiet)
    trace = agent.run(plan)
    print(json.dumps({"success": trace.success, "failed_step": trace.failed_step,
                      "work_dir": str(s.work_dir)}, indent=2))
    return 0 if trace.success else 1


def cmd_verify(args: argparse.Namespace) -> int:
    import shutil
    from pathlib import Path
    s = _settings(args)
    plan = load_plan(args.plan)
    with open(args.trace, encoding="utf-8") as f:
        td = json.load(f)
    trace = _trace_from_dict(td)
    # verify.py reads plan.json/trace.json from its own directory; make sure
    # they exist there when verifying artifacts from another run dir.
    s.ensure_work_dir()
    for src, name in ((args.plan, "plan.json"), (args.trace, "trace.json")):
        dst = s.work_dir / name
        if Path(src).resolve() != dst.resolve():
            shutil.copy2(src, dst)
    llm = None if args.no_llm else LLMClient(s.llm)
    report = VerifierAgent(s, llm).verify(plan, trace)
    print(json.dumps({"ok": report.ok, "passed": report.passed, "total": report.total,
                      "results": report.results}, indent=2))
    return 0 if report.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    s = _settings(args)
    orch = Orchestrator(s, log_to_console=not args.quiet, use_rag=not args.no_rag)
    result = orch.run(args.task, max_replans=args.max_replans, verify=not args.no_verify)
    _print_run(result, s)
    return 0 if result.ok else 1


def cmd_toolsource(args: argparse.Namespace) -> int:
    from .toolsource import DEFAULT_KB_DIR, build_kb
    # Needs the catalog, not a run of its own: reuse one stable dir rather
    # than leaving a catalog-only run dir behind (same reasoning as taskgen).
    if not getattr(args, "run_id", None):
        args.run_id = "toolsource-catalog"
    s = _settings(args)
    catalog = load_catalog(s)
    index = build_kb(s.repo_root, [t.name for t in catalog])
    covered = len(index["tools"]) - len(index["missing_impl"])
    print(f"[toolsource] {covered}/{len(index['tools'])} tools mapped to Python "
          f"impl functions -> {DEFAULT_KB_DIR}")
    if index["missing_impl"]:
        print(f"[toolsource] no impl found: {index['missing_impl']}")
    if index["missing_tests"]:
        print(f"[toolsource] no tests found: {index['missing_tests']}")
    return 0 if not index["missing_impl"] else 1


def cmd_selfcheck(args: argparse.Namespace) -> int:
    s = _settings(args)
    print(f"[selfcheck] provider={s.llm.provider} model={s.llm.model} "
          f"db={'yes' if s.database.available else 'NO'} "
          f"key={'yes' if s.llm.api_key else 'NO'}")
    if not s.llm.api_key:
        print("  ERROR: no LLM API key. Set DEEPSEEK_API_KEY (or chosen provider key) "
              "in env/.env.", file=sys.stderr)
        return 2
    if not s.database.available:
        print("  ERROR: no database URI. Set DATABASE_URI in env/.env, e.g. "
              "postgresql://user:pass@localhost:5432/dbname.", file=sys.stderr)
        return 2
    result = selfcheck(s, task=args.task)
    _print_run(result, s)
    return 0 if result.ok else 1


def _print_run(result: Any, s: Settings) -> None:
    print("\n=== RUN COMPLETE ===")
    print(f"work_dir : {s.work_dir}")
    print(f"plan     : {len(result.plan.steps)} step(s), replans={result.replans}")
    if result.replans:
        print(f"           NOTE: plan.json holds only the FINAL (remaining-work) "
              f"plan. The consolidated from-scratch sequence is in "
              f"plan.effective.json; earlier attempts are archived as "
              f"attemptN-plan.json and summarized in run.json 'attempts'.")
    print(f"execution: {'SUCCESS' if result.trace.success else 'FAILED at ' + str(result.trace.failed_step)}")
    if result.report is not None:
        print(f"verify   : {result.report.passed}/{result.report.total} checks passed "
              f"-> {'OK' if result.report.ok else 'FAIL'}")
    print(f"overall  : {'OK' if result.ok else 'FAIL'}")


def _trace_from_dict(td: dict[str, Any]):
    from .executor import Attempt, ExecutionTrace, StepExecution
    steps = []
    for s in td.get("steps", []):
        steps.append(StepExecution(
            id=s["id"], tool=s["tool"], status=s.get("status", "pending"),
            arguments_resolved=s.get("arguments_resolved"),
            attempts=[Attempt(**a) for a in s.get("attempts", [])],
            result_data=s.get("result_data"), result_content=s.get("result_content"),
            error=s.get("error"), error_details=s.get("error_details"),
            post_action_properties=s.get("post_action_properties", {}),
            description=s.get("description", ""),
        ))
    return ExecutionTrace(
        task=td.get("task", ""), success=td.get("success", False), steps=steps,
        failed_step=td.get("failed_step"), error_context=td.get("error_context"),
        started_at=td.get("started_at", 0.0), finished_at=td.get("finished_at", 0.0),
        server_stderr=td.get("server_stderr", ""),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pkg.agenticmcpe",
                                description="Agentic planner/executor/verifier over postgres-mcp.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("config", help="Show resolved configuration.")
    _add_common(pc); pc.set_defaults(func=cmd_config)

    pp = sub.add_parser("plan", help="Plan a tool-call sequence for a task.")
    _add_common(pp); pp.add_argument("--task", required=True)
    pp.add_argument("--no-rag", action="store_true",
                    help="Disable knowledge-base retrieval; plan purely from the LLM.")
    pp.set_defaults(func=cmd_plan)

    pe = sub.add_parser("exec", help="Execute a plan.json verbatim.")
    _add_common(pe); pe.add_argument("--plan", required=True)
    pe.set_defaults(func=cmd_exec)

    pv = sub.add_parser("verify", help="Verify a plan+trace with generated evaluators.")
    _add_common(pv); pv.add_argument("--plan", required=True)
    pv.add_argument("--trace", required=True)
    pv.add_argument("--no-llm", action="store_true", help="Skip LLM dynamic evaluators.")
    pv.set_defaults(func=cmd_verify)

    pr = sub.add_parser("run", help="Full pipeline: plan -> execute -> verify.")
    _add_common(pr); pr.add_argument("--task", required=True)
    pr.add_argument("--max-replans", type=int, default=2)
    pr.add_argument("--no-verify", action="store_true")
    pr.add_argument("--no-rag", action="store_true",
                    help="Disable knowledge-base retrieval; plan purely from the LLM.")
    pr.set_defaults(func=cmd_run)

    pt = sub.add_parser("toolsource",
                        help="Build the tool->Python source/test JSON KB the "
                             "verifier grounds its dynamic evaluators on.")
    _add_common(pt); pt.set_defaults(func=cmd_toolsource)

    ps = sub.add_parser("selfcheck", help="Live read-only end-to-end self-check.")
    _add_common(ps); ps.add_argument("--task", default=None,
                                     help=f"Override the default self-check task.\nDefault: {SELFCHECK_TASK}")
    ps.set_defaults(func=cmd_selfcheck)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except PlannerError as e:
        print(f"planner error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
