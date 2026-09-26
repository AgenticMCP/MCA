"""Command-line interface for the finance agentic workflow.

Subcommands:

* ``config``       — print the resolved configuration.
* ``catalog``      — spawn the server, list its tool catalog (or extract
                     it from source if ``--from-source``).
* ``plan``         — plan a task, write plan.json + plan.md.
* ``exec``         — execute a plan.json against the server.
* ``verify``       — verify a plan.json + trace.json pair.
* ``run``          — plan + exec + verify with bounded replan.
* ``toolsource``   — extract a per-tool source KB (Python AST -> dict).
* ``selfcheck``    — read-only end-to-end smoke test against a stub task.

Examples::

    python -m pkg.finance_agenticmcpe config
    python -m pkg.finance_agenticmcpe catalog --from-source
    python -m pkg.finance_agenticmcpe plan --task-file run/task_0001.txt
    python -m pkg.finance_agenticmcpe run --task "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .config import AgenticConfig, default_config
from .executor import ExecutionAgent, ExecutionTrace
from .llm import LLMClient
from .orchestrator import Orchestrator
from .planner import Plan, PlannerAgent, PlannerError
from .verifier import VerifierAgent
from .catalog import (
    discover,
    load_catalog,
    save_catalog,
    tool_to_dict,
)


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to a JSON config file")


def _load_config(args: argparse.Namespace) -> AgenticConfig:
    if getattr(args, "config", None):
        return AgenticConfig.load(args.config)
    return default_config()


def _resolve_tools(config: AgenticConfig, prefer: str = "live") -> list:
    return discover(
        source_path=config.tool_definition_source,
        command=config.server_command,
        args=config.server_args,
        cwd=config.server_cwd,
        env=config.server_env,
        prefer=prefer,
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    print(cfg.to_json())
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    prefer = "source" if args.from_source else "live"
    tools = _resolve_tools(cfg, prefer=prefer)
    out = [tool_to_dict(t) for t in tools]
    if args.out:
        save_catalog(tools, args.out)
        print(f"wrote {len(out)} tool(s) to {args.out}", file=sys.stderr)
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    task = _read_task(args)
    llm = _make_llm(cfg)
    tools = _resolve_tools(cfg, prefer="source" if args.from_source else "live")
    planner = PlannerAgent(llm=llm, tools=tools)
    try:
        plan_obj = planner.plan(task)
    except PlannerError as e:
        print(f"planner failed: {e}", file=sys.stderr)
        return 2
    out_path = args.out or "plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan_obj.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    if args.warnings:
        for w in plan_obj.warnings:
            print(f"warning: {w}", file=sys.stderr)
    print(json.dumps({"plan_path": out_path, "warnings": plan_obj.warnings}, indent=2))
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    with open(args.plan, encoding="utf-8") as f:
        plan_obj = Plan.from_dict(json.load(f))
    agent = ExecutionAgent(
        cfg, work_dir=args.work_dir, log_to_console=True,
    )
    trace = agent.run(plan_obj)
    print(json.dumps({
        "success": trace.success,
        "failed_step": trace.failed_step,
        "error_context": trace.error_context,
        "trace_path": os.path.join(args.work_dir or ".", "trace.json"),
    }, indent=2))
    return 0 if trace.success else 3


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    llm = _make_llm(cfg)
    tools = _resolve_tools(cfg, prefer="source")
    with open(args.plan, encoding="utf-8") as f:
        plan_obj = Plan.from_dict(json.load(f))
    with open(args.trace, encoding="utf-8") as f:
        trace_data = json.load(f)
    trace = ExecutionTrace(
        **{k: v for k, v in trace_data.items() if k in ExecutionTrace.__dataclass_fields__}
    )
    verifier = VerifierAgent(cfg, tools=tools, llm=llm)
    report = verifier.verify(plan_obj, trace)
    out_path = args.out or "verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(json.dumps({
        "passed": report.passed,
        "verification_path": out_path,
        "failures": [c.to_dict() for c in report.failures],
    }, indent=2))
    return 0 if report.passed else 4


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    llm = _make_llm(cfg)
    task = _read_task(args)
    orch = Orchestrator(cfg, llm=llm)
    result = orch.run(
        task,
        run_id=args.run_id,
        work_dir=args.work_dir,
        stream=True,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.success else 5


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Read-only end-to-end smoke test against a known-good stub task.

    Picks a single yfinance tool that always works (get_stock_info for
    AAPL), plans -> executes -> verifies, and prints a one-line summary."""
    cfg = _load_config(args)
    llm = _make_llm(cfg)
    tools = _resolve_tools(cfg, prefer="source")
    orch = Orchestrator(cfg, llm=llm, tools=tools)
    task = (
        "Use get_stock_info to fetch the current stock info for "
        "Apple (AAPL) and return the sector field."
    )
    work_dir = args.work_dir or "runs/selfcheck"
    result = orch.run(task, run_id="selfcheck", work_dir=work_dir)
    print(json.dumps({
        "success": result.success,
        "attempts": [a.n for a in result.attempts],
        "work_dir": result.work_dir,
    }, indent=2))
    return 0 if result.success else 6


def cmd_toolsource(args: argparse.Namespace) -> int:
    """Precompute the per-tool source KB (Python AST -> dict)."""
    from .toolsource import extract_tools_from_python_file
    if not args.source:
        print("--source is required", file=sys.stderr)
        return 2
    tools = extract_tools_from_python_file(args.source)
    out = [tool_to_dict(t) for t in tools]
    if args.out:
        save_catalog(tools, args.out)
        print(f"wrote {len(out)} tool(s) to {args.out}", file=sys.stderr)
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _read_task(args: argparse.Namespace) -> str:
    if args.task:
        return args.task
    if args.task_file:
        with open(args.task_file, encoding="utf-8") as f:
            return f.read().strip()
    raise SystemExit("either --task or --task-file is required")


def _make_llm(cfg: AgenticConfig) -> LLMClient:
    return LLMClient.from_env(env_var=cfg.env_anthropic_api_key)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="finance_agenticmcpe",
        description="Agentic workflow for the yahoo_finance MCP server.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cfg = sub.add_parser("config", help="print the resolved configuration")
    _add_common_flags(p_cfg)
    p_cfg.set_defaults(func=cmd_config)

    p_cat = sub.add_parser("catalog", help="spawn the server and dump its catalog")
    _add_common_flags(p_cat)
    p_cat.add_argument(
        "--from-source",
        action="store_true",
        help="extract the catalog from the Python source instead of spawning the server",
    )
    p_cat.add_argument(
        "--out",
        default=None,
        help="path to write the catalog JSON (default: stdout only)",
    )
    p_cat.set_defaults(func=cmd_catalog)

    p_plan = sub.add_parser("plan", help="plan a single task")
    _add_common_flags(p_plan)
    p_plan.add_argument("--task", default=None, help="the task prompt")
    p_plan.add_argument("--task-file", default=None, help="path to a file with the task prompt")
    p_plan.add_argument("--out", default=None, help="path to write plan.json (default: ./plan.json)")
    p_plan.add_argument("--from-source", action="store_true")
    p_plan.add_argument("--warnings", action="store_true", help="print planner warnings to stderr")
    p_plan.set_defaults(func=cmd_plan)

    p_exec = sub.add_parser("exec", help="execute a plan.json")
    _add_common_flags(p_exec)
    p_exec.add_argument("--plan", required=True, help="path to plan.json")
    p_exec.add_argument(
        "--work-dir", default=None, help="directory to write trace.json into (default: cwd)",
    )
    p_exec.set_defaults(func=cmd_exec)

    p_ver = sub.add_parser("verify", help="verify a plan + trace pair")
    _add_common_flags(p_ver)
    p_ver.add_argument("--plan", required=True)
    p_ver.add_argument("--trace", required=True)
    p_ver.add_argument("--out", default=None)
    p_ver.set_defaults(func=cmd_verify)

    p_run = sub.add_parser("run", help="plan + exec + verify with bounded replan")
    _add_common_flags(p_run)
    p_run.add_argument("--task", default=None)
    p_run.add_argument("--task-file", default=None)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--work-dir", default=None)
    p_run.set_defaults(func=cmd_run)

    p_sc = sub.add_parser("selfcheck", help="read-only end-to-end smoke test")
    _add_common_flags(p_sc)
    p_sc.add_argument("--work-dir", default=None)
    p_sc.set_defaults(func=cmd_selfcheck)

    p_ts = sub.add_parser("toolsource", help="extract tool catalog from a Python source file")
    _add_common_flags(p_ts)
    p_ts.add_argument("--source", required=True)
    p_ts.add_argument("--out", default=None)
    p_ts.set_defaults(func=cmd_toolsource)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


__all__ = ["main", "build_parser"]