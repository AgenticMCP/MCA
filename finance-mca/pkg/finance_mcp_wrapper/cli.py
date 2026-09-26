"""Command-line interface for the finance MCP wrapper.

Server-agnostic. Subcommands:

* ``catalog`` — spawn the server, dump tools/list as JSON.
* ``call``    — invoke a single tool by name and print the decoded result.
* ``run``     — execute a step sequence from a JSON file.

Common flags (all subcommands)::

    --command PYTHON           # default: sys.executable
    --arg foo                  # repeatable; passed to the server
    --env KEY=VAL              # repeatable; merged into child env
    --cwd PATH                 # working directory for the child process
    --api-key SECRET           # injected as YFINANCE_API_KEY by default

Examples::

    # Inspect the yfinance catalog
    python -m pkg.finance_mcp_wrapper catalog \\
        --arg -m --arg servers.yahoo_finance --arg --transport --arg stdio

    # Call a single tool directly
    python -m pkg.finance_mcp_wrapper call get_historical_stock_prices \\
        --arg -m --arg servers.yahoo_finance --arg --transport --arg stdio \\
        --input '{"ticker":"AAPL","start_date":"2024-01-01","end_date":"2024-01-10"}'

    # Execute a step sequence
    python -m pkg.finance_mcp_wrapper run steps.json \\
        --arg -m --arg servers.yahoo_finance --arg --transport --arg stdio
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .client import MCPClient
from .sequence import execute_sequence, load_steps_from_json


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--command",
        default=None,
        help="executable to spawn (default: sys.executable)",
    )
    p.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="ARG",
        help="argument to pass to the server (repeatable)",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="extra environment variable KEY=VAL (repeatable)",
    )
    p.add_argument(
        "--cwd",
        default=None,
        help="working directory for the child process",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key injected as YFINANCE_API_KEY (default: not set)",
    )


def _parse_env(entries: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--env expects KEY=VAL, got {entry!r}")
        k, _, v = entry.partition("=")
        out[k] = v
    return out


def _build_client(args: argparse.Namespace) -> MCPClient:
    return MCPClient(
        command=args.command,
        args=list(args.arg),
        env=_parse_env(args.env) or None,
        cwd=args.cwd,
        api_key=args.api_key,
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_catalog(args: argparse.Namespace) -> int:
    with _build_client(args) as mcp:
        tools = mcp.list_tools(refresh=True)
        out = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "annotations": t.annotations,
            }
            for t in tools
        ]
        json.dump(out, sys.stdout, indent=2, ensure_ascii=False, default=str)
        sys.stdout.write("\n")
        return 0


def cmd_call(args: argparse.Namespace) -> int:
    if not args.tool:
        raise SystemExit("`call` requires a tool name")
    try:
        arguments = json.loads(args.input) if args.input else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"--input is not valid JSON: {e}")
    if not isinstance(arguments, dict):
        raise SystemExit(f"--input must decode to an object, got {type(arguments).__name__}")
    with _build_client(args) as mcp:
        result = mcp.call(args.tool, arguments)
        json.dump(
            {
                "tool": result.tool,
                "is_error": result.is_error,
                "text": result.text,
                "data": result.data,
            },
            sys.stdout,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        sys.stdout.write("\n")
    return 0 if not result.is_error else 1


def cmd_run(args: argparse.Namespace) -> int:
    if not args.steps:
        raise SystemExit("`run` requires a path to a steps JSON file")
    steps = load_steps_from_json(args.steps)
    on_error = "continue" if args.continue_on_error else "stop"
    with _build_client(args) as mcp:
        result = execute_sequence(mcp, steps, on_error=on_error)
        json.dump(result.to_dict(), sys.stdout, indent=2, ensure_ascii=False, default=str)
        sys.stdout.write("\n")
    return 0 if result.success else 2


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="finance_mcp_wrapper",
        description="Generic stdio JSON-RPC client for an MCP server.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cat = sub.add_parser("catalog", help="print the server's tool catalog as JSON")
    _add_common_flags(p_cat)
    p_cat.set_defaults(func=cmd_catalog)

    p_call = sub.add_parser("call", help="invoke one tool and print its decoded result")
    p_call.add_argument("tool", help="tool name (e.g. get_historical_stock_prices)")
    p_call.add_argument(
        "--input",
        default="",
        help="JSON object of arguments (e.g. '{\"ticker\":\"AAPL\"}')",
    )
    _add_common_flags(p_call)
    p_call.set_defaults(func=cmd_call)

    p_run = sub.add_parser("run", help="execute a step sequence from a JSON file")
    p_run.add_argument("steps", help="path to a JSON array of steps")
    p_run.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep going after a step fails (default: stop at first failure)",
    )
    _add_common_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


__all__ = ["main", "build_parser"]