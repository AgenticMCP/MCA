"""Minimal CLI for the playwright-mcp wrapper.

    python -m pkg.mcp_wrapper list-tools
    python -m pkg.mcp_wrapper call browser_navigate --args '{"url": "https://example.com"}'
    python -m pkg.mcp_wrapper seq steps.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .client import MCPClient, shell_escape_argv
from .sequence import load_steps_from_json


def _client(args: argparse.Namespace) -> MCPClient:
    return MCPClient(
        headless=not args.headed,
        isolated=not args.profile,
        browser=args.browser,
    )


def cmd_list_tools(args: argparse.Namespace) -> int:
    with _client(args) as pw:
        for t in pw.list_tools():
            ro = "ro" if t.annotations.get("readOnlyHint") else "  "
            print(f"{t.name:32s} [{ro}] {t.description[:90]}")
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    arguments = json.loads(args.args) if args.args else {}
    with _client(args) as pw:
        result = pw.call(args.tool, arguments)
        print(result.text)
    return 0


def cmd_seq(args: argparse.Namespace) -> int:
    steps = load_steps_from_json(args.steps)
    with _client(args) as pw:
        seq = pw.execute_sequence(steps, on_error=args.on_error)
    print(json.dumps(seq.to_dict(), indent=2, default=str))
    return 0 if seq.success else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pkg.mcp_wrapper",
                                description="Thin CLI over the playwright-mcp stdio wrapper.")
    p.add_argument("--headed", action="store_true", help="Run the browser headed.")
    p.add_argument("--profile", action="store_true",
                   help="Use a persistent profile instead of --isolated.")
    p.add_argument("--browser", help="chrome|firefox|webkit|msedge (default: bundled chromium)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list-tools", help="List advertised tools.")
    pl.set_defaults(func=cmd_list_tools)

    pc = sub.add_parser("call", help="Invoke one tool.")
    pc.add_argument("tool")
    pc.add_argument("--args", help="JSON object of tool arguments.")
    pc.set_defaults(func=cmd_call)

    ps = sub.add_parser("seq", help="Run a JSON sequence file with bindings.")
    ps.add_argument("steps")
    ps.add_argument("--on-error", choices=["stop", "continue"], default="stop")
    ps.set_defaults(func=cmd_seq)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
