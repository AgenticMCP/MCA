"""Command-line interface for the github-mcp-server Python wrapper.

Mirrors the ergonomics of cmd/mcpcurl but speaks the protocol from Python::

    python -m pkg.mcp_wrapper list
    python -m pkg.mcp_wrapper schema get_issue
    python -m pkg.mcp_wrapper call get_issue --json '{"owner":"octocat","repo":"hello-world","issue_number":1}'
    python -m pkg.mcp_wrapper call get_issue --arg owner=octocat --arg repo=hello-world --arg issue_number=1
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .client import MCPClient
from .sequence import load_steps_from_json
from .types import MCPError, MCPProtocolError, MCPToolError, MCPTransportError


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--binary",
        help="Path to the github-mcp-server binary "
        "(default: the binary bundled in pkg/mcp_wrapper/bin/).",
    )
    p.add_argument(
        "--token",
        help="GitHub personal access token. Falls back to "
        "GITHUB_PERSONAL_ACCESS_TOKEN / GITHUB_TOKEN env vars.",
    )
    p.add_argument(
        "--toolsets",
        help="Comma-separated toolset IDs to enable (e.g. issues,pull_requests). "
        "Omit to use server defaults; pass 'all' for everything.",
    )
    p.add_argument(
        "--tools",
        help="Comma-separated explicit tool names to enable (overrides toolsets).",
    )
    p.add_argument(
        "--exclude-tools",
        help="Comma-separated tool names to exclude from the default set.",
    )
    p.add_argument(
        "--read-only",
        action="store_true",
        help="Disable every write-capable tool.",
    )
    p.add_argument(
        "--host",
        help="Override GitHub host (for GitHub Enterprise).",
    )
    p.add_argument(
        "--log-file",
        help="Forwarded to the server as --log-file.",
    )
    p.add_argument(
        "--enable-command-logging",
        action="store_true",
        help="Server-side: log every JSON-RPC message it sees.",
    )


def _client_from_args(args: argparse.Namespace) -> MCPClient:
    kwargs: dict[str, Any] = {
        "binary": args.binary,
        "token": args.token,
        "read_only": args.read_only,
        "host": args.host,
        "log_file": args.log_file,
        "enable_command_logging": args.enable_command_logging,
    }
    if args.toolsets:
        kwargs["toolsets"] = [s for s in args.toolsets.split(",") if s]
    if args.tools:
        kwargs["tools"] = [s for s in args.tools.split(",") if s]
    if args.exclude_tools:
        kwargs["exclude_tools"] = [s for s in args.exclude_tools.split(",") if s]
    return MCPClient(**kwargs)


def _parse_kv_args(items: list[str]) -> dict[str, Any]:
    """Parse a list of `key=value` strings. Values are JSON-decoded when
    possible (so `--arg n=1` becomes int, `--arg ok=true` becomes bool,
    `--arg name=octocat` stays a string), and fall back to raw strings."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--arg expected key=value, got: {item!r}")
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _print_json(obj: Any, *, compact: bool) -> None:
    if compact:
        json.dump(obj, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")


def cmd_list(args: argparse.Namespace) -> int:
    with _client_from_args(args) as gh:
        tools = gh.list_tools()
        if args.json:
            _print_json(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                        "annotations": t.annotations,
                    }
                    for t in tools
                ],
                compact=args.compact,
            )
        else:
            for t in tools:
                desc = (t.description or "").splitlines()[0] if t.description else ""
                print(f"{t.name:42s} {desc[:120]}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    with _client_from_args(args) as gh:
        tool = gh.get_tool(args.tool)
        if tool is None:
            print(f"unknown tool: {args.tool}", file=sys.stderr)
            return 2
        _print_json(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "annotations": tool.annotations,
            },
            compact=args.compact,
        )
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    steps = load_steps_from_json(args.file)
    with _client_from_args(args) as gh:
        result = gh.execute_sequence(steps, on_error=args.on_error)
    out = result.to_dict()
    _print_json(out, compact=args.compact)
    return 0 if result.success else 1


def cmd_call(args: argparse.Namespace) -> int:
    arguments: dict[str, Any] = {}
    if args.json:
        try:
            arguments = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"--json is not valid JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(arguments, dict):
            print("--json must decode to an object", file=sys.stderr)
            return 2
    if args.arg:
        arguments.update(_parse_kv_args(args.arg))

    with _client_from_args(args) as gh:
        try:
            result = gh.call(args.tool, arguments)
        except MCPToolError as e:
            print(json.dumps({"error": "tool_error", "tool": e.tool, "text": e.text}, indent=2))
            return 1

    if args.raw:
        _print_json(result.content, compact=args.compact)
    else:
        decoded = result.data
        _print_json(
            {"tool": result.tool, "is_error": result.is_error, "data": decoded},
            compact=args.compact,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pkg.mcp_wrapper",
        description="Local Python driver for the Go github-mcp-server stdio binary.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # `list`
    p_list = sub.add_parser("list", help="List tools advertised by the server.")
    _add_common_flags(p_list)
    p_list.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    p_list.add_argument("--compact", action="store_true", help="Compact JSON (one line).")
    p_list.set_defaults(func=cmd_list)

    # `schema`
    p_schema = sub.add_parser("schema", help="Show a tool's input schema + metadata.")
    _add_common_flags(p_schema)
    p_schema.add_argument("tool", help="Tool name (e.g. get_issue).")
    p_schema.add_argument("--compact", action="store_true", help="Compact JSON.")
    p_schema.set_defaults(func=cmd_schema)

    # `call`
    p_call = sub.add_parser("call", help="Invoke a tool.")
    _add_common_flags(p_call)
    p_call.add_argument("tool", help="Tool name (e.g. get_issue).")
    p_call.add_argument(
        "--json", help="Tool arguments as a JSON object (overridden field-by-field by --arg)."
    )
    p_call.add_argument(
        "--arg",
        action="append",
        metavar="KEY=VALUE",
        help="Tool argument as key=value. Repeat for multiple. Values are JSON-decoded when possible.",
    )
    p_call.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw MCP content array instead of decoding the structured payload.",
    )
    p_call.add_argument("--compact", action="store_true", help="Compact JSON output.")
    p_call.set_defaults(func=cmd_call)

    # `sequence`
    p_seq = sub.add_parser(
        "sequence",
        help="Run a chain of tool calls from a JSON file, with binding resolution.",
    )
    _add_common_flags(p_seq)
    p_seq.add_argument(
        "file",
        help="Path to a JSON file: a list of {tool, arguments, id?, expect_output?, "
        "validate_input?, description?} step objects.",
    )
    p_seq.add_argument(
        "--on-error",
        choices=["stop", "continue"],
        default="stop",
        help="On step failure: stop the sequence (default) or continue.",
    )
    p_seq.add_argument("--compact", action="store_true", help="Compact JSON output.")
    p_seq.set_defaults(func=cmd_sequence)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MCPProtocolError as e:
        print(
            json.dumps(
                {"error": "protocol", "code": e.code, "message": e.message, "data": e.data},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    except MCPTransportError as e:
        print(f"transport error: {e}", file=sys.stderr)
        return 1
    except MCPError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
