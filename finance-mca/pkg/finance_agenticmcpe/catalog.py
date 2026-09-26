"""Tool catalog: live discovery + on-disk cache.

Mirrors the github-mcp-server helper but with no token bootstrap (the
yfinance server doesn't need one). The catalog is the source of truth for
planner prompts, executor input validation, and verifier output checking.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from pkg.finance_mcp_wrapper.client import MCPClient
from pkg.finance_mcp_wrapper.types import Tool

from .toolsource import extract_tools_from_python_file

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def tool_to_dict(t: Tool) -> dict[str, Any]:
    """Plain-dict representation of a Tool, suitable for JSON dumping."""
    return {
        "name": t.name,
        "description": t.description,
        "inputSchema": t.input_schema,
        "annotations": t.annotations,
    }


def dict_to_tool(d: Mapping[str, Any]) -> Tool:
    return Tool(
        name=d["name"],
        description=d.get("description", ""),
        input_schema=d.get("inputSchema") or {},
        annotations=d.get("annotations") or {},
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_live(
    *,
    command: str | None = None,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    api_key: str | None = None,
) -> list[Tool]:
    """Spawn the server once and dump tools/list. No cache."""
    cmd = command or sys.executable
    argv = list(args or ["-m", "servers.yahoo_finance", "--transport", "stdio"])
    with MCPClient(command=cmd, args=argv, cwd=cwd, env=dict(env or {}), api_key=api_key) as mcp:
        return mcp.list_tools(refresh=True)


def discover_from_source(source_path: str) -> list[Tool]:
    """Extract tool definitions directly from a Python MCP server source
    file — useful when running offline or in CI."""
    return extract_tools_from_python_file(source_path)


def discover(
    *,
    source_path: str | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    api_key: str | None = None,
    prefer: str = "live",
) -> list[Tool]:
    """Try live discovery first; fall back to source extraction if the
    subprocess fails or `prefer='source'`."""
    if prefer == "source":
        if not source_path:
            raise ValueError("prefer='source' requires source_path")
        return discover_from_source(source_path)
    try:
        return discover_live(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            api_key=api_key,
        )
    except Exception:
        if source_path:
            return discover_from_source(source_path)
        raise


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def save_catalog(tools: Iterable[Tool], path: str | os.PathLike[str]) -> int:
    """Write the catalog as a JSON array. Returns the number of tools
    written."""
    out = [tool_to_dict(t) for t in tools]
    Path(path).write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return len(out)


def load_catalog(path: str | os.PathLike[str]) -> list[Tool]:
    """Read a cached catalog from disk."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected top-level JSON array of tools, got {type(raw).__name__}")
    return [dict_to_tool(d) for d in raw]


__all__ = [
    "discover_live",
    "discover_from_source",
    "discover",
    "save_catalog",
    "load_catalog",
    "tool_to_dict",
    "dict_to_tool",
]