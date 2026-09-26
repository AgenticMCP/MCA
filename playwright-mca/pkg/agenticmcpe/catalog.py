"""Tool-catalog loading + schema grounding, shared by planner and verifier.

* :func:`load_catalog` spins up the playwright-mcp server (no credentials —
  ``tools/list`` opens no browser page) and returns the real toolset.
* :func:`condense_catalog` produces a compact, token-cheap description for the
  planner prompt (name, one-line doc, required params + types). playwright-mcp
  publishes ``annotations.readOnlyHint`` on every tool, so the read-only
  classification comes straight from the server.
* :func:`tool_grounding` returns a tool's full catalog entry (description +
  inputSchema) for the verifier's dynamic-evaluator prompt. The GitHub port
  grounded on the server's Go source; playwright-mcp's implementation lives in
  the bundled ``playwright-core`` package, so the catalog schema + live
  re-query is the grounding source here (PORTING.md seam 7, "drop" option).
"""

from __future__ import annotations

import json
from typing import Any

from pkg.mcp_wrapper import MCPClient, Tool

from .config import Settings

# Where catalog snapshots are cached so we don't respawn the server every plan.
_CATALOG_CACHE = "catalog.json"


def load_catalog(settings: Settings, *, use_cache: bool = True) -> list[Tool]:
    """Return the live tool catalog. Caches to ``work_dir/catalog.json``."""
    cache = settings.work_dir / _CATALOG_CACHE
    if use_cache and cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [Tool.from_mcp(t) for t in raw]

    with MCPClient(
        command=settings.server_command,
        headless=settings.headless,
        isolated=settings.isolated,
        browser=settings.browser,
    ) as client:
        tools = client.list_tools()

    settings.ensure_work_dir()
    cache.write_text(
        json.dumps(
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                    "annotations": t.annotations,
                }
                for t in tools
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return tools


def _required_params(schema: dict[str, Any]) -> list[dict[str, Any]]:
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    out = []
    for name, sub in props.items():
        out.append(
            {
                "name": name,
                "type": sub.get("type", "any"),
                "required": name in required,
                "enum": sub.get("enum"),
                "desc": (sub.get("description") or "")[:120],
            }
        )
    # required first, then alphabetical
    out.sort(key=lambda p: (not p["required"], p["name"]))
    return out


def condense_catalog(tools: list[Tool]) -> list[dict[str, Any]]:
    """Compact catalog for the planner prompt."""
    condensed = []
    for t in tools:
        doc = (t.description or "").strip().splitlines()
        condensed.append(
            {
                "tool": t.name,
                "summary": doc[0][:160] if doc else "",
                "params": _required_params(t.input_schema),
                "read_only": bool(t.annotations.get("readOnlyHint")),
            }
        )
    return condensed


def tool_grounding(settings: Settings, tool_name: str, *,
                   max_chars: int = 2400) -> dict[str, Any]:
    """One tool's grounding for the dynamic-evaluator prompt: its catalog
    description and inputSchema, read from the cached catalog.json."""
    cache = settings.work_dir / _CATALOG_CACHE
    result: dict[str, Any] = {"tool": tool_name, "description": "", "inputSchema": {}}
    if not cache.is_file():
        return result
    for t in json.loads(cache.read_text(encoding="utf-8")):
        if t.get("name") == tool_name:
            result["description"] = (t.get("description") or "")[:max_chars]
            schema = json.dumps(t.get("inputSchema") or {}, separators=(",", ":"))
            result["inputSchema"] = json.loads(schema[:max_chars]) \
                if len(schema) <= max_chars else {"_truncated": schema[:max_chars]}
            break
    return result


__all__ = ["load_catalog", "condense_catalog", "tool_grounding"]
