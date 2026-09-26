"""Tool-catalog loading + Python source mapping, shared by planner and verifier.

* :func:`load_catalog` spins up the postgres-mcp server (a placeholder
  DATABASE_URI is fine — the server starts and serves ``tools/list`` without a
  reachable database) and returns the real toolset.
* :func:`condense_catalog` produces a compact, token-cheap description for the
  planner prompt (name, one-line doc, required params + types).
* :func:`locate_tool_source` greps ``src/postgres_mcp`` for a tool's Python
  implementation and unit test, giving the verifier ground truth about a
  tool's behaviour (fallback when the prebuilt tool_sources KB has no entry).

The catalog is loaded in UNRESTRICTED mode so ``execute_sql`` is advertised as
the write-capable tool ("Execute any SQL query", no readOnlyHint) — the
read-only classification every safety gate consumes comes straight from the
server's own ``readOnlyHint`` annotations, which postgres-mcp publishes on all
of its read tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pkg.mcp_wrapper import MCPClient, Tool

from .config import Settings

# Where catalog snapshots are cached so we don't respawn the server every plan.
_CATALOG_CACHE = "catalog.json"

# A syntactically valid URI that never resolves: enough for the server to boot
# and answer tools/list when no real database is configured yet.
_PLACEHOLDER_URI = "postgresql://catalog:list-only@127.0.0.1:1/postgres"

# Manual read-only classification, used when the server publishes no
# annotations (the released docker image / older postgres-mcp versions strip
# them). Mirrors the server source: every tool except execute_sql is declared
# with readOnlyHint=True; execute_sql is the write channel in the unrestricted
# mode the catalog is loaded in. The taskgen safety gates and the read-only
# verification client depend on this classification being right.
READ_ONLY_FALLBACK: dict[str, bool] = {
    "list_schemas": True,
    "list_objects": True,
    "get_object_details": True,
    "explain_query": True,
    "analyze_workload_indexes": True,
    "analyze_query_indexes": True,
    "analyze_db_health": True,
    "get_top_queries": True,
    "execute_sql": False,
}


def load_catalog(settings: Settings, *, use_cache: bool = True) -> list[Tool]:
    """Return the live tool catalog. Caches to ``work_dir/catalog.json``."""
    cache = settings.work_dir / _CATALOG_CACHE
    if use_cache and cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [Tool.from_mcp(t) for t in raw]

    # The server refuses to start with NO uri, but starts fine with an
    # unreachable one — tools/list needs no live database.
    uri = settings.database.uri or _PLACEHOLDER_URI
    with MCPClient(
        database_uri=uri,
        server_cmd=settings.server_cmd or None,
        access_mode="unrestricted",
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
        # FastMCP wraps optional params in anyOf; surface the first concrete
        # type so the planner's local type-check has something to work with.
        typ = sub.get("type")
        if typ is None and isinstance(sub.get("anyOf"), list):
            for alt in sub["anyOf"]:
                if isinstance(alt, dict) and alt.get("type") not in (None, "null"):
                    typ = alt["type"]
                    break
        out.append(
            {
                "name": name,
                "type": typ or "any",
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
        if "readOnlyHint" in (t.annotations or {}):
            read_only = bool(t.annotations.get("readOnlyHint"))
        else:
            read_only = READ_ONLY_FALLBACK.get(t.name, False)
        condensed.append(
            {
                "tool": t.name,
                "summary": doc[0][:160] if doc else "",
                "params": _required_params(t.input_schema),
                "read_only": read_only,
            }
        )
    return condensed


def locate_tool_source(repo_root: Path, tool_name: str, *, max_chars: int = 2400) -> dict[str, Any]:
    """Find a tool's Python implementation + unit test by grepping for its name
    in src/postgres_mcp and tests/. Returns paths and a capped source excerpt."""
    src_dir = repo_root / "src" / "postgres_mcp"
    needle = f"def {tool_name}"
    impl_file: Path | None = None
    for py in sorted(src_dir.rglob("*.py")):
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if needle in text or f'"{tool_name}"' in text:
            impl_file = py
            break

    result: dict[str, Any] = {"tool": tool_name, "impl_file": None, "test_file": None,
                              "impl_excerpt": "", "test_excerpt": ""}
    if impl_file is None:
        return result

    result["impl_file"] = str(impl_file.relative_to(repo_root))
    impl_text = impl_file.read_text(encoding="utf-8")
    result["impl_excerpt"] = _excerpt_around(impl_text, needle, max_chars)

    tests_dir = repo_root / "tests"
    for py in sorted(tests_dir.rglob("*.py")) if tests_dir.is_dir() else []:
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if tool_name in text:
            result["test_file"] = str(py.relative_to(repo_root))
            result["test_excerpt"] = _excerpt_around(text, tool_name, max_chars)
            break
    return result


def _excerpt_around(text: str, needle: str, max_chars: int) -> str:
    """Return up to ``max_chars`` of text centred on the first mention of
    ``needle`` (snapped to line boundaries)."""
    idx = text.find(needle)
    if idx == -1:
        return text[:max_chars]
    half = max_chars // 2
    start = max(0, idx - half)
    end = min(len(text), idx + half)
    # snap to line boundaries for readability
    start = text.rfind("\n", 0, start) + 1
    nl = text.find("\n", end)
    end = nl if nl != -1 else end
    prefix = "" if start == 0 else "...\n"
    suffix = "" if end == len(text) else "\n..."
    return prefix + text[start:end] + suffix


__all__ = ["load_catalog", "condense_catalog", "locate_tool_source",
           "READ_ONLY_FALLBACK"]
