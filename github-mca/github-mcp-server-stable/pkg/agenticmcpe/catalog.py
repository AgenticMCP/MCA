"""Tool-catalog loading + Go source mapping, shared by planner and verifier.

* :func:`load_catalog` spins up the Go server (read-only, dummy token is fine —
  ``tools/list`` never touches the GitHub API) and returns the real toolset.
* :func:`condense_catalog` produces a compact, token-cheap description for the
  planner prompt (name, one-line doc, required params + types).
* :func:`locate_tool_source` greps ``pkg/github`` for a tool's Go implementation
  and unit test, giving the verifier ground truth about a tool's behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pkg.mcp_wrapper import MCPClient, Tool

from .config import Settings

# Where catalog snapshots are cached so we don't respawn the server every plan.
_CATALOG_CACHE = "catalog.json"


def load_catalog(settings: Settings, *, toolsets: str = "all", use_cache: bool = True) -> list[Tool]:
    """Return the live tool catalog. Caches to ``work_dir/catalog.json``."""
    cache = settings.work_dir / _CATALOG_CACHE
    if use_cache and cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [Tool.from_mcp(t) for t in raw]

    # A token is required to *start* the server; tools/list needs no real auth.
    token = settings.tokens.tokens[0] if settings.tokens.available else "dummy-list-only"
    with MCPClient(
        token=token,
        binary=settings.binary_path,
        toolsets=[s for s in toolsets.split(",") if s] if toolsets else None,
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


def locate_tool_source(repo_root: Path, tool_name: str, *, max_chars: int = 2400) -> dict[str, Any]:
    """Find a tool's Go implementation + unit test by grepping for its name
    literal in pkg/github. Returns paths and a capped source excerpt."""
    gh_dir = repo_root / "pkg" / "github"
    needle = f'"{tool_name}"'
    impl_file: Path | None = None
    for go in sorted(gh_dir.glob("*.go")):
        if go.name.endswith("_test.go"):
            continue
        try:
            text = go.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if needle in text or f"mcp.NewTool({needle}" in text:
            impl_file = go
            break

    result: dict[str, Any] = {"tool": tool_name, "impl_file": None, "test_file": None,
                              "impl_excerpt": "", "test_excerpt": ""}
    if impl_file is None:
        return result

    result["impl_file"] = str(impl_file.relative_to(repo_root))
    impl_text = impl_file.read_text(encoding="utf-8")
    result["impl_excerpt"] = _excerpt_around(impl_text, tool_name, max_chars)

    test_file = impl_file.with_name(impl_file.stem + "_test.go")
    if test_file.is_file():
        result["test_file"] = str(test_file.relative_to(repo_root))
        test_text = test_file.read_text(encoding="utf-8")
        result["test_excerpt"] = _excerpt_around(test_text, tool_name, max_chars)
    return result


def _excerpt_around(text: str, needle: str, max_chars: int) -> str:
    """Return up to ``max_chars`` of text centred on the first mention of
    ``needle`` (snapped to line boundaries)."""
    idx = text.find(f'"{needle}"')
    if idx == -1:
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


__all__ = ["load_catalog", "condense_catalog", "locate_tool_source"]
