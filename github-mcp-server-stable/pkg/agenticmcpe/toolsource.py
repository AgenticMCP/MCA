"""Prebuilt tool-source knowledge base for the verifier.

``locate_tool_source`` (catalog.py) greps for a tool name at verify time and
clips an excerpt around the FIRST mention — which can land in an alias table or
registry instead of the handler, and routinely cuts off the response-shaping
code the dynamic evaluators most need. This module fixes that by precomputing,
once, a JSON file per tool that maps the tool to its REAL definition
function(s) and unit-test function(s), extracted whole:

* Implementation: every top-level Go function in ``pkg/github`` whose body
  contains ``Name: "<tool>"`` or ``NewTool("<tool>"`` — the actual definition
  site with schema, handler and result marshalling.
* Tests: top-level ``Test*`` functions in ``*_test.go`` referencing the tool
  literal, ranked so functions asserting on ``tool.Name`` (the tool's own
  contract test, with its mock fixtures) come first.

Build with ``python -m pkg.agenticmcpe toolsource``; output lands in
``pkg/agenticmcpe/tool_sources/<tool>.json`` plus an ``index.json``. The
verifier loads these via :func:`load_for_verifier` and falls back to the old
grep when a tool has no prebuilt entry.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .catalog import locate_tool_source

DEFAULT_KB_DIR = Path(__file__).resolve().parent / "tool_sources"

# Definition sites inside a function body. gofmt guarantees top-level funcs
# start at column 0 and close with a lone "}" at column 0.
_DEF_RE = re.compile(r'(?:Name:\s*|NewTool\(\s*)"([a-z0-9_]+)"')
_FUNC_RE = re.compile(r"^func\s+(?:\([^)]*\)\s+)?([A-Za-z0-9_]+)")


def _go_functions(text: str) -> list[dict[str, Any]]:
    """Split a gofmt'd Go file into its top-level functions."""
    lines = text.splitlines()
    funcs: list[dict[str, Any]] = []
    start: int | None = None
    name = ""
    for i, line in enumerate(lines):
        if start is None:
            m = _FUNC_RE.match(line)
            if m:
                start, name = i, m.group(1)
        elif line == "}":
            funcs.append({
                "func": name,
                "start_line": start + 1,
                "end_line": i + 1,
                "code": "\n".join(lines[start:i + 1]),
            })
            start = None
    return funcs


def _cap(code: str, max_chars: int) -> str:
    """Cap a chunk keeping head (schema/setup) and tail (results/asserts)."""
    if len(code) <= max_chars:
        return code
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return code[:head] + "\n// ... [truncated] ...\n" + code[-tail:]


def build_kb(
    repo_root: Path,
    tool_names: list[str],
    *,
    out_dir: Path | None = None,
    impl_cap: int = 9000,
    test_cap: int = 4500,
    tests_per_tool: int = 2,
) -> dict[str, Any]:
    """Scan ``pkg/github`` once and write one JSON per tool + index.json."""
    gh_dir = repo_root / "pkg" / "github"
    out = out_dir or DEFAULT_KB_DIR
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(tool_names)

    impl_by_tool: dict[str, list[dict[str, Any]]] = {}
    tests_by_tool: dict[str, list[dict[str, Any]]] = {}

    for go in sorted(gh_dir.glob("*.go")):
        try:
            text = go.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = str(go.relative_to(repo_root))
        is_test = go.name.endswith("_test.go")
        for fn in _go_functions(text):
            if is_test:
                if not fn["func"].startswith("Test"):
                    continue
                for tool in wanted:
                    if f'"{tool}"' not in fn["code"]:
                        continue
                    # the tool's own contract test asserts on tool.Name
                    score = 2 if re.search(
                        rf'"{tool}",\s*tool\.Name|tool\.Name,\s*"{tool}"',
                        fn["code"]) else 1
                    tests_by_tool.setdefault(tool, []).append(
                        {"file": rel, "score": score, **fn})
            else:
                for tool in set(_DEF_RE.findall(fn["code"])) & wanted:
                    impl_by_tool.setdefault(tool, []).append({"file": rel, **fn})

    index: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": _git_head(repo_root),
        "tools": {},
        "missing_impl": [],
        "missing_tests": [],
    }
    for tool in sorted(wanted):
        impls = impl_by_tool.get(tool, [])
        tests = sorted(tests_by_tool.get(tool, []),
                       key=lambda f: (-f["score"], f["file"], f["start_line"]))
        tests = tests[:tests_per_tool]
        entry = {
            "tool": tool,
            "impl": [
                {"file": f["file"], "func": f["func"],
                 "start_line": f["start_line"], "end_line": f["end_line"],
                 "code": _cap(f["code"], impl_cap // max(1, len(impls)))}
                for f in impls
            ],
            "tests": [
                {"file": f["file"], "func": f["func"],
                 "start_line": f["start_line"], "end_line": f["end_line"],
                 "code": _cap(f["code"], test_cap)}
                for f in tests
            ],
        }
        (out / f"{tool}.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        index["tools"][tool] = {
            "impl": [f"{f['file']}:{f['start_line']}" for f in impls],
            "tests": [f"{f['file']}:{f['start_line']}" for f in tests],
        }
        if not impls:
            index["missing_impl"].append(tool)
        if not tests:
            index["missing_tests"].append(tool)

    (out / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Verifier-facing loader
# ---------------------------------------------------------------------------

def load_for_verifier(repo_root: Path, tool: str, *, budget: int,
                      kb_dir: Path | None = None) -> dict[str, Any]:
    """Return one tool's grounding, shaped for the dynamic-evaluator prompt and
    fitted to ``budget`` chars (impl 60% / tests 40%). Falls back to the legacy
    grep when the prebuilt KB has no entry for the tool."""
    path = (kb_dir or DEFAULT_KB_DIR) / f"{tool}.json"
    if path.is_file():
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("impl"):
            impl_budget = int(budget * 0.6)
            test_budget = budget - impl_budget
            impl = "\n\n".join(f["code"] for f in entry["impl"])
            tests = "\n\n".join(f["code"] for f in entry["tests"])
            return {
                "tool": tool,
                "impl_file": ", ".join(sorted({f["file"] for f in entry["impl"]})),
                "test_file": ", ".join(sorted({f["file"] for f in entry["tests"]})) or None,
                "impl_excerpt": _cap(impl, impl_budget),
                "test_excerpt": _cap(tests, test_budget) if tests else "",
            }
    legacy = locate_tool_source(repo_root, tool, max_chars=budget // 2)
    return legacy


__all__ = ["DEFAULT_KB_DIR", "build_kb", "load_for_verifier"]
