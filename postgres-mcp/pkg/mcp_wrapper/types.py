"""Data types and exceptions for the postgres-mcp Python wrapper.

Engine module — identical to the github-mcp-server original except for ONE
adapter shim in :meth:`ToolResult.data`: postgres-mcp formats every result as
``str(python_object)`` (Python repr with single quotes, not JSON), so the
best-effort decode falls back to ``ast.literal_eval`` after ``json.loads``
fails. That is what makes structured bindings ("$s0[0].schema_name") work
against this server.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any


class MCPError(Exception):
    """Base class for any failure in the wrapper / server."""


class MCPTransportError(MCPError):
    """Subprocess / IO / framing failure (server crashed, EOF, bad JSON, ...)."""


class MCPProtocolError(MCPError):
    """Server returned a JSON-RPC `error` object for a request we sent."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPToolError(MCPError):
    """Tool-level failure: the server answered with `isError: true`, or (the
    postgres-mcp convention) with a normal text block starting "Error:"."""

    def __init__(self, tool: str, text: str, raw: dict[str, Any]):
        super().__init__(f"tool {tool!r} returned an error: {text}")
        self.tool = tool
        self.text = text
        self.raw = raw


@dataclass(frozen=True)
class Tool:
    """A tool advertised by the MCP server via tools/list."""

    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mcp(cls, obj: dict[str, Any]) -> "Tool":
        return cls(
            name=obj["name"],
            description=obj.get("description", ""),
            input_schema=obj.get("inputSchema") or {},
            annotations=obj.get("annotations") or {},
        )


def parse_result_text(raw: str) -> Any:
    """Decode a postgres-mcp text payload into structured data when possible.

    The server emits ``str(python_object)`` — e.g.
    ``[{'schema_name': 'public', ...}]`` — which is not JSON. Order of
    attempts: JSON (future-proof), then a safe Python-literal parse
    (``ast.literal_eval`` evaluates literals only, never code). Values whose
    repr is not a pure literal (datetime.datetime(...), Decimal(...)) make the
    whole parse fail; the raw text is returned unchanged in that case, so
    nothing is ever lost — only left undecoded.
    """
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    stripped = raw.strip()
    if stripped[:1] in "[{(" or stripped in ("True", "False", "None"):
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
    return raw


@dataclass
class ToolResult:
    """Decoded result of a single tools/call invocation.

    The MCP `content` array contains one or more entries. postgres-mcp tools
    emit a single text block whose body is the ``str()`` of a Python object
    (lists/dicts of rows) or free-form text (explain plans, health reports).
    We expose the original `content`, the joined raw text, and a best-effort
    structured decode.
    """

    tool: str
    content: list[dict[str, Any]]
    is_error: bool = False
    structured: Any = None

    @property
    def text(self) -> str:
        """Concatenated text from all content blocks of type 'text'."""
        return "".join(c.get("text", "") for c in self.content if c.get("type") == "text")

    @property
    def resource_texts(self) -> list[str]:
        """Text bodies of embedded resource blocks (kept for MCP-spec
        completeness; postgres-mcp does not currently emit them)."""
        out: list[str] = []
        for c in self.content:
            if c.get("type") == "resource":
                res = c.get("resource") or {}
                if isinstance(res.get("text"), str):
                    out.append(res["text"])
        return out

    @property
    def data(self) -> Any:
        """Best-effort decode of the result payload.

        Order: structured content, then JSON/Python-literal-decoded text
        (postgres-mcp formats results as ``str(python_object)``), then embedded
        resource content, then the raw text, then None.
        """
        if self.structured is not None:
            return self.structured
        raw = self.text
        if raw:
            return parse_result_text(raw)
        resources = self.resource_texts
        if resources:
            return resources[0] if len(resources) == 1 else "\n".join(resources)
        return raw if raw else None

    @classmethod
    def from_mcp(cls, tool: str, result: dict[str, Any]) -> "ToolResult":
        return cls(
            tool=tool,
            content=result.get("content") or [],
            is_error=bool(result.get("isError", False)),
            structured=result.get("structuredContent"),
        )
