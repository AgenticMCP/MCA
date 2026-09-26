"""Data types and exceptions for the playwright-mcp Python wrapper."""

from __future__ import annotations

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
    """Server returned `isError: true` in a tools/call result (a tool-level failure)."""

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


@dataclass
class ToolResult:
    """Decoded result of a single tools/call invocation.

    playwright-mcp tools emit a single markdown-ish text block ("### Ran
    Playwright code ... ### Page ... ### Snapshot ..."), not JSON. We expose
    the original `content`, the joined raw text, and a best-effort decode
    (which for this server is almost always the raw text string).
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
        """Text bodies of embedded resource blocks, when a tool returns one."""
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

        Order: structured content, then JSON-decoded text, then embedded
        resource content, then the raw text, then None.
        """
        if self.structured is not None:
            return self.structured
        raw = self.text
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
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
