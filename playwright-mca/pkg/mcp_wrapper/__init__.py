"""Python wrapper around the Node playwright-mcp stdio server.

Quick start::

    from pkg.mcp_wrapper import MCPClient

    with MCPClient() as pw:
        pw.call("browser_navigate", {"url": "https://example.com"})
        print(pw.call("browser_snapshot").text)

The JSON-RPC engine, binding resolution and schema validation are identical to
the github-mcp-server wrapper this package was ported from; only the spawn
(transport) and credential model differ — playwright-mcp is spawned as
``node cli.js`` (or any command) and needs no token.
"""

from .client import DEFAULT_COMMAND, MCPClient, REPO_ROOT, discover_tools
from .sequence import (
    BindingError,
    SequenceResult,
    Step,
    StepResult,
    execute_sequence,
    load_steps_from_json,
    resolve_bindings,
    validate_against_schema,
)
from .types import (
    MCPError,
    MCPProtocolError,
    MCPToolError,
    MCPTransportError,
    Tool,
    ToolResult,
)

__all__ = [
    "MCPClient",
    "discover_tools",
    "REPO_ROOT",
    "DEFAULT_COMMAND",
    "Tool",
    "ToolResult",
    "MCPError",
    "MCPTransportError",
    "MCPProtocolError",
    "MCPToolError",
    "Step",
    "StepResult",
    "SequenceResult",
    "BindingError",
    "execute_sequence",
    "resolve_bindings",
    "validate_against_schema",
    "load_steps_from_json",
]
