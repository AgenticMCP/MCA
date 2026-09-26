"""Python wrapper around the Go github-mcp-server stdio binary.

Quick start::

    from pkg.mcp_wrapper import MCPClient

    with MCPClient(token="ghp_...") as gh:
        me = gh.call("get_me").data
        issues = gh.list_issues(owner="octocat", repo="hello-world").data

See ``pkg/mcp_wrapper/README.md`` for the full reference.
"""

from .client import MCPClient, discover_tools
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
