"""Python wrapper around an MCP server speaking stdio JSON-RPC 2.0.

Generic — server-agnostic. Configure the spawned command via
``MCPClient(command=..., argv0=..., extra_args=...)``; the yahoo_finance server
ships in ``servers/yahoo_finance`` and is spawned as
``python -m servers.yahoo_finance --transport stdio``. There's no
required credential by default: servers that need one set ``api_key`` /
``env_secret`` / etc. via the constructor.

Quick start::

    from pkg.finance_mcp_wrapper import MCPClient

    with MCPClient(command="python", argv0="-m", extra_args=[
        "servers.yahoo_finance", "--transport", "stdio",
    ]) as mcp:
        print([t.name for t in mcp.list_tools()])
        prices = mcp.get_historical_stock_prices(
            ticker="AAPL", start_date="2024-01-01", end_date="2024-01-10"
        ).data

See ``pkg/finance_mcp_wrapper/README.md`` for the full reference.
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
    "SequenceResult",
    "Step",
    "StepResult",
    "BindingError",
    "execute_sequence",
    "resolve_bindings",
    "validate_against_schema",
    "load_steps_from_json",
]
