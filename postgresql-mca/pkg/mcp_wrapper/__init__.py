"""Python wrapper around the postgres-mcp stdio server.

Quick start::

    from pkg.mcp_wrapper import MCPClient

    with MCPClient(database_uri="postgresql://u:p@localhost:5432/db") as pg:
        schemas = pg.call("list_schemas").data
        rows = pg.execute_sql(sql="SELECT current_user").data

See ``pkg/mcp_wrapper/README.md`` for the full reference.
"""

from .client import MCPClient, discover_tools, resolve_server_cmd
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
    parse_result_text,
)

__all__ = [
    "MCPClient",
    "discover_tools",
    "resolve_server_cmd",
    "Tool",
    "ToolResult",
    "parse_result_text",
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
