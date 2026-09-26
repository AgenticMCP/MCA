"""Python wrapper around the Go `github-mcp-server` stdio binary.

Architecture
------------
`MCPClient` spawns the compiled Go server as a long-lived subprocess and speaks
JSON-RPC 2.0 over its stdin/stdout (one message per line, newline-delimited).

On `start()` the client:
  1. Spawns the server with the requested flags + GITHUB_PERSONAL_ACCESS_TOKEN
     in the environment.
  2. Sends the MCP `initialize` request and reads the response.
  3. Sends the `notifications/initialized` notification.
  4. Sends `tools/list` and caches the advertised tools.

After that the session is reusable: each `call(tool, args)` sends one
`tools/call` request and returns a `ToolResult`. The MCP handshake happens
ONCE per session, not per call, which is the main efficiency win over the
mcpcurl Go CLI.

Concurrency: a `threading.Lock` serializes calls. The wrapper does not pipeline
multiple in-flight requests; that is plenty for an automation harness.
"""

from __future__ import annotations

import itertools
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

from .types import (
    MCPError,
    MCPProtocolError,
    MCPToolError,
    MCPTransportError,
    Tool,
    ToolResult,
)

# Path to the compiled Go binary that ships alongside this package.
_DEFAULT_BIN = Path(__file__).resolve().parent / "bin" / "github-mcp-server"

# MCP protocol version we advertise during the handshake. Matches mcpcurl.
_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "github-mcp-server-python-wrapper", "version": "0.1.0"}


class MCPClient:
    """A live session with a spawned github-mcp-server stdio process.

    Typical usage::

        with MCPClient(token="ghp_...") as gh:
            print([t.name for t in gh.list_tools()])
            me = gh.call("get_me").data
            issues = gh.list_issues(owner="octocat", repo="hello-world").data

    Parameters
    ----------
    token:
        GitHub personal access token. Falls back to the GITHUB_PERSONAL_ACCESS_TOKEN
        or GITHUB_TOKEN environment variables. Required by the server at startup.
    binary:
        Path to the compiled server binary. Defaults to the binary shipped
        alongside this package (pkg/mcp_wrapper/bin/github-mcp-server).
    toolsets:
        Iterable of toolset IDs to enable (e.g. ["issues", "pull_requests"]).
        None means use the server's default toolsets.
    tools:
        Iterable of explicit tool names to enable. Overrides toolsets when set.
    exclude_tools:
        Iterable of tool names to exclude from the default set.
    read_only:
        Forwarded as --read-only. Disables every write-capable tool.
    host:
        Forwarded as --host (e.g. for GitHub Enterprise hosts).
    log_file:
        Forwarded as --log-file. The server writes its own slog output there.
    enable_command_logging:
        Forwarded as --enable-command-logging (server logs JSON-RPC traffic).
    extra_args:
        Additional positional flags appended to the `stdio` subcommand.
    env_extra:
        Extra env vars to merge into the child process environment.
    request_timeout:
        Per-request soft timeout in seconds. None disables the timeout.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        binary: str | os.PathLike[str] | None = None,
        toolsets: Iterable[str] | None = None,
        tools: Iterable[str] | None = None,
        exclude_tools: Iterable[str] | None = None,
        read_only: bool = False,
        host: str | None = None,
        log_file: str | os.PathLike[str] | None = None,
        enable_command_logging: bool = False,
        extra_args: Iterable[str] | None = None,
        env_extra: dict[str, str] | None = None,
        request_timeout: float | None = 120.0,
    ) -> None:
        resolved_token = (
            token
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
        )
        if not resolved_token:
            raise MCPError(
                "no GitHub token: pass token=... or set "
                "GITHUB_PERSONAL_ACCESS_TOKEN / GITHUB_TOKEN"
            )
        self._token = resolved_token

        self._binary = Path(binary) if binary else _DEFAULT_BIN
        if not self._binary.is_file():
            raise MCPError(
                f"server binary not found at {self._binary}. "
                f"Build it with: go build -o {self._binary} ./cmd/github-mcp-server"
            )

        self._toolsets = list(toolsets) if toolsets else None
        self._tools = list(tools) if tools else None
        self._exclude_tools = list(exclude_tools) if exclude_tools else None
        self._read_only = read_only
        self._host = host
        self._log_file = Path(log_file) if log_file else None
        self._enable_command_logging = enable_command_logging
        self._extra_args = list(extra_args) if extra_args else []
        self._env_extra = dict(env_extra or {})
        self._request_timeout = request_timeout

        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._id_counter = itertools.count(1)
        self._tools_by_name: dict[str, Tool] = {}
        self._started = False

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Spawn the server and complete the MCP handshake. Idempotent."""
        if self._started:
            return

        argv = self._build_argv()
        env = os.environ.copy()
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self._token
        env.update(self._env_extra)

        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered
            )
        except OSError as e:
            raise MCPTransportError(f"failed to spawn {argv[0]}: {e}") from e

        # Drain stderr in a background thread so the pipe never fills up.
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="mcp-stderr-drain"
        )
        self._stderr_thread.start()

        try:
            self._handshake()
            self._refresh_tools()
        except Exception:
            self.close()
            raise

        self._started = True

    def close(self) -> None:
        """Terminate the server subprocess. Idempotent."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        self._started = False
        try:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        finally:
            for stream in (proc.stdout, proc.stderr):
                if stream and not stream.closed:
                    try:
                        stream.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------ public API

    def list_tools(self) -> list[Tool]:
        """Return tools advertised by the server (cached after start)."""
        self._ensure_started()
        return list(self._tools_by_name.values())

    def get_tool(self, name: str) -> Tool | None:
        self._ensure_started()
        return self._tools_by_name.get(name)

    def refresh_tools(self) -> list[Tool]:
        """Re-query tools/list. Useful if toolsets change at runtime."""
        self._ensure_started()
        self._refresh_tools()
        return self.list_tools()

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Invoke a tool by name. Returns a ToolResult; raises MCPToolError on
        tool-level failure (isError=true) and MCPProtocolError on JSON-RPC error."""
        self._ensure_started()
        if name not in self._tools_by_name:
            raise MCPError(
                f"unknown tool {name!r} (known: "
                f"{sorted(self._tools_by_name)[:5]}...)"
            )
        result = self._request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        tr = ToolResult.from_mcp(name, result)
        if tr.is_error:
            raise MCPToolError(name, tr.text, result)
        return tr

    def execute_sequence(
        self,
        steps: "Iterable[Any]",
        *,
        on_error: str = "stop",
    ) -> "SequenceResult":
        """Run a chain of tool calls with binding resolution and schema checks.

        See ``pkg.mcp_wrapper.sequence`` for the full spec. Steps may reference
        earlier step outputs via ``"$<step_id>.<path>"`` templates inside
        their arguments. Each step's resolved arguments are pre-validated
        against the tool's `inputSchema`, and an optional `expect_output`
        schema can be supplied per step.
        """
        from .sequence import execute_sequence as _execute_sequence

        self._ensure_started()
        return _execute_sequence(self, steps, on_error=on_error)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        """Dynamic tool dispatch: `client.get_me(...)` → `client.call("get_me", ...)`.

        Only resolves names that match an advertised tool, so genuine attribute
        errors still surface clearly.
        """
        # __getattr__ is only consulted when normal lookup fails. Guard against
        # introspection (dunders, debugger probes) and pre-start access.
        if name.startswith("_"):
            raise AttributeError(name)
        if not self._started or name not in self._tools_by_name:
            raise AttributeError(name)

        def _invoke(**kwargs: Any) -> ToolResult:
            return self.call(name, kwargs)

        _invoke.__name__ = name
        _invoke.__doc__ = self._tools_by_name[name].description
        return _invoke

    def __dir__(self) -> list[str]:
        base = list(super().__dir__())
        if self._started:
            base.extend(self._tools_by_name)
        return sorted(set(base))

    @property
    def stderr_output(self) -> str:
        """Server stderr captured so far (slog output + warnings)."""
        return "".join(self._stderr_lines)

    # ------------------------------------------------------------------ internals

    def _build_argv(self) -> list[str]:
        argv: list[str] = [str(self._binary), "stdio"]
        if self._toolsets:
            argv += ["--toolsets", ",".join(self._toolsets)]
        if self._tools:
            argv += ["--tools", ",".join(self._tools)]
        if self._exclude_tools:
            argv += ["--exclude-tools", ",".join(self._exclude_tools)]
        if self._read_only:
            argv += ["--read-only"]
        if self._host:
            argv += ["--host", self._host]
        if self._log_file:
            argv += ["--log-file", str(self._log_file)]
        if self._enable_command_logging:
            argv += ["--enable-command-logging"]
        argv += list(self._extra_args)
        return argv

    def _ensure_started(self) -> None:
        if not self._started:
            raise MCPError("client not started: call start() or use as context manager")
        if self._proc is None or self._proc.poll() is not None:
            raise MCPTransportError(
                "server process exited unexpectedly; stderr=" + self.stderr_output
            )

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr_lines.append(line)
                # cap memory to a few thousand lines
                if len(self._stderr_lines) > 5000:
                    self._stderr_lines = self._stderr_lines[-2500:]
        except (ValueError, OSError):
            return  # stream closed during shutdown

    def _handshake(self) -> None:
        # 1. initialize request
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
        )
        # 2. initialized notification (no id, no response)
        self._notify("notifications/initialized")

    def _refresh_tools(self) -> None:
        result = self._request("tools/list", {})
        tools = result.get("tools") or []
        self._tools_by_name = {t["name"]: Tool.from_mcp(t) for t in tools}

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = next(self._id_counter)
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        with self._lock:
            self._write_line(payload)
            response = self._read_until_id(req_id)
        if "error" in response and response["error"] is not None:
            err = response["error"]
            raise MCPProtocolError(
                code=err.get("code", -1),
                message=err.get("message", "unknown"),
                data=err.get("data"),
            )
        return response.get("result") or {}

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        with self._lock:
            self._write_line(payload)

    def _write_line(self, obj: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            raise MCPTransportError("server stdin not available")
        try:
            proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPTransportError(
                f"failed to write to server stdin: {e}; stderr={self.stderr_output}"
            ) from e

    def _read_until_id(self, expected_id: int) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise MCPTransportError("server stdout not available")

        # subprocess.Popen with text=True gives us line iteration but no timeout.
        # We use readline() in a loop instead; bufsize=1 makes this prompt.
        # Notifications (no id) are skipped; non-matching ids would indicate a
        # protocol bug, so we surface them.
        while True:
            line = proc.stdout.readline()
            if not line:
                raise MCPTransportError(
                    "server closed stdout unexpectedly; stderr=" + self.stderr_output
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                raise MCPTransportError(
                    f"server emitted non-JSON line: {line!r}: {e}"
                ) from e
            if "id" not in msg:
                # server-initiated notification — ignore
                continue
            if msg["id"] != expected_id:
                # out-of-order response; in this single-flight client this is
                # likely a stale message from a previous failed exchange. Skip.
                continue
            return msg


def discover_tools(**kwargs: Any) -> list[Tool]:
    """Convenience helper: spin up a session purely to list available tools."""
    with MCPClient(**kwargs) as client:
        return client.list_tools()


def shell_escape_argv(argv: list[str]) -> str:
    """Render argv as a shell-quoted command (for debug/print)."""
    return " ".join(shlex.quote(a) for a in argv)


# For type annotations on execute_sequence. Imported lazily inside the method
# to keep the module load order client.py → sequence.py (sequence imports
# ToolResult from .types, which is fine).
from .sequence import SequenceResult  # noqa: E402

__all__ = [
    "MCPClient",
    "discover_tools",
    "shell_escape_argv",
    "Tool",
    "ToolResult",
    "MCPError",
    "MCPTransportError",
    "MCPProtocolError",
    "MCPToolError",
    "SequenceResult",
]


if __name__ == "__main__":  # pragma: no cover
    # Tiny dev helper: `python -m pkg.mcp_wrapper.client` lists tools.
    with MCPClient() as gh:
        for t in gh.list_tools():
            print(f"{t.name:40s} {t.description[:80]}")
    sys.exit(0)
