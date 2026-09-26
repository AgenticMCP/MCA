"""Python wrapper around the Node `playwright-mcp` stdio server.

Architecture
------------
`MCPClient` spawns the playwright-mcp server (``node cli.js`` from this
checkout by default, or any command via ``command=``) as a long-lived
subprocess and speaks JSON-RPC 2.0 over its stdin/stdout (one message per
line, newline-delimited).

On `start()` the client:
  1. Spawns the server with the requested browser flags (no credentials —
     playwright-mcp needs no token; its "auth" is the browser/launch config).
  2. Sends the MCP `initialize` request and reads the response.
  3. Sends the `notifications/initialized` notification.
  4. Sends `tools/list` and caches the advertised tools.

After that the session is reusable: each `call(tool, args)` sends one
`tools/call` request and returns a `ToolResult`. The MCP handshake happens
ONCE per session, and — critically for this server — requests are strictly
SINGLE-FLIGHT: playwright-mcp processes concurrent requests in parallel
(racing a click against an in-flight navigation, opening stray blank pages),
so the per-call lock is not just politeness, it is what keeps the browser
session deterministic.

The browser session lives exactly as long as the subprocess: closing the
client closes the browser, and every new client starts from a blank browser.
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

# The playwright-mcp checkout this package ships inside (repo root is two
# levels up from pkg/mcp_wrapper/client.py). Running `node cli.js` from the
# source tree is the default; there is no compile step (the implementation
# lives in the playwright-core npm dependency).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMAND = ["node", str(REPO_ROOT / "cli.js")]

# MCP protocol version we advertise during the handshake.
_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "playwright-mcp-python-wrapper", "version": "0.1.0"}


class MCPClient:
    """A live session with a spawned playwright-mcp stdio process.

    Typical usage::

        with MCPClient() as pw:
            print([t.name for t in pw.list_tools()])
            pw.call("browser_navigate", {"url": "https://example.com"})
            snap = pw.call("browser_snapshot").text

    Parameters
    ----------
    command:
        Executable + args of the server (before the option flags). Defaults
        to ``["node", "<repo>/cli.js"]`` — the source checkout. Use e.g.
        ``["npx", "-y", "@playwright/mcp@0.0.78"]`` to pin the npm package.
    headless / isolated:
        Browser flags (default True/True: headless, in-memory profile).
    browser:
        ``--browser`` value (chrome|firefox|webkit|msedge). None = bundled
        Chromium default.
    device / viewport_size / user_agent:
        Emulation flags, forwarded verbatim when set.
    caps:
        Extra capabilities for ``--caps`` (e.g. ["vision", "pdf"]).
    output_dir:
        ``--output-dir`` — where the server saves snapshots/screenshots.
        Point it into the run directory so artifacts stay with the run.
    timeout_action / timeout_navigation:
        Server-side timeouts in ms, forwarded when set.
    allowed_origins / blocked_origins:
        Forwarded verbatim when set (semicolon-separated strings).
    allow_tools:
        CLIENT-side allowlist: when set, `call()` refuses any tool not in
        the set. playwright-mcp has no --read-only flag or toolset
        selection, so a read-only verifier session is enforced here in the
        wrapper instead of in the server.
    extra_args:
        Additional raw flags appended to the command.
    env_extra:
        Extra env vars to merge into the child process environment.
    log_file:
        The server has no --log-file flag; when set, the captured stderr is
        written there on close() so every run keeps a server log artifact.
    request_timeout:
        Per-request soft timeout in seconds. None disables the timeout.
    """

    def __init__(
        self,
        *,
        command: Iterable[str] | None = None,
        headless: bool = True,
        isolated: bool = True,
        browser: str | None = None,
        device: str | None = None,
        viewport_size: str | None = None,
        user_agent: str | None = None,
        caps: Iterable[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        timeout_action: int | None = None,
        timeout_navigation: int | None = None,
        allowed_origins: str | None = None,
        blocked_origins: str | None = None,
        allow_tools: Iterable[str] | None = None,
        extra_args: Iterable[str] | None = None,
        env_extra: dict[str, str] | None = None,
        log_file: str | os.PathLike[str] | None = None,
        request_timeout: float | None = 120.0,
    ) -> None:
        self._command = list(command) if command else list(DEFAULT_COMMAND)
        self._headless = headless
        self._isolated = isolated
        self._browser = browser
        self._device = device
        self._viewport_size = viewport_size
        self._user_agent = user_agent
        self._caps = list(caps) if caps else None
        self._output_dir = Path(output_dir) if output_dir else None
        self._timeout_action = timeout_action
        self._timeout_navigation = timeout_navigation
        self._allowed_origins = allowed_origins
        self._blocked_origins = blocked_origins
        self._allow_tools = set(allow_tools) if allow_tools else None
        self._extra_args = list(extra_args) if extra_args else []
        self._env_extra = dict(env_extra or {})
        self._log_file = Path(log_file) if log_file else None
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
        """Terminate the server subprocess (which closes the browser). Idempotent."""
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
            if self._log_file is not None:
                try:
                    self._log_file.parent.mkdir(parents=True, exist_ok=True)
                    self._log_file.write_text(self.stderr_output, encoding="utf-8")
                except OSError:
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
        """Re-query tools/list."""
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
        if self._allow_tools is not None and name not in self._allow_tools:
            raise MCPError(
                f"tool {name!r} is not in this client's allowlist "
                f"({sorted(self._allow_tools)}) — this session is restricted "
                f"to observation tools"
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

        See ``pkg.mcp_wrapper.sequence`` for the full spec.
        """
        from .sequence import execute_sequence as _execute_sequence

        self._ensure_started()
        return _execute_sequence(self, steps, on_error=on_error)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        """Dynamic tool dispatch: `client.browser_snapshot(...)` →
        `client.call("browser_snapshot", ...)`."""
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
        """Server stderr captured so far (browser/driver warnings)."""
        return "".join(self._stderr_lines)

    # ------------------------------------------------------------------ internals

    def _build_argv(self) -> list[str]:
        argv: list[str] = list(self._command)
        if self._headless:
            argv += ["--headless"]
        if self._isolated:
            argv += ["--isolated"]
        if self._browser:
            argv += ["--browser", self._browser]
        if self._device:
            argv += ["--device", self._device]
        if self._viewport_size:
            argv += ["--viewport-size", self._viewport_size]
        if self._user_agent:
            argv += ["--user-agent", self._user_agent]
        if self._caps:
            argv += ["--caps", ",".join(self._caps)]
        if self._output_dir:
            argv += ["--output-dir", str(self._output_dir)]
        if self._timeout_action is not None:
            argv += ["--timeout-action", str(self._timeout_action)]
        if self._timeout_navigation is not None:
            argv += ["--timeout-navigation", str(self._timeout_navigation)]
        if self._allowed_origins:
            argv += ["--allowed-origins", self._allowed_origins]
        if self._blocked_origins:
            argv += ["--blocked-origins", self._blocked_origins]
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
# to keep the module load order client.py → sequence.py.
from .sequence import SequenceResult  # noqa: E402

__all__ = [
    "MCPClient",
    "discover_tools",
    "shell_escape_argv",
    "REPO_ROOT",
    "DEFAULT_COMMAND",
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
    with MCPClient() as pw:
        for t in pw.list_tools():
            print(f"{t.name:32s} {t.description[:90]}")
    sys.exit(0)
