"""JSON-RPC 2.0 stdio client for any MCP server.

Server-agnostic. The wrapper spawns an MCP server as a subprocess and
exchanges `initialize` / `tools/list` / `tools/call` messages over its
stdin/stdout. The default server command targets the bundled yahoo_finance
implementation::

    MCPClient(command="python", argv0="-m", extra_args=[
        "servers.yahoo_finance", "--transport", "stdio",
    ])

Design choices:

* No required credential. Servers that need one can pass ``api_key=...`` or
  ``env={"FOO": "bar"}`` at construction time.
* The server binary is launched via ``sys.executable`` unless ``command=``
  is given explicitly (so venv changes are picked up automatically).
* ``initialize`` advertises a wide-open ``roots`` list and full
  ``sampling`` + ``roots`` capability set — the yfinance server is read-only
  and ignores capabilities it doesn't use, but the broad set ensures
  compatibility with other MCP servers.
* Notifications (messages with no ``id``) are routed to a ``notifier``
  callback on the client; this matters for servers that emit `notifications/
  progress` or `notifications/initialized` (we send the latter explicitly
  to keep the server's handshake tidy).
* Subprocess env defaults to a sanitized copy of ``os.environ`` minus a
  small denylist (PATH-mutating wrappers, debug flags), with caller-supplied
  ``env`` entries merged on top.
* ``shutdown`` and a synchronous 1-second ``exit`` are sent on close.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable

from .types import (
    MCPError,
    MCPProtocolError,
    MCPToolError,
    MCPTransportError,
    Tool,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

# These variables often leak into child processes and confuse stdio servers
# (they make Python buffered, or change stdin/stdout to non-binary). Strip
# them unless the caller explicitly re-adds them via ``env=``.
_DEFAULT_ENV_DENYLIST = frozenset(
    {
        "PYTHONUNBUFFERED",  # we set our own below
        "PYTHONIOENCODING",
        "PYTHONFAULTHANDLER",
        "PYTHONHASHSEED",
        "PYTHONHOME",
        "PYTHONPATH",  # the caller's path may not exist in the child
        "PYTHONDONTWRITEBYTECODE",
        "TERM",
    }
)


def _build_subprocess_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Copy os.environ minus the denylist, then layer extras on top. Always
    force unbuffered binary stdio so JSON-RPC framing isn't disturbed."""
    out = {k: v for k, v in os.environ.items() if k not in _DEFAULT_ENV_DENYLIST}
    out["PYTHONUNBUFFERED"] = "1"
    out["PYTHONIOENCODING"] = "utf-8"
    if extra:
        out.update(extra)
    return out


def _spawn_subprocess(
    command: str,
    args: list[str],
    env: dict[str, str] | None,
    cwd: str | None,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [command, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
    )


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def _frame(payload: dict[str, Any]) -> bytes:
    """MCP stdio framing: one compact JSON message per line.

    (This is *not* LSP framing — the MCP stdio transport has no
    Content-Length header; messages are newline-delimited and must contain
    no embedded newlines.)
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _drain_stderr(proc: subprocess.Popen[bytes], buf: list[str]) -> None:
    """Drain stderr into a buffer for diagnostics. Runs on a daemon thread;
    safe to call once at start."""
    assert proc.stderr is not None

    def _pump() -> None:
        try:
            for line in iter(proc.stderr.readline, b""):
                buf.append(line.decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001 — drainer; ignore.
            pass

    t = threading.Thread(target=_pump, daemon=True, name="mcp-stderr-pump")
    t.start()


def _read_framed(stream) -> dict[str, Any]:
    """Read one newline-delimited JSON-RPC message from ``stream``.

    Blank lines are skipped (some servers pad their output). Raises
    ``MCPTransportError`` on EOF or an unparseable line.
    """
    while True:
        line = stream.readline()
        if not line:
            raise MCPTransportError("EOF while reading MCP message (server closed stdout)")
        text = line.strip()
        if not text:
            continue
        try:
            return json.loads(text.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MCPTransportError(f"server emitted a non-JSON line: {text[:200]!r}: {e}") from e


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


Notifier = Callable[[dict[str, Any]], None]
DEFAULT_NOTIFIER: Notifier = lambda msg: None  # noqa: E731


@dataclass
class MCPClient:
    """A JSON-RPC 2.0 stdio client for an MCP server.

    Lifecycle::

        with MCPClient(...) as mcp:
            mcp.list_tools()
            mcp.call("get_historical_stock_prices", {...})

    The class can also be used without ``with`` if you call :meth:`start`
    manually and :meth:`close` in a ``finally`` block.
    """

    command: str = field(default_factory=lambda: sys.executable)
    args: list[str] = field(default_factory=list)
    api_key: str | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    client_name: str = "finance-mcp-wrapper"
    client_version: str = "0.1.0"
    # Generous: a cold start imports pandas + yfinance before the server
    # answers `initialize`, which can exceed 15s on a loaded machine and
    # was costing whole taskgen attempts as spurious transport failures.
    start_timeout: float = 60.0
    call_timeout: float = 60.0
    notifier: Notifier = field(default_factory=lambda: DEFAULT_NOTIFIER)
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False, compare=False)
    _stderr_buf: list[str] = field(default_factory=list, init=False, repr=False, compare=False)
    _rx_queue: "Queue[dict[str, Any]]" = field(default_factory=Queue, init=False, repr=False, compare=False)
    _rx_thread: threading.Thread | None = field(default=None, init=False, repr=False, compare=False)
    _send_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _next_id: int = field(default=1, init=False, repr=False, compare=False)
    _inflight: dict[int, "threading.Event"] = field(default_factory=dict, init=False, repr=False, compare=False)
    _responses: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _errors: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _tools: dict[str, Tool] = field(default_factory=dict, init=False, repr=False, compare=False)
    _initialized: bool = field(default=False, init=False, repr=False, compare=False)
    _closing: bool = field(default=False, init=False, repr=False, compare=False)

    # ---- public lifecycle -------------------------------------------------

    def __post_init__(self) -> None:
        # Coerce None api_key / env defensively.
        if self.env is None:
            self.env = {}
        if self.api_key is not None:
            # Yfinance doesn't require an API key, but other MCP servers
            # may; expose via a generic YFINANCE_API_KEY env var by default.
            self.env.setdefault("YFINANCE_API_KEY", self.api_key)

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ---- subprocess + framing --------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            return
        env = _build_subprocess_env(self.env or None)
        self._proc = _spawn_subprocess(self.command, self.args, env, self.cwd)
        _drain_stderr(self._proc, self._stderr_buf)
        self._rx_thread = threading.Thread(
            target=self._pump_stdout,
            daemon=True,
            name="mcp-stdout-pump",
        )
        self._rx_thread.start()
        self._initialize()
        self._initialized = True

    def close(self) -> None:
        if self._proc is None or self._closing:
            return
        self._closing = True
        try:
            if self._initialized:
                try:
                    self.request("shutdown", {}, timeout=2.0)
                except MCPError:
                    pass
                # Always send the JSON-RPC ``exit`` so stdio servers actually quit.
                try:
                    self._send_raw({"jsonrpc": "2.0", "method": "exit"})
                except Exception:  # noqa: BLE001
                    pass
        finally:
            proc = self._proc
            self._proc = None
            if proc and proc.poll() is None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()

    @property
    def stderr(self) -> str:
        """Captured server stderr (concatenated). Useful when the server
        crashes on startup."""
        return "".join(self._stderr_buf)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---- framing helpers -------------------------------------------------

    def _send_raw(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        with self._send_lock:
            try:
                self._proc.stdin.write(_frame(payload))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise MCPTransportError(f"failed to write to MCP server: {e}") from e

    def _pump_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        while True:
            try:
                msg = _read_framed(stream)
            except MCPTransportError as e:
                # Server hung up. Wake any waiters with a synthetic error
                # so they don't hang forever.
                with self._send_lock:
                    inflight = list(self._inflight.keys())
                    err = {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}
                    for mid in inflight:
                        self._responses.setdefault(mid, err)
                        self._errors.setdefault(mid, err)
                        ev = self._inflight.pop(mid, None)
                        if ev is not None:
                            ev.set()
                return
            except Exception as e:  # noqa: BLE001
                # Defensive: keep the pump alive under unexpected errors.
                self.notifier({"jsonrpc": "2.0", "method": "internal/transport_error", "params": {"message": str(e)}})
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                mid = msg["id"]
                with self._send_lock:
                    ev = self._inflight.pop(mid, None)
                    if "error" in msg:
                        self._errors[mid] = msg
                    self._responses[mid] = msg
                    if ev is not None:
                        ev.set()
            else:
                # Notification.
                self.notifier(msg)

    def _initialize(self) -> None:
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": False},
                "sampling": {},
            },
            "clientInfo": {"name": self.client_name, "version": self.client_version},
        }
        # Per spec: send ``notifications/initialized`` AFTER ``initialize``.
        deadline = time.monotonic() + self.start_timeout
        init_resp = self.request("initialize", params, timeout=deadline - time.monotonic())
        if not isinstance(init_resp, dict) or "serverInfo" not in init_resp:
            raise MCPTransportError(f"server did not return serverInfo: {init_resp!r}")
        # Send the initialized notification (fire-and-forget; no id).
        self._send_raw({"jsonrpc": "2.0", "method": "notifications/initialized"})

    # ---- request / call ---------------------------------------------------

    def request(self, method: str, params: Any, *, timeout: float | None = None) -> Any:
        """Send a JSON-RPC request and wait for its response.

        Returns ``msg["result"]`` on success. Raises ``MCPProtocolError``
        if the server returns a JSON-RPC ``error`` object, or
        ``MCPTransportError`` if the subprocess dies before responding.
        """
        if self._proc is None:
            raise MCPTransportError("client is not started")
        if timeout is None:
            timeout = self.call_timeout
        with self._send_lock:
            mid = self._next_id
            self._next_id += 1
            ev = threading.Event()
            self._inflight[mid] = ev
        self._send_raw({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        if not ev.wait(timeout=timeout):
            with self._send_lock:
                self._inflight.pop(mid, None)
            raise MCPTransportError(f"timeout waiting for response to {method!r} (id={mid})")
        with self._send_lock:
            msg = self._responses.pop(mid, None)
            self._errors.pop(mid, None)
        if msg is None:
            raise MCPTransportError(f"no response captured for {method!r}")
        if "error" in msg:
            err = msg["error"]
            raise MCPProtocolError(
                code=int(err.get("code", -32000)),
                message=str(err.get("message", "")),
                data=err.get("data"),
            )
        return msg.get("result")

    def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a tool by name with the given arguments.

        Raises ``MCPToolError`` when the server returns ``isError: true``
        in the result (which is distinct from a JSON-RPC ``error`` — that's
        reserved for protocol-level failures). Raises ``MCPProtocolError``
        for protocol-level errors.
        """
        result = self.request(
            "tools/call",
            {"name": tool, "arguments": arguments},
            timeout=self.call_timeout,
        )
        if not isinstance(result, dict):
            raise MCPProtocolError(-32000, "tools/call did not return an object", data=result)
        if result.get("isError"):
            text = ""
            for c in result.get("content") or []:
                if c.get("type") == "text":
                    text += c.get("text", "")
            raise MCPToolError(tool=tool, text=text, raw=result)
        return ToolResult.from_mcp(tool, result)

    # ---- discovery -------------------------------------------------------

    def list_tools(self, *, refresh: bool = False) -> list[Tool]:
        """Return the tool catalog (cached after first call)."""
        if refresh or not self._tools:
            resp = self.request("tools/list", {})
            tools_raw = resp.get("tools") if isinstance(resp, dict) else None
            if not isinstance(tools_raw, list):
                raise MCPTransportError(f"tools/list did not return a 'tools' list: {resp!r}")
            self._tools = {t["name"]: Tool.from_mcp(t) for t in tools_raw}
        return list(self._tools.values())

    def get_tool(self, name: str) -> Tool | None:
        if not self._tools:
            self.list_tools()
        return self._tools.get(name)


def discover_tools(
    *,
    command: str | None = None,
    args: list[str] | None = None,
    **kwargs: Any,
) -> list[Tool]:
    """Convenience: start a client, dump the catalog, return it.

    Example::

        tools = discover_tools(args=["-m", "servers.yahoo_finance", "--transport", "stdio"])
    """
    cmd = command or sys.executable
    argv = list(args or [])
    with MCPClient(command=cmd, args=argv, **kwargs) as mcp:
        return mcp.list_tools(refresh=True)


__all__ = ["MCPClient", "discover_tools"]