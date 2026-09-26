"""Python wrapper around the `postgres-mcp` stdio server.

Architecture
------------
`MCPClient` spawns the postgres-mcp server as a long-lived subprocess and
speaks JSON-RPC 2.0 over its stdin/stdout (one message per line,
newline-delimited).

On `start()` the client:
  1. Spawns the server with the requested access mode + DATABASE_URI in the
     environment.
  2. Sends the MCP `initialize` request and reads the response.
  3. Sends the `notifications/initialized` notification.
  4. Sends `tools/list` and caches the advertised tools.

After that the session is reusable: each `call(tool, args)` sends one
`tools/call` request and returns a `ToolResult`. The MCP handshake happens
ONCE per session, not per call.

postgres-mcp specifics (the adapter seams; the JSON-RPC engine is unchanged
from the github-mcp-server original):

* **Spawn** — postgres-mcp is a Python package with several launch styles. The
  argv is resolved once per process by :func:`resolve_server_cmd`, in order:
  ``AGENTICMCPE_PG_SERVER_CMD`` (shell-split, authoritative), a
  ``postgres-mcp`` executable on PATH, ``<repo>/.venv/bin/postgres-mcp``,
  ``uv run --project <repo> postgres-mcp``, ``uvx postgres-mcp``, and finally
  ``docker run -i --rm crystaldba/postgres-mcp``. Flags appended:
  ``--access-mode`` and ``--transport stdio``.
* **Credentials** — a PostgreSQL connection URI, passed via the DATABASE_URI
  environment variable (never argv, so it can't leak into process listings).
  The server starts and serves tools/list even when the database is
  unreachable, so a catalog load works with a placeholder URI.
* **Read-only mode** — ``access_mode="restricted"`` (the server's SafeSqlDriver
  then permits only SELECT/EXPLAIN/SHOW/ANALYZE/VACUUM and enforces a
  statement timeout). This replaces the GitHub ``--read-only`` flag.
* **Error convention** — postgres-mcp reports tool failures as a NORMAL text
  result whose body starts with ``"Error:"`` (isError stays false). `call()`
  converts those into :class:`MCPToolError` so the executor's error taxonomy
  (tool_error, retries, benign-exists markers) works unchanged.
* **Logging** — the server has no --log-file flag; it logs to stderr. When a
  ``log_file`` is given the captured stderr is written there on close().

Concurrency: a `threading.Lock` serializes calls. The wrapper does not pipeline
multiple in-flight requests; that is plenty for an automation harness.
"""

from __future__ import annotations

import itertools
import json
import os
import shlex
import shutil
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

# Repo root = two levels up from this file (pkg/mcp_wrapper/client.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Docker image published by the postgres-mcp project (used as the last-resort
# spawn when no local install is found). DATABASE_URI is forwarded into the
# container; --network=host is NOT used so a host database must be reachable
# as host.docker.internal (the resolver rewrites localhost URIs accordingly).
_DOCKER_IMAGE = os.environ.get("AGENTICMCPE_PG_DOCKER_IMAGE", "crystaldba/postgres-mcp")

# MCP protocol version we advertise during the handshake.
_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "postgres-mcp-python-wrapper", "version": "0.1.0"}

_ACCESS_MODES = ("unrestricted", "restricted")


def resolve_server_cmd() -> list[str]:
    """The base argv that launches postgres-mcp (without flags), resolved from
    the environment. See the module docstring for the search order. Raises
    :class:`MCPError` when no launch style is available."""
    override = os.environ.get("AGENTICMCPE_PG_SERVER_CMD", "").strip()
    if override:
        return shlex.split(override)
    exe = shutil.which("postgres-mcp")
    if exe:
        return [exe]
    venv_exe = REPO_ROOT / ".venv" / "bin" / "postgres-mcp"
    if venv_exe.is_file():
        return [str(venv_exe)]
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "--project", str(REPO_ROOT), "postgres-mcp"]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "postgres-mcp"]
    docker = shutil.which("docker")
    if docker:
        return [docker, "run", "-i", "--rm", "-e", "DATABASE_URI", _DOCKER_IMAGE]
    raise MCPError(
        "cannot find a way to launch postgres-mcp: install it (pip/uv), or set "
        "AGENTICMCPE_PG_SERVER_CMD to the exact command (e.g. "
        "'/path/to/venv/bin/postgres-mcp'), or install docker"
    )


def _is_docker_cmd(cmd: list[str]) -> bool:
    return bool(cmd) and Path(cmd[0]).name == "docker" and "run" in cmd


def _dockerize_uri(cmd: list[str], uri: str) -> str:
    """When the server runs inside docker, a URI pointing at the HOST's
    localhost must be rewritten to host.docker.internal or the container will
    dial itself. No-op for non-docker commands or non-local URIs."""
    if not cmd or Path(cmd[0]).name != "docker":
        return uri
    return uri.replace("@localhost", "@host.docker.internal").replace(
        "@127.0.0.1", "@host.docker.internal")


# Unique --name suffixes for docker-spawned servers (see MCPClient.close):
# killing the `docker run` CLI does NOT stop the container, and the server
# does not exit on stdin EOF, so without a name to `docker rm -f` every
# session would leak a running container that holds database connections
# (100 leaked containers exhaust PostgreSQL's default max_connections=100).
_container_counter = itertools.count(1)


class MCPClient:
    """A live session with a spawned postgres-mcp stdio process.

    Typical usage::

        with MCPClient(database_uri="postgresql://u:p@localhost:5432/db") as pg:
            print([t.name for t in pg.list_tools()])
            schemas = pg.call("list_schemas").data
            rows = pg.execute_sql(sql="SELECT current_user").data

    Parameters
    ----------
    database_uri:
        PostgreSQL connection URI. Falls back to the DATABASE_URI environment
        variable. Optional: the server starts (and serves tools/list) without
        a reachable database; tool calls then fail until one is configured.
    server_cmd:
        Base argv that launches the server. Defaults to
        :func:`resolve_server_cmd` (env override, PATH, .venv, uv, docker).
    access_mode:
        "unrestricted" (default; execute_sql may run any SQL) or "restricted"
        (read-only SQL only, enforced server-side). Forwarded as
        ``--access-mode``. This is the read-only switch for verification.
    tools:
        Optional allowlist of tool names. postgres-mcp has no server-side tool
        selection flags, so this is enforced CLIENT-side: non-listed tools are
        hidden from list_tools()/get_tool() and rejected by call().
    log_file:
        Where to write the server's captured stderr on close() (the server has
        no --log-file flag of its own).
    extra_args:
        Additional flags appended to the server argv.
    env_extra:
        Extra env vars to merge into the child process environment.
    request_timeout:
        Per-request soft timeout in seconds. None disables the timeout.
    """

    def __init__(
        self,
        *,
        database_uri: str | None = None,
        server_cmd: Iterable[str] | None = None,
        access_mode: str = "unrestricted",
        tools: Iterable[str] | None = None,
        log_file: str | os.PathLike[str] | None = None,
        extra_args: Iterable[str] | None = None,
        env_extra: dict[str, str] | None = None,
        request_timeout: float | None = 120.0,
    ) -> None:
        if access_mode not in _ACCESS_MODES:
            raise MCPError(
                f"access_mode must be one of {_ACCESS_MODES}, got {access_mode!r}")
        self._database_uri = database_uri or os.environ.get("DATABASE_URI") or ""
        self._server_cmd = list(server_cmd) if server_cmd else resolve_server_cmd()
        self._access_mode = access_mode
        self._tool_allowlist = set(tools) if tools else None
        self._log_file = Path(log_file) if log_file else None
        self._extra_args = list(extra_args) if extra_args else []
        self._env_extra = dict(env_extra or {})
        self._request_timeout = request_timeout

        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._id_counter = itertools.count(1)
        self._tools_by_name: dict[str, Tool] = {}
        self._started = False
        # Set when the server runs under `docker run`: the exact container
        # name to force-remove on close (the CLI process dying does not stop
        # the container).
        self._container_name: str | None = None

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
        if self._database_uri:
            env["DATABASE_URI"] = _dockerize_uri(self._server_cmd, self._database_uri)
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
            # Docker-spawned server: remove exactly OUR container. The server
            # does not exit on stdin EOF, so without this every session leaks
            # a running container holding database connections.
            if self._container_name:
                try:
                    subprocess.run(
                        [self._server_cmd[0], "rm", "-f", self._container_name],
                        capture_output=True, timeout=20)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self._container_name = None
            # The server has no --log-file flag; persist its captured stderr
            # ourselves so every run dir still gets a server.log.
            if self._log_file is not None:
                try:
                    self._log_file.parent.mkdir(parents=True, exist_ok=True)
                    with self._log_file.open("a", encoding="utf-8") as f:
                        f.write("".join(self._stderr_lines))
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
        tool-level failure and MCPProtocolError on JSON-RPC error.

        Tool-level failure covers BOTH conventions: ``isError: true`` (e.g.
        FastMCP catching an exception) and postgres-mcp's own style of
        returning a normal text block starting with "Error:"."""
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
        text = tr.text
        if text.lstrip().startswith("Error:"):
            raise MCPToolError(name, text.strip(), result)
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
        """Dynamic tool dispatch: `client.list_schemas(...)` →
        `client.call("list_schemas", ...)`.

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
        """Server stderr captured so far (its logging output + warnings)."""
        return "".join(self._stderr_lines)

    # ------------------------------------------------------------------ internals

    def _build_argv(self) -> list[str]:
        argv = list(self._server_cmd)
        if _is_docker_cmd(argv):
            # Name the container so close() can `docker rm -f` exactly it —
            # terminating the docker CLI alone leaves the container (and its
            # database connection pool) running forever.
            self._container_name = (
                f"agenticmcpe-mcp-{os.getpid()}-{next(_container_counter)}")
            argv = argv[: argv.index("run") + 1] + [
                "--name", self._container_name] + argv[argv.index("run") + 1:]
        argv += ["--access-mode", self._access_mode]
        argv += ["--transport", "stdio"]
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
        # Client-side allowlist: postgres-mcp has no --tools flag, so a
        # restricted tool surface (e.g. a read-only verifier that must not see
        # execute_sql in unrestricted mode) is enforced here.
        if self._tool_allowlist is not None:
            self._tools_by_name = {
                n: t for n, t in self._tools_by_name.items()
                if n in self._tool_allowlist
            }

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
    "resolve_server_cmd",
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
    with MCPClient() as pg:
        for t in pg.list_tools():
            print(f"{t.name:40s} {t.description[:80]}")
    sys.exit(0)
