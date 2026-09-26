"""Offline regression harness for the finance executor.

The github agenticmcpe ships an `idem_selftest.py` to cover its
idempotency / settle-wait machinery. The finance port has neither
idempotency nor settle-wait (the server is read-only), so the harness
here covers the executor's binding-resolution and retry paths against a
scripted fake client. Run it after touching executor.py:

    python3 -m pkg.finance_agenticmcpe.idem_selftest
"""

from __future__ import annotations

import sys
from typing import Any

from pkg.finance_agenticmcpe.config import AgenticConfig
from pkg.finance_agenticmcpe.executor import ExecutionAgent
from pkg.finance_agenticmcpe.planner import Plan, PlanStep


# ---------------------------------------------------------------------------
# FakeClient — scripted MCP client that exercises the executor's branches
# without spawning a real server.
# ---------------------------------------------------------------------------


class FakeClient:
    """A scripted MCP client.

    Each call dispatches to ``self.scripts[tool]`` (a callable taking
    arguments and returning either a ``data`` object or raising an
    ``MCPError`` subclass). ``self.call_log`` records every invocation
    so tests can assert on argument values and ordering."""

    def __init__(self) -> None:
        self.scripts: dict[str, Any] = {}
        self.call_log: list[dict[str, Any]] = []
        self._tools: dict[str, Any] = {}
        self._initialized = True

    # MCPClient-compatible surface
    def get_tool(self, name: str) -> Any:
        return self._tools.get(name)

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        self.call_log.append({"tool": tool, "arguments": arguments})

        class _Result:
            def __init__(self, data, content=None):
                self.data = data
                self.content = content or [{"type": "text", "text": str(data)}]
                self.is_error = False
        scripted = self.scripts.get(tool)
        if scripted is None:
            from pkg.finance_mcp_wrapper.types import MCPToolError
            raise MCPToolError(tool, f"no script for {tool!r}", {"raw": {"content": []}})
        return _Result(*scripted(arguments))

    # Helpers
    def script(self, tool: str, fn: Any) -> None:
        self.scripts[tool] = fn

    def register_tool(self, name: str, input_schema: dict[str, Any]) -> None:
        from pkg.finance_mcp_wrapper.types import Tool
        self._tools[name] = Tool(
            name=name,
            description=f"fake {name}",
            input_schema=input_schema,
            annotations={},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_binding_resolution() -> None:
    fc = FakeClient()
    fc.register_tool("get_historical_stock_prices", {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
        },
        "required": ["ticker", "start_date", "end_date"],
    })
    fc.register_tool("get_stock_info", {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    })
    fc.script(
        "get_stock_info",
        lambda a: ({"ticker": a["ticker"], "sector": "Tech"}, None),
    )
    fc.script(
        "get_historical_stock_prices",
        lambda a: ([{"Date": "2024-01-02", "Open": 1.0, "Close": 2.0}], None),
    )

    plan = Plan(
        task="fetch info then prices with a binding",
        summary="two reads with a binding",
        steps=[
            # s0: get_stock_info returns {ticker, sector, ...}
            PlanStep(id="s0", tool="get_stock_info", arguments={"ticker": "AAPL"}),
            # s1: bind the ticker from s0
            PlanStep(
                id="s1",
                tool="get_historical_stock_prices",
                arguments={
                    "ticker": "$s0.ticker",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-10",
                },
            ),
        ],
    )
    agent = ExecutionAgent(AgenticConfig(), client=fc, work_dir=None, log_to_console=False)
    trace = agent.run(plan)
    assert trace.success, f"trace failed: {trace.error_context}"
    assert len(fc.call_log) == 2
    s1_args = fc.call_log[1]["arguments"]
    assert s1_args["ticker"] == "AAPL", f"binding did not resolve: {s1_args!r}"
    print("ok test_binding_resolution")


def test_transient_retry_recovers() -> None:
    from pkg.finance_mcp_wrapper.types import MCPToolError
    fc = FakeClient()
    fc.register_tool("get_stock_info", {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    })
    calls = {"n": 0}

    def flaky(_a: dict[str, Any]) -> tuple[Any, None]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise MCPToolError("get_stock_info", "flaky transient", {"raw": {"content": []}})
        return ({"ticker": "AAPL", "sector": "Tech"}, None)

    fc.script("get_stock_info", flaky)
    plan = Plan(
        task="fetch info",
        summary="one read",
        steps=[PlanStep(id="s0", tool="get_stock_info", arguments={"ticker": "AAPL"})],
    )
    agent = ExecutionAgent(AgenticConfig(), client=fc, work_dir=None, log_to_console=False)
    trace = agent.run(plan)
    assert trace.success, f"trace failed: {trace.error_context}"
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
    print("ok test_transient_retry_recovers")


def test_validation_error_fail_fast() -> None:
    fc = FakeClient()
    fc.register_tool("get_stock_info", {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    })
    fc.script("get_stock_info", lambda a: ({"ticker": a["ticker"]}, None))
    plan = Plan(
        task="missing required",
        summary="",
        steps=[PlanStep(id="s0", tool="get_stock_info", arguments={})],  # missing 'ticker'
    )
    agent = ExecutionAgent(AgenticConfig(), client=fc, work_dir=None, log_to_console=False)
    trace = agent.run(plan)
    assert not trace.success
    assert trace.steps[0].status == "validation_error"
    assert not fc.call_log, "must not call the server on a schema failure"
    print("ok test_validation_error_fail_fast")


def test_binding_error_fail_fast() -> None:
    fc = FakeClient()
    fc.register_tool("get_stock_info", {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    })
    fc.script("get_stock_info", lambda a: ({"ticker": a["ticker"]}, None))
    plan = Plan(
        task="dangling binding",
        summary="",
        steps=[
            PlanStep(id="s0", tool="get_stock_info", arguments={"ticker": "$s99.ticker"}),
        ],
    )
    agent = ExecutionAgent(AgenticConfig(), client=fc, work_dir=None, log_to_console=False)
    trace = agent.run(plan)
    assert not trace.success
    assert trace.steps[0].status == "binding_error"
    assert not fc.call_log
    print("ok test_binding_error_fail_fast")


def test_unknown_tool_fail_fast() -> None:
    fc = FakeClient()
    plan = Plan(
        task="unknown tool",
        summary="",
        steps=[PlanStep(id="s0", tool="nonexistent_tool", arguments={})],
    )
    agent = ExecutionAgent(AgenticConfig(), client=fc, work_dir=None, log_to_console=False)
    trace = agent.run(plan)
    assert not trace.success
    assert trace.steps[0].status == "validation_error"
    print("ok test_unknown_tool_fail_fast")


def test_embedded_binding_in_string() -> None:
    fc = FakeClient()
    fc.register_tool("get_historical_stock_prices", {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
        },
        "required": ["ticker", "start_date", "end_date"],
    })
    fc.register_tool("get_stock_info", {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    })
    fc.script(
        "get_stock_info",
        lambda a: ({"ticker": a["ticker"], "sector": "Tech"}, None),
    )
    fc.script(
        "get_historical_stock_prices",
        lambda a: ([{"Date": "2024-01-02", "Open": 1.0}], None),
    )

    plan = Plan(
        task="embedded binding in report",
        summary="two reads",
        steps=[
            PlanStep(id="s0", tool="get_stock_info", arguments={"ticker": "AAPL"}),
            PlanStep(
                id="s1",
                tool="get_historical_stock_prices",
                arguments={
                    # Embedded binding: "report: ticker=$s0.ticker first_open=$s0[0].Open"
                    # — composed from s0's ticker field and s0's first bar's Open field.
                    # (We get s0[0].Open from a synthetic bar returned by an
                    # earlier get_historical_stock_prices step — but the simpler
                    # scenario is to bind a non-list value into a report string.
                    # Here we test the simpler case: $s0.ticker embeds.)
                    "ticker": "report: ticker=$s0.ticker",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-10",
                },
            ),
        ],
    )
    agent = ExecutionAgent(AgenticConfig(), client=fc, work_dir=None, log_to_console=False)
    trace = agent.run(plan)
    assert trace.success, trace.error_context
    s1_args = fc.call_log[1]["arguments"]
    assert s1_args["ticker"] == "report: ticker=AAPL", s1_args
    print("ok test_embedded_binding_in_string")


def test_error_context_shape() -> None:
    from pkg.finance_mcp_wrapper.types import MCPToolError
    fc = FakeClient()
    fc.register_tool("get_stock_info", {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    })
    fc.script("get_stock_info", lambda a: (_ for _ in ()).throw(
        MCPToolError("get_stock_info", "ticker may be delisted", {"raw": {"content": []}})
    ))
    plan = Plan(
        task="force a failure",
        summary="",
        steps=[PlanStep(id="s0", tool="get_stock_info", arguments={"ticker": "DELISTED"})],
    )
    agent = ExecutionAgent(AgenticConfig(), client=fc, work_dir=None, log_to_console=False)
    trace = agent.run(plan)
    assert not trace.success
    assert trace.failed_step == "s0"
    assert trace.error_context is not None
    assert "ticker may be delisted" in trace.error_context
    assert "Resolved arguments" in trace.error_context
    # Hint surfacing for the replan
    assert "HINT" in trace.error_context
    print("ok test_error_context_shape")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all() -> int:
    tests = [
        test_binding_resolution,
        test_transient_retry_recovers,
        test_validation_error_fail_fast,
        test_binding_error_fail_fast,
        test_unknown_tool_fail_fast,
        test_embedded_binding_in_string,
        test_error_context_shape,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}", file=sys.stderr)
            return 1
    print(f"\n{len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())