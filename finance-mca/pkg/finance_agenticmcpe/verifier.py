"""Verifier agent: hybrid Format + Static + Dynamic evaluators for finance.

Mirrors the github verifier's three-layer model:

1. **Format** (deterministic, strict, no LLM):
   * Every step ran and reported a success status.
   * Each step's resolved arguments validated against its tool schema
     (re-checks the executor's pre-flight).
   * Each step's decoded result has any fields declared in
     ``post_action_properties.expect_fields``.
   * `expect_output` schema fragment, when supplied by the planner, holds.

2. **Static** (deterministic, strict):
   * Fixed-value invariants declared in ``post_action_properties.invariants``
     (e.g. the `ticker` returned by `get_stock_info` matches the input ticker).
   * Numeric bounds declared in ``post_action_properties.bounds``.
   * Result is non-empty when ``expect_fields`` claims so.
   * Length / cardinality expectations (``min_length``, ``exact_length``).
   * Cross-step invariants (later step's input equals earlier step's output)
     are checked by the binding resolver at execution time — re-validated here
     against the recorded trace.

3. **Dynamic** (LLM-generated, sandboxed):
   * The verifier LLM is prompted with the plan + trace + per-step
     `post_action_properties.metamorphic` notes, and emits a small Python
     program of fresh MCP queries that re-check the truth of the answers.
     The program is `exec`d in a tightly-scoped namespace (an `MCPClient`
     wrapper plus the safe helpers in this module) and any exception
     downgrades a check to a recorded failure — never a crash.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pkg.finance_mcp_wrapper import (
    MCPClient,
    MCPProtocolError,
    MCPToolError,
    MCPTransportError,
)
from pkg.finance_mcp_wrapper.sequence import validate_against_schema
from pkg.finance_mcp_wrapper.types import Tool

from .config import AgenticConfig
from .executor import ExecutionTrace, StepExecution
from .planner import Plan, PlanStep, _binding_refs
from .llm import LLMClient, Message

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    layer: str  # "format" | "static" | "dynamic"
    passed: bool
    detail: str = ""
    # "error" fails the run; "advisory" is recorded for humans but does not.
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "layer": self.layer, "passed": self.passed,
                "detail": self.detail, "severity": self.severity}


@dataclass
class VerificationReport:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    dynamic_program: str | None = None
    dynamic_program_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "dynamic_program": self.dynamic_program,
            "dynamic_program_error": self.dynamic_program_error,
        }

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_path(obj: Any, path: str) -> Any:
    """Traverse ``obj`` by dotted/bracket path (e.g. ``items[0].close``)."""
    tokens: list[str | int] = []
    i = 0
    n = len(path)
    while i < n:
        if path[i] == "[":
            j = path.find("]", i + 1)
            if j == -1:
                return None
            tokens.append(int(path[i + 1 : j].strip()))
            i = j + 1
            if i < n and path[i] == ".":
                i += 1
            continue
        j = i
        while j < n and path[j] not in ".[":
            j += 1
        tokens.append(path[i:j])
        i = j
        if i < n and path[i] == ".":
            i += 1
    cur: Any = obj
    for tok in tokens:
        if cur is None:
            return None
        if isinstance(tok, int):
            if not isinstance(cur, list) or tok < 0 or tok >= len(cur):
                return None
            cur = cur[tok]
        elif isinstance(cur, dict):
            cur = cur.get(tok)
        else:
            return None
    return cur


_MISSING = object()


def _resolve_field(data: Any, fld: str) -> Any:
    """Resolve ``fld`` against a step result, returning :data:`_MISSING`
    when it is absent.

    Eight of the nine yfinance tools return an ARRAY of records, so a
    field named in `post_action_properties` almost always lives on an
    *element*. A bare path lookup against the array yields nothing, which
    is why every `bounds`/`invariants` declaration on a series used to
    fail deterministically.
    """
    if isinstance(data, dict) and fld in data:
        return data[fld]
    if isinstance(data, list) and data and isinstance(data[0], dict) and fld in data[0]:
        return data[0][fld]
    v = _get_path(data, fld)
    return _MISSING if v is None else v


def _resolve_field_series(data: Any, fld: str) -> list[Any]:
    """Every value of ``fld`` in the result: one per element for an array
    result, a single value for an object. Empty when the field is absent."""
    if isinstance(data, list):
        return [e[fld] for e in data if isinstance(e, dict) and fld in e]
    v = _resolve_field(data, fld)
    return [] if v is _MISSING else [v]


# Invariant keys that mean "the cardinality of the result" rather than a
# field to look up. The planner guardrails point at `min_length`, but LLMs
# reach for these spellings often enough to be worth accepting.
_CARDINALITY_KEYS = frozenset({"len", "length", "count", "size", "n"})

_COMPARISON_RE = re.compile(r"^\s*(>=|<=|==|=|>|<|!=)\s*(-?\d+(?:\.\d+)?)\s*$")


def _invariant_actual(data: Any, fld: str) -> Any:
    """Resolve the left-hand side of an invariant: a cardinality keyword
    yields the result's length, anything else is a path lookup."""
    if fld.lower() in _CARDINALITY_KEYS:
        if isinstance(data, (list, str, dict)):
            return len(data)
        return None
    v = _resolve_field(data, fld)
    return None if v is _MISSING else v


def _invariant_holds(actual: Any, expected: Any) -> bool:
    """Compare against ``expected``, which is either a literal (equality,
    with numeric tolerance) or a comparison string such as ``">= 1"``."""
    if isinstance(expected, str):
        m = _COMPARISON_RE.match(expected)
        if m:
            op, raw = m.group(1), m.group(2)
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                return False
            rhs = float(raw)
            return {
                ">=": actual >= rhs,
                "<=": actual <= rhs,
                ">": actual > rhs,
                "<": actual < rhs,
                "==": _eq_close(actual, rhs),
                "=": _eq_close(actual, rhs),
                "!=": not _eq_close(actual, rhs),
            }[op]
    return bool(_eq_close(actual, expected))


def _eq_close(a: Any, b: Any, rel: float = 0.01) -> bool:
    """Numeric comparison with a small tolerance — useful for prices
    that may have minute-level discrepancies across replans."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1e-12)
        return abs(a - b) / denom <= rel
    return a == b


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class VerifierAgent:
    """Plan + Trace -> VerificationReport."""

    def __init__(
        self,
        config: AgenticConfig,
        *,
        tools: list[Tool] | None = None,
        llm: LLMClient | None = None,
        client: MCPClient | None = None,
    ):
        self.config = config
        self.tools = tools
        self.llm = llm
        self._client = client
        # Last dynamic program generated, surfaced on the report for diagnosis.
        self._last_program: str | None = None
        self._task_text: str = ""

    # ----------------------------------------------------------------- public

    def verify(self, plan: Plan, trace: ExecutionTrace) -> VerificationReport:
        checks: list[CheckResult] = []
        plan_steps_by_id = {st.id: st for st in plan.steps}
        # Used to tell a task requirement from a planner guess.
        self._task_text = (plan.task or trace.task or "").lower()
        # Step ids some other step binds into; see the non_empty check.
        self._consumed_step_ids = set().union(
            *(_binding_refs(st.arguments) for st in plan.steps)
        ) if plan.steps else set()

        for se in trace.steps:
            plan_step = plan_steps_by_id.get(se.id)
            checks.extend(self._format_checks(se, plan_step))
            checks.extend(self._static_checks(se, plan_step))

        # Plan-level invariants: cross-step consistency. The executor
        # validates every binding at execution time, so a successful step
        # already proves its bound arguments equal the referenced outputs.
        # We re-check declared cross-step relations here as a defense in
        # depth.
        for rel in _declared_cross_step_relations(plan):
            checks.extend(self._cross_step_checks(rel, trace))

        # Dynamic layer: ask the verifier LLM to write a Python program
        # that re-queries the server. Optional but recommended; missing LLM
        # only skips the dynamic layer.
        self._last_program = None
        if self.llm is not None and self._client is not None:
            checks.extend(self._dynamic_checks(plan, trace))

        # Advisory checks are recorded but never fail the run: they encode a
        # planner GUESS rather than a task requirement, and a deterministic
        # evaluator must trust the trace, not the guess.
        passed = all(c.passed for c in checks if c.severity != "advisory")
        return VerificationReport(
            passed=passed, checks=checks, dynamic_program=self._last_program
        )

    # ---------------------------------------------------------------- format

    def _format_checks(
        self, se: StepExecution, plan_step: PlanStep | None
    ) -> list[CheckResult]:
        checks: list[CheckResult] = []
        checks.append(
            CheckResult(
                name=f"{se.id}:step_succeeded",
                layer="format",
                passed=se.status == "success",
                detail=f"status={se.status}" if se.status != "success" else "",
            )
        )
        if plan_step is None:
            return checks
        props = plan_step.post_action_properties or {}
        # Schema re-validation of the resolved arguments (executor already
        # did this; we re-do it as a defense-in-depth check).
        if se.arguments_resolved is not None:
            tool = self._tool_for(plan_step.tool)
            if tool is not None:
                errs = validate_against_schema(se.arguments_resolved, tool.input_schema)
                checks.append(
                    CheckResult(
                        name=f"{se.id}:schema_valid",
                        layer="format",
                        passed=not errs,
                        detail="; ".join(errs) if errs else "",
                    )
                )
        # expect_fields — the result must contain each named field (or the
        # equivalent path when present in a nested list element).
        #
        # Skipped entirely for an empty list: "every element carries a Close"
        # has no counterexample over zero elements, so asserting it is vacuous,
        # not failed. A calendar-window price query that lands on a weekend or
        # market holiday legitimately returns [], and emitting one error per
        # expected field there fails a run whose data is correct — eight such
        # checks sank an otherwise-passing task. Whether the step SHOULD have
        # returned rows is the cardinality question, answered below.
        if not (isinstance(se.result_data, list) and not se.result_data):
            for fld in props.get("expect_fields") or []:
                if not isinstance(fld, str):
                    continue
                present = self._field_present(se.result_data, fld)
                checks.append(
                    CheckResult(
                        name=f"{se.id}:has_field:{fld}",
                        layer="format",
                        passed=bool(present),
                        detail=f"field {fld!r} absent" if not present else "",
                    )
                )
        # expect_output schema, if the planner supplied one.
        if plan_step.expect_output:
            errs = validate_against_schema(se.result_data, plan_step.expect_output)
            checks.append(
                CheckResult(
                    name=f"{se.id}:expect_output_schema",
                    layer="format",
                    passed=not errs,
                    detail="; ".join(errs) if errs else "",
                )
            )
        return checks

    def _field_present(self, data: Any, fld: str) -> bool:
        """True when ``fld`` is addressable in ``data``: top-level, in the
        first element of an array result, or via path traversal.

        Presence means key membership, not truthiness — a bar's
        ``"Dividends": 0.0`` is present.
        """
        return _resolve_field(data, fld) is not _MISSING

    # ----------------------------------------------------------------- static

    def _static_checks(
        self, se: StepExecution, plan_step: PlanStep | None
    ) -> list[CheckResult]:
        checks: list[CheckResult] = []
        if plan_step is None:
            return checks
        if se.status != "success":
            return checks
        props = plan_step.post_action_properties or {}
        # Invariants — fixed values, plus the cardinality/comparison forms
        # planners keep emitting ({"len": ">= 1"}). Treating those as an
        # equality check against a missing path produced false failures on
        # every well-formed plan.
        for fld, expected in (props.get("invariants") or {}).items():
            actual = _invariant_actual(se.result_data, fld)
            ok = _invariant_holds(actual, expected)
            checks.append(
                CheckResult(
                    name=f"{se.id}:invariant:{fld}",
                    layer="static",
                    passed=bool(ok),
                    detail=f"expected {expected!r}, got {actual!r}" if not ok else "",
                    severity=self._invariant_severity(expected),
                )
            )
        # Bounds — min/max on numeric fields. For an array result every
        # element's value must hold, not just the first.
        for fld, limits in (props.get("bounds") or {}).items():
            if not isinstance(limits, dict):
                continue
            values = _resolve_field_series(se.result_data, fld)
            numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not numeric:
                checks.append(
                    CheckResult(
                        name=f"{se.id}:bound:{fld}",
                        layer="static",
                        passed=False,
                        detail=(
                            f"no numeric value for {fld!r} in the result"
                            if not values
                            else f"value at {fld!r} is not numeric: {values[0]!r}"
                        ),
                    )
                )
                continue
            low = limits.get("min")
            high = limits.get("max")
            ok = True
            detail = ""
            if low is not None:
                below = [v for v in numeric if v < low]
                if below:
                    ok = False
                    detail = f"{fld}={below[0]} < min={low} ({len(below)}/{len(numeric)} values)"
            if high is not None:
                above = [v for v in numeric if v > high]
                if above:
                    ok = False
                    detail = f"{fld}={above[0]} > max={high} ({len(above)}/{len(numeric)} values)"
            checks.append(
                CheckResult(
                    name=f"{se.id}:bound:{fld}",
                    layer="static",
                    passed=ok,
                    detail=detail,
                )
            )
        # Cardinality expectations on list results.
        #
        # An EMPTY result is a hard failure — the step fetched nothing and any
        # later binding into it is meaningless. A specific count, though, is
        # the planner GUESSING how many trading days a calendar window holds;
        # weekends and market holidays make that guess wrong routinely, and
        # failing a run over it rejects correct data. Those are advisory, and
        # a genuine "did we fetch enough for a 50-day SMA?" belongs to the
        # dynamic layer, which derives the requirement from the task.
        if isinstance(se.result_data, list):
            n = len(se.result_data)
            if "min_length" in props:
                want = props["min_length"]
                ok = n >= want
                checks.append(
                    CheckResult(
                        name=f"{se.id}:length_at_least",
                        layer="static",
                        passed=ok,
                        detail="" if ok else f"len={n} < min_length={want}",
                        severity="error" if want <= 1 else "advisory",
                    )
                )
            if "exact_length" in props and n != props["exact_length"]:
                checks.append(
                    CheckResult(
                        name=f"{se.id}:length_exact",
                        layer="static",
                        passed=False,
                        detail=f"len={n} != exact_length={props['exact_length']}",
                        severity="error" if props["exact_length"] <= 1 else "advisory",
                    )
                )
            if n == 0 and "min_length" not in props:
                # Error only when another step actually binds into this one —
                # that is the whole reason an empty result is a hard failure
                # ("any later binding into it is meaningless"). When nothing
                # consumes the step, an empty result makes it a redundant
                # query, not a wrong one, and the same weekend/holiday window
                # that makes count guesses unreliable produces it legitimately.
                consumed = se.id in getattr(self, "_consumed_step_ids", set())
                checks.append(
                    CheckResult(
                        name=f"{se.id}:non_empty",
                        layer="static",
                        passed=False,
                        detail="step returned an empty list"
                        + ("" if consumed else " (unconsumed by any later step)"),
                        severity="error" if consumed else "advisory",
                    )
                )
        # Single-string length bounds.
        if isinstance(se.result_data, str):
            slen = len(se.result_data)
            if "min_length" in props and slen < props["min_length"]:
                checks.append(
                    CheckResult(
                        name=f"{se.id}:string_length_at_least",
                        layer="static",
                        passed=False,
                        detail=f"len={slen} < min_length={props['min_length']}",
                    )
                )
        return checks

    def _invariant_severity(self, expected: Any) -> str:
        """Is this invariant a task REQUIREMENT or a planner GUESS?

        An expected value that appears in the task text (a ticker the user
        named, a date they gave) is something the run must honour. One that
        does not — "the largest holder is Vanguard", "the sector is
        Technology" — is the planner predicting the data, and the answer may
        legitimately be something else. Guesses are recorded, not enforced.
        Numbers are exempt: they are usually echo-checks of an input.
        """
        if not isinstance(expected, str) or not expected.strip():
            return "error"
        return "error" if expected.strip().lower() in self._task_text else "advisory"

    def _cross_step_checks(
        self, rel: dict[str, Any], trace: ExecutionTrace
    ) -> list[CheckResult]:
        """One check per cross-step relation declared in the plan."""
        kind = rel.get("kind")
        if kind == "equal":
            src_step = rel.get("from_step")
            src_path = rel.get("from_path", "")
            dst_step = rel.get("to_step")
            dst_path = rel.get("to_path", "")
            src_data = next(
                (s.result_data for s in trace.steps if s.id == src_step), None
            )
            dst_data = next(
                (s.result_data for s in trace.steps if s.id == dst_step), None
            )
            if src_data is None or dst_data is None:
                return [
                    CheckResult(
                        name=f"cross:{src_step}->{dst_step}:equal",
                        layer="static",
                        passed=False,
                        detail="missing step data",
                    )
                ]
            src_v = _get_path(src_data, src_path)
            dst_v = _get_path(dst_data, dst_path)
            ok = _eq_close(src_v, dst_v)
            return [
                CheckResult(
                    name=f"cross:{src_step}.{src_path}={dst_step}.{dst_path}",
                    layer="static",
                    passed=bool(ok),
                    detail=f"src={src_v!r} dst={dst_v!r}" if not ok else "",
                )
            ]
        return []

    # ---------------------------------------------------------------- dynamic

    def _dynamic_checks(
        self, plan: Plan, trace: ExecutionTrace
    ) -> list[CheckResult]:
        """Ask the verifier LLM to write a small Python program that
        re-queries the server and re-checks the answers."""
        assert self.llm is not None and self._client is not None
        checks: list[CheckResult] = []
        system = _DYNAMIC_SYSTEM
        prompt_blob = {
            "plan_summary": plan.summary,
            "steps": [
                {
                    "id": st.id,
                    "tool": st.tool,
                    "description": st.description,
                    "arguments": st.arguments,
                    "post_action_properties": st.post_action_properties,
                    "trace": {
                        "status": trace.steps[i].status,
                        "data_preview": _preview(trace.steps[i].result_data, 400)
                        if trace.steps[i].result_data is not None
                        else None,
                    },
                }
                for i, st in enumerate(plan.steps)
            ],
            "metamorphics": [
                {
                    "step_id": st.id,
                    "metamorphic": st.post_action_properties.get("metamorphic"),
                }
                for st in plan.steps
                if st.post_action_properties.get("metamorphic")
            ],
        }
        user = (
            "Generate a Python program that uses the helpers in scope "
            "(see system prompt) to re-verify the truth of this run. "
            "Return a SINGLE JSON object with this exact shape:\n"
            "{\n  'program': '<python source string>'\n}\n\n"
            "Plan + trace:\n"
            + json.dumps(prompt_blob, default=str, ensure_ascii=False)
        )
        try:
            reply = self.llm.complete(
                system=system,
                messages=[Message("user", user)],
                temperature=self.config.verifier_kwargs().get("temperature", 0.0),
                json_mode=True,
            )
        except Exception as e:  # noqa: BLE001 — skip dynamic layer on transport failure.
            return [
                CheckResult(
                    name="dynamic_layer",
                    layer="dynamic",
                    passed=False,
                    detail=f"verifier LLM call failed: {e}",
                )
            ]

        program = _program_from_reply(reply)
        if program is None:
            return [
                CheckResult(
                    name="dynamic_layer",
                    layer="dynamic",
                    passed=False,
                    detail="verifier LLM reply contained no usable program",
                )
            ]
        # Compile-check before running: a syntactically broken body becomes one
        # recorded failure instead of an opaque exception mid-execution.
        try:
            compile(program, "<verify>", "exec")
        except SyntaxError as e:
            return [
                CheckResult(
                    name="dynamic_layer",
                    layer="dynamic",
                    passed=False,
                    detail=f"verifier program failed to compile: {e}",
                )
            ]

        # Surfaced on the report so the generated source lands in
        # verification.json. Without it, a program that emitted no checks (or
        # asserted the wrong thing) was completely undiagnosable after the run.
        self._last_program = program
        report = VerificationReport(
            passed=True,
            checks=[],
            dynamic_program=program,
        )
        # Sandbox execution. The program is given a tightly-scoped namespace
        # containing the MCP client, safe helpers, and nothing else.
        namespace = {
            "client": self._client,
            "call": _safe_call,
            "get_path": _get_path,
            "preview": _preview,
            "report_checks": [],
            # The system prompt sanctions these modules, so pre-bind them
            # rather than relying on the program to import them — generated
            # bodies routinely use `json.dumps` with no import line, which
            # cost the whole dynamic layer a NameError.
            "json": json,
            "re": re,
            "math": math,
            "datetime": datetime,
        }
        try:
            exec(program, namespace)  # noqa: S102 — verifier-generated code; sandboxed.
        # SystemExit is a BaseException, so a generated `exit()` / `quit()` /
        # `raise SystemExit(0)` used as an early return would sail through an
        # `except Exception`, through run_batch's per-task handler as well, and
        # terminate the whole benchmark process mid-run with a clean exit
        # status and no traceback. KeyboardInterrupt is deliberately
        # NOT caught — Ctrl-C must still stop the run.
        except (Exception, SystemExit) as e:  # noqa: BLE001 — sandbox: downgrade.
            report.dynamic_program_error = f"{type(e).__name__}: {e}"
            return [
                CheckResult(
                    name="dynamic_layer",
                    layer="dynamic",
                    passed=False,
                    detail=f"verifier program raised: {e}",
                )
            ]
        emitted = namespace.get("report_checks") or []
        if not emitted:
            # The program ran cleanly but asserted nothing. That is the
            # verifier failing to do its job, not the RUN being wrong —
            # failing here rejected otherwise-correct executions. Record it
            # as advisory so the deterministic layers still decide the run.
            return [
                CheckResult(
                    name="dynamic_layer",
                    layer="dynamic",
                    passed=False,
                    detail="verifier program produced no report_checks "
                           "(dynamic layer contributed nothing)",
                    severity="advisory",
                )
            ]
        for c in emitted:
            if not isinstance(c, dict):
                continue
            checks.append(
                CheckResult(
                    name=str(c.get("name", "dynamic_check")),
                    layer="dynamic",
                    passed=bool(c.get("passed")),
                    detail=str(c.get("detail", "")),
                )
            )
        return checks

    # ---------------------------------------------------------------- helpers

    def _tool_for(self, name: str) -> Tool | None:
        if not self.tools:
            return None
        for t in self.tools:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Helpers exported to dynamic-layer sandbox
# ---------------------------------------------------------------------------


def _safe_call(client: MCPClient, tool: str, **arguments: Any) -> Any:
    """Wrapper around MCPClient.call used by the dynamic program.

    Catches and converts MCP exceptions into a structured error return so
    the verifier program can degrade gracefully (rather than crashing the
    sandbox)."""
    try:
        return client.call(tool, arguments).data
    except MCPToolError as e:
        return {"_error": "tool_error", "text": e.text}
    except MCPProtocolError as e:
        return {"_error": "protocol_error", "code": e.code, "message": e.message}
    except MCPTransportError as e:
        return {"_error": "transport_error", "message": str(e)}


def _preview(data: Any, limit: int = 200) -> str:
    if isinstance(data, str):
        s = data
    else:
        s = json.dumps(data, default=str)
    return s[:limit] + ("..." if len(s) > limit else "")


# ---------------------------------------------------------------------------
# Plan-level cross-step relations
# ---------------------------------------------------------------------------


def _declared_cross_step_relations(plan: Plan) -> list[dict[str, Any]]:
    """Cross-step relations declared under ``plan.summary_cross_steps`` or
    embedded in step post_action_properties. The planner is free to add
    relations under any key; we look for a stable name pattern."""
    rels: list[dict[str, Any]] = []
    for st in plan.steps:
        cross = (st.post_action_properties or {}).get("cross_step")
        if isinstance(cross, dict):
            cross.setdefault("from_step", st.id)
            rels.append(cross)
    return rels


# ---------------------------------------------------------------------------
# Verifier LLM system prompt
# ---------------------------------------------------------------------------


_DYNAMIC_SYSTEM = """\
You are the verifier LLM in an agentic workflow that automates a \
read-only yahoo_finance MCP server. Given a plan, the per-step trace, \
and a list of metamorphic relations, write a small Python program that \
re-queries the server and re-checks the answers.

Helpers in scope:
- client         : an MCPClient already started against the server
- call(client, "<tool_name>", **kwargs)  -> result data
- get_path(obj, "items[0].close")         -> walk a JSON-like object
- preview(obj, limit=200)                 -> short JSON preview string
- report_checks : list; append {"name": ..., "passed": bool, "detail": "..."}

WHAT YOU ARE CHECKING — read this first. You verify that the run FETCHED \
THE RIGHT DATA and that the data still holds on a fresh query. You are NOT \
grading whether the market behaved the way the user hoped. If the task asks \
"did earnings rise for four straight quarters?", the correct outcome may be \
NO — check that the quarters were retrieved and match a re-query, never \
assert that they rose. Turning the task's premise into a pass condition \
fails correct runs.

Result shapes (authoritative — the tools return these directly):
- Every tool returns its payload DIRECTLY. `call(...)` already gives you the \
decoded value: a LIST for price bars, statements, holders, actions, \
recommendations and option chains; an OBJECT for get_stock_info; a STRING \
for news. There is no wrapper — never look for `.data`, `.result`, \
`.items` or `["rows"]` on it.
- Price bars: {"Date","Open","High","Low","Close","Volume","Dividends",\
"Stock Splits"} — capitalised, and "Date" is ISO like \
"2023-01-09T05:00:00.000Z".
- get_holder_info rows use "Date Reported" as EPOCH MILLISECONDS (an int, \
e.g. 1782777600000), NOT a date string. Convert before comparing, and \
never string-match it.

Trading-day rules (these caused more false failures than anything else):
- NEVER assert that a specific calendar date has a bar. Markets close on \
weekends and holidays, so "2022-04-15" (Good Friday) or any Saturday \
legitimately has no row. If you need a date, assert the NEAREST trading \
day on/before it, or that the returned range brackets it.
- NEVER assert a specific NUMBER of bars for a date window. The count \
depends on holidays. Assert non-empty, or that the range is covered.
- If a check's own evidence shows nothing is missing, it must PASS. Do not \
fail on a count when the dates you expected are all present.

Re-query rules:
- Re-query with the SAME arguments the step used (same ticker, same dates, \
same interval). Different filters produce different results and a \
meaningless comparison.
- For numeric price comparisons allow a 1% tolerance. If a value differs by \
more than that, prefer reporting it in the detail over failing, unless the \
task depends on that exact number.

Rules:
- Use ONLY tools in the yfinance catalog. Look at each step's `tool` field \
in the trace and re-invoke those that have a `metamorphic` note.
- Catch every exception inside your program — do NOT let it crash the \
sandbox. On any failure, append a check with passed=False and a clear \
detail.
- The program must be self-contained: only the helpers above, plus \
the standard library. json, re, math and datetime are ALREADY BOUND — use \
them directly, no import needed.
- You MUST append at least one report_checks entry before the program \
ends, and at least one per step that has a metamorphic relation. A program \
that asserts nothing is a failed verification.
- Emit the STATEMENTS THEMSELVES, executable at top level. Do NOT wrap them \
in an assignment like `program = \"\"\"...\"\"\"`, a function, or a \
`if __name__` guard — the sandbox exec()s what you return, so a wrapper \
runs nothing and silently verifies nothing.
- `report_checks` already exists in the namespace. Append to it; do not \
rebind it with `report_checks = []` at the top.
- Put the evidence in `detail` (the values you compared), so a human can \
tell a real defect from a bad check.
- Return JSON only, with shape {"program": "<full python source>"}.

The sandbox will exec your program with exec(); keep it short.
"""


_FENCE_RE = re.compile(r"```(?:python|json)?\s*\n(.*?)```", re.DOTALL)

# Salvages the program value out of a JSON envelope that failed to parse
# (unescaped newlines/quotes inside the embedded source).
_BROKEN_ENVELOPE_RE = re.compile(
    r'"program"\s*:\s*"(.*)"\s*\}\s*$', re.DOTALL
)

# Models sometimes emit the body wrapped in an assignment:
#     program = r"""  report_checks.append(...)  """
# exec()ing that only binds a string and runs nothing, so the dynamic layer
# silently contributed zero checks. Unwrap to the inner source.
_PROGRAM_ASSIGN_RE = re.compile(
    r"""^\s*(?:program|code|source|script)\s*=\s*[rbuf]{0,2}(\"\"\"|''')(?P<body>.*)\1\s*$""",
    re.DOTALL | re.IGNORECASE,
)


def _unwrap_program_assignment(src: str) -> str:
    """Return the inner body when ``src`` is just `program = \"\"\"...\"\"\"`."""
    seen = set()
    while True:
        m = _PROGRAM_ASSIGN_RE.match(src)
        if not m:
            return src
        body = m.group("body")
        if not body.strip() or body in seen:
            return src
        seen.add(body)
        src = body


def _program_from_reply(reply: str) -> str | None:
    """Recover the verifier program from an LLM reply, or None.

    Tried in order:

    1. strict JSON  -> ``{"program": "..."}``
    2. fence/prose-tolerant JSON extraction
    3. the reply treated as raw Python source

    Step 3 matters because the requested envelope asks the model to embed a
    multi-line program *inside* a JSON string. Escaping that correctly is
    exactly what models get wrong, and a malformed envelope used to discard a
    perfectly good program. A fenced code block, or bare source, is accepted.
    """
    if not reply or not reply.strip():
        return None
    text = reply.strip()
    if "</think>" in text:
        text = text.rpartition("</think>")[2].strip()

    for parse in (lambda t: json.loads(t), _extract_json):
        try:
            obj = parse(text)
        except Exception:  # noqa: BLE001 — each strategy is best-effort.
            continue
        if isinstance(obj, dict):
            prog = obj.get("program")
            if isinstance(prog, str) and prog.strip():
                return _unwrap_program_assignment(prog)

    # Fall back to raw source: prefer a fenced block, else the whole reply.
    blocks = _FENCE_RE.findall(text)
    candidate = blocks[0] if blocks else text
    candidate = candidate.strip()
    if candidate and not candidate.startswith("{"):
        return _unwrap_program_assignment(candidate)

    # Still a JSON envelope, so it failed to parse — almost always because the
    # embedded program contains raw newlines or unescaped quotes. Salvage the
    # program value textually and undo the standard escapes.
    m = _BROKEN_ENVELOPE_RE.search(text)
    if m:
        salvaged = m.group(1)
        for esc, real in (("\\n", "\n"), ("\\t", "\t"), ('\\"', '"'), ("\\'", "'")):
            salvaged = salvaged.replace(esc, real)
        salvaged = salvaged.replace("\\\\", "\\").strip()
        if salvaged:
            return _unwrap_program_assignment(salvaged)
    return None


def _extract_json(text: str) -> Any:
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if esc:
                esc = False
                continue
            if in_str:
                if ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(s[start : i + 1])
    return json.loads(s)


__all__ = ["VerifierAgent", "VerificationReport", "CheckResult"]