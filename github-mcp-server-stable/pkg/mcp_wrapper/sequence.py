"""Run a sequence of MCP tool calls with binding resolution + schema checks.

A `Step` declares one tool invocation. Steps run in order. Each step can:

* Pull values from a previous step's result via the binding syntax
  ``"$<step_id>.<dotted.path[index]>"`` (or ``"$<step_id>"`` for the whole
  decoded payload). The path traverses the step's `result.data` (i.e. the
  best-effort JSON decode of the MCP text content).

  A binding that is the **whole value** of an argument keeps its native type
  (e.g. ``{"issue_number": "$s1.number"}`` becomes the integer 42, not the
  string "42"). ``$sN.path`` segments may ALSO be **embedded** inside a longer
  string (``"repo:$s1.items[0].full_name is:issue"``, CSV/report content like
  ``"$s1.items[0].full_name,$s2.totalCount"``): each segment is substituted
  with the value rendered as text (strings as-is, everything else as JSON).
  Embedded matching is restricted to numbered step ids (``$s0``…) with at
  least one path segment, so ordinary text like ``"$scope.foo"`` is never
  touched; an embedded segment that cannot be resolved raises
  ``BindingError`` instead of silently passing the template through.
* Be pre-validated against the tool's advertised `inputSchema` (caught here
  with descriptive messages instead of being rejected by the server).
* Be post-validated against a caller-supplied `expect_output` schema.

The orchestrator records per-step status (success / validation_error /
binding_error / tool_error / protocol_error / skipped) and either stops at
the first failure or continues, per the `on_error` policy.

This module is intentionally schema-aware but does NOT depend on the
``jsonschema`` package — it implements the subset of JSON Schema that the
github-mcp-server actually uses in its tool catalog (type, properties,
required, additionalProperties, enum, minimum, maximum, minLength,
maxLength, pattern, items, anyOf, oneOf).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from .types import MCPProtocolError, MCPToolError, ToolResult

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

StepStatus = Literal[
    "pending",
    "success",
    "skipped",
    "validation_error",
    "binding_error",
    "tool_error",
    "protocol_error",
    "output_mismatch",
]

OnError = Literal["stop", "continue"]


@dataclass
class Step:
    """One tool invocation in a sequence.

    Attributes
    ----------
    tool:
        Name of the MCP tool to invoke (e.g. ``"create_issue"``).
    arguments:
        Arguments to pass. Values may be binding templates like
        ``"$s1.number"`` that are resolved against earlier step outputs.
    id:
        Unique identifier for this step inside the sequence. Used both as
        the binding source name (``$<id>.<path>``) and as the lookup key
        in `SequenceResult.outputs`. If omitted, an auto id ``step_<N>`` is
        assigned by the orchestrator.
    expect_output:
        Optional JSON Schema fragment to validate the decoded step result
        against. Failures produce status ``output_mismatch``.
    validate_input:
        When True (default), the resolved arguments are pre-validated
        against the tool's declared `inputSchema`. Set to False to skip.
    description:
        Free-form human-readable label. Echoed in `StepResult` for logging.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    expect_output: dict[str, Any] | None = None
    validate_input: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "Step":
        if "tool" not in obj:
            raise ValueError(f"step dict missing required key 'tool': {obj!r}")
        return cls(
            tool=obj["tool"],
            arguments=dict(obj.get("arguments") or obj.get("args") or {}),
            id=obj.get("id", ""),
            expect_output=obj.get("expect_output") or obj.get("expect"),
            validate_input=bool(obj.get("validate_input", True)),
            description=obj.get("description", ""),
        )


@dataclass
class StepResult:
    """Outcome of one step in a sequence."""

    id: str
    tool: str
    status: StepStatus
    arguments_resolved: dict[str, Any] | None = None
    result: ToolResult | None = None
    error: str | None = None
    error_details: list[str] | None = None
    description: str = ""

    @property
    def data(self) -> Any:
        """Convenience accessor: decoded data of the result, or None."""
        return self.result.data if self.result is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "status": self.status,
            "description": self.description,
            "arguments_resolved": self.arguments_resolved,
            "result": (
                {
                    "is_error": self.result.is_error,
                    "data": self.result.data,
                    "content": self.result.content,
                }
                if self.result is not None
                else None
            ),
            "error": self.error,
            "error_details": self.error_details,
        }


@dataclass
class SequenceResult:
    """Outcome of executing a sequence."""

    steps: list[StepResult] = field(default_factory=list)

    @property
    def outputs(self) -> dict[str, Any]:
        """Map of step id → decoded result data, for completed steps."""
        return {
            sr.id: sr.result.data
            for sr in self.steps
            if sr.result is not None and sr.status in ("success",)
        }

    @property
    def success(self) -> bool:
        return all(sr.status == "success" for sr in self.steps)

    def by_id(self, step_id: str) -> StepResult | None:
        for sr in self.steps:
            if sr.id == step_id:
                return sr
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Binding resolution
# ---------------------------------------------------------------------------

# Binding syntax: "$<step_id>" or "$<step_id><path>". Path supports
# dot-separated keys and bracket indexing, beginning with either: e.g.
# "$s2.items[0].id" or "$s3[2]".
_BINDING_RE = re.compile(
    r"^\$(?P<step>[A-Za-z_][A-Za-z0-9_-]*)(?P<path>[.\[].*)?$"
)

# Embedded form: "$sN.path" segments INSIDE a longer string (search queries,
# report/file content composed from prior outputs). Restricted to numbered
# step ids so ordinary text like "$scope.foo" is never touched, and requires
# at least one path segment (a bare "$s1" embedded in prose stays literal).
# Key charset excludes "-" on purpose: "$s1.name-suffix" must read as the
# binding ".name" followed by the literal "-suffix".
_EMBEDDED_RE = re.compile(
    r"\$(?P<step>s\d+)(?P<path>(?:\.[A-Za-z0-9_]+|\[\d+\])+)"
)


class BindingError(Exception):
    """Raised when a binding template cannot be resolved against prior outputs."""


def _split_path(path: str) -> list[str | int]:
    """Split a path like ``items[0].name`` into ``['items', 0, 'name']``."""
    tokens: list[str | int] = []
    i = 0
    n = len(path)
    while i < n:
        # bracket index
        if path[i] == "[":
            j = path.find("]", i + 1)
            if j == -1:
                raise BindingError(f"unterminated '[' in path {path!r}")
            inner = path[i + 1 : j].strip()
            try:
                tokens.append(int(inner))
            except ValueError as e:
                raise BindingError(
                    f"non-integer array index {inner!r} in path {path!r}"
                ) from e
            i = j + 1
            if i < n and path[i] == ".":
                i += 1
            continue
        # dotted key
        j = i
        while j < n and path[j] not in ".[":
            j += 1
        key = path[i:j]
        if not key:
            raise BindingError(f"empty segment in path {path!r}")
        tokens.append(key)
        i = j
        if i < n and path[i] == ".":
            i += 1
    return tokens


def _traverse(obj: Any, tokens: Sequence[str | int], original: str) -> Any:
    cur = obj
    for tok in tokens:
        if isinstance(tok, int):
            if not isinstance(cur, list):
                raise BindingError(
                    f"binding {original!r}: expected list at segment [{tok}], got {type(cur).__name__}"
                )
            if tok < 0 or tok >= len(cur):
                raise BindingError(
                    f"binding {original!r}: index {tok} out of range (len={len(cur)})"
                )
            cur = cur[tok]
        else:
            if isinstance(cur, Mapping):
                if tok not in cur:
                    raise BindingError(
                        f"binding {original!r}: key {tok!r} not found "
                        f"(available: {sorted(cur.keys())[:8]})"
                    )
                cur = cur[tok]
            else:
                raise BindingError(
                    f"binding {original!r}: cannot index {type(cur).__name__} with key {tok!r}"
                )
    return cur


def _looks_like_binding(s: str) -> bool:
    return bool(s) and s.startswith("$") and bool(_BINDING_RE.match(s))


def _render(v: Any) -> str:
    """A resolved value as text for embedded substitution: strings verbatim,
    everything else (numbers, booleans, null, objects, arrays) as JSON."""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, default=str)


def _interpolate(s: str, outputs: Mapping[str, Any]) -> str:
    """Substitute every embedded ``$sN.path`` segment in ``s`` with its
    resolved value. Unresolvable segments raise BindingError — a template
    must never silently pass through as literal text."""

    def repl(m: "re.Match[str]") -> str:
        step_id = m.group("step")
        if step_id not in outputs:
            raise BindingError(
                f"embedded binding {m.group(0)!r}: step id {step_id!r} not in "
                f"outputs (available: {sorted(outputs.keys())})"
            )
        path = m.group("path")
        if path.startswith("."):
            path = path[1:]
        tokens = _split_path(path)
        return _render(_traverse(outputs[step_id], tokens, original=m.group(0)))

    return _EMBEDDED_RE.sub(repl, s)


def resolve_bindings(value: Any, outputs: Mapping[str, Any]) -> Any:
    """Recursively resolve binding templates inside ``value`` against
    ``outputs`` (a map of step id → decoded step result data).

    Strings of the form ``"$step_id"`` or ``"$step_id.path"`` are replaced
    by the looked-up value (type preserved). Other strings have any embedded
    ``$sN.path`` segments substituted as text. Dicts and lists are walked
    recursively. Any other value is returned unchanged.
    """
    if isinstance(value, str):
        if not _looks_like_binding(value):
            # Not a whole-value template; substitute any embedded segments.
            if "$s" in value:
                return _interpolate(value, outputs)
            return value
        m = _BINDING_RE.match(value)
        assert m is not None
        step_id = m.group("step")
        path = m.group("path")
        if step_id not in outputs:
            raise BindingError(
                f"binding {value!r}: step id {step_id!r} not in outputs "
                f"(available: {sorted(outputs.keys())})"
            )
        root = outputs[step_id]
        if not path:
            return root
        # Strip the leading "." when present; "[..." is handled directly by
        # _split_path.
        if path.startswith("."):
            path = path[1:]
        tokens = _split_path(path)
        return _traverse(root, tokens, original=value)
    if isinstance(value, Mapping):
        return {k: resolve_bindings(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_bindings(v, outputs) for v in value]
    return value


# ---------------------------------------------------------------------------
# Minimal JSON Schema validator
# ---------------------------------------------------------------------------

_TYPE_CHECKERS: dict[str, callable] = {  # type: ignore[type-arg]
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _matches_type(value: Any, declared: Any) -> bool:
    if declared is None:
        return True
    if isinstance(declared, str):
        checker = _TYPE_CHECKERS.get(declared)
        return bool(checker and checker(value))
    if isinstance(declared, list):
        return any(_matches_type(value, t) for t in declared)
    return True  # unknown declaration shape — be permissive


def validate_against_schema(
    value: Any,
    schema: Mapping[str, Any] | None,
    *,
    path: str = "",
) -> list[str]:
    """Return a list of validation error messages. Empty list means valid.

    Supports the subset of JSON Schema actually used in github-mcp-server's
    tool catalog: ``type``, ``properties``, ``required``,
    ``additionalProperties``, ``enum``, ``minimum``, ``maximum``,
    ``minLength``, ``maxLength``, ``pattern``, ``items``, ``anyOf``, ``oneOf``.
    """
    if not schema:
        return []
    errs: list[str] = []
    here = path or "<root>"

    # type
    declared_type = schema.get("type")
    if declared_type is not None and not _matches_type(value, declared_type):
        errs.append(f"{here}: expected type {declared_type!r}, got {type(value).__name__}")
        return errs  # later checks would compound the confusion

    # enum
    if "enum" in schema:
        if value not in schema["enum"]:
            errs.append(f"{here}: value {value!r} not in enum {schema['enum']!r}")

    # numeric bounds
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{here}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{here}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errs.append(f"{here}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errs.append(f"{here}: {value} >= exclusiveMaximum {schema['exclusiveMaximum']}")

    # string bounds
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{here}: length {len(value)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{here}: length {len(value)} > maxLength {schema['maxLength']}")
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    errs.append(f"{here}: {value!r} does not match pattern {schema['pattern']!r}")
            except re.error as e:
                errs.append(f"{here}: invalid regex pattern {schema['pattern']!r} ({e})")

    # object
    if isinstance(value, Mapping):
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        for k in required:
            if k not in value:
                errs.append(f"{here}: missing required property {k!r}")
        for k, v in value.items():
            sub = props.get(k)
            if sub is not None:
                errs.extend(validate_against_schema(v, sub, path=f"{here}.{k}"))
        addl = schema.get("additionalProperties")
        if addl is False:
            extras = set(value.keys()) - set(props.keys())
            if extras:
                errs.append(f"{here}: unexpected properties {sorted(extras)!r}")
        elif isinstance(addl, Mapping):
            for k, v in value.items():
                if k in props:
                    continue
                errs.extend(validate_against_schema(v, addl, path=f"{here}.{k}"))

    # array
    if isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, Mapping):
            for i, v in enumerate(value):
                errs.extend(validate_against_schema(v, items_schema, path=f"{here}[{i}]"))
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{here}: length {len(value)} < minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{here}: length {len(value)} > maxItems {schema['maxItems']}")

    # combinators
    if "anyOf" in schema:
        matched = any(
            not validate_against_schema(value, sub, path=here)
            for sub in schema["anyOf"]
            if isinstance(sub, Mapping)
        )
        if not matched:
            errs.append(f"{here}: did not match any of anyOf schemas")
    if "oneOf" in schema:
        matches = sum(
            1
            for sub in schema["oneOf"]
            if isinstance(sub, Mapping) and not validate_against_schema(value, sub, path=here)
        )
        if matches != 1:
            errs.append(f"{here}: matched {matches} of oneOf schemas (expected exactly 1)")

    return errs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _normalize_steps(steps: Iterable[Step | Mapping[str, Any]]) -> list[Step]:
    norm: list[Step] = []
    for raw in steps:
        if isinstance(raw, Step):
            norm.append(raw)
        elif isinstance(raw, Mapping):
            norm.append(Step.from_dict(raw))
        else:
            raise TypeError(f"step must be Step or dict, got {type(raw).__name__}")
    seen_ids: set[str] = set()
    for i, st in enumerate(norm):
        if not st.id:
            st.id = f"step_{i + 1}"
        if st.id in seen_ids:
            raise ValueError(f"duplicate step id {st.id!r}")
        seen_ids.add(st.id)
    return norm


def execute_sequence(
    client: Any,  # MCPClient — typed loosely to avoid the import cycle
    steps: Iterable[Step | Mapping[str, Any]],
    *,
    on_error: OnError = "stop",
) -> SequenceResult:
    """Run a sequence of tool calls in order, with binding resolution and
    schema validation. Returns a `SequenceResult` summarizing every step.

    Parameters
    ----------
    client:
        A started `MCPClient` instance.
    steps:
        Iterable of `Step` objects or dicts (see `Step.from_dict`).
    on_error:
        ``"stop"`` (default) marks every remaining step as ``skipped`` after
        the first failure. ``"continue"`` keeps going.
    """
    normalized = _normalize_steps(steps)
    seq = SequenceResult()
    outputs: dict[str, Any] = {}
    abort = False

    for st in normalized:
        sr = StepResult(
            id=st.id,
            tool=st.tool,
            status="pending",
            description=st.description,
        )

        if abort:
            sr.status = "skipped"
            sr.error = "prior step failed and on_error='stop'"
            seq.steps.append(sr)
            continue

        # 1. Resolve bindings
        try:
            resolved = resolve_bindings(st.arguments, outputs)
        except BindingError as e:
            sr.status = "binding_error"
            sr.error = str(e)
            seq.steps.append(sr)
            abort = on_error == "stop"
            continue
        if not isinstance(resolved, Mapping):
            sr.status = "binding_error"
            sr.error = f"resolved arguments must be an object, got {type(resolved).__name__}"
            seq.steps.append(sr)
            abort = on_error == "stop"
            continue
        sr.arguments_resolved = dict(resolved)

        # 2. Validate input schema
        if st.validate_input:
            tool = client.get_tool(st.tool)
            if tool is None:
                sr.status = "validation_error"
                sr.error = f"unknown tool {st.tool!r}"
                seq.steps.append(sr)
                abort = on_error == "stop"
                continue
            errs = validate_against_schema(sr.arguments_resolved, tool.input_schema)
            if errs:
                sr.status = "validation_error"
                sr.error = f"input failed schema validation ({len(errs)} issue(s))"
                sr.error_details = errs
                seq.steps.append(sr)
                abort = on_error == "stop"
                continue

        # 3. Invoke the tool
        try:
            tr = client.call(st.tool, sr.arguments_resolved)
        except MCPToolError as e:
            sr.status = "tool_error"
            sr.error = e.text
            sr.result = ToolResult.from_mcp(st.tool, e.raw)
            seq.steps.append(sr)
            abort = on_error == "stop"
            continue
        except MCPProtocolError as e:
            sr.status = "protocol_error"
            sr.error = f"JSON-RPC error {e.code}: {e.message}"
            seq.steps.append(sr)
            abort = on_error == "stop"
            continue
        sr.result = tr

        # 4. Validate output schema (caller-supplied)
        if st.expect_output:
            errs = validate_against_schema(tr.data, st.expect_output)
            if errs:
                sr.status = "output_mismatch"
                sr.error = f"output failed expect_output schema ({len(errs)} issue(s))"
                sr.error_details = errs
                seq.steps.append(sr)
                abort = on_error == "stop"
                continue

        # 5. Record success and feed forward
        sr.status = "success"
        outputs[st.id] = tr.data
        seq.steps.append(sr)

    return seq


def load_steps_from_json(path: str) -> list[Step]:
    """Load a sequence definition from a JSON file (a list of step dicts)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a top-level JSON array of steps")
    return [Step.from_dict(s) for s in raw]


__all__ = [
    "Step",
    "StepStatus",
    "StepResult",
    "SequenceResult",
    "BindingError",
    "execute_sequence",
    "resolve_bindings",
    "validate_against_schema",
    "load_steps_from_json",
]
