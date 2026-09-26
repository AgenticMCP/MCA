"""Extract Tool definitions from a Python MCP server source file.

The yfinance server (and many other FastMCP-style servers) define tools
by decorating async functions with ``@server.tool(...)``. We parse those
functions out of the source and synthesize a `Tool` from each, including
its docstring (description) and a JSON Schema inferred from the function
signature.

This is for offline catalog use: when the subprocess can't be spawned
(e.g. in CI), the orchestrator can still know the tool set. The MCP
catalog remains authoritative at runtime.
"""

from __future__ import annotations

import ast
import inspect
import re
from typing import Any, Mapping

from pkg.finance_mcp_wrapper.types import Tool


_PRIMITIVE_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
}


def _annotation_to_schema(ann: ast.AST | None) -> dict[str, Any]:
    """Translate a single AST annotation node into a JSON Schema fragment.

    Handles the common cases: bare primitives, ``list[X]`` / ``List[X]``,
    ``dict[str, X]`` / ``Dict[str, X]``, ``Optional[X]`` / ``Union[X, None]``,
    ``Literal[...]``, and bare class names by leaving them opaque.
    """
    if ann is None:
        return {}
    if isinstance(ann, ast.Name):
        return {"type": _PRIMITIVE_TO_JSON_TYPE.get(ann.id, "string")}
    if isinstance(ann, ast.Constant):
        return {"type": _PRIMITIVE_TO_JSON_TYPE.get(str(ann.value), "string")}
    if isinstance(ann, ast.Subscript):
        base = ann.value
        slice_ = ann.slice
        base_name = getattr(base, "id", None) or getattr(base, "attr", None)
        if base_name in {"list", "List"}:
            inner = _annotation_to_schema(slice_)
            return {"type": "array", "items": inner}
        if base_name in {"dict", "Dict"}:
            if isinstance(slice_, ast.Tuple) and len(slice_.elts) == 2:
                inner = _annotation_to_schema(slice_.elts[1])
            else:
                inner = _annotation_to_schema(slice_)
            return {"type": "object", "additionalProperties": inner}
        if base_name in {"Optional", "Union"}:
            # Optional[X] is Union[X, None]; strip None and keep the rest.
            if isinstance(slice_, ast.Tuple):
                variants = [
                    _annotation_to_schema(e)
                    for e in slice_.elts
                    if not (isinstance(e, ast.Constant) and e.value is None)
                ]
                if len(variants) == 1:
                    return variants[0]
                return {"anyOf": variants}
            return _annotation_to_schema(slice_)
        # Generic[...] subscript — fall through
        return {}
    if isinstance(ann, ast.Attribute):
        return {}  # unknown class — opaque
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        # PEP 604 union: `int | None`
        left = _annotation_to_schema(ann.left)
        right = _annotation_to_schema(ann.right)
        merged: list[Any] = []
        for v in (left, right):
            if v and "anyOf" in v:
                merged.extend(v["anyOf"])
            elif v:
                merged.append(v)
        return {"anyOf": merged} if len(merged) > 1 else (merged[0] if merged else {})
    return {}


def _signature_to_schema(sig: inspect.Signature, defaults: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Build an inputSchema from an inspect.Signature, honoring type
    annotations and default values. Returns (schema, required_field_names)."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else None
        if isinstance(ann, str):
            try:
                ann_node = ast.parse(ann, mode="eval").body
            except SyntaxError:
                ann_node = ast.Name(id=ann)
            schema = _annotation_to_schema(ann_node)
        else:
            schema = _annotation_to_schema(ann)
        # Default value: prefer the AST default captured at decoration
        # time (defaults[name]); fall back to the inspect-level default.
        if name in defaults:
            schema["default"] = defaults[name]
        elif param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = schema
    out: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        out["required"] = required
    return out, required


def _parse_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if not node.body:
        return ""
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return inspect.cleandoc(first.value.value)
    return ""


def _capture_decorator_kwargs(decorators: list[ast.expr]) -> dict[str, Any]:
    """Pull keyword args out of ``@server.tool(name=..., description=...)``.

    FastMCP's ``@server.tool`` decorator accepts a small handful of
    kwargs; we capture the ones that influence the public Tool shape.
    """
    out: dict[str, Any] = {}
    for dec in decorators:
        # Match either `@server.tool` or `@some_var.tool`.
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        attr_ok = isinstance(func, ast.Attribute) and func.attr == "tool"
        if not attr_ok:
            continue
        for kw in dec.keywords:
            value = kw.value
            if isinstance(value, ast.Constant):
                out[kw.arg] = value.value
            elif isinstance(value, ast.Name):
                out[kw.arg] = value.id  # reference; resolve at runtime
    return out


def _default_from_decorator_kwargs(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, decorator_args: dict[str, Any]
) -> dict[str, Any]:
    """Return a {param_name: default} map derived from the AST literal
    defaults and any decorator-supplied values.

    FastMCP accepts a single positional argument to ``@server.tool``: a
    default ``{}`` for the first parameter (used for params-free tools).
    We capture both literal ``arg=...`` kwargs and that positional case.
    """
    out: dict[str, Any] = {}
    # First, kw defaults from the decorator itself
    out.update({k: v for k, v in decorator_args.items() if k in fn.args.args})
    # Then, positional defaults from the AST function signature.
    args = fn.args
    positional = args.args
    defaults = args.defaults  # right-aligned list of defaults for positional args
    if defaults:
        offset = len(positional) - len(defaults)
        for i, default in enumerate(defaults):
            param_index = offset + i
            if param_index < 0 or param_index >= len(positional):
                continue
            param_name = positional[param_index].arg
            if param_name in out:
                continue  # decorator-supplied default wins
            try:
                out[param_name] = ast.literal_eval(default)
            except (ValueError, SyntaxError):
                pass
    return out


def extract_tools_from_source(source: str) -> list[Tool]:
    """Parse the Python source string and return the Tool definitions
    synthesized from each ``@server.tool``-decorated function."""
    tree = ast.parse(source)
    out: list[Tool] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        kwargs = _capture_decorator_kwargs(node.decorator_list)
        if not kwargs and not node.decorator_list:
            continue
        # Only treat as a tool if at least one decorator ends with `.tool(...)`
        is_tool = any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "tool"
            for d in node.decorator_list
        )
        if not is_tool:
            continue
        defaults = _default_from_decorator_kwargs(node, kwargs)
        sig = _signature_from_ast(node)
        if sig is None:
            continue
        schema, required = _signature_to_schema(sig, defaults)
        description = kwargs.get("description") or _parse_docstring(node)
        name = kwargs.get("name") or node.name
        out.append(
            Tool(
                name=name,
                description=description,
                input_schema=schema,
                annotations={"required": required},
            )
        )
    return out


def extract_tools_from_python_file(path: str) -> list[Tool]:
    with open(path, encoding="utf-8") as f:
        source = f.read()
    return extract_tools_from_source(source)


def _signature_from_ast(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> inspect.Signature | None:
    """Reconstruct an inspect.Signature from AST for schema inference."""
    try:
        parameters: list[inspect.Parameter] = []
        pos = [a.arg for a in fn.args.args]
        defaults_nodes = list(fn.args.defaults)
        defaults: list[Any] = []
        offset = len(pos) - len(defaults_nodes)
        for i, dn in enumerate(defaults_nodes):
            if i + offset < 0:
                defaults.append(inspect.Parameter.empty)
                continue
            try:
                defaults.append(ast.literal_eval(dn))
            except (ValueError, SyntaxError):
                defaults.append(inspect.Parameter.empty)
        for i, name in enumerate(pos):
            d = defaults[i] if i < len(defaults) and i >= offset else inspect.Parameter.empty
            a = fn.args.args[i].annotation
            ann: Any = inspect.Parameter.empty
            if a is not None:
                ann = ast.unparse(a)
            parameters.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=d,
                    annotation=ann,
                )
            )
        if fn.args.vararg:
            parameters.append(
                inspect.Parameter(
                    name=fn.args.vararg.arg,
                    kind=inspect.Parameter.VAR_POSITIONAL,
                    annotation=ast.unparse(fn.args.vararg.annotation)
                    if fn.args.vararg.annotation
                    else inspect.Parameter.empty,
                )
            )
        for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            parameters.append(
                inspect.Parameter(
                    name=a.arg,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=ast.literal_eval(d) if d is not None else inspect.Parameter.empty,
                    annotation=ast.unparse(a.annotation)
                    if a.annotation
                    else inspect.Parameter.empty,
                )
            )
        ret: Any = inspect.Signature.empty
        if fn.returns is not None:
            try:
                ret = ast.unparse(fn.returns)
            except Exception:  # noqa: BLE001
                ret = inspect.Signature.empty
        return inspect.Signature(parameters, return_annotation=ret)
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "extract_tools_from_source",
    "extract_tools_from_python_file",
]