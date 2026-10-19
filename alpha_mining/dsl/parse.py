"""Parse, validate and canonicalize single-line factor expressions.

Expressions are Python syntax restricted to the whitelist in `spec`. Parsing goes
through `ast` and only the node types handled below are accepted, so nothing in
the string is ever executed. Error messages are written to be returned verbatim
to the LLM.
"""
import ast
import math
from dataclasses import dataclass

from alpha_mining.dsl import spec


class DSLError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    op: str                       # feature name, "const", operator name, or add/sub/mul/div/neg
    args: tuple["Node", ...] = ()
    window: int | None = None
    value: float | None = None

    @property
    def depth(self) -> int:
        return 1 + max((a.depth for a in self.args), default=0)

    @property
    def size(self) -> int:
        """Node count; window parameters are not nodes."""
        return 1 + sum(a.size for a in self.args)

    def __str__(self) -> str:
        return canonical(self)


_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def _convert(n: ast.AST) -> Node:
    if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
        return Node(_BINOPS[type(n.op)], (_convert(n.left), _convert(n.right)))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        inner = _convert(n.operand)
        if isinstance(n.op, ast.UAdd):
            return inner
        if inner.op == "const":
            return Node("const", value=-inner.value)
        return Node("neg", (inner,))
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        if not math.isfinite(n.value):
            raise DSLError("numeric constants must be finite")
        return Node("const", value=float(n.value))
    if isinstance(n, ast.Name):
        if n.id not in spec.FEATURES:
            raise DSLError(f"unknown feature '{n.id}'; allowed: {', '.join(spec.FEATURES)}")
        return Node(n.id)
    if isinstance(n, ast.Call):
        return _convert_call(n)
    raise DSLError(f"unsupported syntax: {ast.unparse(n)!r}")


def _convert_call(n: ast.Call) -> Node:
    if not isinstance(n.func, ast.Name):
        raise DSLError(f"unsupported call: {ast.unparse(n.func)!r}")
    name = n.func.id
    if n.keywords:
        raise DSLError(f"{name}: keyword arguments are not allowed")
    args = n.args
    if name in spec.CROSS_SECTIONAL or name in spec.ELEMENTWISE:
        if len(args) != 1:
            raise DSLError(f"{name}(x) takes exactly 1 argument, got {len(args)}")
        return Node(name, (_convert(args[0]),))
    if name in spec.ELEMENTWISE_BINARY:
        if len(args) != 2:
            raise DSLError(f"{name}(x, y) takes exactly 2 arguments, got {len(args)}")
        return Node(name, (_convert(args[0]), _convert(args[1])))
    if name in spec.TIME_SERIES:
        n_series, windows = spec.TIME_SERIES[name]
        sig = f"{name}({', '.join('xy'[:n_series])}, d)"
        if len(args) != n_series + 1:
            raise DSLError(f"{sig} takes {n_series + 1} arguments, got {len(args)}")
        d = args[-1]
        if not (isinstance(d, ast.Constant) and type(d.value) is int and d.value in windows):
            raise DSLError(f"{sig}: d must be one of {list(windows)}, got {ast.unparse(d)!r}")
        return Node(name, tuple(_convert(a) for a in args[:-1]), window=d.value)
    known = spec.CROSS_SECTIONAL + spec.ELEMENTWISE + spec.ELEMENTWISE_BINARY + tuple(spec.TIME_SERIES)
    raise DSLError(f"unknown operator '{name}'; allowed: {', '.join(known)}")


def parse(expr: str) -> Node:
    """Parse and validate an expression; raises DSLError with an LLM-readable message."""
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("empty expression")
    if "\n" in expr.strip():
        raise DSLError("expression must be a single line")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise DSLError(f"syntax error: {e.msg}") from None
    node = _convert(tree.body)
    if node.op == "const":
        raise DSLError("expression is a constant")
    if not _has_feature(node):
        raise DSLError("expression uses no feature")
    if node.depth > spec.MAX_DEPTH:
        raise DSLError(f"expression depth {node.depth} exceeds the limit of {spec.MAX_DEPTH}")
    if node.size > spec.MAX_NODES:
        raise DSLError(f"expression has {node.size} nodes, exceeding the limit of {spec.MAX_NODES}")
    return node


def _has_feature(node: Node) -> bool:
    return node.op in spec.FEATURES or any(_has_feature(a) for a in node.args)


def _flatten(node: Node, op: str) -> list[Node]:
    if node.op == op:
        return [x for a in node.args for x in _flatten(a, op)]
    return [node]


def canonical(node: Node) -> str:
    """Canonical text: commutative operands sorted, +/* chains flattened.

    Two expressions that differ only in operand order or parenthesization of
    + and * map to the same string; it is used for duplicate detection.
    """
    op = node.op
    if op == "const":
        return format(node.value, "g")
    if op in spec.FEATURES:
        return op
    if op in ("add", "mul"):
        parts = sorted(canonical(a) for a in _flatten(node, op))
        return "(" + f" {_SYMBOLS[op]} ".join(parts) + ")"
    if op in ("sub", "div"):
        return f"({canonical(node.args[0])} {_SYMBOLS[op]} {canonical(node.args[1])})"
    if op == "neg":
        return f"-{canonical(node.args[0])}"
    args = [canonical(a) for a in node.args]
    if op in spec.SYMMETRIC:
        args.sort()
    if node.window is not None:
        args.append(str(node.window))
    return f"{op}({', '.join(args)})"
