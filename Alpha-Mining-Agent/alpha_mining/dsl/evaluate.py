"""Evaluate a parsed expression tree on feature panels."""
import numpy as np

from alpha_mining.dsl import ops, spec
from alpha_mining.dsl.parse import Node, canonical, parse

_TS = {name: getattr(ops, name) for name in spec.TIME_SERIES}
_CS = {"rank": ops.rank, "zscore": ops.zscore, "scale": ops.scale}


def evaluate(expr: str | Node, features: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Compute a factor as a (T, N) array, NaN outside `mask`.

    `features` maps every name in spec.FEATURES to a (T, N) array on the full,
    unmasked panel; `mask` is the boolean membership mask of the same shape.
    """
    node = parse(expr) if isinstance(expr, str) else node_or_raise(expr)
    shape = mask.shape
    memo: dict[str, np.ndarray] = {}

    def as_array(v):
        return np.full(shape, v, dtype=np.float64) if np.isscalar(v) else v

    def ev(n: Node):
        if n.op == "const":
            return n.value
        key = canonical(n)
        if key in memo:
            return memo[key]
        if n.op in spec.FEATURES:
            out = np.asarray(features[n.op], dtype=np.float64)
        elif n.op in ("add", "sub", "mul", "div"):
            a, b = ev(n.args[0]), ev(n.args[1])
            if n.op == "add":
                out = a + b
            elif n.op == "sub":
                out = a - b
            elif n.op == "mul":
                out = a * b
            else:
                out = ops.protected_div(as_array(a), as_array(b))
            out = ops._clean(as_array(out))
        elif n.op == "neg":
            out = -as_array(ev(n.args[0]))
        elif n.op == "abs":
            out = np.abs(as_array(ev(n.args[0])))
        elif n.op == "sign":
            out = np.sign(as_array(ev(n.args[0])))
        elif n.op in ("max", "min"):
            fn = np.maximum if n.op == "max" else np.minimum
            out = fn(as_array(ev(n.args[0])), as_array(ev(n.args[1])))
        elif n.op == "log":
            out = ops.protected_log(as_array(ev(n.args[0])))
        elif n.op in _CS:
            out = _CS[n.op](as_array(ev(n.args[0])), mask)
        elif n.op in _TS:
            out = _TS[n.op](*(as_array(ev(a)) for a in n.args), n.window)
        else:
            raise AssertionError(f"unhandled op {n.op}")
        memo[key] = out
        return out

    return np.where(mask, as_array(ev(node)), np.nan)


def node_or_raise(node: Node) -> Node:
    # Re-validate trees built elsewhere (e.g. by GP) through the same text path.
    return parse(canonical(node))
