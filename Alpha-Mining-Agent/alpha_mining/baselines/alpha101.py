"""Interpreter for the formula language of Kakushadze (2016), "101 Formulaic Alphas".

Formulas are taken verbatim from data/alpha101_formulas.csv (extracted from the
paper) and evaluated with a small Pratt parser, so no formula is retyped by hand.
Semantics follow the paper's Appendix A.1:
  * window arguments d may be non-integer and are floored;
  * min(x, d) / max(x, d) with a numeric d are ts_min / ts_max; with two series
    (Alphas 71, 73, ...) they are taken elementwise;
  * ts_argmax / ts_argmin return the 1-based position of the extreme within the
    window, counted from the oldest day;
  * signedpower(x, a) = sign(x) * |x|^a;
  * vwap is approximated by the typical price (high + low + close) / 3, and
    adv{d} is the d-day mean dollar volume.
Cross-sectional rank/scale use the membership mask, as in the DSL.
"""
import math
import re

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from alpha_mining.config import ROOT
from alpha_mining.dsl import ops

FORMULAS_FILE = ROOT / "data" / "alpha101_formulas.csv"

_TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|([A-Za-z_][A-Za-z_0-9.]*)|(\|\||&&|==|<=|>=|[-+*/^<>?:(),]))")


def tokenize(s: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot tokenize at {s[pos:pos + 20]!r}")
        num, name, op = m.groups()
        out.append(("num", num) if num else ("name", name.lower()) if name else ("op", op))
        pos = m.end()
    return out


# Binding powers: ternary < || < && < comparison < additive < multiplicative < power.
_BINARY = {"||": 2, "&&": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "==": 4,
           "+": 5, "-": 5, "*": 6, "/": 6, "^": 8}


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("eof", "")

    def take(self, value=None):
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise ValueError(f"expected {value!r}, got {tok[1]!r}")
        self.i += 1
        return tok

    def expr(self, min_bp: int = 0):
        kind, v = self.take()
        if kind == "num":
            left = ("num", float(v))
        elif kind == "name":
            if self.peek()[1] == "(":
                self.take("(")
                args = [self.expr()]
                while self.peek()[1] == ",":
                    self.take(",")
                    args.append(self.expr())
                self.take(")")
                left = ("call", v, args)
            else:
                left = ("var", v)
        elif v == "(":
            left = self.expr()
            self.take(")")
        elif v == "-":
            left = ("neg", self.expr(7))
        else:
            raise ValueError(f"unexpected token {v!r}")
        while True:
            op = self.peek()[1]
            if op == "?" and min_bp <= 1:
                self.take("?")
                a = self.expr(1)
                self.take(":")
                b = self.expr(1)  # right-associative
                left = ("if", left, a, b)
                continue
            bp = _BINARY.get(op)
            if bp is None or bp < min_bp:
                return left
            self.take()
            left = ("bin", op, left, self.expr(bp + 1 if op != "^" else bp))


def parse(formula: str):
    p = _Parser(tokenize(formula))
    tree = p.expr()
    if p.peek()[0] != "eof":
        raise ValueError(f"trailing input at {p.peek()[1]!r}")
    return tree


def _window(d) -> int:
    if not np.isscalar(d):
        raise ValueError("window argument must be a number")
    return max(1, int(math.floor(d)))


def _rolling(x, d, fn):
    return ops._clean(getattr(pd.DataFrame(x).rolling(d, min_periods=d), fn)().to_numpy())


def _ts_arg(x, d, extreme):
    out = np.full(x.shape, np.nan)
    if d > len(x):
        return out
    win = sliding_window_view(x, d, axis=0)  # (T-d+1, N, d), a view
    full = np.isfinite(win).all(axis=-1)
    idx = extreme(np.where(np.isfinite(win), win, -np.inf if extreme is np.argmax else np.inf), axis=-1)
    out[d - 1:] = np.where(full, idx + 1.0, np.nan)
    return out


def _product(x, d):
    return ops._clean(pd.DataFrame(x).rolling(d, min_periods=d).apply(np.prod, raw=True).to_numpy())


class Alpha101:
    def __init__(self, features: dict[str, np.ndarray], mask: np.ndarray):
        self.mask = mask
        f = features
        self.vars = {"open": f["open"], "high": f["high"], "low": f["low"], "close": f["close"],
                     "volume": f["volume"], "returns": f["returns"], "vwap": f["typical"]}
        self.dollar_vol = f["dollar_vol"]
        self.shape = mask.shape

    def var(self, name):
        if name in self.vars:
            return self.vars[name]
        m = re.fullmatch(r"adv(\d+)", name)
        if m:
            d = int(m.group(1))
            self.vars[name] = ops.ts_mean(self.dollar_vol, d)
            return self.vars[name]
        raise ValueError(f"unsupported input {name!r}")

    def arr(self, v):
        return np.full(self.shape, float(v)) if np.isscalar(v) else v

    def call(self, name, args):
        a = [self.ev(x) for x in args]
        series = lambda i: self.arr(a[i])
        if name in ("rank",):
            return ops.rank(series(0), self.mask)
        if name == "scale":
            return ops.scale(series(0), self.mask) * (a[1] if len(a) > 1 else 1.0)
        if name in ("abs", "log", "sign"):
            fn = {"abs": np.abs, "sign": np.sign, "log": lambda z: np.log(np.where(z > 0, z, np.nan))}[name]
            return ops._clean(fn(series(0)))
        if name == "signedpower":
            x = series(0)
            return ops._clean(np.sign(x) * np.abs(x) ** a[1])
        if name in ("delay", "delta"):
            return getattr(ops, "ts_" + name)(series(0), _window(a[1]))
        if name in ("correlation", "covariance"):
            fn = ops.ts_corr if name == "correlation" else ops.ts_cov
            return fn(series(0), series(1), _window(a[2]))
        if name in ("min", "max") and len(a) == 2 and not np.isscalar(a[1]):
            return (np.fmin if name == "min" else np.fmax)(series(0), series(1))
        d = _window(a[1])
        if name in ("ts_min", "min"):
            return ops.ts_min(series(0), d)
        if name in ("ts_max", "max"):
            return ops.ts_max(series(0), d)
        if name == "ts_rank":
            return ops.ts_rank(series(0), d)
        if name == "sum":
            return _rolling(series(0), d, "sum")
        if name == "stddev":
            return _rolling(series(0), d, "std")
        if name == "product":
            return series(0) if d == 1 else _product(series(0), d)
        if name == "decay_linear":
            return ops.ts_decay_linear(series(0), d)
        if name == "ts_argmax":
            return _ts_arg(series(0), d, np.argmax)
        if name == "ts_argmin":
            return _ts_arg(series(0), d, np.argmin)
        raise ValueError(f"unsupported function {name!r}")

    def ev(self, node):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "var":
            return self.var(node[1])
        if kind == "neg":
            return -self.ev(node[1])
        if kind == "call":
            return self.call(node[1], node[2])
        if kind == "if":
            c = self.arr(self.ev(node[1]))
            return np.where(c > 0, self.arr(self.ev(node[2])), self.arr(self.ev(node[3])))
        op, x, y = node[1], self.ev(node[2]), self.ev(node[3])
        with np.errstate(all="ignore"):
            if op == "+":
                r = x + y
            elif op == "-":
                r = x - y
            elif op == "*":
                r = x * y
            elif op == "/":
                r = x / y
            elif op == "^":
                r = np.power(x, y)
            elif op in ("<", ">", "<=", ">=", "=="):
                r = {"<": np.less, ">": np.greater, "<=": np.less_equal,
                     ">=": np.greater_equal, "==": np.equal}[op](x, y).astype(np.float64)
            elif op == "||":
                r = ((self.arr(x) > 0) | (self.arr(y) > 0)).astype(np.float64)
            else:
                r = ((self.arr(x) > 0) & (self.arr(y) > 0)).astype(np.float64)
        return ops._clean(r) if not np.isscalar(r) else r

    def compute(self, formula: str) -> np.ndarray:
        return np.where(self.mask, self.arr(self.ev(parse(formula))), np.nan)


def eligible_formulas() -> pd.DataFrame:
    df = pd.read_csv(FORMULAS_FILE)
    return df[df["eligible"]].reset_index(drop=True)
