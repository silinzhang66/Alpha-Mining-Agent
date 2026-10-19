"""Genetic-programming baseline (DEAP) over the same DSL, fitness and budget rules.

Strongly typed trees: series-valued nodes, plus window terminals of two types
(statistics windows and lag windows) so every generated tree maps onto a valid
DSL call signature. Trees are rendered to DSL text and submitted through the
shared MiningSession, so validation, duplicate detection and budget accounting
are identical to the LLM groups. Survival is (mu + lambda): the best `pop_size`
of parents and offspring.
"""
import random

from deap import base, creator, gp, tools

from alpha_mining.dsl import spec
from alpha_mining.mining.session import MiningSession

TREE_HEIGHT_LIMIT = 5  # DEAP height; DSL depth/size limits are enforced on submit
INVALID_FITNESS = -1e9


class Series:
    pass


class Win(int):
    pass


class Lag(int):
    pass


def _noop(*args):
    return None


def build_pset() -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("factor", [], Series)
    for name in ("add", "sub", "mul", "div") + spec.ELEMENTWISE_BINARY:
        pset.addPrimitive(_noop, [Series, Series], Series, name=name)
    for name in ("neg",) + spec.ELEMENTWISE + spec.CROSS_SECTIONAL:
        pset.addPrimitive(_noop, [Series], Series, name=name)
    for name, (n_series, windows) in spec.TIME_SERIES.items():
        wtype = Lag if windows == spec.LAG_WINDOWS else Win
        pset.addPrimitive(_noop, [Series] * n_series + [wtype], Series, name=name)
    for f in spec.FEATURES:
        pset.addTerminal(f, Series, name=f)
    for w in spec.WINDOWS:
        pset.addTerminal(w, Win, name=f"w{w}")
    for w in spec.LAG_WINDOWS:
        pset.addTerminal(w, Lag, name=f"l{w}")
    return pset


def generate(pset, min_, max_, type_=None):
    """Typed 'grow' generation: series nodes become primitives until the sampled
    height is reached; window arguments are always terminals."""
    type_ = type_ or pset.ret
    height = random.randint(min_, max_)
    expr, stack = [], [(0, type_)]
    while stack:
        depth, t = stack.pop()
        prims, terms = pset.primitives[t], pset.terminals[t]
        stop = not prims or depth >= height or (depth >= min_ and random.random() < 0.3)
        if stop:
            node = random.choice(terms)
            expr.append(node() if isinstance(node, type) else node)
        else:
            prim = random.choice(prims)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def to_dsl(tree: gp.PrimitiveTree) -> str:
    def render(i: int) -> tuple[str, int]:
        node = tree[i]
        if isinstance(node, gp.Terminal):
            name = node.name  # "w20" / "l1" for window terminals, else a feature
            return (name[1:] if name[0] in "wl" and name[1:].isdigit() else name), i + 1
        args, j = [], i + 1
        for _ in range(node.arity):
            s, j = render(j)
            args.append(s)
        if node.name in _INFIX:
            return f"({args[0]} {_INFIX[node.name]} {args[1]})", j
        if node.name == "neg":
            return f"-({args[0]})", j
        return f"{node.name}({', '.join(args)})", j

    return render(0)[0]


def _setup(pset) -> base.Toolbox:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register("expr", generate, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", generate, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    limit = gp.staticLimit(key=lambda ind: ind.height, max_value=TREE_HEIGHT_LIMIT)
    tb.decorate("mate", limit)
    tb.decorate("mutate", limit)
    return tb


def run_gp(session: MiningSession, pop_size: int, seed: int, cxpb: float = 0.5) -> None:
    """Evolve until the session budget is spent.

    Each generation creates `pop_size` new individuals, each by crossover (prob
    cxpb) or else by mutation (DEAP varOr without reproduction), so every
    offspring is a new proposal and spends one unit of budget, as an LLM
    candidate does.
    """
    random.seed(seed)
    pset = build_pset()
    tb = _setup(pset)

    def evaluate(inds, gen):
        for ind in inds:
            if session.remaining <= 0:
                ind.fitness.values = (INVALID_FITNESS,)
                continue
            c = session.submit(to_dsl(ind), gen, meta={"method": "gp"})
            fit = session.seen[c.canonical].fitness if c.status == "duplicate" else c.fitness
            ind.fitness.values = (fit if fit > -float("inf") else INVALID_FITNESS,)

    pop = [tb.individual() for _ in range(pop_size)]
    evaluate(pop, 1)
    gen = 1
    while session.remaining > 0:
        gen += 1
        offspring = []
        for _ in range(pop_size):
            if random.random() < cxpb:
                a, b = (tb.clone(i) for i in tb.select(pop, 2))
                child = tb.mate(a, b)[0]
            else:
                child = tb.mutate(tb.clone(tb.select(pop, 1)[0]))[0]
            del child.fitness.values
            offspring.append(child)
        evaluate(offspring, gen)
        pop = tools.selBest(pop + offspring, pop_size)
