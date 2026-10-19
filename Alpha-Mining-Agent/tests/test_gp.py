import random

from alpha_mining.baselines.gp import build_pset, generate, to_dsl
from alpha_mining.dsl.parse import DSLError, parse
from deap import gp


def test_generated_trees_render_to_valid_dsl():
    random.seed(0)
    pset = build_pset()
    ok = 0
    for _ in range(300):
        tree = gp.PrimitiveTree(generate(pset, 1, 4))
        text = to_dsl(tree)
        try:
            parse(text)
            ok += 1
        except DSLError as e:
            # Only size/depth limits may fail; the signature is always valid.
            assert "depth" in str(e) or "nodes" in str(e), (text, e)
    assert ok > 250
