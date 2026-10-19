"""Shared experiment setup: panel, industries, market regimes, evaluator, config."""
import json
from functools import lru_cache

from alpha_mining.config import ROOT
from alpha_mining.data.panel import benchmark, load_panel
from alpha_mining.data.sectors import industry_codes
from alpha_mining.evaluation.evaluator import Evaluator
from alpha_mining.evaluation.regimes import regime_labels
from alpha_mining.mining.diagnostics import Thresholds
from alpha_mining.mining.pool import TrainingPool

CONFIG_FILE = ROOT / "configs" / "experiment.json"


@lru_cache
def config() -> dict:
    return json.loads(CONFIG_FILE.read_text())


@lru_cache
def evaluator() -> Evaluator:
    cfg = config()
    panel = load_panel()
    regimes = regime_labels(benchmark()["adj_close"], panel.dates)
    return Evaluator(panel, industry_codes(panel.entities), regimes, cfg["lambda"], cfg["tau"])


def thresholds() -> Thresholds:
    return Thresholds(**config()["diagnostics"])


def new_pool() -> TrainingPool:
    cfg = config()
    gates = json.loads((ROOT / cfg["pool_gate_file"]).read_text())
    return TrainingPool(gates["train_abs_ic_median"], gates["train_abs_icir_median"], cfg["pool_rho_max"])
