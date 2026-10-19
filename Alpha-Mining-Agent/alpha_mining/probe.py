"""Knowledge-cutoff probe (paper Section 4).

210 monthly questions (2025-01 to 2026-09, 10 per month) and 20 control
questions from 2023, four options each. Options are reshuffled with a fixed
seed, questions are mixed across months into batches of 10 (one call each),
and the models answer with thinking and search off at temperature 0.

Pre-specified decision rule: per month, one-sided binomial test of accuracy
> 25%; the effective cutoff C is the last month with p < 0.05 such that every
later month is not significant. A missing or unparseable answer counts as wrong.
"""
import json
import random
import re
from pathlib import Path

from scipy.stats import binomtest

from alpha_mining.config import ROOT

PROBE_DIR = ROOT / "data" / "probe"
RAW_DIR = PROBE_DIR / "raw"
BANK_FILE = PROBE_DIR / "questions.jsonl"
SEED = 20261018
BATCH = 10
CHANCE = 0.25
ALPHA = 0.05
LETTERS = "ABCD"
CONTROL = "2023-ctrl"
FIELDS = ("id", "month", "event_date", "category", "question", "options", "answer",
          "source_url", "source_title", "evidence")

MONTHS = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)
          if (y, m) <= (2026, 9)]


def validate(q: dict) -> None:
    missing = [f for f in FIELDS if not q.get(f)]
    if missing:
        raise ValueError(f"{q.get('id')}: missing {missing}")
    if sorted(q["options"]) != list(LETTERS) or q["answer"] not in LETTERS:
        raise ValueError(f"{q['id']}: options must be A-D with one answer letter")
    if len({v.strip().lower() for v in q["options"].values()}) != 4:
        raise ValueError(f"{q['id']}: duplicate options")
    if not q["source_url"].startswith("http"):
        raise ValueError(f"{q['id']}: bad source_url")
    if q["month"] != CONTROL and not q["event_date"].startswith(q["month"]):
        raise ValueError(f"{q['id']}: event_date {q['event_date']} outside {q['month']}")


def shuffle_options(q: dict, rng: random.Random) -> dict:
    texts = [q["options"][k] for k in LETTERS]
    correct = q["options"][q["answer"]]
    rng.shuffle(texts)
    return {**q, "options": dict(zip(LETTERS, texts)), "answer": LETTERS[texts.index(correct)]}


def build_bank(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Read raw/*.jsonl, check counts and fields, reshuffle options deterministically."""
    expected = {m: 10 for m in MONTHS} | {CONTROL: 20}
    qs = []
    for month, n in expected.items():
        path = raw_dir / f"{month}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != n:
            raise ValueError(f"{path.name}: {len(rows)} questions, expected {n}")
        for q in rows:
            if q["month"] != month:
                raise ValueError(f"{q['id']}: month {q['month']} in {path.name}")
            validate(q)
        qs += rows
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    rng = random.Random(SEED)
    return [shuffle_options(q, rng) for q in qs]


def stem_leaks(qs: list[dict]) -> list[tuple[str, str, str]]:
    """(answer id, stem id, text): a correct option quoted in another question's stem."""
    def norm(s):
        return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "
    out = []
    for q in qs:
        a = norm(q["options"][q["answer"]])
        if len(a.strip()) < 5 or not re.search(r"[a-z]", a):
            continue
        out += [(q["id"], o["id"], a.strip()) for o in qs if o is not q and a in norm(o["question"])]
    return out


def batches(qs: list[dict]) -> list[list[dict]]:
    """Mix months so no batch is a single month; fixed order for both models."""
    order = list(qs)
    random.Random(SEED + 1).shuffle(order)
    return [order[i:i + BATCH] for i in range(0, len(order), BATCH)]


SYSTEM = ("You are taking a multiple-choice quiz about real-world events. Use only your "
          "own knowledge. Every question has exactly one correct option. If you are not "
          "sure, give your best guess; never skip a question or refuse.")


def prompt(batch: list[dict]) -> str:
    parts = []
    for i, q in enumerate(batch, 1):
        opts = "\n".join(f"{k}. {q['options'][k]}" for k in LETTERS)
        parts.append(f"Q{i}. {q['question']}\n{opts}")
    return ("\n\n".join(parts) + f"\n\nAnswer all {len(batch)} questions. Reply with exactly "
            f"{len(batch)} lines and nothing else, in the form `Q1: B`.")


def parse_answers(text: str, n: int) -> list[str | None]:
    found = {}
    for m in re.finditer(r"Q\s*(\d+)\s*[:.)\-]\s*\**\s*([ABCD])\b", text, flags=re.I):
        found.setdefault(int(m.group(1)), m.group(2).upper())
    return [found.get(i) for i in range(1, n + 1)]


def month_stats(results: list[dict]) -> dict:
    """results: [{month, correct}] -> {month: {n, k, acc, p}}."""
    out = {}
    for month in MONTHS + [CONTROL]:
        xs = [r["correct"] for r in results if r["month"] == month]
        if not xs:
            continue
        k = sum(xs)
        out[month] = {"n": len(xs), "k": k, "acc": k / len(xs),
                      "p": binomtest(k, len(xs), CHANCE, alternative="greater").pvalue}
    return out


def effective_cutoff(stats: dict) -> str | None:
    """Last significant month with every later month not significant; None = before 2025-01."""
    sig = [m for m in MONTHS if m in stats and stats[m]["p"] < ALPHA]
    return sig[-1] if sig else None
