import json
import random

import pytest

from alpha_mining import probe


def _q(qid, month, answer="A"):
    return {"id": qid, "month": month, "event_date": f"{month}-15" if month != probe.CONTROL else "2023-05-01",
            "category": "world", "question": "In ...?", "options": {"A": "w", "B": "x", "C": "y", "D": "z"},
            "answer": answer, "source_url": "https://example.org", "source_title": "t", "evidence": "e"}


def test_shuffle_keeps_correct_text():
    rng = random.Random(0)
    for _ in range(50):
        q = probe.shuffle_options(_q("1", "2025-01", "C"), rng)
        assert q["options"][q["answer"]] == "y"


def test_validate_rejects_bad_questions():
    q = _q("1", "2025-01")
    probe.validate(q)
    with pytest.raises(ValueError):
        probe.validate({**q, "event_date": "2025-02-01"})
    with pytest.raises(ValueError):
        probe.validate({**q, "options": {"A": "w", "B": "w", "C": "y", "D": "z"}})
    with pytest.raises(ValueError):
        probe.validate({**q, "source_url": ""})


def test_build_bank_and_batches(tmp_path):
    for m in probe.MONTHS:
        (tmp_path / f"{m}.jsonl").write_text("\n".join(json.dumps(_q(f"{m}-{i}", m)) for i in range(10)))
    (tmp_path / f"{probe.CONTROL}.jsonl").write_text(
        "\n".join(json.dumps(_q(f"c{i}", probe.CONTROL)) for i in range(20)))
    bank = probe.build_bank(tmp_path)
    assert len(bank) == 230
    assert bank == probe.build_bank(tmp_path)  # deterministic
    assert {q["answer"] for q in bank} == set("ABCD")
    bs = probe.batches(bank)
    assert len(bs) == 23 and all(len(b) == 10 for b in bs)
    assert all(len({q["month"] for q in b}) > 1 for b in bs)


def test_parse_answers():
    assert probe.parse_answers("Q1: B\nQ2: **c**\nQ3) D\nQ5 - A", 5) == ["B", "C", "D", None, "A"]


def test_effective_cutoff():
    res = []
    for i, m in enumerate(probe.MONTHS):
        k = 8 if i < 5 else (6 if i == 8 else 2)   # 2025-09 significant again after a gap
        res += [{"month": m, "correct": j < k} for j in range(10)]
    stats = probe.month_stats(res)
    assert stats["2025-01"]["p"] < 0.05 and stats["2025-06"]["p"] > 0.05
    assert probe.effective_cutoff(stats) == "2025-09"
    assert probe.effective_cutoff({m: {"p": 0.5} for m in probe.MONTHS}) is None


def test_stem_leaks():
    a = {**_q("a", "2025-01"), "options": {"A": "Netflix", "B": "x", "C": "y", "D": "z"}}
    b = {**_q("b", "2026-02"), "question": "After Netflix declined to match, who won?"}
    c = {**_q("c", "2026-03"), "question": "What did Netflixers do?"}
    assert probe.stem_leaks([a, b, c]) == [("a", "b", "netflix")]
