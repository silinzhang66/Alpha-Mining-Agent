"""Knowledge-cutoff probe.

  uv run python scripts/run_probe.py build                  # raw/*.jsonl -> questions.jsonl + review CSV
  uv run python scripts/run_probe.py run --model deepseek   # 23 calls, temperature 0, thinking/search off
  uv run python scripts/run_probe.py analyze                # per-month accuracy, binomial p, cutoff C
"""
import argparse
import json

import pandas as pd

from alpha_mining import probe
from alpha_mining.config import ROOT
from alpha_mining.llm.client import ChatClient

OUT = ROOT / "results" / "probe"

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["build", "run", "analyze"])
p.add_argument("--model", choices=["deepseek", "qwen"])
args = p.parse_args()

if args.cmd == "build":
    bank = probe.build_bank()
    probe.BANK_FILE.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in bank))
    rows = [{"id": q["id"], "month": q["month"], "category": q["category"], "question": q["question"],
             **{k: q["options"][k] for k in probe.LETTERS}, "answer": q["answer"],
             "event_date": q["event_date"], "source_url": q["source_url"], "evidence": q["evidence"]}
            for q in bank]
    pd.DataFrame(rows).to_csv(probe.PROBE_DIR / "questions_review.csv", index=False)
    for a, o, text in probe.stem_leaks(bank):
        print(f"check: answer of {a} ({text!r}) appears in the stem of {o}")
    print(f"{len(bank)} questions; answer letters:",
          pd.Series([q["answer"] for q in bank]).value_counts().sort_index().to_dict())

elif args.cmd == "run":
    bank = [json.loads(line) for line in probe.BANK_FILE.read_text().splitlines()]
    out = OUT / f"{args.model}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; the probe is run once per model")
    OUT.mkdir(parents=True, exist_ok=True)
    client = ChatClient(args.model, thinking=False, temperature=0.0, log_name=f"probe_{args.model}")
    results = []
    for b, batch in enumerate(probe.batches(bank)):
        msgs = [{"role": "system", "content": probe.SYSTEM},
                {"role": "user", "content": probe.prompt(batch)}]
        reply = client.chat(msgs, max_tokens=200, batch=b)
        answers = probe.parse_answers(reply.text, len(batch))
        for q, a in zip(batch, answers):
            results.append({"id": q["id"], "month": q["month"], "batch": b, "given": a,
                            "answer": q["answer"], "correct": a == q["answer"]})
        print(f"batch {b}: {sum(a is None for a in answers)} unparsed")
    out.write_text("".join(json.dumps(r) + "\n" for r in results))
    (OUT / f"{args.model}.meta.json").write_text(json.dumps(
        {"served_model": client.served_model, "base_url": client.base_url, "temperature": 0.0,
         "thinking": False, "n_questions": len(results)}, indent=2))

else:
    summary = {}
    for model in ("deepseek", "qwen"):
        path = OUT / f"{model}.jsonl"
        if not path.exists():
            continue
        res = [json.loads(line) for line in path.read_text().splitlines()]
        stats = probe.month_stats(res)
        summary[model] = {"cutoff": probe.effective_cutoff(stats), "months": stats,
                          "unparsed": sum(r["given"] is None for r in res)}
        print(f"\n{model}: effective cutoff C = {summary[model]['cutoff'] or 'before 2025-01'}"
              f" (unparsed answers: {summary[model]['unparsed']})")
        print(pd.DataFrame(stats).T[["k", "n", "acc", "p"]].round(4).to_string())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
