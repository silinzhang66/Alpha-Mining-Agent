"""Closed-loop LLM mining: R rounds x K candidates on a shared MiningSession.

Each round is a fresh two-message prompt (no chat history): the shared DSL and
scoring text, the feedback numbers for three expressions, last round's
failures, and for Group C the diagnoses. The three expressions shown are the
two best by training fitness so far plus the best new candidate of the previous
round (the next one down if it is already among the top two).
"""
from alpha_mining.llm.client import ChatClient
from alpha_mining.llm.prompts import (DIAGNOSED, build_messages, diagnosis_block, feedback_block,
                                      parse_reply, stats_block)
from alpha_mining.mining.diagnostics import R1_ADVICE, R1_ADVICE_DEPTH, Thresholds, diagnose
from alpha_mining.mining.session import Candidate, MiningSession

PARSE_RETRIES = 2


def select_shown(session: MiningSession, round_: int) -> list[Candidate]:
    scored = sorted(session.evaluated(), key=lambda c: c.fitness, reverse=True)
    top = scored[:2]
    prev = [c for c in scored if c.round == round_ - 1 and all(c is not t for t in top)]
    return top + prev[:1]


def run_llm(session: MiningSession, client: ChatClient, group: str, rounds: int, k: int,
            lam: float, tau: float, thresholds: Thresholds = Thresholds()) -> None:
    for rnd in range(1, rounds + 1):
        shown = select_shown(session, rnd)
        failures = [{"formula": c.expr or "(missing)", "message": c.message}
                    for c in session.candidates if c.round == rnd - 1 and c.status != "ok"]
        feedback = feedback_block([c.feedback_numbers() for c in shown], failures, rnd, rounds)
        diagnosis = None
        if group in DIAGNOSED:
            diags = diagnose(shown, thresholds, R1_ADVICE_DEPTH if group == "Cd" else R1_ADVICE)
            diagnosis = stats_block(diags) if group == "Cs" else diagnosis_block(diags)
        messages = build_messages(group, k, lam, tau, feedback, diagnosis)

        items, error = [], ""
        for attempt in range(PARSE_RETRIES + 1):
            reply = client.chat(messages, round=rnd, attempt=attempt, group=group)
            try:
                items = parse_reply(reply.text)
                break
            except (ValueError, TypeError) as e:  # json errors are ValueErrors
                error = f"unparseable response: {e}"
        items = items[:k]
        for it in items:
            session.submit(it["formula"], rnd, meta={"hypothesis": it["hypothesis"], "group": group})
        for _ in range(k - len(items)):  # missing slots still spend budget
            c = session.submit("", rnd, meta={"group": group})
            c.message = error or "fewer formulas returned than requested"
