"""
delta_C: the computational proxy.

STATUS: Phase 2 (minimal working implementation, not yet validated). This
module is intentionally simpler than delta_s.py -- it establishes the
harness shape so that a more careful retrieval backend can be swapped in
later, per Table 2 of the paper ("retrieval/tool-augmented accuracy gain"
as a proxy for the worst-case computability gap).

Methodology: for each fact, ask the model the same closed-book question
twice:
  (1) with no supporting context (closed-book; tests parametric recall)
  (2) with the fact's bio_context provided directly in the prompt
      (open-book; a stand-in for "retrieval succeeded and surfaced the
      right document")
The retrieval-gain proxy is accuracy(open-book) - accuracy(closed-book).
A large gap is read as evidence of a large *closeable* computational
deficit (the model lacked the fact parametrically but could use it once
supplied) -- consistent with the oracle-escape framing in Shi et al.
(2025) that the paper's Section 4.1 and Table 2 describe.

KNOWN LIMITATION (be upfront about this in any write-up): this measures
whether providing the *correct* document helps, not whether a *real*
retrieval system would have found and surfaced that correct document in
the first place. It is a proxy for the upper bound of what retrieval could
buy you, not a measurement of any actual retrieval pipeline's recall. A
more complete v2 would wire in a real retriever (e.g. BM25 or embedding
search over a corpus containing both correct and distractor documents)
so that retrieval failure modes are captured too, not just the best case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from grounding_deficit.data.facts_dataset import FactRecord
from grounding_deficit.models import BaseModel
from grounding_deficit.deltas.delta_s import _extract_year


@dataclass
class DeltaCResult:
    model_name: str
    n_facts: int
    accuracy_closed_book: float
    accuracy_open_book: float
    retrieval_gain: float            # open-book accuracy minus closed-book accuracy
    per_fact: list = field(default_factory=list)


_CLOSED_BOOK_PROMPT = (
    "What year was {full_name} ({profession}, based in {city}) born?\n"
    "If you do not know, make your best guess. Answer with only the four-digit year."
)

_OPEN_BOOK_PROMPT = (
    "Context: {bio}\n\n"
    "Question: What year was {first_name} born?\n"
    "Answer with only the four-digit year, nothing else."
)


def evaluate_delta_c(model: BaseModel, facts: list[FactRecord], verbose: bool = False) -> DeltaCResult:
    per_fact = []

    for fact in facts:
        first_name = fact.entity_name.split()[0]

        closed_prompt = _CLOSED_BOOK_PROMPT.format(
            full_name=fact.entity_name, profession=fact.profession, city=fact.city)
        closed_resp = model.complete(closed_prompt, temperature=0.0, max_tokens=10)
        closed_pred = _extract_year(closed_resp.text)
        closed_correct = (closed_pred == fact.attribute_value)

        open_prompt = _OPEN_BOOK_PROMPT.format(bio=fact.bio_context, first_name=first_name)
        open_resp = model.complete(open_prompt, temperature=0.0, max_tokens=10)
        open_pred = _extract_year(open_resp.text)
        open_correct = (open_pred == fact.attribute_value)

        per_fact.append({
            "fact_id": fact.fact_id,
            "closed_book_pred": closed_pred,
            "closed_book_correct": closed_correct,
            "open_book_pred": open_pred,
            "open_book_correct": open_correct,
            "ground_truth": fact.attribute_value,
        })

        if verbose:
            print(f"[{fact.fact_id}] closed={closed_correct} open={open_correct}")

    closed_acc = float(np.mean([r["closed_book_correct"] for r in per_fact])) if per_fact else float("nan")
    open_acc = float(np.mean([r["open_book_correct"] for r in per_fact])) if per_fact else float("nan")

    return DeltaCResult(
        model_name=model.name,
        n_facts=len(facts),
        accuracy_closed_book=closed_acc,
        accuracy_open_book=open_acc,
        retrieval_gain=open_acc - closed_acc,
        per_fact=per_fact,
    )
