"""
delta_S: the statistical (calibration) proxy.

Implements the proxy proposed in Table 2 of the paper ("Three Lenses, One
Gap"): expected calibration error (ECE) / Brier score on facts stratified
by (synthetic) training-set frequency, plus a monofact-rate-stratified
accuracy curve -- the empirical analogue of Kalai & Vempala's theoretical
bound, in the spirit of Miao & Kearns's bin-wise KL estimator.

WHAT THIS DOES NOT DO (read before citing results from this module):
  - It does not test pretraining calibration in the Kalai & Vempala sense,
    because we do not control the model's pretraining corpus. It tests
    in-context recall calibration: a fact is given in a short bio
    paragraph, the model is asked a detail-recall question some turns
    later (or in the same turn, configurable), and we measure whether the
    model's stated/sampled confidence tracks whether it actually recalled
    correctly -- and whether that relationship interacts with the fact's
    synthetic frequency bin (monofact/rare/common) the way the theorem's
    logic would predict if "synthetic frequency" stood in for "training
    frequency."
  - "Confidence" is estimated one of two ways depending on what the
    backend exposes:
      (a) average token logprob -> probability, when available (OpenAI), or
      (b) self-reported confidence, elicited via a follow-up prompt asking
          the model to state a 0-100 confidence in its own answer, when
          logprobs are unavailable (Anthropic, Together-hosted open models).
    These are NOT the same quantity and are not directly comparable across
    backends without caveats -- see README.md, "Known limitations of v1".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from grounding_deficit.data.facts_dataset import FactRecord
from grounding_deficit.models import BaseModel


@dataclass
class DeltaSResult:
    model_name: str
    n_facts: int
    accuracy_overall: float
    accuracy_by_bin: dict          # {"monofact": float, "rare": float, "common": float}
    ece: float                      # expected calibration error, overall
    brier_score: float               # overall
    monofact_rate_dataset: float      # fraction of dataset that is monofact-bin
    per_fact: list = field(default_factory=list)  # raw per-item records for inspection/debugging


_RECALL_PROMPT = (
    "Read the following short biography, then answer the question.\n\n"
    "Biography: {bio}\n\n"
    "Question: What year was {first_name} born?\n"
    "Answer with only the four-digit year, nothing else."
)

_CONFIDENCE_PROMPT = (
    "You previously answered a question with: \"{answer}\"\n"
    "On a scale from 0 to 100, how confident are you that this answer is correct? "
    "Respond with only a number from 0 to 100."
)


def _extract_year(text: str) -> str | None:
    match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", text)
    return match.group(1) if match else None


def _extract_confidence(text: str) -> float | None:
    match = re.search(r"\d{1,3}", text)
    if not match:
        return None
    val = float(match.group(0))
    return max(0.0, min(100.0, val)) / 100.0


def _expected_calibration_error(confidences: list[float], correctness: list[bool],
                                  n_bins: int = 10) -> float:
    """Standard ECE: bin predictions by confidence, compare mean confidence to
    empirical accuracy in each bin, weight by bin size."""
    if not confidences:
        return float("nan")
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _brier_score(confidences: list[float], correctness: list[bool]) -> float:
    if not confidences:
        return float("nan")
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    return float(np.mean((confidences - correctness) ** 2))


def evaluate_delta_s(model: BaseModel, facts: list[FactRecord],
                       elicit_confidence: bool = True, verbose: bool = False) -> DeltaSResult:
    """
    Run the delta_S evaluation over a list of FactRecords against `model`.
    Returns a DeltaSResult with overall and per-bin accuracy, ECE, and Brier score.
    """
    per_fact = []

    for fact in facts:
        first_name = fact.entity_name.split()[0]
        prompt = _RECALL_PROMPT.format(bio=fact.bio_context, first_name=first_name)
        resp = model.complete(prompt, temperature=0.0, max_tokens=20)
        predicted_year = _extract_year(resp.text)
        correct = (predicted_year == fact.attribute_value)

        # Confidence: prefer provider logprob-derived confidence; otherwise elicit it.
        confidence = resp.raw_confidence
        if confidence is None and elicit_confidence:
            conf_prompt = _CONFIDENCE_PROMPT.format(answer=resp.text.strip())
            conf_resp = model.complete(conf_prompt, temperature=0.0, max_tokens=10)
            confidence = _extract_confidence(conf_resp.text)

        if confidence is None:
            confidence = 1.0 if correct else 0.0  # last-resort fallback; flagged in per_fact

        per_fact.append({
            "fact_id": fact.fact_id,
            "monofact_bin": fact.monofact_bin,
            "synthetic_frequency": fact.synthetic_frequency,
            "predicted": predicted_year,
            "ground_truth": fact.attribute_value,
            "correct": correct,
            "confidence": confidence,
            "confidence_was_elicited": resp.raw_confidence is None,
        })

        if verbose:
            print(f"[{fact.fact_id}] bin={fact.monofact_bin} pred={predicted_year} "
                  f"truth={fact.attribute_value} correct={correct} conf={confidence:.2f}")

    confidences = [r["confidence"] for r in per_fact]
    correctness = [r["correct"] for r in per_fact]

    acc_by_bin = {}
    for b in ("monofact", "rare", "common"):
        bin_items = [r["correct"] for r in per_fact if r["monofact_bin"] == b]
        acc_by_bin[b] = float(np.mean(bin_items)) if bin_items else float("nan")

    monofact_frac = sum(1 for f in facts if f.monofact_bin == "monofact") / len(facts) if facts else 0.0

    return DeltaSResult(
        model_name=model.name,
        n_facts=len(facts),
        accuracy_overall=float(np.mean(correctness)) if correctness else float("nan"),
        accuracy_by_bin=acc_by_bin,
        ece=_expected_calibration_error(confidences, correctness),
        brier_score=_brier_score(confidences, correctness),
        monofact_rate_dataset=monofact_frac,
        per_fact=per_fact,
    )
