import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grounding_deficit.data.facts_dataset import generate_fact_dataset
from grounding_deficit.deltas.delta_s import (
    evaluate_delta_s, _expected_calibration_error, _brier_score, _extract_year,
    _extract_confidence,
)
from grounding_deficit.models import BaseModel, ModelResponse


class _PerfectMockModel(BaseModel):
    """Always answers correctly with full confidence -- sanity-check that
    evaluate_delta_s reports zero ECE/Brier and perfect accuracy in the trivial case."""

    def __init__(self):
        self.name = "perfect-mock"
        self._facts_by_year = {}

    def complete(self, prompt, system=None, **kwargs):
        # Pull the year directly out of the prompt's biography text (the mock
        # "knows" everything perfectly), or answer a confidence query at 100.
        if "confident" in prompt.lower():
            return ModelResponse(text="100")
        import re
        match = re.search(r"born in (\d{4})", prompt)
        year = match.group(1) if match else "0000"
        return ModelResponse(text=year, raw_confidence=None)


class _AlwaysWrongMockModel(BaseModel):
    def __init__(self):
        self.name = "wrong-mock"

    def complete(self, prompt, system=None, **kwargs):
        if "confident" in prompt.lower():
            return ModelResponse(text="100")  # confidently wrong -> should inflate ECE
        return ModelResponse(text="1900", raw_confidence=None)


def test_extract_year():
    assert _extract_year("The answer is 1987.") == "1987"
    assert _extract_year("no year here") is None


def test_extract_confidence():
    assert _extract_confidence("85") == 0.85
    assert _extract_confidence("I am 100% sure") == 1.0
    assert _extract_confidence("no digits") is None


def test_ece_perfect_calibration_is_zero():
    confidences = [1.0, 1.0, 0.0, 0.0]
    correctness = [True, True, False, False]
    assert _expected_calibration_error(confidences, correctness) == 0.0


def test_brier_score_perfect_is_zero():
    confidences = [1.0, 0.0]
    correctness = [True, False]
    assert _brier_score(confidences, correctness) == 0.0


def test_evaluate_delta_s_perfect_model():
    facts = generate_fact_dataset(n=8, seed=4)
    model = _PerfectMockModel()
    result = evaluate_delta_s(model, facts, elicit_confidence=True)
    assert result.accuracy_overall == 1.0
    assert abs(result.ece) < 1e-6
    assert abs(result.brier_score) < 1e-6


def test_evaluate_delta_s_always_wrong_confident_model():
    facts = generate_fact_dataset(n=8, seed=5)
    model = _AlwaysWrongMockModel()
    result = evaluate_delta_s(model, facts, elicit_confidence=True)
    assert result.accuracy_overall < 1.0
    # Confidently wrong -> high ECE and high Brier (worst-case miscalibration)
    assert result.ece > 0.5
    assert result.brier_score > 0.5
