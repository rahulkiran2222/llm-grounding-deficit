import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grounding_deficit.data.facts_dataset import (
    generate_fact_dataset, monofact_rate, save_dataset, load_dataset, _bin_frequency
)


def test_generate_fact_dataset_basic_shape():
    facts = generate_fact_dataset(n=50, alpha=1.2, seed=1)
    assert len(facts) == 50
    ids = {f.fact_id for f in facts}
    assert len(ids) == 50  # all unique
    names = {f.entity_name for f in facts}
    assert len(names) == 50  # no duplicate entities


def test_alpha_changes_monofact_rate():
    # Verified by direct simulation (see _sample_pareto_frequencies docstring): with
    # this implementation's parameterization, HIGHER alpha concentrates more mass near
    # 1, producing a HIGHER monofact rate. We assert the direction is monotonic and
    # non-trivial (not that both are equal), rather than hardcoding the direction's
    # justification here -- the docstring is the source of truth for *why*.
    low_alpha = generate_fact_dataset(n=300, alpha=0.8, seed=2)
    high_alpha = generate_fact_dataset(n=300, alpha=3.0, seed=2)
    assert monofact_rate(high_alpha) > monofact_rate(low_alpha)


def test_bin_frequency_boundaries():
    assert _bin_frequency(1) == "monofact"
    assert _bin_frequency(2) == "rare"
    assert _bin_frequency(5) == "rare"
    assert _bin_frequency(6) == "common"
    assert _bin_frequency(100) == "common"


def test_save_and_load_roundtrip(tmp_path):
    facts = generate_fact_dataset(n=10, seed=3)
    path = tmp_path / "facts.jsonl"
    save_dataset(facts, path)
    loaded = load_dataset(path)
    assert len(loaded) == 10
    assert loaded[0].fact_id == facts[0].fact_id
    assert loaded[0].attribute_value == facts[0].attribute_value


def test_monofact_rate_empty():
    assert monofact_rate([]) == 0.0
