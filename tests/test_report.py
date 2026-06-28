import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grounding_deficit.deltas.delta_s import DeltaSResult
from grounding_deficit.deltas.delta_c import DeltaCResult
from grounding_deficit.deltas.delta_e import DeltaEResult
from grounding_deficit.report import to_deficit_triple, save_result, load_all_results, latest_result_per_model


def _make_dummy_results():
    ds = DeltaSResult(model_name="m", n_facts=10, accuracy_overall=0.8,
                       accuracy_by_bin={"monofact": 0.5, "rare": 0.8, "common": 0.95},
                       ece=0.12, brier_score=0.1, monofact_rate_dataset=0.3, per_fact=[])
    dc = DeltaCResult(model_name="m", n_facts=10, accuracy_closed_book=0.4,
                       accuracy_open_book=0.9, retrieval_gain=0.5, per_fact=[])
    de = DeltaEResult(model_name="m", n_facts=10, faithfulness_rate=0.7,
                       citation_rate=0.6, per_fact=[])
    return ds, dc, de


def test_to_deficit_triple_mapping():
    ds, dc, de = _make_dummy_results()
    triple = to_deficit_triple(ds, dc, de)
    assert triple["delta_s"] == ds.ece
    assert abs(triple["delta_c"] - (1.0 - dc.accuracy_open_book)) < 1e-9
    assert abs(triple["delta_e"] - (1.0 - de.faithfulness_rate)) < 1e-9


def test_to_deficit_triple_partial():
    ds, _, _ = _make_dummy_results()
    triple = to_deficit_triple(ds, None, None)
    assert triple["delta_s"] is not None
    assert triple["delta_c"] is None
    assert triple["delta_e"] is None


def test_save_and_load_results(tmp_path):
    ds, dc, de = _make_dummy_results()
    triple = to_deficit_triple(ds, dc, de)
    save_result("test-model", triple, results_dir=tmp_path)

    all_results = load_all_results(tmp_path)
    assert len(all_results) == 1
    assert all_results[0]["model_name"] == "test-model"

    latest = latest_result_per_model(tmp_path)
    assert "test-model" in latest
