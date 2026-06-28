"""
Aggregates delta_S, delta_C, delta_E results into the Delta(M, F) triple
from the paper (Definition 3, Section 7.2) and handles saving/loading
results to/from JSON for the dashboard to consume.

Note on interpretation: the paper defines each delta as a DEFICIT (higher
= worse, bounded in [0,1]). Our raw measurements are framed as accuracy/
faithfulness (higher = better) because that's the natural unit for the
underlying metrics. `to_deficit_triple()` below does the inversion and
documents exactly how, so a reader can check the mapping rather than take
it on faith.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone

from grounding_deficit.deltas.delta_s import DeltaSResult
from grounding_deficit.deltas.delta_c import DeltaCResult
from grounding_deficit.deltas.delta_e import DeltaEResult


def to_deficit_triple(delta_s: DeltaSResult | None,
                       delta_c: DeltaCResult | None,
                       delta_e: DeltaEResult | None) -> dict:
    """
    Maps raw measurements onto the [0,1] deficit scale used in the paper.
    Returns a dict with 'delta_s', 'delta_c', 'delta_e' (each None if that
    axis wasn't measured) plus the raw values they were derived from, so
    the mapping is auditable rather than opaque.

    Mapping used (documented, not "correct" in any deep sense -- this is
    exactly the kind of proxy-validation question Section 7.3/Limitation 1
    of the paper flags as open):
      delta_s = ECE                      (already in [0,1], already a "badness" measure)
      delta_c = 1 - accuracy_open_book    (residual inaccuracy even with the right
                                            document supplied, i.e. the part retrieval
                                            access alone cannot fix)
      delta_e = 1 - faithfulness_rate      (fraction of citations that do NOT actually
                                            support the claim attributed to them)
    """
    out = {"delta_s": None, "delta_c": None, "delta_e": None, "_raw": {}}

    if delta_s is not None:
        out["delta_s"] = delta_s.ece
        out["_raw"]["delta_s"] = asdict(delta_s)

    if delta_c is not None:
        out["delta_c"] = 1.0 - delta_c.accuracy_open_book
        out["_raw"]["delta_c"] = asdict(delta_c)

    if delta_e is not None:
        out["delta_e"] = 1.0 - delta_e.faithfulness_rate
        out["_raw"]["delta_e"] = asdict(delta_e)

    return out


def save_result(model_name: str, triple: dict, results_dir: str | Path = "results") -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = model_name.replace("/", "_")
    path = results_dir / f"{safe_name}_{timestamp}.json"

    payload = {"model_name": model_name, "timestamp": timestamp, **triple}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_all_results(results_dir: str | Path = "results") -> list[dict]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    out = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out


def latest_result_per_model(results_dir: str | Path = "results") -> dict[str, dict]:
    """Collapse multiple runs into the most recent result per model_name,
    convenient for the dashboard's comparison table."""
    all_results = load_all_results(results_dir)
    latest: dict[str, dict] = {}
    for r in all_results:
        name = r["model_name"]
        if name not in latest or r["timestamp"] > latest[name]["timestamp"]:
            latest[name] = r
    return latest
