"""
Synthetic "arbitrary facts" dataset generator.

Methodology adapted from Kalai & Vempala (2024) "Calibrated Language Models
Must Hallucinate" and the empirical bin-wise KL approach of Miao & Kearns
(2025/2026) "Hallucination, Monofacts, and Miscalibration" -- but adapted
for evaluating already-deployed, API-accessed models rather than models
trained from scratch on a controlled corpus.

KEY DIFFERENCE FROM MIAO & KEARNS: they control the *training* corpus of a
model they train themselves, so "frequency in training data" is something
they set directly. We cannot do that for GPT-4o or Claude -- we don't
control their training data. So v1 of this dataset uses a deliberately
FICTIONAL frequency model: we invent entities and facts about them, assign
each a synthetic "world frequency" drawn from a Pareto distribution, and
score whether the assigned frequency is associated with model confidence
and accuracy *for facts we then tell the model about in-context* (a
controlled, closed-book-then-open-book design -- see `mode` below).

This is explicitly a v1 placeholder methodology, not a recreation of the
original theorem's setting. See README.md, "Known limitations of v1" for
the honest accounting of what this does and does not establish. v1.1 is
planned to replace synthetic frequency with a real corpus-frequency proxy
(e.g. a held-out Wikipedia/Common Crawl n-gram frequency count) so that
facts and frequencies are real rather than fabricated.

Two evaluation modes are supported:
  - "open_book": the fact is given in-context (e.g. a short bio paragraph),
    and the model is asked to recall a specific detail from it. This tests
    whether stated confidence tracks how memorable/salient the fact was in
    context, which is the closest in-context analogue we have to monofact
    sensitivity without controlling pretraining data.
  - "closed_book": the model is asked the fact directly with no context,
    testing whether it fabricates a plausible-sounding but ungrounded
    answer. This is the more direct hallucination test but does not let us
    control "frequency" at all (the model either knows it from pretraining
    or doesn't) -- useful as a qualitative companion, not for the
    monofact-rate quantitative claim.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

FIRST_NAMES = ["Elena", "Marcus", "Priya", "Tobias", "Aiko", "Daniela", "Kwame", "Ines",
                "Soren", "Yuki", "Mateus", "Anya", "Liam", "Farah", "Niko", "Sana"]
LAST_NAMES = ["Vasquez", "Lindqvist", "Okafor", "Nakamura", "Petrov", "Almeida", "Krause",
               "Singh", "Dubois", "Larsen", "Mwangi", "Castellano", "Hoang", "Ferreira"]
PROFESSIONS = ["a structural engineer", "a marine biologist", "a typeface designer",
               "a viola player", "a glaciologist", "a ceramicist", "an actuary",
               "a long-distance cyclist", "a cartographer", "a beekeeper"]
CITIES = ["Porto", "Bandung", "Tartu", "Cuenca", "Hobart", "Kumasi", "Chiang Mai",
          "Ljubljana", "Valdivia", "Gdansk", "Mysore", "Trondheim"]


@dataclass
class FactRecord:
    fact_id: str
    entity_name: str
    profession: str
    city: str
    attribute: str          # which detail is being probed, e.g. "birth_year"
    attribute_value: str    # the ground-truth value
    synthetic_frequency: int  # fabricated "world frequency" (Pareto-sampled), v1 placeholder
    monofact_bin: str       # "monofact" (freq==1) | "rare" | "common", derived from frequency
    bio_context: str        # short paragraph used in open_book mode


def _sample_pareto_frequencies(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n frequencies from a Pareto(alpha) distribution, rounded to >=1 integer counts.
    Empirically (verified by direct simulation, not assumed): with this parameterization
    -- numpy.random.Generator.pareto(alpha) shifted by +1 and rounded -- HIGHER alpha
    concentrates more mass near 1, producing a HIGHER monofact rate; lower alpha spreads
    mass into a heavier tail of larger counts, producing a LOWER monofact rate but a few
    very high-frequency outliers. This is the opposite direction from the loose verbal
    description in some Zipf/Pareto literature, where "lower shape parameter = heavier
    tail = more rare events" is also true in a *relative* sense -- but the monofact rate
    (exactly-one count) specifically depends on where the bulk of the distribution's mass
    sits after the +1 shift and rounding, which is what we verify here empirically rather
    than asserting from the general heavy-tail intuition. See tests/test_facts_dataset.py
    for the regression test that pins this direction down."""
    raw = rng.pareto(alpha, size=n) + 1.0
    counts = np.clip(np.round(raw), 1, None).astype(int)
    return counts


def _bin_frequency(freq: int) -> str:
    if freq <= 1:
        return "monofact"
    if freq <= 5:
        return "rare"
    return "common"


def generate_fact_dataset(n: int = 200, alpha: float = 1.2, seed: int = 7) -> list[FactRecord]:
    """
    Generate n synthetic FactRecords with a Pareto-distributed synthetic frequency.
    `alpha` controls the monofact rate: with this implementation's parameterization,
    HIGHER alpha => higher monofact rate (see _sample_pareto_frequencies docstring for
    the verified, non-obvious direction). Kalai & Vempala's theorem predicts hallucination
    rate (in open_book mode, recall error rate) should track the monofact rate this induces,
    regardless of which direction of alpha produces it.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    freqs = _sample_pareto_frequencies(n, alpha, rng)
    records = []
    used_names = set()
    max_unique = len(FIRST_NAMES) * len(LAST_NAMES)

    for i in range(n):
        if n > max_unique:
            # Combinatorial pool of plain "First Last" names is exhausted for this n;
            # disambiguate with a numeric suffix so generation always terminates rather
            # than retrying indefinitely against a full name pool.
            name = f"{py_rng.choice(FIRST_NAMES)} {py_rng.choice(LAST_NAMES)}-{i}"
        else:
            attempts = 0
            while True:
                name = f"{py_rng.choice(FIRST_NAMES)} {py_rng.choice(LAST_NAMES)}"
                attempts += 1
                if name not in used_names or attempts > max_unique * 4:
                    used_names.add(name)
                    break

        profession = py_rng.choice(PROFESSIONS)
        city = py_rng.choice(CITIES)
        birth_year = py_rng.randint(1955, 1998)
        freq = int(freqs[i])

        bio = (f"{name} is {profession} based in {city}. "
               f"{name.split()[0]} was born in {birth_year}.")

        records.append(FactRecord(
            fact_id=f"fact_{i:04d}",
            entity_name=name,
            profession=profession,
            city=city,
            attribute="birth_year",
            attribute_value=str(birth_year),
            synthetic_frequency=freq,
            monofact_bin=_bin_frequency(freq),
            bio_context=bio,
        ))
    return records


def save_dataset(records: list[FactRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def load_dataset(path: str | Path) -> list[FactRecord]:
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(FactRecord(**json.loads(line)))
    return records


def monofact_rate(records: list[FactRecord]) -> float:
    """Fraction of records with synthetic_frequency == 1, mirroring the definition
    in Kalai & Vempala (2024), Section 3."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.synthetic_frequency == 1) / len(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate the synthetic arbitrary-facts dataset.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="data/arbitrary_facts_v1.jsonl")
    args = parser.parse_args()

    recs = generate_fact_dataset(n=args.n, alpha=args.alpha, seed=args.seed)
    save_dataset(recs, args.out)
    print(f"Wrote {len(recs)} facts to {args.out}")
    print(f"Monofact rate: {monofact_rate(recs):.3f}")
    print(f"Bin counts: monofact={sum(1 for r in recs if r.monofact_bin=='monofact')}, "
          f"rare={sum(1 for r in recs if r.monofact_bin=='rare')}, "
          f"common={sum(1 for r in recs if r.monofact_bin=='common')}")
