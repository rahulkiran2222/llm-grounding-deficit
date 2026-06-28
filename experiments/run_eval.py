#!/usr/bin/env python
"""
Main CLI entrypoint: run grounding-deficit evaluation against a model.

Examples:
    python experiments/run_eval.py --backend openai --model gpt-4o-mini --axis delta_s
    python experiments/run_eval.py --backend anthropic --model claude-haiku-4-5-20251001 --axis all --n-facts 50
    python experiments/run_eval.py --backend together --model meta-llama/Llama-3.3-70B-Instruct-Turbo --axis delta_c

Requires the relevant API key as an environment variable
(OPENAI_API_KEY / ANTHROPIC_API_KEY / TOGETHER_API_KEY) depending on
--backend. Set it before running, e.g.:
    export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grounding_deficit.data.facts_dataset import load_dataset, generate_fact_dataset, save_dataset
from grounding_deficit.models import get_model
from grounding_deficit.deltas.delta_s import evaluate_delta_s
from grounding_deficit.deltas.delta_c import evaluate_delta_c
from grounding_deficit.deltas.delta_e import evaluate_delta_e
from grounding_deficit.report import to_deficit_triple, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, choices=["openai", "anthropic", "together"])
    parser.add_argument("--model", required=True, help="Model name/ID for the chosen backend")
    parser.add_argument("--axis", default="all", choices=["delta_s", "delta_c", "delta_e", "all"])
    parser.add_argument("--dataset", default="data/arbitrary_facts_v1.jsonl",
                         help="Path to the facts dataset. Generated automatically if missing.")
    parser.add_argument("--n-facts", type=int, default=40,
                         help="Number of facts to evaluate (subset of the dataset, for cost control)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--judge-backend", default=None,
                         help="Optional separate backend for delta_e's faithfulness judge "
                              "(recommended over self-judging). E.g. --judge-backend openai")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}, generating a default one...")
        recs = generate_fact_dataset(n=200, alpha=1.2, seed=7)
        save_dataset(recs, dataset_path)

    facts = load_dataset(dataset_path)[: args.n_facts]
    print(f"Loaded {len(facts)} facts from {dataset_path}")

    try:
        model = get_model(args.backend, args.model)
    except Exception as e:
        print(f"\nERROR: failed to initialize backend='{args.backend}' model='{args.model}'.")
        print(f"  {type(e).__name__}: {e}")
        print(f"  Did you set the right API key as an environment variable? "
              f"See .env.example for the expected variable name per backend.")
        sys.exit(1)
    print(f"Evaluating model: {model.name} (backend={args.backend})")

    delta_s_res = delta_c_res = delta_e_res = None

    if args.axis in ("delta_s", "all"):
        print("\n--- Running delta_S (calibration) ---")
        delta_s_res = evaluate_delta_s(model, facts, verbose=args.verbose)
        print(f"  accuracy_overall = {delta_s_res.accuracy_overall:.3f}")
        print(f"  accuracy_by_bin  = {delta_s_res.accuracy_by_bin}")
        print(f"  ECE              = {delta_s_res.ece:.3f}")
        print(f"  Brier score      = {delta_s_res.brier_score:.3f}")

    if args.axis in ("delta_c", "all"):
        print("\n--- Running delta_C (retrieval gain) ---")
        delta_c_res = evaluate_delta_c(model, facts, verbose=args.verbose)
        print(f"  accuracy_closed_book = {delta_c_res.accuracy_closed_book:.3f}")
        print(f"  accuracy_open_book   = {delta_c_res.accuracy_open_book:.3f}")
        print(f"  retrieval_gain       = {delta_c_res.retrieval_gain:.3f}")

    if args.axis in ("delta_e", "all"):
        print("\n--- Running delta_E (citation faithfulness) ---")
        judge_model = None
        if args.judge_backend:
            judge_model = get_model(args.judge_backend, args.judge_model)
            print(f"  using separate judge: {judge_model.name} (backend={args.judge_backend})")
        else:
            print("  WARNING: no --judge-backend given; model will judge its own citations "
                  "(self-judging). Results are weaker evidence -- see README.")
        delta_e_res = evaluate_delta_e(model, facts, judge_model=judge_model, verbose=args.verbose)
        print(f"  citation_rate     = {delta_e_res.citation_rate:.3f}")
        print(f"  faithfulness_rate = {delta_e_res.faithfulness_rate:.3f}")

    triple = to_deficit_triple(delta_s_res, delta_c_res, delta_e_res)
    out_path = save_result(model.name, triple, results_dir=args.results_dir)

    print(f"\nDeficit triple Delta(M, F) for {model.name}:")
    print(f"  delta_S = {triple['delta_s']}")
    print(f"  delta_C = {triple['delta_c']}")
    print(f"  delta_E = {triple['delta_e']}")
    print(f"\nSaved full result to {out_path}")
    print("Run `streamlit run dashboard/app.py` to view this in the dashboard.")


if __name__ == "__main__":
    main()
