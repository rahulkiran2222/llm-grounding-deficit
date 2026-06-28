# LLM Grounding Deficit

An empirical companion tool for the paper **"Three Lenses, One Gap: A Unified
Theoretical Account of Hallucination in Large Language Models."** The paper
surveys three theoretical traditions explaining why LLMs hallucinate
(statistical learning theory, computability/information theory, and
epistemology) and proposes a unifying construct, the *grounding deficit*
Δ(M, F) = (δS, δC, δE), along with a set of **proposed but unvalidated**
proxy metrics for measuring it (Table 2 of the paper).

This repo implements those proxies as a real, runnable harness against
actual deployed LLMs (via API), plus a small dashboard to visualize results.

**Status: early, working, and intentionally honest about its own limits.**
This is a v1 demo built to *become* a real eval tool, not a finished
benchmark. Read "Known limitations of v1" below before drawing any
conclusions from a run.

---

## What this measures

| Axis | What it's a proxy for | How it's measured here |
|---|---|---|
| **δS** (statistical) | Calibration / monofact-rate sensitivity (Kalai & Vempala, 2024) | Expected Calibration Error (ECE) and Brier score on a synthetic fact-recall task, stratified by a fabricated "monofact bin" |
| **δC** (computational) | Residual gap that retrieval/tool access can't close (Shi et al., 2025; Guo & Li, 2026) | Accuracy gap between closed-book and open-book (correct document supplied) answers to the same question |
| **δE** (epistemic) | Testimonial warrant / citation faithfulness (Heersmink et al., 2024) | LLM-as-judge check of whether a model's cited source actually supports its claim |

Each module lives in `grounding_deficit/deltas/` and is independently runnable.

## Quickstart

```bash
git clone <this-repo>
cd llm-grounding-deficit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,dev]"

cp .env.example .env   # fill in the API key(s) you have
export $(grep -v '^#' .env | xargs)   # or use python-dotenv / direnv

# generate the seed dataset (or just use the committed data/arbitrary_facts_v1.jsonl)
python grounding_deficit/data/facts_dataset.py --n 200 --alpha 1.2 --out data/arbitrary_facts_v1.jsonl

# run an evaluation
python experiments/run_eval.py --backend openai --model gpt-4o-mini --axis all --n-facts 40

# view results
streamlit run dashboard/app.py
```

Run the test suite (pure logic, no API calls, no keys needed):
```bash
pytest tests/ -v
```

## Repo structure

```
grounding_deficit/
  models.py              # unified interface over OpenAI / Anthropic / Together (OpenAI-compatible) backends
  data/facts_dataset.py  # synthetic Pareto-distributed "arbitrary facts" generator
  deltas/
    delta_s.py           # calibration proxy (Phase 1 — most developed)
    delta_c.py           # retrieval-gain proxy (Phase 2 — minimal working version)
    delta_e.py           # citation-faithfulness proxy (Phase 3 — skeleton, LLM-judge based)
  report.py              # aggregates results into Delta(M,F), saves/loads JSON
experiments/run_eval.py  # CLI entrypoint
dashboard/app.py         # Streamlit dashboard (reads results/, no API calls itself)
data/arbitrary_facts_v1.jsonl   # committed seed dataset (200 synthetic facts)
results/                 # output JSON per run (gitignored by default; .gitkeep keeps the dir)
tests/                   # pytest suite, all using mocked models — no API keys required
```

## Build phases (where things actually stand)

- **Phase 1 — δS (calibration): working, most validated of the three.**
  Clean implementation of ECE/Brier on a controlled synthetic dataset. The
  weakest link is the dataset itself (see limitations below), not the
  calibration math, which is standard and tested.
- **Phase 2 — δC (retrieval gain): working, minimal.**
  Measures the *best case* — what happens when the model is handed the
  exact right document. Does not yet wire in a real retriever, so it
  cannot capture retrieval *failure* (wrong document retrieved, document
  not found at all). That's the natural v2 extension.
- **Phase 3 — δE (citation faithfulness): working, least validated.**
  Uses LLM-as-judge rather than a dedicated NLI/entailment model, and the
  "did it cite anything" detector is a crude keyword heuristic, not a
  trained classifier. Treat this axis's numbers as the most exploratory of
  the three — consistent with the paper's own characterization of δE as
  "the least operationalized" of the framework's three coordinates.

## Known limitations of v1 (read this before citing any numbers)

This list exists because the paper this repo implements was itself
criticized — correctly — for proposing proxies without validating them.
Building the harness is a step toward validation, not validation itself.
Specifically:

1. **The δS dataset is synthetic, not real.** Kalai & Vempala's theorem is
   about *pretraining* frequency, which we cannot control for API-only
   models like GPT-4o or Claude. This harness instead tests **in-context
   recall calibration** — a fact is given in a short bio paragraph in the
   prompt, and we measure whether confidence tracks recall accuracy,
   stratified by a *fabricated* frequency label. This is a different
   (related, but not identical) phenomenon from the one the theorem is
   about. **Planned v1.1:** replace fabricated frequency with a real
   corpus-frequency proxy (e.g., a Wikipedia or Common Crawl n-gram
   frequency lookup) so that "rare fact" reflects something true about the
   world rather than a label we invented.
2. **δC measures an upper bound, not a real retrieval pipeline.** Supplying
   the correct document directly tells you what retrieval *could* buy you
   in the best case, not whether a real retriever (BM25, embeddings, etc.)
   would actually find and surface that document. A model could score well
   here and still hallucinate badly in a real RAG deployment if retrieval
   itself fails.
3. **δE's citation detector is a keyword heuristic**, and its faithfulness
   judge is an LLM call, not a validated NLI model. Self-judging (using the
   same model to answer and to judge) is supported but explicitly
   discouraged — always pass `--judge-backend` pointing at a different
   model if you want results worth reporting.
4. **Confidence elicitation differs by backend.** OpenAI's API exposes
   token logprobs, which we use directly as a confidence proxy. Anthropic's
   API does not, so we fall back to asking the model to self-report a
   0–100 confidence score in a follow-up turn. These are not the same
   underlying quantity, and cross-backend ECE comparisons should be read
   with that caveat front and center, not as a footnote.
5. **None of the three δ proxies has been validated against the others, or
   against any ground truth, on a shared benchmark.** That validation is
   the actual research contribution still missing — this repo makes that
   work possible to *do*, it does not claim to have done it.
6. **Sample sizes in a typical quickstart run (n=40) are small.** Treat any
   single run as exploratory/diagnostic, not as a statistically powered
   comparison between models, unless you scale up `--n-facts` and report
   confidence intervals (not currently computed automatically — a good
   first contribution if you want to extend this).

## Relationship to the paper

This repo operationalizes **Table 2** ("Proposed proxy metrics for each
axis of the grounding deficit") and is a direct response to peer-review
feedback that the original proxies were "not yet measurable in practice."
It also is designed to test the conditions described in the paper's
**Remark 7 (Falsifiability)**: if a model achieves near-zero deficit on one
axis without the predicted blind spot appearing on the other two (or vice
versa — closing one axis without affecting the others, contrary to
Claim 1's prediction), that's a finding worth writing up, not a result to
discard.

## License

MIT 
