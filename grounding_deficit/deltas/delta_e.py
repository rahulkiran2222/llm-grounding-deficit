"""
delta_E: the epistemic proxy.

STATUS: Phase 3 (skeleton + working LLM-as-judge implementation; the
weakest-validated of the three axes, consistent with the paper's own
characterization of delta_E as "the least operationalized of the three"
-- Section 7.2, Remark 4).

Methodology: citation/attribution faithfulness rate, per Table 2 of the
paper. We ask the model a question that invites it to cite a source for
its answer (using the same fact bios as delta_s/delta_c, framed as
"documents"), then check whether the cited document actually supports the
claim attributed to it. Faithfulness is checked via LLM-as-judge (asking a
separate judge call whether the claim follows from the cited document),
which is the practical stand-in for a dedicated NLI/entailment model.

IMPORTANT CAVEAT (the paper says this explicitly in Section 7.3 and we
repeat it here so the limitation travels with the code, not just the
prose): citation faithfulness captures only one specific operationalization
of "epistemic warrant." It is closest to the testimony-warrant literature
(Heersmink et al. 2024) and does NOT probe the bullshit-framing argument's
claim about the underlying generative process (Hicks et al. 2024) -- a
model can be perfectly citation-faithful by this metric while still being,
on that argument, indifferent to truth by construction. Treat a good score
here as evidence against ONE specific epistemic failure mode, not as
evidence of "epistemic grounding" in general.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from grounding_deficit.data.facts_dataset import FactRecord
from grounding_deficit.models import BaseModel


@dataclass
class DeltaEResult:
    model_name: str
    n_facts: int
    faithfulness_rate: float          # fraction of cited claims judged as supported
    citation_rate: float               # fraction of answers that actually included a citation
    per_fact: list = field(default_factory=list)


_CITED_ANSWER_PROMPT = (
    "Here is a source document:\n\n"
    "Document: \"{bio}\"\n\n"
    "Question: What year was {first_name} born, and what supports your answer? "
    "Cite the document directly in your response."
)

_JUDGE_PROMPT = (
    "Document: \"{bio}\"\n\n"
    "Claim made (citing the document above): \"{claim}\"\n\n"
    "Does the document actually support this claim? "
    "Answer with only YES or NO."
)


def _looks_like_citation(text: str) -> bool:
    """Crude heuristic: did the answer reference the document at all
    (quote marks, 'the document', 'according to', 'states that', etc.)?
    A real implementation should replace this with a more careful classifier;
    flagged in README known limitations."""
    markers = ["document", "according to", "states that", "says that", '"']
    lowered = text.lower()
    return any(m in lowered for m in markers)


def _judge_supported(model: BaseModel, bio: str, claim: str) -> bool | None:
    judge_prompt = _JUDGE_PROMPT.format(bio=bio, claim=claim)
    resp = model.complete(judge_prompt, temperature=0.0, max_tokens=5)
    text = resp.text.strip().upper()
    if "YES" in text:
        return True
    if "NO" in text:
        return False
    return None  # unparseable judge response; excluded from the rate, logged in per_fact


def evaluate_delta_e(model: BaseModel, facts: list[FactRecord],
                       judge_model: BaseModel | None = None, verbose: bool = False) -> DeltaEResult:
    """
    judge_model: optional separate (typically stronger/cheaper-to-trust) model used
    to adjudicate faithfulness. Defaults to using `model` itself as the judge if not
    provided -- note this is a weaker design (self-judging) and is flagged as such
    in the README; for any result you intend to report, pass a distinct judge_model.
    """
    judge = judge_model or model
    per_fact = []

    for fact in facts:
        first_name = fact.entity_name.split()[0]
        prompt = _CITED_ANSWER_PROMPT.format(bio=fact.bio_context, first_name=first_name)
        resp = model.complete(prompt, temperature=0.0, max_tokens=150)

        cited = _looks_like_citation(resp.text)
        supported = _judge_supported(judge, fact.bio_context, resp.text) if cited else None

        per_fact.append({
            "fact_id": fact.fact_id,
            "answer": resp.text,
            "cited": cited,
            "judged_supported": supported,
        })

        if verbose:
            print(f"[{fact.fact_id}] cited={cited} supported={supported}")

    cited_items = [r for r in per_fact if r["cited"]]
    judged_items = [r for r in cited_items if r["judged_supported"] is not None]

    citation_rate = len(cited_items) / len(per_fact) if per_fact else float("nan")
    faithfulness_rate = (
        float(np.mean([r["judged_supported"] for r in judged_items]))
        if judged_items else float("nan")
    )

    return DeltaEResult(
        model_name=model.name,
        n_facts=len(facts),
        faithfulness_rate=faithfulness_rate,
        citation_rate=citation_rate,
        per_fact=per_fact,
    )
