"""
Streamlit dashboard for the grounding-deficit harness.

Run with:
    streamlit run dashboard/app.py

Reads whatever result JSON files exist in results/ (produced by
experiments/run_eval.py) and renders:
  - A comparison table of Delta(M, F) across all evaluated models
  - A radar/bar chart per model showing the three-axis profile
  - Drill-down into per-fact results for debugging/inspection

This dashboard reads results only -- it does not call any model API
itself, so it has no API-key requirements and is safe to run without
internet access once results/ has been populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from grounding_deficit.report import latest_result_per_model, load_all_results

st.set_page_config(page_title="LLM Grounding Deficit", layout="wide")

st.title("LLM Grounding Deficit Dashboard")
st.caption(
    "Companion tool for *Three Lenses, One Gap: A Unified Theoretical Account of "
    "Hallucination in Large Language Models*. Renders the empirical proxy measurements "
    "for \u03b4S (statistical), \u03b4C (computational), and \u03b4E (epistemic) deficits "
    "proposed in Table 2 of the paper."
)

with st.expander("⚠️ Read before interpreting these numbers", expanded=False):
    st.markdown(
        """
These are **proposed, unvalidated proxies**, not validated measurements of the
constructs defined in the paper. Specifically:

- **δS** uses a *synthetic* fact dataset with fabricated frequency counts, not real
  training-data frequency. It tests in-context recall calibration, not pretraining
  calibration in the Kalai & Vempala (2024) sense.
- **δC** measures the gap between closed-book and open-book accuracy when the
  *correct* document is supplied directly — it is an upper bound on what retrieval
  could buy you, not a measurement of any real retrieval pipeline's actual recall.
- **δE** uses LLM-as-judge to check citation faithfulness, which captures only one
  of several conditions the paper's epistemic lens discusses (closest to the
  testimony-warrant literature; it does not probe the "bullshit framing" argument
  about the underlying generative process).

Treat everything below as illustrative and exploratory, not as a benchmark result
ready to cite. See the repo README for the full list of known limitations.
        """
    )

results_dir = st.sidebar.text_input("Results directory", value="results")
latest = latest_result_per_model(results_dir)

if not latest:
    st.info(
        f"No results found in `{results_dir}/`. Run an evaluation first, e.g.:\n\n"
        "```\npython experiments/run_eval.py --backend openai --model gpt-4o-mini --axis all\n```"
    )
    st.stop()

# --- Comparison table ---
st.header("Model comparison")

rows = []
for name, r in latest.items():
    rows.append({
        "model": name,
        "delta_S (calibration)": r.get("delta_s"),
        "delta_C (computational)": r.get("delta_c"),
        "delta_E (epistemic)": r.get("delta_e"),
        "timestamp": r.get("timestamp"),
    })
df = pd.DataFrame(rows).set_index("model")
st.dataframe(df.style.format({
    "delta_S (calibration)": "{:.3f}",
    "delta_C (computational)": "{:.3f}",
    "delta_E (epistemic)": "{:.3f}",
}, na_rep="—"), use_container_width=True)

st.caption("Lower is better on all three axes (these are *deficits*, per the paper's Definition 3).")

# --- Radar chart ---
st.header("Grounding-deficit profile per model")

axes = ["delta_S (calibration)", "delta_C (computational)", "delta_E (epistemic)"]
fig = go.Figure()
for name, r in latest.items():
    values = [r.get("delta_s"), r.get("delta_c"), r.get("delta_e")]
    if all(v is None for v in values):
        continue
    values_clean = [v if v is not None else 0 for v in values]
    fig.add_trace(go.Scatterpolar(
        r=values_clean + [values_clean[0]],
        theta=axes + [axes[0]],
        fill="toself",
        name=name,
    ))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# --- Per-axis bar comparison ---
st.header("Per-axis bar comparison")
melted = df.reset_index().melt(id_vars=["model", "timestamp"], value_vars=axes,
                                 var_name="axis", value_name="deficit")
melted = melted.dropna(subset=["deficit"])
if not melted.empty:
    import plotly.express as px
    fig2 = px.bar(melted, x="model", y="deficit", color="axis", barmode="group",
                  range_y=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

# --- Drill-down ---
st.header("Drill-down: per-fact results")
selected_model = st.selectbox("Select a model", list(latest.keys()))
selected = latest[selected_model]
raw = selected.get("_raw", {})

tab_s, tab_c, tab_e = st.tabs(["delta_S detail", "delta_C detail", "delta_E detail"])

with tab_s:
    if "delta_s" in raw:
        per_fact = pd.DataFrame(raw["delta_s"]["per_fact"])
        st.dataframe(per_fact, use_container_width=True)
        st.metric("Accuracy by bin", str(raw["delta_s"]["accuracy_by_bin"]))
    else:
        st.write("No delta_S data for this model.")

with tab_c:
    if "delta_c" in raw:
        per_fact = pd.DataFrame(raw["delta_c"]["per_fact"])
        st.dataframe(per_fact, use_container_width=True)
    else:
        st.write("No delta_C data for this model.")

with tab_e:
    if "delta_e" in raw:
        per_fact = pd.DataFrame(raw["delta_e"]["per_fact"])
        st.dataframe(per_fact, use_container_width=True)
    else:
        st.write("No delta_E data for this model.")

st.divider()
st.caption(
    "Source: *Three Lenses, One Gap* (paper). This dashboard implements the proxy "
    "metrics proposed in Table 2 as a starting point for empirical validation, "
    "per the paper's Section 9, Limitation 5 and Remark 7 (Falsifiability)."
)
