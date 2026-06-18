"""Model diagnostics: convergence, posterior predictive checks, LOO, placebo."""

from __future__ import annotations

import streamlit as st

from _shared import PATHS, load_diagnostics, load_robustness, page_header

st.set_page_config(page_title="Model Diagnostics", page_icon="🩺", layout="wide")
page_header("Model Diagnostics", "Did the sampler converge, and do the models predict the data?")


def diag_block(title: str, d: dict) -> None:
    if not d:
        st.info(f"No diagnostics for {title} yet — run the corresponding fit.")
        return
    st.markdown(f"#### {title}")
    cols = st.columns(5)
    cols[0].metric("max R-hat", f"{d['max_r_hat']:.4f}", help="target < 1.01")
    cols[1].metric("min bulk ESS", f"{d['min_ess_bulk']:.0f}")
    cols[2].metric("min tail ESS", f"{d['min_ess_tail']:.0f}")
    cols[3].metric("divergences", f"{d['divergences']}")
    cols[4].metric("min BFMI", f"{d['min_bfmi']:.2f}", help="target > 0.3")
    if d["max_r_hat"] >= 1.01 or d["divergences"] > 0:
        st.warning(
            "Some diagnostics are outside target. In `--quick` demo mode this is expected "
            "(very short chains). Re-run without `--quick` for production estimates."
        )


st.markdown("### Convergence")
s1 = load_diagnostics("stage1")
diag_block("Stage 1A — total fouls", s1.get("total", {}))
diag_block("Stage 1B — directional lean", s1.get("lean", {}))
diag_block("Stage 2 — point differential", load_diagnostics("stage2").get("pointdiff", {}))

st.markdown("### Posterior predictive checks")
st.caption("Observed distribution (black) vs simulated datasets (blue). Good fit ≈ overlap.")
ppc_imgs = [
    ("ppc_total_total_fouls.png", "Total fouls"),
    ("ppc_lean_home_pf.png", "Home personal fouls"),
    ("ppc_lean_away_pf.png", "Away personal fouls"),
]
cols = st.columns(3)
any_ppc = False
for (fname, label), col in zip(ppc_imgs, cols, strict=True):
    p = PATHS.figures / fname
    if p.exists():
        col.image(str(p), caption=label, width="stretch")
        any_ppc = True
if not any_ppc:
    st.info("No PPC figures yet — run `python -m refball.models.fit_stage1`.")

st.markdown("### Baseline model comparison (PSIS-LOO)")
loo = load_robustness("loo_compare")
if len(loo):
    st.dataframe(loo, width="stretch")
    st.caption(
        "M0 = no referee effects, M1 = referee volume effect, M2 = volume + directional lean. "
        "If referee effects don't improve out-of-sample fit, the ranking and `elpd_diff` show it."
    )
else:
    st.info("Run `python -m refball.models.robustness --quick` to populate LOO comparison.")

st.markdown("### Permutation placebo")
placebo = load_robustness("placebo")
if len(placebo):
    st.dataframe(placebo, width="stretch")
    real = placebo.loc[placebo["run"] == "real", "sigma_ref_lean_mean"]
    others = placebo.loc[placebo["run"] != "real", "sigma_ref_lean_mean"]
    if len(real) and len(others):
        st.caption(
            f"Real referee-lean sigma = {real.iloc[0]:.4f}; shuffled-crew placebo mean = "
            f"{others.mean():.4f}. Under no real signal these should be indistinguishable."
        )
else:
    st.info("Run the placebo check to populate this.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Leave-one-season-out")
    loso = load_robustness("loso")
    st.dataframe(loso, width="stretch") if len(loso) else st.info("Run LOSO check.")
with c2:
    st.markdown("### Sensitivity (odds variants)")
    sens = load_robustness("sensitivity")
    st.dataframe(sens, width="stretch") if len(sens) else st.info("Run sensitivity check.")
