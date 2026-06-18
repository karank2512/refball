"""Ref Ball? — Streamlit home page."""

from __future__ import annotations

import streamlit as st

from _shared import CONVENTIONS, DISCLAIMER, is_demo, load_table

st.set_page_config(page_title="Ref Ball?", page_icon="🏀", layout="wide")

st.title("Ref Ball? 🏀")
st.subheader("Quantifying NBA Playoff Referee Impact Through Fouls and Point Differential")
st.markdown("##### Refs → fouls → point differential")

if is_demo():
    st.info(
        "🧪 **Demo mode** — currently showing **synthetic** data with planted structure so the "
        "whole app runs in minutes. Replace it with the real `nba_api` pull when ready."
    )

st.markdown(
    """
The motivating fan question — the *Scott Foster* narrative is the cultural hook — is:

> *Are certain referees or crews associated with NBA playoff foul patterns in ways that
> appear to "swing" outcomes?*

This project answers it **honestly**. It estimates, with full Bayesian uncertainty:

1. whether individual officials are associated with **higher/lower total foul volume**,
2. whether they are associated with a directional **home-vs-away foul margin** (a *lean*),
3. how foul margin and free-throw margin are associated with **final point differential**, and
4. a **mediated** referee → foul-margin → scoreboard composite — propagated over the posterior.

The causal spine we explore is:
"""
)
st.markdown(
    """
```text
Referee crew  →  foul environment / foul margin  →  free throws + game flow  →  point differential
```
"""
)

st.warning(DISCLAIMER)
st.markdown(CONVENTIONS)

st.markdown("### Why this is *not* clean causal identification")
st.markdown(
    """
- Referee assignments are **not random**; playoff assignments are **endogenous**.
- Better/veteran crews may be assigned to bigger, more physical, more competitive games.
- Betting lines control for team strength but **do not** remove selection bias.
- A statistical outlier is a **screening signal**, never proof of intent or misconduct.

Use the pages on the left to explore coverage, referee effects with credible intervals, a
crew-level game explorer, model diagnostics, and the full methodology and limitations.
"""
)

try:
    table = load_table()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Playoff games", f"{len(table):,}")
    c2.metric("Seasons", f"{table['season'].nunique()}")
    c3.metric("Games with officials", f"{100 * table['has_officials'].mean():.0f}%")
    c4.metric("Games with odds", f"{100 * table['has_odds'].mean():.0f}%")
except Exception:  # noqa: BLE001
    st.info("No modeling table yet — run the pipeline (see any page for the demo commands).")

st.caption(
    "Built with PyMC, ArviZ, pandas, Plotly, and Streamlit. Code and methodology are open; "
    "see the Methodology & Limitations page."
)
