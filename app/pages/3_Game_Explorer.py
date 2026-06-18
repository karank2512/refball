"""Crew-level game explorer with season/team/round/referee filters."""

from __future__ import annotations

import numpy as np
import streamlit as st

from _shared import PATHS, load_ref_effects, load_table, page_header, require

st.set_page_config(page_title="Game Explorer", page_icon="🔎", layout="wide")
page_header(
    "Game Explorer", "Filter playoff games and see the crew, the whistle, and the scoreboard."
)

if not require(PATHS.modeling_table, "Modeling table"):
    st.stop()

table = load_table().copy()
ref_eff = load_ref_effects() if PATHS.referee_stage1_effects.exists() else None

# Estimated crew lean per game = mean of assigned officials' lean (if model has run).
if ref_eff is not None:
    lean_by_id = dict(zip(ref_eff["ref_id"], ref_eff["lean_mean"], strict=False))

    def crew_lean(row):
        vals = [lean_by_id.get(row[c]) for c in ("ref_1_id", "ref_2_id", "ref_3_id")]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    table["est_crew_lean"] = table.apply(crew_lean, axis=1)

# --- filters ---
c1, c2, c3, c4 = st.columns(4)
seasons = sorted(table["season"].unique().tolist())
sel_seasons = c1.multiselect("Season", seasons, default=seasons)
teams = sorted(set(table["home_tricode"]) | set(table["away_tricode"]))
sel_team = c2.selectbox("Team (home or away)", ["(any)"] + teams)
rounds = sorted([int(r) for r in table["playoff_round"].dropna().unique().tolist()])
sel_round = c3.multiselect("Playoff round", rounds, default=rounds)
officials = sorted(
    set(table["official_1"].dropna())
    | set(table["official_2"].dropna())
    | set(table["official_3"].dropna())
)
sel_ref = c4.selectbox("Referee", ["(any)"] + officials)

f = table[table["season"].isin(sel_seasons)]
if sel_round:
    f = f[f["playoff_round"].isin(sel_round)]
if sel_team != "(any)":
    f = f[(f["home_tricode"] == sel_team) | (f["away_tricode"] == sel_team)]
if sel_ref != "(any)":
    f = f[
        (f["official_1"] == sel_ref) | (f["official_2"] == sel_ref) | (f["official_3"] == sel_ref)
    ]

st.caption(f"{len(f):,} games match the filters.")

f = f.assign(
    matchup=f["away_tricode"] + " @ " + f["home_tricode"],
    score=f["away_score"].astype(str) + "–" + f["home_score"].astype(str),
    crew=f[["official_1", "official_2", "official_3"]].fillna("?").agg(" · ".join, axis=1),
)
show_cols = [
    "game_date",
    "season",
    "playoff_round",
    "matchup",
    "score",
    "point_diff_home",
    "spread_home",
    "total_fouls",
    "foul_diff_home",
    "ftm_diff_home",
    "crew",
]
if "est_crew_lean" in f.columns:
    show_cols.append("est_crew_lean")

st.dataframe(
    f[show_cols]
    .sort_values("game_date")
    .rename(
        columns={
            "playoff_round": "round",
            "point_diff_home": "home pt diff",
            "spread_home": "spread (home)",
            "total_fouls": "total fouls",
            "foul_diff_home": "foul margin (home)",
            "ftm_diff_home": "FT made margin",
            "est_crew_lean": "est. crew lean",
        }
    ),
    width="stretch",
    height=520,
)

st.download_button(
    "⬇️ Download filtered games (CSV)",
    data=f[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="games_filtered.csv",
    mime="text/csv",
)

if len(f) > 1:
    st.markdown("### Foul margin vs final point differential (filtered)")
    import importlib.util

    import plotly.express as px

    has_statsmodels = importlib.util.find_spec("statsmodels") is not None
    fig = px.scatter(
        f,
        x="foul_diff_home",
        y="point_diff_home",
        hover_data=["matchup", "score", "crew"],
        labels={
            "foul_diff_home": "home foul margin (home_pf − away_pf)",
            "point_diff_home": "home point differential",
        },
        trendline="ols" if (len(f) >= 5 and has_statsmodels) else None,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, width="stretch")
    st.caption("Descriptive scatter for the current filter — not the adjusted model estimate.")
