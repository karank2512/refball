"""Data coverage, missingness, and source provenance."""

from __future__ import annotations

import streamlit as st

from _shared import PATHS, load_provenance, load_table, page_header, require

st.set_page_config(page_title="Data Coverage", page_icon="📊", layout="wide")
page_header("Data Coverage", "What's in the dataset, how complete it is, and where it came from.")

if not require(PATHS.modeling_table, "Modeling table"):
    st.stop()

table = load_table()


def distinct_refs(df) -> int:
    import pandas as pd

    ids = pd.concat([df["ref_1_id"], df["ref_2_id"], df["ref_3_id"]], ignore_index=True)
    return int(ids.dropna().nunique())


c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", f"{len(table):,}")
c2.metric("Seasons", f"{table['season'].nunique()}")
c3.metric("Referees (distinct ids)", f"{distinct_refs(table)}")
c4.metric("Teams", f"{table['home_tricode'].nunique()}")

st.markdown("### Coverage by season")
by_season = (
    table.groupby("season")
    .agg(
        games=("game_id", "count"),
        with_officials=("has_officials", "mean"),
        with_odds=("has_odds", "mean"),
        mean_total_fouls=("total_fouls", "mean"),
    )
    .reset_index()
)
by_season["with_officials"] = (100 * by_season["with_officials"]).round(1)
by_season["with_odds"] = (100 * by_season["with_odds"]).round(1)
st.dataframe(by_season, width="stretch")

st.markdown("### Match rates & missingness")
colA, colB = st.columns(2)
colA.metric("Officials match rate", f"{100 * table['has_officials'].mean():.1f}%")
colB.metric("Odds match rate", f"{100 * table['has_odds'].mean():.1f}%")

key_cols = [
    "home_pf",
    "away_pf",
    "home_fta",
    "away_fta",
    "home_ftm",
    "away_ftm",
    "spread_home",
    "total_market",
    "ref_1_id",
    "ref_2_id",
    "ref_3_id",
]
miss = (
    table[key_cols]
    .isna()
    .mean()
    .mul(100)
    .round(1)
    .rename("missing_%")
    .reset_index()
    .rename(columns={"index": "column"})
)
st.dataframe(miss, width="stretch")

st.markdown("### Source provenance")
prov = load_provenance()
if len(prov):
    st.dataframe(prov, width="stretch")
else:
    st.info("No provenance log yet.")

st.caption(
    "If the odds match rate is low, the with-odds models are weaker and the no-odds "
    "sensitivity model is the fallback — see Methodology & Limitations."
)
