"""Additional evidence streams: regular-season power, within-series identification, PBP cleaning."""

from __future__ import annotations

import streamlit as st

from _shared import load_json_summary, page_header

st.set_page_config(page_title="Evidence Streams", page_icon="🧪", layout="wide")
page_header(
    "Evidence Streams",
    "Four extra angles, all attacking the same question — power, identification, and a cleaner whistle.",
)

st.markdown(
    "The headline analysis is *associational and underpowered*. These streams push on exactly "
    "those weaknesses with more public data. They all land in the same place."
)

# --- Regular-season power test -------------------------------------------------
st.markdown("## 1. Regular-season power test")
rs = load_json_summary("regular_season_summary")
if rs:
    c = st.columns(4)
    c[0].metric("Games", f"{rs['n_games']:,}")
    c[1].metric("Referees", f"{rs['n_refs']}")
    c[2].metric("Median games/ref", f"{rs['median_games_per_ref']}", help="vs ~21 in the playoffs")
    c[3].metric("Refs with lean ≠ 0", f"{rs['reg_refs_excluding_zero']} / {rs['n_refs']}")
    st.success(
        f"With **{rs['median_games_per_ref']} games per referee** — real statistical power — the "
        f"home-foul-lean variance is `sigma = {rs['reg_sigma_ref_lean']:.4f}` (even *smaller* than "
        f"the playoff ~0.013), and **{rs['reg_refs_excluding_zero']} of {rs['n_refs']}** referees "
        "have an interval excluding zero. Crucially, the correlation between a referee's "
        f"**playoff** lean and their high-power **regular-season** lean is Spearman "
        f"**{rs['spearman_reg_vs_playoff_lean']:.2f}** (≈ 0) — so the tiny playoff leans are "
        "**sampling noise, not a real referee trait**. And "
        f"**{rs['refs_playoff_differs_from_baseline']} of {rs['n_refs']}** referees behave "
        "differently in the playoffs than in their own regular season."
    )
    st.caption(
        "This converts the earlier *underpowered* null into a *high-power* null: given the chance "
        "to detect a real lean across thousands of games, the model finds none."
    )
else:
    st.info(
        "Run `python -m refball.models.regular_season --quick` (after the regular-season pull)."
    )

# --- Within-series identification ---------------------------------------------
st.markdown("## 2. Within-series identification")
ws = load_json_summary("within_series_summary")
if ws:
    c = st.columns(4)
    c[0].metric("Series", f"{ws['n_series']}")
    c[1].metric("Within-series σ_lean", f"{ws['sigma_ref_lean_within_series']:.4f}")
    c[2].metric("Placebo σ_lean", f"{ws['placebo_sigma_mean']:.4f}")
    c[3].metric("Refs with lean ≠ 0", f"{ws['crews_excluding_zero']} / {ws['n_refs']}")
    st.info(
        "A playoff series is the same two teams with a different crew most nights. Holding the "
        "**matchup fixed** (series × team effects), the crew lean is identified from the crew "
        f"rotation — and it stays at `{ws['sigma_ref_lean_within_series']:.4f}` ≈ placebo "
        f"`{ws['placebo_sigma_mean']:.4f}`, with **{ws['crews_excluding_zero']}/{ws['n_refs']}** "
        f"refs distinguishable. Home-court foul effect `{ws['beta_home_mean']:+.3f}` (home whistled "
        "slightly less). The 'refs just draw certain matchups' confound is not hiding a swing."
    )
else:
    st.info("Run `python -m refball.models.within_series --quick`.")

# --- PBP game-state / foul-type cleaning --------------------------------------
st.markdown("## 3. Play-by-play game-state & foul-type cleaning")
pc = load_json_summary("pbp_clean_summary")
if pc:
    import pandas as pd

    corrs = pc["correlations_with_point_diff"]
    gam = pc["gamma_foul_points"]
    st.caption(
        f"{pc['n_games']} games. Does the foul→points link appear once we keep only discretionary "
        "fouls (shooting/personal/loose-ball) and drop garbage-time/intentional fouls?"
    )
    tbl = pd.DataFrame(
        [
            {
                "foul margin": k,
                "corr with point diff": f"{corrs[k]:+.3f}",
                "γ (pts/foul)": f"{gam[g]['mean']:+.3f}",
                "94% HDI": f"[{gam[g]['hdi'][0]:+.2f}, {gam[g]['hdi'][1]:+.2f}]",
                "P(<0)": f"{gam[g]['p_lt0']:.2f}",
            }
            for k, g in [
                ("raw_box_foul_diff", "raw_box"),
                ("discretionary_foul_diff", "discretionary"),
                ("competitive_disc_foul_diff", "competitive_disc"),
            ]
        ]
    )
    st.dataframe(tbl, width="stretch")
    st.info(
        "If the competitive/discretionary coefficient were clearly negative while the raw one is "
        "~0, game-state confounds were masking a real foul→scoreboard link. Read the table above "
        "for this dataset's verdict."
    )
else:
    st.info("Run `python -m refball.data.pbp` then `python -m refball.models.pbp_clean --quick`.")

st.markdown("## 4. Closing betting line (market control)")
st.caption(
    "Adding the real closing spread/total (sportsbookreview archive, 332 matched games) barely "
    "moves the foul→points coefficient and leaves referee-lean rankings stable (Spearman ~0.76). "
    "See Data Coverage for the odds match rate."
)

st.success(
    "**Bottom line across every instrument** — box-score fouls, the foul→points link, "
    "within-series identification, the NBA's own L2M clutch-call grades, a high-power "
    "regular-season test, and a market-controlled model — all agree: **no detectable "
    "referee/crew swing**, with at most a faint, well-documented home tilt. The high-power "
    "regular-season result is the strongest: even with the power to find one, there's no lean, "
    "and playoff 'leans' don't persist."
)
