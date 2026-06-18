"""L2M clutch-call evidence: the NBA's own graded officiating errors in close-game endings."""

from __future__ import annotations

import streamlit as st

from _shared import (
    caterpillar,
    load_l2m_crew_effects,
    load_l2m_summary,
    page_header,
)

st.set_page_config(page_title="Clutch Calls (L2M)", page_icon="⏱️", layout="wide")
page_header(
    "Clutch Calls — NBA Last Two Minute reports",
    "The most direct public evidence: the NBA's own graded calls/non-calls, signed by who they hurt.",
)

summary = load_l2m_summary()
if not summary:
    st.warning(
        "L2M results not found. Build them with:\n\n"
        "```bash\npython -m refball.data.l2m\npython -m refball.models.l2m_model --quick\n```"
    )
    st.stop()

st.markdown(
    """
The NBA publicly grades every officiated event in the final ~2 minutes of close games as
**Correct Call**, **Correct Non-Call**, **Incorrect Call**, or **Incorrect Non-Call**. We treat
the **incorrect** calls/non-calls (IC + INC) as graded *errors* and sign each one by the team it
disadvantaged — so an error that hurt the away team is an error that **favored the home team**.
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Close games covered",
    f"{summary['n_games']}",
    help="of 585 playoff games — only close-and-late games get an L2M report",
)
c2.metric("Graded clutch errors", f"{summary['total_errors']}")
c3.metric(
    "Errors favoring home", f"{summary['err_against_away']}", help="errors that hurt the away team"
)
c4.metric(
    "Errors favoring away", f"{summary['err_against_home']}", help="errors that hurt the home team"
)

st.markdown("### Is there an overall clutch home-error bias?")
lo, hi = summary["home_error_bias_hdi"]
b = summary["home_error_bias_mean"]
p = summary["p_bias_favors_home"]
rr = summary["favor_home_to_away_rate_ratio"]
verdict = (
    "directionally home-leaning but **not conclusive**"
    if (lo < 0 < hi)
    else "**distinguishable from zero**"
)
st.info(
    f"Overall bias `b = {b:+.3f}` (log), 94% HDI `[{lo:+.3f}, {hi:+.3f}]`, "
    f"**P(favors home) = {p:.2f}**, favor-home : favor-away error-rate ratio **{rr:.2f}×**. "
    f"This is {verdict} — consistent with the small home bias documented in the literature, "
    f"but {summary['n_games']} close games is too few to nail it."
)

st.markdown("### Per-crew clutch-error lean")
eff = load_l2m_crew_effects()
if len(eff):
    st.caption(
        f"**{summary['crews_lean_excludes_zero']} of {summary['n_crews']}** crews have a 94% "
        "interval excluding zero. Positive lean = that crew's clutch errors tend to favor the home "
        "team. (Crew-level, not individual official — L2M does not say which ref made each call.)"
    )
    st.plotly_chart(
        caterpillar(
            eff,
            mean_col="error_lean_mean",
            low_col="hdi_low",
            high_col="hdi_high",
            label_col="referee",
            title="Crew clutch-error lean (positive = errors favor home)",
            x_title="error lean (log-rate)",
            max_rows=40,
        ),
        width="stretch",
    )
    st.dataframe(
        eff[
            [
                "referee",
                "games",
                "error_lean_mean",
                "hdi_low",
                "hdi_high",
                "p_favor_home",
                "excludes_zero",
            ]
        ],
        width="stretch",
        height=360,
    )
    st.download_button(
        "⬇️ Download L2M crew effects (CSV)",
        data=eff.to_csv(index=False).encode("utf-8"),
        file_name="l2m_crew_effects.csv",
        mime="text/csv",
    )

st.markdown("### Placebo")
st.caption(
    f"Shuffling crews across games (within season) gives a placebo crew-lean σ of "
    f"**{summary['placebo_sigma_crew_lean_mean']:.4f}** vs the real **{summary['real_sigma_crew_lean']:.4f}** — "
    "indistinguishable, i.e. no crew-specific signal beyond chance."
)

st.warning(
    "**Read this carefully.** L2M is the most *direct* public measure, but: (1) it only covers "
    "games that were **close and late** (selection on the outcome — you can't read 'no report' as "
    "'no errors'); (2) grades attribute to the **crew**, not an individual official; (3) it's the "
    "**NBA grading its own calls**, not an independent referee; (4) it covers the last ~2 minutes "
    "only. So this answers *'do crews' clutch errors net-favor a side'* — and the answer here is "
    "**no detectable crew effect, and at most a faint, inconclusive leaguewide home tilt.** Not "
    "proof of bias or intent."
)
