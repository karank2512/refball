"""Referee effects: total-foul, directional lean, and mediated scoreboard — with intervals."""

from __future__ import annotations

import streamlit as st

from _shared import (
    CONVENTIONS,
    DISCLAIMER,
    PATHS,
    SETTINGS,
    caterpillar,
    load_mediated,
    load_ref_effects,
    page_header,
    require,
)

st.set_page_config(page_title="Referee Effects", page_icon="🧑‍⚖️", layout="wide")
page_header(
    "Referee Effects",
    "Partial-pooled, multi-membership estimates — every number carries a credible interval.",
)

if not (require(PATHS.referee_stage1_effects, "Stage 1 referee effects")):
    st.stop()

eff = load_ref_effects()
med = load_mediated() if PATHS.referee_mediated_effects.exists() else None

st.warning(DISCLAIMER)
st.markdown(CONVENTIONS)

min_games = st.slider(
    "Minimum games worked (display filter only — the model still partial-pools everyone)",
    min_value=1,
    max_value=int(eff["games"].max()),
    value=int(SETTINGS.min_games_for_display),
)
view = eff[eff["games"] >= min_games].copy()

# Merge mediated effect if present.
if med is not None:
    view = view.merge(
        med[
            [
                "ref_id",
                "mediated_points_mean",
                "mediated_hdi_low",
                "mediated_hdi_high",
                "p_mediated_lt0",
            ]
        ],
        on="ref_id",
        how="left",
    )


def hdi_str(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"


prob = int(round(SETTINGS.hdi_prob * 100))
display = view.copy()
display[f"total {prob}% HDI"] = [
    hdi_str(a, b) for a, b in zip(display["total_hdi_low"], display["total_hdi_high"], strict=True)
]
display[f"lean {prob}% HDI"] = [
    hdi_str(a, b) for a, b in zip(display["lean_hdi_low"], display["lean_hdi_high"], strict=True)
]
cols = {
    "referee": "referee",
    "games": "games",
    "total_effect_mean": "total foul effect (log)",
    f"total {prob}% HDI": f"total {prob}% HDI",
    "p_total_gt0": "P(total>0)",
    "extra_fouls_per100_mean": "extra fouls/100 poss",
    "lean_mean": "home-foul lean (log)",
    f"lean {prob}% HDI": f"lean {prob}% HDI",
    "p_lean_gt0": "P(lean>0)",
    "foul_margin_mean": "exp. foul margin",
}
if med is not None:
    display[f"mediated {prob}% HDI"] = [
        hdi_str(a, b)
        for a, b in zip(display["mediated_hdi_low"], display["mediated_hdi_high"], strict=True)
    ]
    cols.update(
        {
            "mediated_points_mean": "mediated home pts",
            f"mediated {prob}% HDI": f"mediated {prob}% HDI",
        }
    )

st.markdown("### Referee effects table")
st.dataframe(
    display[list(cols)].rename(columns=cols).sort_values("home-foul lean (log)", ascending=False),
    width="stretch",
    height=480,
)
st.caption(
    "`total foul effect` and `home-foul lean` are on the log-rate scale (the model's native "
    "units). `extra fouls/100 poss` and `exp. foul margin` translate a single official's "
    "marginal contribution (effect ÷ 3, since a crew is 3 officials) into countable units."
)

st.download_button(
    "⬇️ Download referee effects (CSV)",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="referee_effects.csv",
    mime="text/csv",
)

st.markdown("### Caterpillar plots")
tab1, tab2, tab3 = st.tabs(["Total foul effect", "Home-foul lean", "Mediated scoreboard effect"])
with tab1:
    st.plotly_chart(
        caterpillar(
            view,
            mean_col="total_effect_mean",
            low_col="total_hdi_low",
            high_col="total_hdi_high",
            title="Referee total-foul effect (log-rate)",
            x_title="effect (log-rate)",
        ),
        width="stretch",
    )
with tab2:
    st.plotly_chart(
        caterpillar(
            view,
            mean_col="lean_mean",
            low_col="lean_hdi_low",
            high_col="lean_hdi_high",
            title="Referee home-foul lean (positive = more home fouls)",
            x_title="lean (log-rate)",
        ),
        width="stretch",
    )
with tab3:
    if med is not None:
        st.plotly_chart(
            caterpillar(
                view,
                mean_col="mediated_points_mean",
                low_col="mediated_hdi_low",
                high_col="mediated_hdi_high",
                title="Mediated referee → scoreboard effect",
                x_title="home point differential (pts)",
            ),
            width="stretch",
        )
    else:
        st.info("Run `python -m refball.models.mediation` to populate mediated effects.")

st.caption(
    "Intervals that comfortably straddle zero mean no detectable referee-associated signal. "
    "Ranking by point estimate alone — without the interval — is exactly the mistake this "
    "project is built to avoid."
)
