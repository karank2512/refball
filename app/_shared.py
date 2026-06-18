"""Shared helpers for the Ref Ball Streamlit app.

Adds ``src`` to the path (so ``import refball`` works when launched via
``streamlit run app/Home.py``), provides cached artifact loaders, the responsible-
interpretation disclaimer, and reusable Plotly figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from refball.config import get_settings  # noqa: E402
from refball.utils.provenance import read_provenance  # noqa: E402

SETTINGS = get_settings()
PATHS = SETTINGS.paths

DISCLAIMER = (
    "**Responsible interpretation.** This is an *observational* project. Referees are **not** "
    "randomly assigned, and playoff assignments are endogenous — better or more experienced "
    "crews may be sent to bigger, more physical, more competitive games. Betting lines help "
    "control for team strength but do **not** solve selection bias. Every number here is a "
    "**screening metric reported with uncertainty, not proof**. Nothing on this site claims that "
    "any referee, crew, team, or league fixed, rigged, swung, or manipulated a game, or acted "
    "with intent. A statistical outlier is not evidence of misconduct."
)

CONVENTIONS = (
    "**Conventions.** `foul_diff_home = home_pf − away_pf`. A *positive* referee **lean** means "
    "games with that official are associated with **more home-team fouls** relative to away — "
    "that is the home team being *whistled more*, not favored. `spread_home < 0` means the home "
    "team was favored."
)


# --- artifact existence / demo detection ------------------------------------
def artifact_exists(path: Path) -> bool:
    return Path(path).exists()


def require(path: Path, what: str) -> bool:
    """Show a friendly message + return False if an artifact is missing."""
    if artifact_exists(path):
        return True
    st.warning(
        f"**{what}** not found at `{path.relative_to(REPO_ROOT)}`.\n\n"
        "Run the pipeline first (demo mode takes ~1 minute):\n\n"
        "```bash\npython -m refball.data.synthetic\npython -m refball.features.build_table\n"
        "python -m refball.models.fit_stage1 --quick\npython -m refball.models.fit_stage2 --quick\n"
        "python -m refball.models.mediation\n```"
    )
    return False


@st.cache_data(show_spinner=False)
def is_demo() -> bool:
    if PATHS.synthetic_truth.exists():
        return True
    return any(r.get("source") == "synthetic" for r in read_provenance())


# --- cached loaders ----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_table() -> pd.DataFrame:
    return pd.read_parquet(PATHS.modeling_table)


@st.cache_data(show_spinner=False)
def load_ref_effects() -> pd.DataFrame:
    return pd.read_parquet(PATHS.referee_stage1_effects)


@st.cache_data(show_spinner=False)
def load_mediated() -> pd.DataFrame:
    return pd.read_parquet(PATHS.referee_mediated_effects)


@st.cache_data(show_spinner=False)
def load_stage2_coef() -> pd.DataFrame:
    return pd.read_parquet(PATHS.stage2_coefficients)


@st.cache_data(show_spinner=False)
def load_eda_raw() -> pd.DataFrame:
    p = PATHS.processed / "eda_referee_raw.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_provenance() -> pd.DataFrame:
    recs = read_provenance()
    return pd.DataFrame(recs) if recs else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_diagnostics(stage: str) -> dict:
    p = PATHS.processed / f"diagnostics_{stage}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_robustness(name: str) -> pd.DataFrame:
    p = PATHS.robustness_dir / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_l2m_summary() -> dict:
    p = PATHS.processed / "l2m_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_l2m_crew_effects() -> pd.DataFrame:
    p = PATHS.processed / "l2m_crew_effects.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_l2m_game_table() -> pd.DataFrame:
    p = PATHS.processed / "l2m_game_table.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# --- reusable figures --------------------------------------------------------
def caterpillar(
    df: pd.DataFrame,
    *,
    mean_col: str,
    low_col: str,
    high_col: str,
    label_col: str = "referee",
    title: str = "",
    x_title: str = "effect",
    zero_line: bool = True,
    max_rows: int | None = None,
):
    """Horizontal point + credible-interval ('caterpillar') plot, sorted by mean."""
    import plotly.graph_objects as go

    d = df.dropna(subset=[mean_col, low_col, high_col]).sort_values(mean_col)
    if max_rows:
        d = d.tail(max_rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d[mean_col],
            y=d[label_col],
            error_x=dict(
                type="data",
                symmetric=False,
                array=d[high_col] - d[mean_col],
                arrayminus=d[mean_col] - d[low_col],
                thickness=1,
            ),
            mode="markers",
            marker=dict(size=7, color="#3b6fb0"),
            hovertemplate="%{y}<br>mean %{x:.3f}<extra></extra>",
        )
    )
    if zero_line:
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        height=max(360, 18 * len(d)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    if is_demo():
        st.info(
            "🧪 **Demo mode** — the loaded dataset is **synthetic** (planted structure for "
            "testing). Numbers are illustrative, not real NBA results. Run the real `nba_api` "
            "pull to replace it."
        )
