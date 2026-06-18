"""Static (matplotlib) report figures, mirroring the app's interactive Plotly charts.

Used for README screenshots / `reports/figures` artifacts. The Streamlit app builds its own
interactive Plotly versions in ``app/_shared.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from refball.config import get_settings
from refball.utils.logging import get_logger

logger = get_logger(__name__)


def caterpillar_mpl(
    df,
    *,
    mean_col: str,
    low_col: str,
    high_col: str,
    label_col: str = "referee",
    title: str = "",
    x_title: str = "effect",
    path: Path | None = None,
):
    """Horizontal point + credible-interval plot sorted by mean; saved if ``path`` given."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=[mean_col, low_col, high_col]).sort_values(mean_col)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.22 * len(d))))
    ax.errorbar(
        d[mean_col],
        y,
        xerr=[d[mean_col] - d[low_col], d[high_col] - d[mean_col]],
        fmt="o",
        color="#3b6fb0",
        ecolor="#9bb8d8",
        elinewidth=1,
        capsize=2,
        markersize=4,
    )
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col], fontsize=7)
    ax.set(title=title, xlabel=x_title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)
    return fig


def save_referee_caterpillars(ref_effects=None, mediated=None) -> list[str]:
    """Write total/lean (and mediated, if available) caterpillars to ``reports/figures``."""
    import pandas as pd

    s = get_settings()
    s.paths.ensure()
    if ref_effects is None:
        ref_effects = pd.read_parquet(s.paths.referee_stage1_effects)
    saved = [
        caterpillar_mpl(
            ref_effects,
            mean_col="total_effect_mean",
            low_col="total_hdi_low",
            high_col="total_hdi_high",
            title="Referee total-foul effect (log-rate)",
            x_title="effect (log-rate)",
            path=s.paths.figures / "caterpillar_total_effect.png",
        ),
        caterpillar_mpl(
            ref_effects,
            mean_col="lean_mean",
            low_col="lean_hdi_low",
            high_col="lean_hdi_high",
            title="Referee home-foul lean (positive = more home fouls)",
            x_title="lean (log-rate)",
            path=s.paths.figures / "caterpillar_lean.png",
        ),
    ]
    if mediated is None and s.paths.referee_mediated_effects.exists():
        mediated = pd.read_parquet(s.paths.referee_mediated_effects)
    if mediated is not None:
        saved.append(
            caterpillar_mpl(
                mediated,
                mean_col="mediated_points_mean",
                low_col="mediated_hdi_low",
                high_col="mediated_hdi_high",
                title="Mediated referee -> scoreboard effect",
                x_title="home point differential (pts)",
                path=s.paths.figures / "caterpillar_mediated.png",
            )
        )
    logger.info("Saved %d referee caterpillar figures.", len(saved))
    return saved
