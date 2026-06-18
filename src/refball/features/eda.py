"""Exploratory data analysis.

Produces figures under ``reports/figures`` and a raw per-referee summary table. Every raw
referee average here is **unadjusted** — it ignores pace, team strength, and market
expectation, and it credits all three officials with the whole-game outcome. These are
descriptive only; the Bayesian models in Stage 7+ are what actually adjust and shrink.
"""

from __future__ import annotations

import argparse

import numpy as np

from refball.config import get_settings
from refball.features.multimembership import build_ref_index
from refball.utils.logging import get_logger

logger = get_logger(__name__)


def _load_table():
    import pandas as pd

    s = get_settings()
    if not s.paths.modeling_table.exists():
        raise FileNotFoundError("modeling_table.parquet not found; run build_table first.")
    return pd.read_parquet(s.paths.modeling_table)


def raw_referee_averages(table):
    """Unadjusted per-referee mean total fouls and mean home foul margin."""
    import pandas as pd

    index = build_ref_index(table)
    recs = []
    id_cols = ["ref_1_id", "ref_2_id", "ref_3_id"]
    for rid in index.ref_ids:
        mask = np.zeros(len(table), dtype=bool)
        for c in id_cols:
            mask |= (
                pd.to_numeric(table[c], errors="coerce").fillna(-1).astype(int).to_numpy() == rid
            )
        sub = table[mask]
        if len(sub) == 0:
            continue
        recs.append(
            {
                "ref_id": rid,
                "referee": index.names[rid],
                "games": int(len(sub)),
                "raw_mean_total_fouls": float(sub["total_fouls"].mean()),
                "raw_mean_foul_diff_home": float(sub["foul_diff_home"].mean()),
                "raw_mean_point_diff_home": float(sub["point_diff_home"].mean()),
            }
        )
    out = pd.DataFrame(recs).sort_values("games", ascending=False).reset_index(drop=True)
    return out


def make_figures(table, raw_ref) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = get_settings()
    s.paths.ensure()
    saved = []

    def _save(fig, name):
        path = s.paths.figures / name
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(table["total_fouls"], bins=30, color="#3b6fb0")
    ax.set(title="Total fouls per game (unadjusted)", xlabel="total fouls", ylabel="games")
    _save(fig, "foul_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(table["foul_diff_home"], bins=30, color="#b05a3b")
    ax.axvline(0, color="k", lw=1)
    ax.set(
        title="Home minus away foul margin (unadjusted)", xlabel="home_pf - away_pf", ylabel="games"
    )
    _save(fig, "foul_margin_distribution.png")

    by_season = table.groupby("season")["total_fouls"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(by_season.index, by_season.values, marker="o")
    ax.set(
        title="Mean total fouls by season", xlabel="season (start year)", ylabel="mean total fouls"
    )
    _save(fig, "fouls_by_season.png")

    top = raw_ref.head(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["referee"][::-1], top["games"][::-1], color="#4a7")
    ax.set(title="Top referees by games worked (in sample)", xlabel="games")
    _save(fig, "top_referees_by_games.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(raw_ref["games"], raw_ref["raw_mean_foul_diff_home"], alpha=0.6)
    ax.axhline(0, color="k", lw=1)
    ax.set(
        title="Raw (unadjusted) referee home foul margin vs games",
        xlabel="games worked",
        ylabel="mean home foul margin",
    )
    _save(fig, "referee_raw_foul_margin.png")

    return saved


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Run EDA and write figures.").parse_args(argv)
    s = get_settings()
    table = _load_table()
    raw_ref = raw_referee_averages(table)
    out_path = s.paths.processed / "eda_referee_raw.parquet"
    raw_ref.to_parquet(out_path, index=False)
    figs = make_figures(table, raw_ref)
    print(f"[eda] referees={len(raw_ref)} figures={len(figs)}")
    print("NOTE: raw referee averages are UNADJUSTED (no pace/strength/market controls).")
    for f in figs:
        print("  wrote", f)
    print("  wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
