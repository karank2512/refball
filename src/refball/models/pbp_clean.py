"""Does cleaning fouls (discretionary + competitive) uncover a foul -> scoreboard link?

Raw box-score foul margin is ~uncorrelated with the final margin (r~0.02) — plausibly because
intentional/garbage-time fouls swamp the signal. Using play-by-play (:mod:`refball.data.pbp`)
we rebuild the foul margin from **discretionary** fouls only (shooting/personal/loose-ball) and,
separately, from those in **competitive** game states (dropping late-game blowout fouls), then
re-estimate the foul-margin -> point-differential association. If cleaning materially strengthens
it, the raw null was a confound artifact; if not, fouls genuinely don't move the scoreboard much.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from refball.config import get_settings
from refball.models.diagnostics import hdi_interval
from refball.utils.logging import get_logger

logger = get_logger(__name__)


def _merged():
    import pandas as pd

    s = get_settings()
    pbp_path = s.paths.processed / "pbp_foul_table.parquet"
    if not pbp_path.exists():
        raise FileNotFoundError(
            "pbp_foul_table.parquet not found; run `python -m refball.data.pbp`."
        )
    mt = pd.read_parquet(s.paths.modeling_table)
    pbp = pd.read_parquet(pbp_path)
    df = mt.merge(pbp, on="game_id", how="inner").reset_index(drop=True)
    logger.info("PBP-clean data: %d games (of %d playoff games)", len(df), len(mt))
    return df


def _fit_gamma(df, margin_col, quick):
    """StudentT point-diff on a single (centered) foul-margin column + team strength + season."""
    import pymc as pm

    teams = sorted(set(df["home_tricode"]) | set(df["away_tricode"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    seasons = sorted(df["season"].unique().tolist())
    s_idx = {s: i for i, s in enumerate(seasons)}
    home = df["home_tricode"].map(t_idx).to_numpy()
    away = df["away_tricode"].map(t_idx).to_numpy()
    sidx = df["season"].map(s_idx).to_numpy()
    margin = df[margin_col].to_numpy(float)
    margin = margin - margin.mean()
    y = df["point_diff_home"].to_numpy(float)

    with pm.Model(coords={"team": teams, "season": [str(x) for x in seasons]}):
        g0 = pm.Normal("g0", 0, 5)
        g_foul = pm.Normal("g_foul", 0, 1)
        sig_str = pm.HalfNormal("sig_str", 5)
        strength = pm.Normal("strength", 0, sig_str, dims="team")
        sig_seas = pm.HalfNormal("sig_seas", 3)
        seas = pm.Normal("seas", 0, sig_seas, dims="season")
        mu = g0 + g_foul * margin + strength[home] - strength[away] + seas[sidx]
        nu = pm.Gamma("nu", 2, 0.1)
        sigma = pm.HalfNormal("sigma", 15)
        pm.StudentT("pd", nu=nu, mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(**get_settings().sampler.resolve(quick), progressbar=False)
    g = idata.posterior["g_foul"].values.ravel()
    lo, hi = hdi_interval(g, get_settings().hdi_prob)
    return float(g.mean()), lo, hi, float((g < 0).mean())


def run(quick: bool):
    import pandas as pd

    s = get_settings()
    df = _merged()

    def corr(a, b):
        return float(np.corrcoef(df[a], df[b])[0, 1])

    corrs = {
        "raw_box_foul_diff": corr("foul_diff_home", "point_diff_home"),
        "discretionary_foul_diff": corr("foul_diff_home_disc", "point_diff_home"),
        "competitive_disc_foul_diff": corr("foul_diff_home_comp", "point_diff_home"),
    }
    logger.info("Fitting foul->points for raw vs cleaned margins...")
    gammas = {
        "raw_box": _fit_gamma(df, "foul_diff_home", quick),
        "discretionary": _fit_gamma(df, "foul_diff_home_disc", quick),
        "competitive_disc": _fit_gamma(df, "foul_diff_home_comp", quick),
    }

    summary = {
        "n_games": int(len(df)),
        "correlations_with_point_diff": corrs,
        "gamma_foul_points": {
            k: {"mean": v[0], "hdi": [v[1], v[2]], "p_lt0": v[3]} for k, v in gammas.items()
        },
    }
    s.paths.ensure()
    (s.paths.processed / "pbp_clean_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"margin": k, "gamma_mean": v[0], "hdi_low": v[1], "hdi_high": v[2], "p_lt0": v[3]}
            for k, v in gammas.items()
        ]
    ).to_parquet(s.paths.processed / "pbp_clean_gammas.parquet", index=False)

    print("\n=== PBP GAME-STATE / FOUL-TYPE CLEANING (does the foul->points link appear?) ===")
    print(f"games: {summary['n_games']}")
    print("correlation of foul margin with final point differential:")
    for k, v in corrs.items():
        print(f"  {k:28s} r = {v:+.3f}")
    print("\nfoul-margin -> point-diff coefficient (StudentT, +team strength +season):")
    for k, v in gammas.items():
        print(
            f"  {k:18s} gamma = {v[0]:+.3f}  94% HDI [{v[1]:+.3f}, {v[2]:+.3f}]  P(<0) = {v[3]:.2f}"
        )
    print(
        "Read: if the competitive/discretionary gamma is clearly negative while the raw is ~0, "
        "game-state confounds were masking a real foul->scoreboard link."
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PBP cleaned foul -> points analysis.")
    p.add_argument("--quick", action="store_true")
    ns = p.parse_args(argv)
    run(quick=ns.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
