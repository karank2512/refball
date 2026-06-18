"""Stage 2: point-differential model (PyMC v5).

Outcome ``point_diff_home`` ~ StudentT (robust to blowouts). We estimate how foul margin
and free-throw margins are *associated* with the final scoreboard, controlling for the
betting line, total, and season:

    mu = gamma_0
       + gamma_line  * z_spread_home
       + gamma_total * z_total_market
       + gamma_foul_diff * foul_diff_home
       + gamma_fta_diff  * fta_diff_home
       + gamma_ftm_diff  * ftm_diff_home
       + season_effect
       [ + optional (home_strength - away_strength) team effects ]

This model is **associational**, not causal: fouls are themselves driven by game state,
aggression, strategy, and intentional fouling. ``gamma_fta_diff``/``gamma_ftm_diff`` capture
the mechanical free-throw scoring channel; ``gamma_foul_diff`` is the residual foul-margin
association after the line and free throws are accounted for.

Note: ``fta_diff`` and ``ftm_diff`` are strongly collinear (made ≈ 0.77 × attempted), so we
center all predictors and report their joint behavior with caution.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from refball.config import get_settings
from refball.models.data_prep import ModelData, prepare
from refball.models.diagnostics import hdi_interval, print_diagnostics, summarize_diagnostics
from refball.utils.logging import get_logger

logger = get_logger(__name__)

_COEFS = ["gamma_foul_diff", "gamma_fta_diff", "gamma_ftm_diff", "gamma_line", "gamma_total"]


def build_pointdiff_model(md: ModelData, team_effects: bool):
    import pymc as pm

    s = get_settings()

    def c(x):  # center a predictor
        x = np.asarray(x, dtype=float)
        return x - x.mean()

    foul_diff = c(md.foul_diff_home)
    fta_diff = c(md.fta_diff_home)
    ftm_diff = c(md.ftm_diff_home)

    with pm.Model(coords=md.coords) as model:
        gamma_0 = pm.Normal("gamma_0", 0.0, 5.0)
        gamma_foul_diff = pm.Normal("gamma_foul_diff", 0.0, 1.0)
        gamma_fta_diff = pm.Normal("gamma_fta_diff", 0.0, 1.0)
        gamma_ftm_diff = pm.Normal("gamma_ftm_diff", 0.0, 1.0)

        sigma_season = pm.HalfNormal("sigma_season", s.sigma_season_prior * 10)  # points scale
        season_eff = pm.Deterministic(
            "season_effect",
            pm.Normal("season_z", 0.0, 1.0, dims="season") * sigma_season,
            dims="season",
        )

        mu = (
            gamma_0
            + gamma_foul_diff * foul_diff
            + gamma_fta_diff * fta_diff
            + gamma_ftm_diff * ftm_diff
            + season_eff[md.season_idx]
        )
        if md.use_odds:
            gamma_line = pm.Normal("gamma_line", 0.0, 5.0)
            gamma_total = pm.Normal("gamma_total", 0.0, 3.0)
            mu = mu + gamma_line * md.z_spread_home + gamma_total * md.z_total_market
        if team_effects:
            sigma_strength = pm.HalfNormal("sigma_strength", 5.0)
            strength = pm.Deterministic(
                "team_strength",
                pm.Normal("strength_z", 0.0, 1.0, dims="team") * sigma_strength,
                dims="team",
            )
            mu = mu + strength[md.home_team_idx] - strength[md.away_team_idx]

        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
        sigma = pm.HalfNormal("sigma", 15.0)
        pm.StudentT(
            "point_diff_home", nu=nu, mu=mu, sigma=sigma, observed=md.point_diff_home, dims="game"
        )
    return model


def _fit(model, quick: bool):
    import pymc as pm

    with model:
        idata = pm.sample(**get_settings().sampler.resolve(quick), progressbar=False)
        pm.compute_log_likelihood(idata, progressbar=False)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)
    return idata


def summarize_coefficients(idata):
    import pandas as pd

    prob = get_settings().hdi_prob
    present = [c for c in _COEFS if c in idata.posterior]
    recs = []
    for name in present:
        draws = idata.posterior[name].values.ravel()
        lo, hi = hdi_interval(draws, prob)
        recs.append(
            {
                "coefficient": name,
                "mean": float(draws.mean()),
                "hdi_low": lo,
                "hdi_high": hi,
                "p_gt0": float((draws > 0).mean()),
                "p_lt0": float((draws < 0).mean()),
            }
        )
    return pd.DataFrame(recs)


def run(quick: bool, use_odds: bool, team_effects: bool | None = None):
    s = get_settings()
    md = prepare(use_odds=use_odds, require_officials=False)
    if team_effects is None:
        team_effects = not use_odds  # add strength when the line is unavailable
    logger.info(
        "Fitting Stage 2 point-diff (use_odds=%s team_effects=%s)...", use_odds, team_effects
    )
    idata = _fit(build_pointdiff_model(md, team_effects), quick)

    diag = summarize_diagnostics(
        idata, var_names=[c for c in _COEFS if c in idata.posterior] + ["gamma_0"]
    )
    print_diagnostics("Stage 2 point differential", diag)

    s.paths.ensure()
    idata.to_netcdf(str(s.paths.stage2_nc))
    coefs = summarize_coefficients(idata)
    coefs.to_parquet(s.paths.stage2_coefficients, index=False)
    (s.paths.processed / "diagnostics_stage2.json").write_text(
        json.dumps(
            {"pointdiff": diag, "use_odds": use_odds, "team_effects": team_effects, "quick": quick},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nStage 2 coefficients (points per unit; foul/fta/ftm diff are home-minus-away):")
    print(coefs.to_string(index=False))
    return idata, coefs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Stage 2 point-differential model.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-odds", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--team-effects", dest="team_effects", action="store_true")
    group.add_argument("--no-team-effects", dest="team_effects", action="store_false")
    parser.set_defaults(team_effects=None)
    ns = parser.parse_args(argv)
    run(quick=ns.quick, use_odds=not ns.no_odds, team_effects=ns.team_effects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
