"""Stage 1 Bayesian foul models (PyMC v5).

Stage 1A  — total foul volume (NegativeBinomial), one effect per referee.
Stage 1B  — directional home/away fouls (two NegativeBinomial likelihoods sharing a crew
            *volume* effect and a crew *lean* effect).

Both use:
* multi-membership referee effects (``crew = R @ ref_effect``, R = 1/k weights),
* non-centered parameterization (``effect = z * sigma``),
* hard half-normal priors on the referee sigmas (playoff samples are small),
* hierarchical team + season effects and optional standardized market controls.

Interpretation guardrails (see README / Methodology):
* ``ref_lean > 0`` => games with this official are associated with MORE home-team fouls
  relative to away (because ``foul_diff_home = home_pf - away_pf``). That is a foul-margin
  association, *not* "bias", "favoring", or intent.
* Referee assignments are not random; these are screening estimates with uncertainty.
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


# ----------------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------------
def build_total_model(md: ModelData):
    """Stage 1A: total fouls ~ NegativeBinomial with multi-membership ref effects."""
    import pymc as pm

    s = get_settings()
    mean_total = float(np.mean(md.total_fouls))
    base_intercept = np.log(mean_total) - float(np.mean(md.log_possessions))

    with pm.Model(coords=md.coords) as model:
        alpha = pm.Normal("alpha", mu=base_intercept, sigma=0.5)

        sigma_ref = pm.HalfNormal("sigma_ref_total", s.sigma_ref_total_prior)
        ref_z = pm.Normal("ref_total_z", 0.0, 1.0, dims="ref")
        ref_total = pm.Deterministic("ref_total_effect", ref_z * sigma_ref, dims="ref")
        crew_total = pm.math.dot(md.R, ref_total)

        sigma_team = pm.HalfNormal("sigma_team", s.sigma_team_prior)
        home_team_eff = pm.Deterministic(
            "home_team_effect",
            pm.Normal("home_team_z", 0.0, 1.0, dims="team") * sigma_team,
            dims="team",
        )
        away_team_eff = pm.Deterministic(
            "away_team_effect",
            pm.Normal("away_team_z", 0.0, 1.0, dims="team") * sigma_team,
            dims="team",
        )

        sigma_season = pm.HalfNormal("sigma_season", s.sigma_season_prior)
        season_eff = pm.Deterministic(
            "season_effect",
            pm.Normal("season_z", 0.0, 1.0, dims="season") * sigma_season,
            dims="season",
        )

        eta = (
            alpha
            + md.log_possessions
            + crew_total
            + home_team_eff[md.home_team_idx]
            + away_team_eff[md.away_team_idx]
            + season_eff[md.season_idx]
        )
        if md.use_odds:
            beta_spread = pm.Normal("beta_spread", 0.0, 0.1)
            beta_total = pm.Normal("beta_total", 0.0, 0.1)
            eta = eta + beta_spread * md.z_spread_home + beta_total * md.z_total_market

        mu = pm.math.exp(eta)
        alpha_nb = pm.Gamma("alpha_nb", alpha=3.0, beta=0.1)
        pm.NegativeBinomial(
            "total_fouls", mu=mu, alpha=alpha_nb, observed=md.total_fouls, dims="game"
        )
    return model


def build_lean_model(md: ModelData):
    """Stage 1B: home_pf & away_pf ~ NegativeBinomial with shared crew volume + lean."""
    import pymc as pm

    s = get_settings()
    base_home = np.log(max(float(np.mean(md.home_pf)), 1.0)) - float(np.mean(md.log_possessions))
    base_away = np.log(max(float(np.mean(md.away_pf)), 1.0)) - float(np.mean(md.log_possessions))

    with pm.Model(coords=md.coords) as model:
        alpha_home = pm.Normal("alpha_home", mu=base_home, sigma=0.5)
        alpha_away = pm.Normal("alpha_away", mu=base_away, sigma=0.5)

        sigma_ref_vol = pm.HalfNormal("sigma_ref_vol", s.sigma_ref_total_prior)
        ref_vol = pm.Deterministic(
            "ref_vol_effect",
            pm.Normal("ref_vol_z", 0.0, 1.0, dims="ref") * sigma_ref_vol,
            dims="ref",
        )
        crew_vol = pm.math.dot(md.R, ref_vol)

        sigma_ref_lean = pm.HalfNormal("sigma_ref_lean", s.sigma_ref_lean_prior)
        ref_lean = pm.Deterministic(
            "ref_lean_effect",
            pm.Normal("ref_lean_z", 0.0, 1.0, dims="ref") * sigma_ref_lean,
            dims="ref",
        )
        crew_lean = pm.math.dot(md.R, ref_lean)

        sigma_team = pm.HalfNormal("sigma_team", s.sigma_team_prior)
        team_commit = pm.Deterministic(
            "team_commit_effect",
            pm.Normal("commit_z", 0.0, 1.0, dims="team") * sigma_team,
            dims="team",
        )
        team_draw = pm.Deterministic(
            "team_draw_effect", pm.Normal("draw_z", 0.0, 1.0, dims="team") * sigma_team, dims="team"
        )

        sigma_season = pm.HalfNormal("sigma_season", s.sigma_season_prior)
        season_eff = pm.Deterministic(
            "season_effect",
            pm.Normal("season_z", 0.0, 1.0, dims="season") * sigma_season,
            dims="season",
        )

        log_mu_home = (
            alpha_home
            + md.log_possessions
            + crew_vol
            + crew_lean
            + team_commit[md.home_team_idx]
            + team_draw[md.away_team_idx]
            + season_eff[md.season_idx]
        )
        log_mu_away = (
            alpha_away
            + md.log_possessions
            + crew_vol
            - crew_lean
            + team_commit[md.away_team_idx]
            + team_draw[md.home_team_idx]
            + season_eff[md.season_idx]
        )
        if md.use_odds:
            beta_spread_home = pm.Normal("beta_spread_home", 0.0, 0.1)
            beta_total_home = pm.Normal("beta_total_home", 0.0, 0.1)
            beta_spread_away = pm.Normal("beta_spread_away", 0.0, 0.1)
            beta_total_away = pm.Normal("beta_total_away", 0.0, 0.1)
            log_mu_home = (
                log_mu_home
                + beta_spread_home * md.z_spread_home
                + beta_total_home * md.z_total_market
            )
            log_mu_away = (
                log_mu_away
                + beta_spread_away * md.z_spread_home
                + beta_total_away * md.z_total_market
            )

        alpha_nb_home = pm.Gamma("alpha_nb_home", alpha=3.0, beta=0.1)
        alpha_nb_away = pm.Gamma("alpha_nb_away", alpha=3.0, beta=0.1)
        pm.NegativeBinomial(
            "home_pf",
            mu=pm.math.exp(log_mu_home),
            alpha=alpha_nb_home,
            observed=md.home_pf,
            dims="game",
        )
        pm.NegativeBinomial(
            "away_pf",
            mu=pm.math.exp(log_mu_away),
            alpha=alpha_nb_away,
            observed=md.away_pf,
            dims="game",
        )
    return model


# ----------------------------------------------------------------------------
# Fit + summarize
# ----------------------------------------------------------------------------
def _fit(model, quick: bool, posterior_predictive: bool = True):
    import pymc as pm

    sampler_kwargs = get_settings().sampler.resolve(quick)
    with model:
        idata = pm.sample(**sampler_kwargs, progressbar=False)
        pm.compute_log_likelihood(idata, progressbar=False)
        if posterior_predictive:
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)
    return idata


def summarize_referees(md: ModelData, idata_total, idata_lean):
    """Combine total + lean posteriors into a per-referee effects table."""
    import pandas as pd

    s = get_settings()
    prob = s.hdi_prob

    total = (
        idata_total.posterior["ref_total_effect"].stack(s=("chain", "draw")).values
    )  # [ref, draws]
    lean = idata_lean.posterior["ref_lean_effect"].stack(s=("chain", "draw")).values  # [ref, draws]

    baseline_per100 = float(
        np.mean(md.total_fouls / md.table["possessions_avg"].to_numpy()) * 100.0
    )
    f_bar = float(np.mean((md.home_pf + md.away_pf) / 2.0))
    k = float(get_settings().officials_per_game)

    recs = []
    for i, rid in enumerate(md.ref_index.ref_ids):
        t = total[i]
        ll = lean[i]
        # marginal-add convention: a single ref contributes effect/k to the crew average.
        extra_fouls_per100 = baseline_per100 * (np.exp(t / k) - 1.0)
        foul_margin = f_bar * (np.exp(ll / k) - np.exp(-ll / k))  # home-minus-away fouls
        t_lo, t_hi = hdi_interval(t, prob)
        l_lo, l_hi = hdi_interval(ll, prob)
        fm_lo, fm_hi = hdi_interval(foul_margin, prob)
        recs.append(
            {
                "ref_id": rid,
                "ref_idx": i,
                "referee": md.ref_index.names[rid],
                "games": md.ref_index.games_count[rid],
                "total_effect_mean": float(t.mean()),
                "total_hdi_low": t_lo,
                "total_hdi_high": t_hi,
                "p_total_gt0": float((t > 0).mean()),
                "extra_fouls_per100_mean": float(extra_fouls_per100.mean()),
                "lean_mean": float(ll.mean()),
                "lean_hdi_low": l_lo,
                "lean_hdi_high": l_hi,
                "p_lean_gt0": float((ll > 0).mean()),
                "foul_margin_mean": float(foul_margin.mean()),
                "foul_margin_hdi_low": fm_lo,
                "foul_margin_hdi_high": fm_hi,
            }
        )
    df = pd.DataFrame(recs).sort_values("lean_mean").reset_index(drop=True)
    df.attrs["baseline_fouls_per100"] = baseline_per100
    df.attrs["f_bar"] = f_bar
    return df


def _save_ppc_plots(idata, var_names: list[str], prefix: str, n_overlay: int = 60) -> None:
    """Manual posterior-predictive overlay (az.plot_ppc was removed in ArviZ 1.x).

    Plots the observed distribution against ``n_overlay`` simulated datasets.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = get_settings()
        pp = getattr(idata, "posterior_predictive", None)
        obs = getattr(idata, "observed_data", None)
        if pp is None or obs is None:
            logger.warning("No posterior_predictive group; skipping PPC plots.")
            return
        for v in var_names:
            observed = np.asarray(obs[v].values, dtype=float).ravel()
            mat = pp[v].stack(s=("chain", "draw")).values  # (n_obs, S)
            ns = mat.shape[1]
            idxs = np.linspace(0, ns - 1, min(n_overlay, ns)).astype(int)
            lo, hi = np.percentile(observed, [0.5, 99.5])
            bins = np.linspace(min(lo, mat.min()), max(hi, mat.max()), 30)

            fig, ax = plt.subplots(figsize=(7, 4))
            for j in idxs:
                ax.hist(
                    mat[:, j], bins=bins, density=True, histtype="step", color="#3b6fb0", alpha=0.15
                )
            ax.hist(
                observed,
                bins=bins,
                density=True,
                histtype="step",
                color="k",
                lw=2,
                label="observed",
            )
            ax.plot([], [], color="#3b6fb0", alpha=0.6, label="posterior predictive")
            ax.set(title=f"PPC: {v}", xlabel=v, ylabel="density")
            ax.legend()
            fig.tight_layout()
            fig.savefig(s.paths.figures / f"ppc_{prefix}_{v}.png", dpi=110)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PPC plotting skipped: %s", exc)


def run(quick: bool, use_odds: bool, plots: bool = True):
    s = get_settings()
    md = prepare(use_odds=use_odds, require_officials=True)

    logger.info("Fitting Stage 1A (total fouls)...")
    idata_total = _fit(build_total_model(md), quick)
    diag_total = summarize_diagnostics(
        idata_total, var_names=["alpha", "sigma_ref_total", "ref_total_effect"]
    )
    print_diagnostics("Stage 1A total fouls", diag_total)

    logger.info("Fitting Stage 1B (directional lean)...")
    idata_lean = _fit(build_lean_model(md), quick)
    diag_lean = summarize_diagnostics(
        idata_lean, var_names=["alpha_home", "alpha_away", "sigma_ref_lean", "ref_lean_effect"]
    )
    print_diagnostics("Stage 1B directional lean", diag_lean)

    s.paths.ensure()
    idata_total.to_netcdf(str(s.paths.stage1_total_nc))
    idata_lean.to_netcdf(str(s.paths.stage1_lean_nc))
    (s.paths.processed / "diagnostics_stage1.json").write_text(
        json.dumps(
            {"total": diag_total, "lean": diag_lean, "use_odds": use_odds, "quick": quick}, indent=2
        ),
        encoding="utf-8",
    )

    ref_table = summarize_referees(md, idata_total, idata_lean)
    ref_table.to_parquet(s.paths.referee_stage1_effects, index=False)
    logger.info(
        "Saved referee Stage 1 effects: %s (%d refs)",
        s.paths.referee_stage1_effects,
        len(ref_table),
    )

    if plots:
        _save_ppc_plots(idata_total, ["total_fouls"], "total")
        _save_ppc_plots(idata_lean, ["home_pf", "away_pf"], "lean")

    print("\nTop positive home-foul lean (more home fouls; screening only, wide intervals):")
    print(
        ref_table.sort_values("lean_mean", ascending=False)[
            ["referee", "games", "lean_mean", "lean_hdi_low", "lean_hdi_high", "p_lean_gt0"]
        ]
        .head(5)
        .to_string(index=False)
    )
    return idata_total, idata_lean, ref_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Stage 1 foul models.")
    parser.add_argument("--quick", action="store_true", help="fast, rough sampling for demo/CI")
    parser.add_argument("--no-odds", action="store_true", help="sensitivity: drop market controls")
    parser.add_argument("--no-plots", action="store_true")
    ns = parser.parse_args(argv)
    run(quick=ns.quick, use_odds=not ns.no_odds, plots=not ns.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
