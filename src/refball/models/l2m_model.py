"""Bayesian model of clutch officiating *errors* (L2M) by crew.

This is the project's most *direct* test: instead of noisy whole-game foul totals, we model
the NBA's own graded errors (IC + INC) in the final ~2 minutes of close games, signed by
which team they hurt. Two questions:

1. **Overall clutch bias** ``b``: do incorrect calls/non-calls *net-favor the home team*
   leaguewide? (The literature expects a small home bias.)
2. **Crew lean** ``crew_lean[r]``: does any crew's clutch errors persistently favor home,
   beyond the leaguewide level — with partial pooling and a permutation placebo?

Model (Poisson pair, multi-membership crew, non-centered):
    err_against_away[g] ~ Poisson(exp(a + b + crew_vol + crew_lean))   # favors HOME
    err_against_home[g] ~ Poisson(exp(a - b + crew_vol - crew_lean))   # favors AWAY

Honest limits: only ~183 close playoff games are covered (selection on closeness); errors
are sparse; grades attribute to the **crew** (not one official) and are the **NBA grading
its own calls**. So this answers "do crews' clutch errors net-favor a side," not "official X
decides games." Estimates are reported with intervals and a placebo, never as proof.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from refball.config import get_settings
from refball.features.multimembership import build_membership_matrix, build_ref_index
from refball.models.diagnostics import hdi_interval, print_diagnostics, summarize_diagnostics
from refball.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_l2m():
    """Merge the modeling table with the L2M per-game errors; return (df, ref_index, R)."""
    import pandas as pd

    s = get_settings()
    l2m_path = s.paths.processed / "l2m_game_table.parquet"
    if not s.paths.modeling_table.exists() or not l2m_path.exists():
        raise FileNotFoundError("Need modeling_table.parquet and l2m_game_table.parquet (run data.l2m).")
    mt = pd.read_parquet(s.paths.modeling_table)
    l2m = pd.read_parquet(l2m_path)[
        ["game_id", "l2m_graded", "l2m_errors", "err_against_home", "err_against_away", "net_home_error_adv"]
    ]
    df = mt.merge(l2m, on="game_id", how="inner")
    df = df[df["has_officials"]].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No games with both officials and L2M coverage.")
    index = build_ref_index(df)
    R, _ = build_membership_matrix(df, index)
    logger.info("L2M model data: %d games, %d crews/refs", len(df), len(index.ref_ids))
    return df, index, R


def build_model(df, index, R):
    import pymc as pm

    against_away = df["err_against_away"].to_numpy(dtype=int)
    against_home = df["err_against_home"].to_numpy(dtype=int)
    base = float(np.log(max(np.mean(np.r_[against_away, against_home]), 0.1)))
    coords = {"ref": [str(r) for r in index.ref_ids], "game": np.arange(len(df))}

    with pm.Model(coords=coords) as model:
        a = pm.Normal("a", base, 1.0)
        b = pm.Normal("home_error_bias", 0.0, 0.5)  # >0 => clutch errors net-favor home (leaguewide)

        sigma_vol = pm.HalfNormal("sigma_crew_vol", 0.3)
        ref_vol = pm.Deterministic("ref_vol_effect", pm.Normal("vol_z", 0, 1, dims="ref") * sigma_vol, dims="ref")
        crew_vol = pm.math.dot(R, ref_vol)

        sigma_lean = pm.HalfNormal("sigma_crew_lean", 0.3)
        ref_lean = pm.Deterministic("ref_error_lean", pm.Normal("lean_z", 0, 1, dims="ref") * sigma_lean, dims="ref")
        crew_lean = pm.math.dot(R, ref_lean)

        mu_away = pm.math.exp(a + b + crew_vol + crew_lean)  # errors favoring home
        mu_home = pm.math.exp(a - b + crew_vol - crew_lean)  # errors favoring away
        pm.Poisson("err_against_away", mu=mu_away, observed=against_away, dims="game")
        pm.Poisson("err_against_home", mu=mu_home, observed=against_home, dims="game")
    return model


def _fit(model, quick: bool):
    import pymc as pm

    with model:
        idata = pm.sample(**get_settings().sampler.resolve(quick), progressbar=False)
    return idata


def _placebo(df, index, R, quick: bool, n: int = 3):
    """Shuffle crews within season; refit; collect placebo sigma_crew_lean."""
    rng = np.random.default_rng(99)
    season = df["season"].to_numpy()
    vals = []
    for _ in range(n):
        R_perm = R.copy()
        for sv in np.unique(season):
            rows = np.where(season == sv)[0]
            R_perm[rows] = R[rng.permutation(rows)]
        idata = _fit(build_model(df, index, R_perm), quick)
        vals.append(float(idata.posterior["sigma_crew_lean"].values.mean()))
    return vals


def run(quick: bool, n_placebo: int = 3):
    import pandas as pd

    s = get_settings()
    df, index, R = prepare_l2m()
    prob = s.hdi_prob

    idata = _fit(build_model(df, index, R), quick)
    diag = summarize_diagnostics(idata, var_names=["a", "home_error_bias", "sigma_crew_lean", "ref_error_lean"])
    print_diagnostics("L2M clutch-error crew model", diag)

    # Overall leaguewide clutch home-error bias
    b = idata.posterior["home_error_bias"].values.ravel()
    b_lo, b_hi = hdi_interval(b, prob)
    rate_ratio = float(np.exp(2 * np.mean(b)))  # favor-home / favor-away error rate ratio

    # Per-crew lean
    lean = idata.posterior["ref_error_lean"].stack(s=("chain", "draw")).transpose("ref", "s").values
    recs = []
    for i, rid in enumerate(index.ref_ids):
        d = lean[i]
        lo, hi = hdi_interval(d, prob)
        recs.append({
            "ref_id": rid, "referee": index.names[rid], "games": index.games_count[rid],
            "error_lean_mean": float(d.mean()), "hdi_low": lo, "hdi_high": hi,
            "p_favor_home": float((d > 0).mean()), "excludes_zero": bool(lo > 0 or hi < 0),
        })
    effects = pd.DataFrame(recs).sort_values("error_lean_mean", ascending=False).reset_index(drop=True)

    placebo = _placebo(df, index, R, quick, n_placebo)
    real_sigma = float(idata.posterior["sigma_crew_lean"].values.mean())

    s.paths.ensure()
    idata.to_netcdf(str(s.paths.models / "l2m_crew.nc"))
    effects.to_parquet(s.paths.processed / "l2m_crew_effects.parquet", index=False)
    summary = {
        "n_games": int(len(df)), "n_crews": int(len(index.ref_ids)),
        "total_errors": int(df["l2m_errors"].sum()),
        "err_against_home": int(df["err_against_home"].sum()),
        "err_against_away": int(df["err_against_away"].sum()),
        "home_error_bias_mean": float(np.mean(b)), "home_error_bias_hdi": [b_lo, b_hi],
        "p_bias_favors_home": float((b > 0).mean()), "favor_home_to_away_rate_ratio": rate_ratio,
        "crews_lean_excludes_zero": int(effects["excludes_zero"].sum()),
        "real_sigma_crew_lean": real_sigma, "placebo_sigma_crew_lean_mean": float(np.mean(placebo)),
        "diagnostics": diag,
    }
    (s.paths.processed / "l2m_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== L2M CLUTCH-ERROR FINDINGS (final ~2 min of close playoff games) ===")
    print(f"games covered:            {summary['n_games']} (of 585 playoff games)")
    print(f"clutch errors:            {summary['total_errors']} "
          f"(against home {summary['err_against_home']} / against away {summary['err_against_away']})")
    print(f"overall home-error bias:  {summary['home_error_bias_mean']:+.3f} log "
          f"(94% HDI [{b_lo:+.3f}, {b_hi:+.3f}]); P(favors home)={summary['p_bias_favors_home']:.2f}")
    print(f"  -> favor-home:favor-away error-rate ratio = {rate_ratio:.2f} (1.0 = no bias)")
    print(f"crews whose lean HDI excludes 0: {summary['crews_lean_excludes_zero']} / {summary['n_crews']}")
    print(f"placebo check: real sigma_crew_lean={real_sigma:.4f} vs shuffled mean={np.mean(placebo):.4f}")
    print("NOTE: crew-level (not individual ref); NBA grades its own calls; only close-late games "
          "(selection). Screening evidence with uncertainty, not proof.")
    return idata, effects, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit the L2M clutch-error crew model.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--placebos", type=int, default=3)
    ns = p.parse_args(argv)
    run(quick=ns.quick, n_placebo=ns.placebos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
