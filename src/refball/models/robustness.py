"""Stage 10 robustness checks.

1. Baseline model comparison (PSIS-LOO) over a single comparable outcome: home/away
   fouls in long format, with three nested models:
     M0 — no referee effects
     M1 — referee *volume* effect (symmetric "total foul" effect)
     M2 — referee volume + directional *lean* effect
   If referee effects don't improve out-of-sample fit, az.compare will say so.

2. Permutation placebo: shuffle referee crews across games *within season* and refit M2.
   Under no real signal, the referee lean sigma should collapse toward the placebo
   distribution. ``--quick`` runs 3 placebos, ``--full`` runs 20+.

3. Leave-one-season-out: refit M2 excluding each playoff season; check whether the
   referee lean sigma (and top-ref leans) are stable or driven by one postseason.

4. Sensitivity: with-odds vs no-odds vs high-confidence-odds-only; compare referee
   rankings and the lean sigma.

All outputs are written under ``data/processed/robustness/``.
"""

from __future__ import annotations

import argparse

import numpy as np

from refball.config import get_settings
from refball.models.data_prep import ModelData, prepare
from refball.models.diagnostics import hdi_interval
from refball.utils.logging import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# Long-format home/away foul model with toggles (single comparable likelihood)
# ----------------------------------------------------------------------------
def build_pf_long_model(md: ModelData, *, use_vol: bool, use_lean: bool, R_override=None):
    """home_pf & away_pf stacked into one NegativeBinomial likelihood of length 2N.

    A single observed RV ('pf') makes PSIS-LOO directly comparable across M0/M1/M2.
    ``R_override`` lets the placebo pass in a permuted membership matrix.
    """
    import pymc as pm

    s = get_settings()
    R = md.R if R_override is None else R_override
    n = md.n_games

    obs = np.concatenate([md.home_pf, md.away_pf]).astype(int)
    log_poss = np.concatenate([md.log_possessions, md.log_possessions])
    is_home = np.concatenate([np.ones(n), np.zeros(n)])
    sign_lean = np.concatenate([np.ones(n), -np.ones(n)])
    commit_idx = np.concatenate([md.home_team_idx, md.away_team_idx]).astype(int)
    draw_idx = np.concatenate([md.away_team_idx, md.home_team_idx]).astype(int)
    season_long = np.concatenate([md.season_idx, md.season_idx]).astype(int)
    R_long = np.vstack([R, R])
    base = np.log(max(obs.mean(), 1.0)) - float(log_poss.mean())

    with pm.Model(coords=md.coords) as model:
        alpha = pm.Normal("alpha", base, 0.5)
        beta_home = pm.Normal("beta_home", 0.0, 0.2)
        eta = alpha + beta_home * is_home + log_poss

        if use_vol:
            sigma_vol = pm.HalfNormal("sigma_ref_vol", s.sigma_ref_total_prior)
            ref_vol = pm.Deterministic(
                "ref_vol_effect",
                pm.Normal("ref_vol_z", 0.0, 1.0, dims="ref") * sigma_vol,
                dims="ref",
            )
            eta = eta + pm.math.dot(R_long, ref_vol)
        if use_lean:
            sigma_lean = pm.HalfNormal("sigma_ref_lean", s.sigma_ref_lean_prior)
            ref_lean = pm.Deterministic(
                "ref_lean_effect",
                pm.Normal("ref_lean_z", 0.0, 1.0, dims="ref") * sigma_lean,
                dims="ref",
            )
            eta = eta + sign_lean * pm.math.dot(R_long, ref_lean)

        sigma_team = pm.HalfNormal("sigma_team", s.sigma_team_prior)
        commit = pm.Deterministic(
            "team_commit_effect",
            pm.Normal("commit_z", 0.0, 1.0, dims="team") * sigma_team,
            dims="team",
        )
        draw = pm.Deterministic(
            "team_draw_effect", pm.Normal("draw_z", 0.0, 1.0, dims="team") * sigma_team, dims="team"
        )
        sigma_season = pm.HalfNormal("sigma_season", s.sigma_season_prior)
        season_eff = pm.Deterministic(
            "season_effect",
            pm.Normal("season_z", 0.0, 1.0, dims="season") * sigma_season,
            dims="season",
        )
        eta = eta + commit[commit_idx] + draw[draw_idx] + season_eff[season_long]

        if md.use_odds:
            beta_spread = pm.Normal("beta_spread", 0.0, 0.1)
            z_spread_long = np.concatenate([md.z_spread_home, md.z_spread_home])
            eta = eta + beta_spread * z_spread_long

        alpha_nb = pm.Gamma("alpha_nb", 3.0, 0.1)
        pm.NegativeBinomial("pf", mu=pm.math.exp(eta), alpha=alpha_nb, observed=obs)
    return model


def _fit_ll(model, quick: bool):
    import pymc as pm

    with model:
        idata = pm.sample(**get_settings().sampler.resolve(quick), progressbar=False)
        pm.compute_log_likelihood(idata, progressbar=False)
    return idata


# ----------------------------------------------------------------------------
# 1. LOO comparison
# ----------------------------------------------------------------------------
def loo_comparison(md: ModelData, quick: bool):
    import arviz as az

    logger.info("LOO: fitting M0/M1/M2...")
    idatas = {
        "M0_no_ref": _fit_ll(build_pf_long_model(md, use_vol=False, use_lean=False), quick),
        "M1_ref_vol": _fit_ll(build_pf_long_model(md, use_vol=True, use_lean=False), quick),
        "M2_ref_vol_lean": _fit_ll(build_pf_long_model(md, use_vol=True, use_lean=True), quick),
    }
    cmp = az.compare(idatas)
    df = cmp.reset_index().rename(columns={"index": "model"})
    return df


# ----------------------------------------------------------------------------
# 2. Permutation placebo
# ----------------------------------------------------------------------------
def _permute_R_within_season(R, season_idx, rng):
    R_perm = R.copy()
    for s_val in np.unique(season_idx):
        rows = np.where(season_idx == s_val)[0]
        R_perm[rows] = R[rng.permutation(rows)]
    return R_perm


def _sigma_lean_mean(idata) -> float:
    return float(idata.posterior["sigma_ref_lean"].values.mean())


def placebo_test(md: ModelData, quick: bool, n_placebo: int):
    import pandas as pd

    logger.info("Placebo: fitting real M2 + %d permutations...", n_placebo)
    real = _fit_ll(build_pf_long_model(md, use_vol=True, use_lean=True), quick)
    real_sigma = _sigma_lean_mean(real)

    rng = np.random.default_rng(2024)
    rows = [{"run": "real", "sigma_ref_lean_mean": real_sigma}]
    for i in range(n_placebo):
        R_perm = _permute_R_within_season(md.R, md.season_idx, rng)
        idata = _fit_ll(
            build_pf_long_model(md, use_vol=True, use_lean=True, R_override=R_perm), quick
        )
        rows.append({"run": f"placebo_{i + 1}", "sigma_ref_lean_mean": _sigma_lean_mean(idata)})
        logger.info(
            "  placebo %d/%d sigma_lean=%.4f", i + 1, n_placebo, rows[-1]["sigma_ref_lean_mean"]
        )
    df = pd.DataFrame(rows)
    placebo_vals = df[df["run"] != "real"]["sigma_ref_lean_mean"]
    df.attrs["placebo_mean"] = float(placebo_vals.mean())
    df.attrs["real_exceeds_placebo_frac"] = float((real_sigma > placebo_vals).mean())
    return df


# ----------------------------------------------------------------------------
# 3. Leave-one-season-out
# ----------------------------------------------------------------------------
def leave_one_season_out(quick: bool, use_odds: bool = True):
    import pandas as pd

    table = prepare(use_odds=use_odds, require_officials=True).table
    seasons = sorted(table["season"].unique().tolist())
    rows = []
    for yr in seasons:
        sub = table[table["season"] != yr]
        md = prepare(use_odds=use_odds, require_officials=True, table=sub)
        idata = _fit_ll(build_pf_long_model(md, use_vol=True, use_lean=True), quick)
        rows.append(
            {
                "excluded_season": yr,
                "n_games": md.n_games,
                "sigma_ref_lean_mean": _sigma_lean_mean(idata),
            }
        )
        logger.info("  LOSO excl %s sigma_lean=%.4f", yr, rows[-1]["sigma_ref_lean_mean"])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Sensitivity (odds variants)
# ----------------------------------------------------------------------------
def _ref_lean_means(idata, ref_ids):
    da = idata.posterior["ref_lean_effect"]
    means = da.mean(dim=("chain", "draw")).values
    coord = [int(x) for x in da.coords["ref"].values]
    return dict(zip(coord, means, strict=True))


def sensitivity(quick: bool, odds_available: bool = True):
    import pandas as pd

    variants = {}
    mds = {}
    # B (no-odds) always runs; A/C require odds coverage.
    md_b = prepare(use_odds=False, require_officials=True)
    variants["B_no_odds"] = _fit_ll(build_pf_long_model(md_b, use_vol=True, use_lean=True), quick)
    mds["B_no_odds"] = md_b
    if odds_available:
        md_a = prepare(use_odds=True, require_officials=True)
        variants["A_with_odds"] = _fit_ll(
            build_pf_long_model(md_a, use_vol=True, use_lean=True), quick
        )
        mds["A_with_odds"] = md_a
        try:
            md_c = prepare(use_odds=True, require_officials=True, high_conf_odds_only=True)
            variants["C_high_conf_odds"] = _fit_ll(
                build_pf_long_model(md_c, use_vol=True, use_lean=True), quick
            )
            mds["C_high_conf_odds"] = md_c
        except Exception as exc:  # noqa: BLE001
            logger.warning("High-conf-odds variant skipped: %s", exc)
    else:
        logger.warning("No odds coverage; sensitivity runs the no-odds model only.")

    summary_rows = []
    lean_by_variant = {}
    for name, idata in variants.items():
        summary_rows.append(
            {
                "variant": name,
                "n_games": mds[name].n_games,
                "sigma_ref_lean_mean": _sigma_lean_mean(idata),
            }
        )
        lean_by_variant[name] = _ref_lean_means(idata, mds[name].ref_index.ref_ids)

    # Spearman correlation of A vs B referee lean rankings (shared refs).
    corr = float("nan")
    if "A_with_odds" in lean_by_variant and "B_no_odds" in lean_by_variant:
        a, b = lean_by_variant["A_with_odds"], lean_by_variant["B_no_odds"]
        shared = sorted(set(a) & set(b))
        if len(shared) >= 3:
            from scipy.stats import spearmanr

            corr = float(spearmanr([a[r] for r in shared], [b[r] for r in shared]).statistic)
    summ = pd.DataFrame(summary_rows)
    summ.attrs["spearman_A_vs_B_lean"] = corr
    return summ


# ----------------------------------------------------------------------------
# 5. Statistical power: analytic MDE + injection-recovery
# ----------------------------------------------------------------------------
def _ref_lean_draws(idata):
    da = idata.posterior["ref_lean_effect"].stack(s=("chain", "draw"))
    return da.transpose("ref", "s").values  # [n_refs, draws]


def power_check(quick: bool, shifts: tuple[int, ...] = (1, 2)):
    """Quantify whether the design could DETECT a real referee lean.

    A null is only informative if the method has power. We (1) report the analytic
    minimum-detectable-effect (median 94% HDI half-width of the fitted ref leans) and
    (2) run an injection-recovery: add a known directional lean to the most-sampled
    referee's games and check whether the model recovers it (94% HDI excludes 0).

    ``shift = a`` adds ``a`` to home_pf and subtracts ``a`` from away_pf on that ref's
    games -> a +2a home foul-margin/game lean. a=1 is already a clearly swing-relevant
    effect (~0.9 pts/game). If a=1 is NOT recovered, the null is "no DETECTABLE swing",
    not "no swing".
    """
    import arviz as az
    import pandas as pd

    from refball.models.fit_stage1 import build_lean_model

    s = get_settings()
    rows = []

    # (1) analytic MDE from the saved lean trace, if present
    if s.paths.stage1_lean_nc.exists():
        draws = _ref_lean_draws(az.from_netcdf(str(s.paths.stage1_lean_nc)))
        halfwidths = [
            (hi - lo) / 2
            for lo, hi in (hdi_interval(draws[i], s.hdi_prob) for i in range(draws.shape[0]))
        ]
        mde_log = float(np.median(halfwidths))
        rows.append(
            {
                "test": "analytic_MDE",
                "shift_fouls": np.nan,
                "lean_mean": np.nan,
                "hdi_low": -mde_log,
                "hdi_high": mde_log,
                "detected": np.nan,
                "note": f"median 94% HDI half-width = {mde_log:.4f} log; ~{2 * 20.7 * mde_log / 3:.2f} foul-margin",
            }
        )

    # (2) injection-recovery on the most-sampled referee
    md = prepare(use_odds=False, require_officials=True)
    idx = int(np.argmax([md.ref_index.games_count[r] for r in md.ref_index.ref_ids]))
    target_id = md.ref_index.ref_ids[idx]
    target_rows = np.where(md.R[:, idx] > 0)[0]
    base_home, base_away = md.home_pf.copy(), md.away_pf.copy()
    logger.info(
        "Injection-recovery on ref %s (%d games)", md.ref_index.names[target_id], len(target_rows)
    )

    for a in shifts:
        md.home_pf = base_home.copy()
        md.away_pf = base_away.copy()
        md.home_pf[target_rows] = base_home[target_rows] + a
        md.away_pf[target_rows] = np.maximum(base_away[target_rows] - a, 0)
        idata = _fit_ll(build_lean_model(md), quick)
        d = _ref_lean_draws(idata)[idx]
        lo, hi = hdi_interval(d, s.hdi_prob)
        detected = bool(lo > 0 or hi < 0)
        rows.append(
            {
                "test": "injection",
                "shift_fouls": a,
                "lean_mean": float(d.mean()),
                "hdi_low": lo,
                "hdi_high": hi,
                "detected": detected,
                "note": f"+{2 * a} foul-margin/game injected; P(>0)={float((d > 0).mean()):.2f}",
            }
        )
        logger.info(
            "  shift=+%d -> lean_mean=%.3f HDI[%.3f,%.3f] detected=%s",
            a,
            d.mean(),
            lo,
            hi,
            detected,
        )

    md.home_pf, md.away_pf = base_home, base_away  # restore
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _odds_available() -> bool:
    """True if the modeling table has any games with odds."""
    import pandas as pd

    table = pd.read_parquet(get_settings().paths.modeling_table)
    return bool(table["has_odds"].any())


def run(
    quick: bool,
    n_placebo: int,
    do_loo: bool,
    do_placebo: bool,
    do_loso: bool,
    do_sens: bool,
    do_power: bool = True,
    use_odds: bool | None = None,
):
    s = get_settings()
    s.paths.robustness_dir.mkdir(parents=True, exist_ok=True)
    odds_ok = _odds_available()
    if use_odds is None:
        use_odds = odds_ok
    if use_odds and not odds_ok:
        logger.warning("Requested with-odds robustness but no games have odds; using no-odds.")
        use_odds = False
    logger.info("Robustness running with use_odds=%s (odds available=%s)", use_odds, odds_ok)
    md = prepare(use_odds=use_odds, require_officials=True)

    if do_loo:
        df = loo_comparison(md, quick)
        df.to_csv(s.paths.robustness_dir / "loo_compare.csv", index=False)
        print("\n=== LOO comparison (M0 no-ref / M1 vol / M2 vol+lean) ===")
        print(df.to_string(index=False))

    if do_placebo:
        df = placebo_test(md, quick, n_placebo)
        df.to_csv(s.paths.robustness_dir / "placebo.csv", index=False)
        print("\n=== Permutation placebo (sigma_ref_lean) ===")
        print(df.to_string(index=False))
        print(
            f"real value: {df[df.run == 'real']['sigma_ref_lean_mean'].iloc[0]:.4f} | "
            f"placebo mean: {df.attrs['placebo_mean']:.4f} | "
            f"real exceeds {100 * df.attrs['real_exceeds_placebo_frac']:.0f}% of placebos"
        )

    if do_loso:
        df = leave_one_season_out(quick, use_odds=use_odds)
        df.to_csv(s.paths.robustness_dir / "loso.csv", index=False)
        print("\n=== Leave-one-season-out (sigma_ref_lean) ===")
        print(df.to_string(index=False))

    if do_sens:
        df = sensitivity(quick, odds_available=use_odds)
        df.to_csv(s.paths.robustness_dir / "sensitivity.csv", index=False)
        print("\n=== Sensitivity (odds variants) ===")
        print(df.to_string(index=False))
        print(
            f"Spearman corr of A-vs-B referee lean rankings: {df.attrs['spearman_A_vs_B_lean']:.3f}"
        )

    if do_power:
        df = power_check(quick)
        df.to_csv(s.paths.robustness_dir / "power.csv", index=False)
        print("\n=== Statistical power: minimum-detectable-effect + injection-recovery ===")
        print(df.to_string(index=False))
        inj = df[df["test"] == "injection"]
        detected = inj[inj["detected"] == True]  # noqa: E712
        smallest = int(detected["shift_fouls"].min()) if len(detected) else None
        if smallest is None:
            print(
                "No injected lean was detected at the tested sizes -> design is severely underpowered."
            )
        else:
            print(
                f"Smallest detected injection: +{smallest} fouls/game (= +{2 * smallest} foul-margin). "
                f"Leans smaller than this are UNDETECTABLE -> the null means 'no DETECTABLE swing', "
                f"not 'no swing'."
            )
    print(f"\n[robustness] outputs in {s.paths.robustness_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 10 robustness checks.")
    parser.add_argument("--quick", action="store_true", help="fast sampling + 3 placebos")
    parser.add_argument("--full", action="store_true", help="full sampling + 20 placebos")
    parser.add_argument("--placebos", type=int, default=None, help="override placebo count")
    parser.add_argument(
        "--only", choices=["loo", "placebo", "loso", "sensitivity", "power"], default=None
    )
    parser.add_argument(
        "--no-odds", action="store_true", help="force the no-odds model (auto-detected otherwise)"
    )
    ns = parser.parse_args(argv)

    quick = ns.quick or not ns.full
    n_placebo = ns.placebos if ns.placebos is not None else (3 if quick else 20)
    sel = ns.only
    run(
        quick=quick,
        n_placebo=n_placebo,
        do_loo=sel in (None, "loo"),
        do_placebo=sel in (None, "placebo"),
        do_loso=sel in (None, "loso"),
        do_sens=sel in (None, "sensitivity"),
        do_power=sel in (None, "power"),
        use_odds=False if ns.no_odds else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
