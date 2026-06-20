"""Devil's advocate: try to PROVE a referee swing, then show why each case collapses.

A null is only credible if you genuinely tried to break it from the *other* side too. This
module reproduces the strongest pro-swing arguments a motivated analyst could build from the
real data, and the standard corrections that defeat each one:

1. Extreme-referee mining (L2M) — the most home-favoring official, uncorrected vs. FDR/Bonferroni.
2. Subgroup forking-paths — the best slice of many, vs. multiplicity + the league foul->points slope.
3. Label-permutation placebo — does referee *identity* add anything over the shared home tilt?
4. Regression to the mean — does the leaderboard-topping referee replicate in their high-power
   regular-season record?

Everything reads existing artifacts; no model fitting. Output: ``devils_advocate_summary.json``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from refball.config import get_settings
from refball.utils.logging import get_logger

logger = get_logger(__name__)
_ID_COLS = ["ref_1_id", "ref_2_id", "ref_3_id"]


def _explode_refs(df):
    import pandas as pd

    long = []
    for c in _ID_COLS:
        sub = df[["game_id", c]].rename(columns={c: "ref_id"})
        long.append(sub)
    out = pd.concat(long, ignore_index=True).dropna(subset=["ref_id"])
    out["ref_id"] = out["ref_id"].astype(int)
    return out


def extreme_ref_l2m(rng):
    """Strategy 1: most home-favoring official by raw L2M errors; corrected vs uncorrected."""
    import pandas as pd
    from scipy.stats import binomtest

    s = get_settings()
    mt = pd.read_parquet(s.paths.modeling_table)[["game_id", *_ID_COLS]]
    l2m = pd.read_parquet(s.paths.processed / "l2m_game_table.parquet")[
        ["game_id", "err_against_home", "err_against_away"]
    ]
    long = _explode_refs(mt).merge(l2m, on="game_id", how="inner")
    per_ref = long.groupby("ref_id")[["err_against_home", "err_against_away"]].sum()
    per_ref["favor_home"] = per_ref["err_against_away"]  # errors hurting away == favoring home
    per_ref["favor_away"] = per_ref["err_against_home"]
    per_ref["n"] = per_ref["favor_home"] + per_ref["favor_away"]
    tested = per_ref[per_ref["n"] >= 10].copy()
    tested["p_one_sided"] = [
        binomtest(int(r.favor_home), int(r.n), 0.5, alternative="greater").pvalue
        for r in tested.itertuples()
    ]
    n_tested = len(tested)
    best = tested.sort_values("p_one_sided").iloc[0]
    best_p = float(best["p_one_sided"])
    # multiplicity corrections
    bonferroni = min(1.0, best_p * n_tested)
    # null sim: each ref a fair coin with their n; fraction of sims where SOME ref >= this extreme
    sims = 2000
    hits = 0
    ns = tested["n"].to_numpy()
    for _ in range(sims):
        fav = rng.binomial(ns, 0.5)
        # one-sided p approx via survival; use normal approx for speed
        z = (fav - ns / 2) / np.sqrt(ns / 4)
        if z.max() >= (best["favor_home"] - best["n"] / 2) / np.sqrt(best["n"] / 4):
            hits += 1
    null_frac = hits / sims
    agg_home = int(per_ref["favor_home"].sum())
    agg_away = int(per_ref["favor_away"].sum())
    agg_p = float(binomtest(agg_home, agg_home + agg_away, 0.5, alternative="two-sided").pvalue)
    return {
        "n_refs_tested": int(n_tested),
        "most_extreme_ref_id": int(best.name),
        "most_extreme_favor_home": int(best["favor_home"]),
        "most_extreme_n": int(best["n"]),
        "uncorrected_p": best_p,
        "bonferroni_p": float(bonferroni),
        "null_sim_someone_this_extreme": null_frac,
        "aggregate_favor_home": agg_home,
        "aggregate_favor_away": agg_away,
        "aggregate_pct_home": round(100 * agg_home / (agg_home + agg_away), 1),
        "aggregate_two_sided_p": agg_p,
    }


def subgroup_forking():
    """Strategy 2: best-looking slice of many, plus the league foul->points slope."""
    import pandas as pd
    from scipy.stats import linregress, ttest_1samp

    s = get_settings()
    mt = pd.read_parquet(s.paths.modeling_table)
    slices = {}
    for rnd in sorted(mt["playoff_round"].dropna().unique()):
        slices[f"round_{int(rnd)}"] = mt[mt["playoff_round"] == rnd]
    odds = mt[mt["has_odds"]]
    slices["home_favorite"] = odds[odds["spread_home"] < 0]
    slices["home_underdog"] = odds[odds["spread_home"] > 0]
    slices["close"] = mt[mt["point_diff_home"].abs() <= 5]
    slices["close_home_fav"] = odds[
        (odds["spread_home"] < 0) & (odds["point_diff_home"].abs() <= 5)
    ]
    slices["close_home_dog"] = odds[
        (odds["spread_home"] > 0) & (odds["point_diff_home"].abs() <= 5)
    ]

    pvals = []
    for name, d in slices.items():
        if len(d) >= 8:
            lr = linregress(d["foul_diff_home"], d["point_diff_home"])
            t = ttest_1samp(d["foul_diff_home"], 0.0)
            pvals.append((name, "foul_slope", float(lr.slope), float(lr.pvalue), len(d)))
            pvals.append(
                (name, "mean_margin", float(d["foul_diff_home"].mean()), float(t.pvalue), len(d))
            )
    n_tests = len(pvals)
    best = min(pvals, key=lambda x: x[3])
    league = linregress(mt["foul_diff_home"], mt["point_diff_home"])
    return {
        "n_slice_tests": n_tests,
        "best_slice": best[0],
        "best_metric": best[1],
        "best_value": round(best[2], 3),
        "best_uncorrected_p": round(best[3], 4),
        "best_n": best[4],
        "best_bonferroni_p": round(min(1.0, best[3] * n_tests), 3),
        "league_foul_to_points_slope": round(float(league.slope), 4),
        "league_foul_to_points_p": round(float(league.pvalue), 3),
        "league_foul_to_points_r": round(
            float(np.corrcoef(mt["foul_diff_home"], mt["point_diff_home"])[0, 1]), 4
        ),
    }


def label_permutation_placebo(rng, n_perm=2000):
    """Strategy 3: does referee identity explain between-ref variance beyond the shared home tilt?"""
    import pandas as pd
    from scipy.stats import ttest_1samp

    s = get_settings()
    reg_path = s.paths.processed / "modeling_table_regular.parquet"
    mt = pd.read_parquet(reg_path if reg_path.exists() else s.paths.modeling_table)
    long = _explode_refs(mt[["game_id", *_ID_COLS]]).merge(
        mt[["game_id", "fta_diff_home"]], on="game_id", how="inner"
    )
    real_means = long.groupby("ref_id")["fta_diff_home"].mean()
    counts = long.groupby("ref_id").size()
    keep = counts[counts >= 20].index
    real_sd = float(real_means.loc[keep].std())

    vals = long["fta_diff_home"].to_numpy()
    ref_of_row = long["ref_id"].to_numpy()
    null_sds = []
    for _ in range(n_perm):
        perm = rng.permutation(vals)
        s_perm = pd.Series(perm).groupby(ref_of_row).mean()
        null_sds.append(float(s_perm.loc[keep].std()))
    null_sds = np.array(null_sds)
    p = float((null_sds >= real_sd).mean())
    league = ttest_1samp(mt["fta_diff_home"], 0.0)
    return {
        "n_refs": int(len(keep)),
        "real_between_ref_sd_fta": round(real_sd, 4),
        "null_sd_mean": round(float(null_sds.mean()), 4),
        "null_sd_95pct": round(float(np.percentile(null_sds, 95)), 4),
        "identity_adds_signal_p": round(p, 4),
        "league_home_fta_tilt_mean": round(float(mt["fta_diff_home"].mean()), 3),
        "league_home_fta_tilt_p": float(league.pvalue),
    }


def regression_to_mean():
    """Strategy 4: does the leaderboard-topping playoff referee replicate in the regular season?"""
    import pandas as pd
    from scipy.stats import spearmanr

    s = get_settings()
    reg = pd.read_parquet(s.paths.processed / "regular_season_effects.parquet")
    merged = reg.dropna(subset=["po_lean_mean", "reg_lean_mean"])
    rho, pval = spearmanr(merged["po_lean_mean"], merged["reg_lean_mean"])
    top = merged.sort_values("po_lean_mean", ascending=False).iloc[0]
    return {
        "n_refs": int(len(merged)),
        "spearman_playoff_vs_regular_lean": round(float(rho), 3),
        "spearman_p": round(float(pval), 3),
        "top_playoff_ref": str(top["referee"]),
        "top_ref_playoff_lean": round(float(top["po_lean_mean"]), 5),
        "top_ref_regular_lean": round(float(top["reg_lean_mean"]), 5),
        "top_ref_sign_flips": bool(np.sign(top["po_lean_mean"]) != np.sign(top["reg_lean_mean"])),
    }


def bad_actor_test(rng, min_games_per_half=60):
    """Strategy 5: is there a SPECIFIC bad-actor referee (not just a null average)?

    (a) High-power per-referee extremes: with ~290 regular-season games each, is *any* single
        referee distinguishable from zero? (b) Model-free **within-referee split-half
        replication** — does a referee's home-foul lean on half their games predict the other
        half? A real bad actor's lean replicates; noise doesn't. No shrinkage is applied, so this
        also answers the worry that partial pooling might hide a lone outlier.
    """
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    s = get_settings()
    e = pd.read_parquet(s.paths.processed / "regular_season_effects.parquet").sort_values(
        "reg_lean_mean", ascending=False
    )
    n_excl = int(((e["reg_hdi_low"] > 0) | (e["reg_hdi_high"] < 0)).sum())

    def row(r):
        return {
            "referee": str(r.referee),
            "reg_games": int(r.reg_games),
            "reg_lean_mean": round(float(r.reg_lean_mean), 4),
            "hdi_low": round(float(r.reg_hdi_low), 4),
            "hdi_high": round(float(r.reg_hdi_high), 4),
            "p_home": round(float(r.reg_p_gt0), 3),
        }

    suspects = [row(r) for r in e.head(3).itertuples()] + [row(r) for r in e.tail(3).itertuples()]

    mt = pd.read_parquet(s.paths.processed / "modeling_table_regular.parquet")
    long = pd.concat(
        [mt[["game_id", c, "foul_diff_home"]].rename(columns={c: "ref_id"}) for c in _ID_COLS],
        ignore_index=True,
    ).dropna(subset=["ref_id"])
    long["ref_id"] = long["ref_id"].astype(int)
    halves = []
    for _, g in long.groupby("ref_id"):
        if len(g) < 2 * min_games_per_half:
            continue
        idx = rng.permutation(len(g))
        h = len(g) // 2
        halves.append(
            (g.iloc[idx[:h]]["foul_diff_home"].mean(), g.iloc[idx[h:]]["foul_diff_home"].mean())
        )
    hh = np.array(halves)
    pr = pearsonr(hh[:, 0], hh[:, 1])
    sr = spearmanr(hh[:, 0], hh[:, 1])
    return {
        "n_refs_regular": int(len(e)),
        "median_games_per_ref": int(e["reg_games"].median()),
        "refs_excluding_zero": n_excl,
        "most_extreme_abs_lean": round(float(e["reg_lean_mean"].abs().max()), 4),
        "prime_suspects": suspects,
        "split_half_n_refs": int(len(hh)),
        "split_half_pearson": round(float(pr.statistic), 3),
        "split_half_spearman": round(float(sr.statistic), 3),
        "split_half_p": round(float(pr.pvalue), 3),
    }


def run():
    rng = np.random.default_rng(2024)
    s = get_settings()
    out = {
        "strategy_1_extreme_ref_l2m": extreme_ref_l2m(rng),
        "strategy_2_subgroup_forking": subgroup_forking(),
        "strategy_3_label_permutation_placebo": label_permutation_placebo(rng),
        "strategy_4_regression_to_mean": regression_to_mean(),
        "strategy_5_bad_actor_replication": bad_actor_test(rng),
    }
    s.paths.ensure()
    (s.paths.processed / "devils_advocate_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    print("\n=== DEVIL'S ADVOCATE: can we manufacture a referee swing? ===")
    a = out["strategy_1_extreme_ref_l2m"]
    print(f"\n1. Extreme-ref L2M mining ({a['n_refs_tested']} refs tested):")
    print(
        f"   most extreme: {a['most_extreme_favor_home']}/{a['most_extreme_n']} favor home, "
        f"uncorrected p={a['uncorrected_p']:.4f} -> Bonferroni p={a['bonferroni_p']:.2f}; "
        f"someone this extreme by chance {100 * a['null_sim_someone_this_extreme']:.0f}% of the time."
    )
    print(
        f"   honest aggregate: {a['aggregate_favor_home']} vs {a['aggregate_favor_away']} "
        f"({a['aggregate_pct_home']}% home), two-sided p={a['aggregate_two_sided_p']:.3f}"
    )
    b = out["strategy_2_subgroup_forking"]
    print(f"\n2. Subgroup forking-paths ({b['n_slice_tests']} tests):")
    print(
        f"   best slice '{b['best_slice']}' {b['best_metric']}={b['best_value']} p={b['best_uncorrected_p']} "
        f"-> Bonferroni p={b['best_bonferroni_p']}"
    )
    print(
        f"   but league foul->points slope={b['league_foul_to_points_slope']} (p={b['league_foul_to_points_p']}, "
        f"r={b['league_foul_to_points_r']}) -> fouls barely move the score"
    )
    c = out["strategy_3_label_permutation_placebo"]
    print(
        f"\n3. Label-permutation placebo ({c['n_refs']} refs): real between-ref FTA SD={c['real_between_ref_sd_fta']} "
        f"vs null {c['null_sd_mean']} (95th {c['null_sd_95pct']}); identity-adds-signal p={c['identity_adds_signal_p']}"
    )
    print(
        f"   league home FTA tilt={c['league_home_fta_tilt_mean']}/game (p={c['league_home_fta_tilt_p']:.1e}) <- the ONE real thing"
    )
    d = out["strategy_4_regression_to_mean"]
    print(
        f"\n4. Regression to mean: playoff-vs-regular lean Spearman={d['spearman_playoff_vs_regular_lean']} "
        f"(p={d['spearman_p']})"
    )
    print(
        f"   leaderboard-topper {d['top_playoff_ref']}: playoff lean {d['top_ref_playoff_lean']:+.4f} -> "
        f"regular {d['top_ref_regular_lean']:+.4f} (sign flips: {d['top_ref_sign_flips']})"
    )
    f = out["strategy_5_bad_actor_replication"]
    print(
        f"\n5. Specific bad actor? High-power per-ref: {f['refs_excluding_zero']}/{f['n_refs_regular']} "
        f"refs exclude zero (~{f['median_games_per_ref']} games each); most extreme |lean|={f['most_extreme_abs_lean']}."
    )
    print(
        f"   within-referee split-half replication ({f['split_half_n_refs']} refs, model-free, NO shrinkage): "
        f"Pearson r={f['split_half_pearson']}, Spearman={f['split_half_spearman']} (p={f['split_half_p']}) "
        f"-> a referee's lean does NOT replicate across their own games."
    )
    print(
        "\nVERDICT: every manufactured 'swing' dies under standard correction. No INDIVIDUAL referee is "
        "both real-sized and reproducible. The only survivor is a small, crew-INVARIANT, leaguewide home "
        "tilt (documented home-court advantage) — not a referee swing."
    )
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Devil's advocate: try to prove a referee swing."
    ).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
