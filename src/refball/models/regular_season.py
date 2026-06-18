"""Regular-season power model + playoff-vs-baseline comparison.

The playoff null was *underpowered* (~21 games/ref). The regular season gives ~290 games/ref
(102 refs over 2017-2023), so this is where a real referee foul-lean would actually show up if
it exists. We:

1. Fit the directional lean model on the full regular-season sample (the **power test**): with
   this much data, how many referees have a home-foul-lean 94% interval excluding zero?
2. Compare each referee's **playoff** lean to their own **regular-season** baseline — the
   "do crews behave differently in the playoffs / get 'brought in' for big games" question.

Inputs: ``data/interim/games_regular.parquet`` (from ``pull --season-type "Regular Season"``).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from refball.config import get_settings
from refball.models.data_prep import prepare
from refball.models.diagnostics import hdi_interval, print_diagnostics, summarize_diagnostics
from refball.utils.logging import get_logger

logger = get_logger(__name__)


def build_regular_table():
    import pandas as pd

    from refball.features.build_table import assemble_modeling_table
    from refball.schema import ODDS_COLUMNS

    s = get_settings()
    if not s.paths.games_regular_interim.exists():
        raise FileNotFoundError(
            "games_regular.parquet not found. Run "
            "`python -m refball.data.pull --season-type 'Regular Season' --season-start 2017 --season-end 2023`."
        )
    games = pd.read_parquet(s.paths.games_regular_interim)
    table = assemble_modeling_table(games, pd.DataFrame(columns=ODDS_COLUMNS))
    out = s.paths.processed / "modeling_table_regular.parquet"
    table.to_parquet(out, index=False)
    logger.info("Regular-season modeling table: %s (%d games)", out, len(table))
    return table


def _fit(model, quick):
    import pymc as pm

    with model:
        return pm.sample(**get_settings().sampler.resolve(quick), progressbar=False)


def run(quick: bool):
    import pandas as pd

    from refball.models.fit_stage1 import build_lean_model

    s = get_settings()
    prob = s.hdi_prob
    table = build_regular_table()
    md = prepare(table=table, use_odds=False, require_officials=True)
    logger.info("Fitting regular-season lean model on %d games, %d refs...", md.n_games, md.n_refs)

    idata = _fit(build_lean_model(md), quick)
    diag = summarize_diagnostics(
        idata, var_names=["alpha_home", "alpha_away", "sigma_ref_lean", "ref_lean_effect"]
    )
    print_diagnostics("Regular-season lean model (power test)", diag)

    lean = (
        idata.posterior["ref_lean_effect"].stack(z=("chain", "draw")).transpose("ref", "z").values
    )
    recs = []
    for i, rid in enumerate(md.ref_index.ref_ids):
        d = lean[i]
        lo, hi = hdi_interval(d, prob)
        recs.append(
            {
                "ref_id": rid,
                "referee": md.ref_index.names[rid],
                "reg_games": md.ref_index.games_count[rid],
                "reg_lean_mean": float(d.mean()),
                "reg_hdi_low": lo,
                "reg_hdi_high": hi,
                "reg_p_gt0": float((d > 0).mean()),
                "reg_excludes_zero": bool(lo > 0 or hi < 0),
            }
        )
    reg = pd.DataFrame(recs).sort_values("reg_lean_mean", ascending=False).reset_index(drop=True)
    reg_sigma = float(idata.posterior["sigma_ref_lean"].values.mean())

    # Compare to playoff leans (if available)
    comp_corr = float("nan")
    n_diff = 0
    po_path = s.paths.referee_stage1_effects
    if po_path.exists():
        po = pd.read_parquet(po_path)[
            ["ref_id", "lean_mean", "lean_hdi_low", "lean_hdi_high", "games"]
        ]
        po = po.rename(
            columns={
                "lean_mean": "po_lean_mean",
                "games": "po_games",
                "lean_hdi_low": "po_hdi_low",
                "lean_hdi_high": "po_hdi_high",
            }
        )
        merged = reg.merge(po, on="ref_id", how="inner")
        if len(merged) >= 3:
            from scipy.stats import spearmanr

            comp_corr = float(spearmanr(merged["reg_lean_mean"], merged["po_lean_mean"]).statistic)
        # "differs in playoffs" = playoff lean HDI excludes the regular-season mean
        merged["playoff_differs_from_baseline"] = (
            merged["po_hdi_low"] > merged["reg_lean_mean"]
        ) | (merged["po_hdi_high"] < merged["reg_lean_mean"])
        n_diff = int(merged["playoff_differs_from_baseline"].sum())
        reg = reg.merge(po[["ref_id", "po_lean_mean", "po_games"]], on="ref_id", how="left")

    s.paths.ensure()
    idata.to_netcdf(str(s.paths.models / "regular_season_lean.nc"))
    reg.to_parquet(s.paths.processed / "regular_season_effects.parquet", index=False)
    summary = {
        "n_games": int(md.n_games),
        "n_refs": int(md.n_refs),
        "median_games_per_ref": int(np.median(list(md.ref_index.games_count.values()))),
        "reg_sigma_ref_lean": reg_sigma,
        "reg_refs_excluding_zero": int(reg["reg_excludes_zero"].sum()),
        "spearman_reg_vs_playoff_lean": comp_corr,
        "refs_playoff_differs_from_baseline": n_diff,
        "diagnostics": diag,
    }
    (s.paths.processed / "regular_season_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== REGULAR-SEASON POWER TEST (the high-power check the playoffs couldn't do) ===")
    print(
        f"games / refs:               {summary['n_games']} / {summary['n_refs']} "
        f"(median {summary['median_games_per_ref']} games/ref)"
    )
    print(f"regular-season sigma_lean:  {reg_sigma:.4f}")
    print(
        f"refs whose lean HDI excludes 0: {summary['reg_refs_excluding_zero']} / {summary['n_refs']}"
    )
    print(f"reg-vs-playoff lean Spearman:   {comp_corr:.3f}")
    print(
        f"refs whose PLAYOFF lean differs from their own regular-season baseline: {n_diff} / {summary['n_refs']}"
    )
    print("Top regular-season home-foul leans (now with real precision):")
    cols = [
        "referee",
        "reg_games",
        "reg_lean_mean",
        "reg_hdi_low",
        "reg_hdi_high",
        "reg_excludes_zero",
    ]
    print(reg.head(5)[cols].to_string(index=False))
    print(reg.tail(3)[cols].to_string(index=False))
    return idata, reg, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regular-season power model + playoff comparison.")
    p.add_argument("--quick", action="store_true")
    ns = p.parse_args(argv)
    run(quick=ns.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
