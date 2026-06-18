"""Stage 9: mediated (referee -> foul margin -> scoreboard) effects.

We compose two posteriors:

* the per-referee directional foul-margin association from Stage 1B, expressed in *count*
  units (home-minus-away fouls attributable to adding this official to an otherwise-average
  crew: ``foul_margin_r = f_bar * (exp(lean_r/k) - exp(-lean_r/k))``), and
* the foul-margin -> point-differential coefficient ``gamma_foul_diff`` from Stage 2.

The mediated scoreboard effect is the product, propagated over the FULL posterior (not
point estimates):

    indirect_points_r = foul_margin_r  x  gamma_foul_diff

The two models are fit separately, so we treat their posteriors as independent and form the
Monte-Carlo product by resampling. The result is a screening metric with a wide interval —
explicitly **not** evidence of intent or that any official changed an outcome on purpose.
"""

from __future__ import annotations

import argparse

import numpy as np

from refball.config import get_settings
from refball.models.diagnostics import hdi_interval
from refball.utils.logging import get_logger

logger = get_logger(__name__)


def _lean_draws_by_ref(idata_lean):
    """Return (ref_ids:list[int], draws:np.ndarray[ref, draw])."""
    da = idata_lean.posterior["ref_lean_effect"]
    ref_coord = [int(x) for x in da.coords["ref"].values]
    draws = da.stack(s=("chain", "draw")).transpose("ref", "s").values
    return ref_coord, draws


def _f_bar(k: float) -> float:
    import pandas as pd

    s = get_settings()
    table = pd.read_parquet(s.paths.modeling_table)
    table = table[table["has_officials"]]
    return float(np.mean((table["home_pf"] + table["away_pf"]) / 2.0))


def compute(seed: int = 12345):
    import arviz as az
    import pandas as pd

    s = get_settings()
    for p in (s.paths.stage1_lean_nc, s.paths.stage2_nc, s.paths.referee_stage1_effects):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {p}. Run Stage 1 and Stage 2 first."
            )

    idata_lean = az.from_netcdf(str(s.paths.stage1_lean_nc))
    idata_stage2 = az.from_netcdf(str(s.paths.stage2_nc))
    ref_effects = pd.read_parquet(s.paths.referee_stage1_effects)
    name_by_id = dict(zip(ref_effects["ref_id"], ref_effects["referee"], strict=False))
    games_by_id = dict(zip(ref_effects["ref_id"], ref_effects["games"], strict=False))

    k = float(s.officials_per_game)
    f_bar = _f_bar(k)
    prob = s.hdi_prob

    ref_ids, lean = _lean_draws_by_ref(idata_lean)
    gamma = idata_stage2.posterior["gamma_foul_diff"].values.ravel()

    rng = np.random.default_rng(seed)
    n_draws = lean.shape[1]
    gamma_rs = rng.choice(gamma, size=n_draws, replace=True)  # independent posteriors -> resample

    recs = []
    for i, rid in enumerate(ref_ids):
        lean_r = lean[i]
        foul_margin_r = f_bar * (np.exp(lean_r / k) - np.exp(-lean_r / k))
        indirect = foul_margin_r * gamma_rs
        h_lo, h_hi = hdi_interval(indirect, prob)
        recs.append(
            {
                "ref_id": rid,
                "referee": name_by_id.get(rid, f"Ref {rid}"),
                "games": int(games_by_id.get(rid, 0)),
                "foul_margin_mean": float(foul_margin_r.mean()),
                "mediated_points_mean": float(indirect.mean()),
                "mediated_hdi_low": h_lo,
                "mediated_hdi_high": h_hi,
                "p_mediated_gt0": float((indirect > 0).mean()),
                "p_mediated_lt0": float((indirect < 0).mean()),
            }
        )
    out = pd.DataFrame(recs).sort_values("mediated_points_mean").reset_index(drop=True)
    out.attrs["f_bar"] = f_bar
    out.attrs["gamma_foul_diff_mean"] = float(gamma.mean())
    return out


def run():
    s = get_settings()
    out = compute()
    s.paths.ensure()
    out.to_parquet(s.paths.referee_mediated_effects, index=False)
    logger.info("Saved mediated effects: %s (%d refs)", s.paths.referee_mediated_effects, len(out))
    print("\nMediated referee scoreboard effects (home points; screening metric, wide intervals):")
    cols = [
        "referee",
        "games",
        "mediated_points_mean",
        "mediated_hdi_low",
        "mediated_hdi_high",
        "p_mediated_lt0",
    ]
    print("Most negative (associated with fewer home points via foul margin):")
    print(out.head(5)[cols].to_string(index=False))
    print("Most positive (associated with more home points via foul margin):")
    print(out.tail(5)[cols].to_string(index=False))
    print(
        "\nReminder: a wide interval that straddles 0 means no detectable mediated signal. "
        "None of this implies intent or misconduct."
    )
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Compute mediated referee scoreboard effects.").parse_args(
        argv
    )
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
