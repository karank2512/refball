"""Shared MCMC diagnostics helpers (R-hat, ESS, divergences, BFMI) and a robust HDI.

Written to work across the ArviZ 0.x / 1.x split: we avoid version-fragile kwargs
(``az.hdi(hdi_prob=...)`` was removed in 1.x) by computing the highest-density interval
directly, and we compute BFMI from the sampler energy rather than relying on ``az.bfmi``'s
changing return type.
"""

from __future__ import annotations

import numpy as np

from refball.utils.logging import get_logger

logger = get_logger(__name__)


def hdi_interval(samples, prob: float = 0.94) -> tuple[float, float]:
    """Highest-density interval of a 1-D sample via the standard sorted-window method."""
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(x[0]), float(x[0]))
    width_idx = int(np.floor(prob * n))
    width_idx = min(max(width_idx, 1), n - 1)
    n_windows = n - width_idx
    widths = x[width_idx:] - x[:n_windows]
    lo = int(np.argmin(widths))
    return (float(x[lo]), float(x[lo + width_idx]))


def _bfmi_min_from_energy(idata) -> float:
    """Per-chain BFMI from sample-stats energy; return the worst (min) chain."""
    ss = getattr(idata, "sample_stats", None)
    if ss is None or "energy" not in ss:
        return float("nan")
    energy = np.asarray(ss["energy"].values)  # (chain, draw)
    if energy.ndim == 1:
        energy = energy[None, :]
    vals = []
    for chain in energy:
        denom = np.sum((chain - chain.mean()) ** 2)
        if denom <= 0:
            continue
        vals.append(np.sum(np.diff(chain) ** 2) / denom)
    return float(np.nanmin(vals)) if vals else float("nan")


def summarize_diagnostics(idata, var_names: list[str] | None = None) -> dict:
    """Return a compact diagnostics dict for an InferenceData/DataTree object."""
    import arviz as az

    summary = az.summary(idata, var_names=var_names)
    rhat = np.asarray(summary["r_hat"], dtype=float)
    ess_bulk = np.asarray(summary["ess_bulk"], dtype=float)
    ess_tail = np.asarray(summary["ess_tail"], dtype=float)

    divergences = 0
    ss = getattr(idata, "sample_stats", None)
    if ss is not None and "diverging" in ss:
        divergences = int(np.asarray(ss["diverging"].values).sum())

    return {
        "max_r_hat": float(np.nanmax(rhat)),
        "min_ess_bulk": float(np.nanmin(ess_bulk)),
        "min_ess_tail": float(np.nanmin(ess_tail)),
        "divergences": divergences,
        "min_bfmi": _bfmi_min_from_energy(idata),
        "n_params": int(len(summary)),
        "rhat_gt_1_01": int(np.nansum(rhat > 1.01)),
    }


def print_diagnostics(name: str, diag: dict) -> None:
    print(f"\n=== DIAGNOSTICS: {name} ===")
    print(f"max R-hat:            {diag['max_r_hat']:.4f}  (target < 1.01)")
    print(f"params R-hat > 1.01:  {diag['rhat_gt_1_01']}")
    print(f"min bulk ESS:         {diag['min_ess_bulk']:.0f}")
    print(f"min tail ESS:         {diag['min_ess_tail']:.0f}")
    print(f"divergences:          {diag['divergences']}")
    print(f"min BFMI:             {diag['min_bfmi']:.3f}  (target > 0.3)")
    status = "OK" if (diag["max_r_hat"] < 1.01 and diag["divergences"] == 0) else "CHECK"
    print(f"status:               {status}")
    print("==============================\n")
