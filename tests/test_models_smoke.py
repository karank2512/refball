"""Smoke tests for the Bayesian models — verify the graphs compile and sample.

These intentionally use tiny draws; they check *correctness of construction*, not
convergence. Skipped automatically if PyMC is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymc")

import pymc as pm  # noqa: E402

from refball.models.data_prep import prepare  # noqa: E402
from refball.models.fit_stage1 import build_lean_model, build_total_model  # noqa: E402
from refball.models.fit_stage2 import build_pointdiff_model  # noqa: E402
from refball.models.robustness import build_pf_long_model  # noqa: E402

TINY = dict(draws=15, tune=15, chains=1, cores=1, progressbar=False, random_seed=0)


@pytest.fixture(scope="module")
def md(modeling_table):
    return prepare(use_odds=True, require_officials=True, table=modeling_table)


def test_total_model_samples(md):
    with build_total_model(md):
        idata = pm.sample(**TINY)
    assert "ref_total_effect" in idata.posterior
    assert idata.posterior["ref_total_effect"].sizes["ref"] == md.n_refs


def test_lean_model_samples(md):
    with build_lean_model(md):
        idata = pm.sample(**TINY)
    assert "ref_lean_effect" in idata.posterior
    assert "ref_vol_effect" in idata.posterior


def test_pointdiff_model_samples(md):
    with build_pointdiff_model(md, team_effects=False):
        idata = pm.sample(**TINY)
    assert "gamma_foul_diff" in idata.posterior


def test_pf_long_nested_models(md):
    for use_vol, use_lean in [(False, False), (True, False), (True, True)]:
        with build_pf_long_model(md, use_vol=use_vol, use_lean=use_lean):
            idata = pm.sample(**TINY)
        if use_lean:
            assert "ref_lean_effect" in idata.posterior
