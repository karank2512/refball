"""Feature engineering, odds join, and multi-membership matrix."""

from __future__ import annotations

import numpy as np

from refball.features.build_table import add_features
from refball.features.multimembership import build_membership_matrix, build_ref_index


def test_derived_features(modeling_table):
    t = modeling_table
    assert (t["total_fouls"] == t["home_pf"] + t["away_pf"]).all()
    assert (t["foul_diff_home"] == t["home_pf"] - t["away_pf"]).all()
    assert (t["point_diff_home"] == t["home_score"] - t["away_score"]).all()


def test_possessions_formula(synth):
    games, _, _ = synth
    feat = add_features(games)
    row = feat.iloc[0]
    expected = row["home_fga"] + 0.44 * row["home_fta"] - row["home_oreb"] + row["home_tov"]
    assert abs(row["possessions_home"] - expected) < 1e-9


def test_odds_join_rate_high(modeling_table):
    # Synthetic odds share exact dates/teams, so the join should be ~complete.
    assert modeling_table["has_odds"].mean() > 0.9


def test_membership_matrix_weights(synth):
    games, _, _ = synth
    index = build_ref_index(games)
    R, k = build_membership_matrix(games, index)
    assert R.shape == (len(games), len(index.ref_ids))
    assert np.allclose(R.sum(axis=1), 1.0)  # 3 officials x 1/3
    assert (k == 3).all()
    # weights are exactly 1/3 where present
    nonzero = R[R > 0]
    assert np.allclose(nonzero, 1.0 / 3.0)
