"""Synthetic generator schema + planted-structure sanity."""

from __future__ import annotations

from refball.schema import GAME_COLUMNS, ODDS_COLUMNS


def test_schema_and_counts(synth):
    games, odds, truth = synth
    assert list(games.columns) == GAME_COLUMNS
    assert list(odds.columns) == ODDS_COLUMNS
    assert len(games) == 40
    assert truth["n_games"] == 40


def test_planted_outliers_declared(synth):
    _, _, truth = synth
    names = set(truth["ref_names"])
    for key in ("high_total_fouls_ref", "positive_lean_ref", "negative_lean_ref"):
        assert truth["planted_outliers"][key] in names


def test_scores_and_fouls_nonnegative(synth):
    games, _, _ = synth
    assert (games["home_pf"] >= 0).all()
    assert (games["away_pf"] >= 0).all()
    assert (games["home_score"] > 0).all()
    assert (games["home_ftm"] <= games["home_fta"]).all()
