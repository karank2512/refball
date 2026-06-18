"""Team-name normalization."""

from __future__ import annotations

from refball.utils.teams import CANONICAL, normalize_team


def test_canonical_has_30_teams():
    assert len(CANONICAL) == 30


def test_aliases_resolve():
    cases = {
        "GS Warriors": "GSW",
        "Golden State": "GSW",
        "Los Angeles Lakers": "LAL",
        "LA Lakers": "LAL",
        "LA Clippers": "LAC",
        "Sixers": "PHI",
        "76ers": "PHI",
        "Brooklyn": "BKN",
        "BKN": "BKN",
        "New Orleans": "NOP",
        "Wolves": "MIN",
    }
    for name, tri in cases.items():
        assert normalize_team(name) == tri, name


def test_unknown_returns_none():
    assert normalize_team("Not A Real Team 123") is None
    assert normalize_team(None) is None
