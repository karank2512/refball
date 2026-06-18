"""Canonical column contracts shared across sources.

The real nba_api pull (:mod:`refball.data.pull`) and the synthetic generator
(:mod:`refball.data.synthetic`) both emit a DataFrame with :data:`GAME_COLUMNS`, so every
downstream stage is source-agnostic. The odds adapter emits :data:`ODDS_COLUMNS`. The
join produces :data:`MODELING_COLUMNS`.
"""

from __future__ import annotations

# One row per playoff game (pre-odds, pre-feature-engineering).
GAME_COLUMNS: list[str] = [
    "game_id",
    "season",  # starting calendar year, e.g. 2017 -> 2017-18
    "season_str",
    "playoff_round",  # 1..4 if decodable, else <NA>
    "game_date",  # python date
    "home_tricode",
    "away_tricode",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_pf",
    "away_pf",
    "home_fta",
    "away_fta",
    "home_ftm",
    "away_ftm",
    "home_fga",
    "away_fga",
    "home_oreb",
    "away_oreb",
    "home_tov",
    "away_tov",
    "official_1",
    "official_2",
    "official_3",
    "ref_1_id",
    "ref_2_id",
    "ref_3_id",
]

# Odds feed (swappable adapter). Spread is signed from the HOME perspective:
#   spread_home < 0  => home favored;  spread_home > 0 => home underdog.
ODDS_COLUMNS: list[str] = [
    "game_date",
    "home_tricode",
    "away_tricode",
    "spread_home",
    "total_market",
    "odds_source",
]

# Final modeling table (after join + feature engineering). Superset of GAME_COLUMNS.
DERIVED_COLUMNS: list[str] = [
    "point_diff_home",
    "total_fouls",
    "foul_diff_home",
    "fta_diff_home",
    "ftm_diff_home",
    "possessions_home",
    "possessions_away",
    "possessions_avg",
    "log_possessions",
    "spread_home",
    "total_market",
    "odds_source",
    "has_odds",
    "has_officials",
    "z_spread_home",
    "z_total_market",
    "z_possessions",
]


def validate_columns(df, required: list[str], name: str) -> None:
    """Raise a clear error if ``df`` is missing any ``required`` column."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. Present: {list(df.columns)}"
        )
