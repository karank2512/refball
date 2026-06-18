"""Build the clean modeling table: games + odds join + engineered features + QC.

Output: ``data/processed/modeling_table.parquet`` (one row per playoff game).

QC philosophy:
* The odds join is tolerant to a +/-1 day date mismatch (timezone slack) but never joins
  on nba_api game_id — odds and nba_api do not share an id namespace.
* After the join we print left/right/matched/unmatched counts, the match rate, and example
  unmatched games. If the match rate is below the configured threshold we warn loudly
  (the no-odds sensitivity model exists precisely for this case).
"""

from __future__ import annotations

import argparse

import numpy as np

from refball.config import get_settings
from refball.schema import GAME_COLUMNS, validate_columns
from refball.utils.logging import get_logger

logger = get_logger(__name__)


def _zscore(series, mask=None):
    """Standardize; if ``mask`` given, compute mean/std on the masked subset only."""
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce")
    ref = s[mask] if mask is not None else s
    mu = ref.mean()
    sd = ref.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def add_features(games):
    """Add point-diff, foul, free-throw, and possession features (no odds yet)."""
    import pandas as pd

    g = games.copy()
    g["point_diff_home"] = g["home_score"] - g["away_score"]
    g["total_fouls"] = g["home_pf"] + g["away_pf"]
    g["foul_diff_home"] = g["home_pf"] - g["away_pf"]
    g["fta_diff_home"] = g["home_fta"] - g["away_fta"]
    g["ftm_diff_home"] = g["home_ftm"] - g["away_ftm"]

    g["possessions_home"] = g["home_fga"] + 0.44 * g["home_fta"] - g["home_oreb"] + g["home_tov"]
    g["possessions_away"] = g["away_fga"] + 0.44 * g["away_fta"] - g["away_oreb"] + g["away_tov"]
    g["possessions_avg"] = (g["possessions_home"] + g["possessions_away"]) / 2.0
    g["log_possessions"] = np.log(g["possessions_avg"].clip(lower=60.0))

    g["has_officials"] = g[["ref_1_id", "ref_2_id", "ref_3_id"]].notna().any(axis=1)
    g["z_possessions"] = _zscore(g["possessions_avg"])
    g["game_date"] = pd.to_datetime(g["game_date"])
    return g


def join_odds(games, odds):
    """Tolerant (+/-1 day) join of odds onto games by (home, away). Prints QC. Returns games+odds."""
    import pandas as pd

    g = games.sort_values("game_date").copy()
    g["game_date"] = pd.to_datetime(g["game_date"])

    if len(odds) == 0:
        logger.warning("No odds rows; marking all games has_odds=False.")
        g["spread_home"] = np.nan
        g["total_market"] = np.nan
        g["odds_source"] = None
        g["has_odds"] = False
        _print_join_qc(len(games), 0, 0)
        return g

    o = odds.copy()
    o["game_date"] = pd.to_datetime(o["game_date"])
    o = o.sort_values("game_date")

    merged = pd.merge_asof(
        g,
        o[
            [
                "game_date",
                "home_tricode",
                "away_tricode",
                "spread_home",
                "total_market",
                "odds_source",
            ]
        ],
        on="game_date",
        by=["home_tricode", "away_tricode"],
        direction="nearest",
        tolerance=pd.Timedelta("1D"),
    )
    merged["has_odds"] = merged["spread_home"].notna() & merged["total_market"].notna()

    matched = int(merged["has_odds"].sum())
    _print_join_qc(len(games), len(odds), matched)

    unmatched = merged[~merged["has_odds"]]
    if len(unmatched):
        ex = unmatched[["game_date", "home_tricode", "away_tricode"]].head(8)
        logger.info("Example unmatched games:\n%s", ex.to_string(index=False))

    rate = matched / max(len(games), 1)
    if rate < get_settings().min_join_match_rate:
        logger.warning(
            "ODDS MATCH RATE %.1f%% is below threshold %.1f%%. "
            "Treat with-odds results cautiously; the no-odds sensitivity model is the fallback.",
            100 * rate,
            100 * get_settings().min_join_match_rate,
        )
    return merged


def _print_join_qc(left: int, right: int, matched: int) -> None:
    rate = matched / max(left, 1)
    print("=== ODDS JOIN QC ===")
    print(f"left rows (games):       {left}")
    print(f"right rows (odds):       {right}")
    print(f"matched rows:            {matched}")
    print(f"unmatched rows:          {left - matched}")
    print(f"match rate:              {100 * rate:.1f}%")
    print("====================")


def assemble_modeling_table(games, odds):
    """Pure (no-IO) assembly: add features, join odds, add standardized market controls.

    Used by :func:`build` and by the test-suite (which passes small in-memory frames).
    """
    table = join_odds(add_features(games), odds)
    has = table["has_odds"].to_numpy()
    table["z_spread_home"] = _zscore(table["spread_home"], mask=has)
    table["z_total_market"] = _zscore(table["total_market"], mask=has)
    return table


def build(odds_path: str | None = None, *, source: str = "auto") -> object:
    """Assemble the modeling table from interim games + odds. Returns the DataFrame."""
    import pandas as pd

    from refball.data.odds import load_odds

    s = get_settings()
    if not s.paths.games_interim.exists():
        raise FileNotFoundError(
            f"{s.paths.games_interim} not found. Run `python -m refball.data.pull ...` "
            f"or `python -m refball.data.synthetic` (demo) first."
        )
    games = pd.read_parquet(s.paths.games_interim)
    validate_columns(games, GAME_COLUMNS, "interim games")
    logger.info("Loaded %d interim games", len(games))

    odds = load_odds(odds_path)
    table = assemble_modeling_table(games, odds)

    # QC summary -----------------------------------------------------------
    n = len(table)
    print("\n=== MODELING TABLE QC ===")
    print(f"games:                   {n}")
    print(f"seasons:                 {sorted(table['season'].unique().tolist())}")
    print(
        f"with officials:          {int(table['has_officials'].sum())} ({100 * table['has_officials'].mean():.1f}%)"
    )
    print(
        f"with odds:               {int(table['has_odds'].sum())} ({100 * table['has_odds'].mean():.1f}%)"
    )
    print("missingness (key cols):")
    for c in [
        "home_pf",
        "away_pf",
        "home_fta",
        "away_fta",
        "spread_home",
        "total_market",
        "ref_1_id",
    ]:
        print(f"  {c:16s} missing {int(table[c].isna().sum()):5d}")
    print("=========================\n")

    s.paths.ensure()
    table.to_parquet(s.paths.modeling_table, index=False)
    logger.info("Wrote modeling table: %s (%d rows)", s.paths.modeling_table, n)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the clean modeling table.")
    parser.add_argument("--odds", type=str, default=None, help="optional odds CSV path")
    ns = parser.parse_args(argv)
    table = build(ns.odds)
    print(f"[build-table] {len(table)} games -> {get_settings().paths.modeling_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
