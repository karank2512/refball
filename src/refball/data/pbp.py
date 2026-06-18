"""Play-by-play foul detail: foul *types* and *game-state* cleaning (via PlayByPlayV3).

Box-score foul totals are blunt and confounded: a trailing team fouls intentionally late, and
garbage-time fouls don't matter — which is why raw foul margin is ~uncorrelated with the final
margin (r~0.02). Play-by-play lets us (a) keep only *discretionary* fouls (Shooting / Personal /
Loose Ball — where referee judgment lives) and (b) drop *non-competitive* fouls (4th-quarter/OT
fouls in a blowout), isolating the whistle that could actually swing a game.

Source: nba_api ``PlayByPlayV3`` (``playbyplayv2`` is broken in this nba_api build). v3 events:
``actionType == "Foul"``; ``subType`` is the foul kind (string); ``teamTricode`` is the committing
team; ``scoreHome``/``scoreAway`` are running scores (only on scoring plays -> forward-filled);
``clock`` is ISO8601 ("PT07M47.00S"). Cached per game.
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from refball.config import get_settings
from refball.utils.cache import cached_parquet, polite_sleep, raw_path, with_retries
from refball.utils.logging import get_logger
from refball.utils.teams import normalize_team

logger = get_logger(__name__)

# Discretionary contact fouls — what a biased whistle would actually move.
DISCRETIONARY = {"shooting", "personal", "loose ball"}
_CLOCK_RE = re.compile(r"PT0*(\d+)M0*(\d+(?:\.\d+)?)S")


def _pbp_for_game(game_id: str, force_refresh: bool):
    path = raw_path("nba", "pbp", f"{game_id}.parquet")

    def _build():
        from nba_api.stats.endpoints import playbyplayv3

        s = get_settings()

        def _call():
            polite_sleep()
            return playbyplayv3.PlayByPlayV3(
                game_id=game_id, timeout=int(s.request_timeout_s)
            ).get_data_frames()[0]

        return with_retries(_call, what=f"PBP {game_id}")

    return cached_parquet(path, _build, force_refresh=force_refresh)


def _clock_seconds(clock: str) -> float:
    m = _CLOCK_RE.search(str(clock))
    return float(m.group(1)) * 60 + float(m.group(2)) if m else np.nan


def parse_fouls(pbp, home_tri: str, away_tri: str):
    """Return a tidy frame of foul events for one game with side + subtype + game-state."""
    import pandas as pd

    if pbp is None or len(pbp) == 0:
        return pd.DataFrame()
    df = pbp.copy()
    # Forward-fill the running score (only populated on scoring plays) to get margin at any event.
    sh = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0)
    sa = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0)
    df["margin_abs"] = (sh - sa).abs()

    f = df[df["actionType"].astype(str).str.strip().str.lower() == "foul"].copy()
    if len(f) == 0:
        return pd.DataFrame()
    f["subtype"] = f["subType"].astype(str).str.strip().str.lower()
    f["is_discretionary"] = f["subtype"].isin(DISCRETIONARY)
    f["committing_tri"] = f["teamTricode"].map(lambda x: normalize_team(x))
    f["side"] = np.where(
        f["committing_tri"] == home_tri,
        "home",
        np.where(f["committing_tri"] == away_tri, "away", None),
    )
    f["period"] = pd.to_numeric(f["period"], errors="coerce").astype("Int64")
    f["clock_s"] = f["clock"].map(_clock_seconds)
    # Competitive = NOT a late-game blowout foul (period>=4 and margin>=12) -> drops garbage time
    # and big-deficit intentional fouling, the main fouls->scoreboard confounds.
    f["competitive"] = ~((f["period"] >= 4) & (f["margin_abs"] >= 12))
    return f


def build_foul_table(games, force_refresh: bool = False):
    """Per-game cleaned foul features, joinable on game_id. ``games`` needs game_id+tricodes."""
    import pandas as pd

    rows = []
    n = len(games)
    for i, (_, g) in enumerate(games.iterrows(), 1):
        gid = str(g["game_id"])
        try:
            fouls = parse_fouls(
                _pbp_for_game(gid, force_refresh), g["home_tricode"], g["away_tricode"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PBP parse failed for %s: %s", gid, exc)
            continue
        if len(fouls) == 0:
            continue
        disc = fouls[fouls["is_discretionary"] & fouls["side"].notna()]
        comp = disc[disc["competitive"]]

        def by_side(d, side):
            return int((d["side"] == side).sum())

        rec = {
            "game_id": gid,
            "home_disc_fouls": by_side(disc, "home"),
            "away_disc_fouls": by_side(disc, "away"),
            "home_disc_fouls_comp": by_side(comp, "home"),
            "away_disc_fouls_comp": by_side(comp, "away"),
            "n_fouls_pbp": int(len(fouls)),
        }
        rec["foul_diff_home_disc"] = rec["home_disc_fouls"] - rec["away_disc_fouls"]
        rec["foul_diff_home_comp"] = rec["home_disc_fouls_comp"] - rec["away_disc_fouls_comp"]
        rows.append(rec)
        if i % 50 == 0 or i == n:
            logger.info("pbp fouls: %d/%d games", i, n)

    table = pd.DataFrame(rows)
    out = get_settings().paths.processed / "pbp_foul_table.parquet"
    table.to_parquet(out, index=False)
    logger.info("Wrote PBP foul table: %s (%d games)", out, len(table))
    return table


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    p = argparse.ArgumentParser(description="Build the PBP cleaned-foul table for playoff games.")
    p.add_argument("--limit", type=int, default=None, help="limit games (for a quick sample)")
    p.add_argument("--force-refresh", action="store_true")
    ns = p.parse_args(argv)

    s = get_settings()
    games = pd.read_parquet(s.paths.modeling_table)[["game_id", "home_tricode", "away_tricode"]]
    if ns.limit:
        games = games.head(ns.limit)
    table = build_foul_table(games, force_refresh=ns.force_refresh)
    print(
        f"[pbp] built foul table for {len(table)} games -> {s.paths.processed / 'pbp_foul_table.parquet'}"
    )
    if len(table):
        print(
            f"mean raw discretionary foul margin (home): {table['foul_diff_home_disc'].mean():+.3f}"
        )
        print(
            f"mean competitive disc foul margin (home):  {table['foul_diff_home_comp'].mean():+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
