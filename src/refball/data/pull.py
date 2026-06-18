"""Pull NBA playoff games + officials from nba_api, with on-disk caching.

Two sources are combined into the canonical one-row-per-game table:

1. ``LeagueGameLog`` (SeasonType=Playoffs, team level) -> scores, personal fouls, free
   throws, and the box-score components needed for the possessions estimate.
2. ``BoxScoreSummaryV2.officials`` -> the assigned crew per game (names + ids).

Everything is cached under ``data/raw`` so re-runs are offline unless ``--force-refresh``.
If nba_api is unavailable or a season returns nothing, the failure is logged and you can
fall back to ``python -m refball.data.synthetic`` for demo mode.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

import numpy as np

from refball.config import get_settings, season_str
from refball.schema import GAME_COLUMNS
from refball.utils.cache import cached_json, cached_parquet, polite_sleep, raw_path, with_retries
from refball.utils.logging import get_logger
from refball.utils.provenance import log_source
from refball.utils.teams import CANONICAL, normalize_team

logger = get_logger(__name__)


def decode_playoff_round(game_id: str) -> int | None:
    """Decode the playoff round (1..4) from a 10-digit playoff GAME_ID.

    Playoff ids look like ``0041700405``: ``004`` prefix, ``17`` season, ``00`` filler,
    then round/series/game. The round digit is index 7.
    """
    gid = str(game_id)
    if len(gid) == 10 and gid.startswith("004"):
        try:
            r = int(gid[7])
            return r if 1 <= r <= 4 else None
        except ValueError:
            return None
    return None


def _parse_date(val: str) -> date:
    return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()


# --- 1. Team game logs --------------------------------------------------------
def _type_tag(season_type: str) -> str:
    return "reg" if season_type.lower().startswith("regular") else "po"


def _season_team_log(start_year: int, force_refresh: bool, season_type: str = "Playoffs"):
    import pandas as pd

    ss = season_str(start_year)
    path = raw_path("nba", f"team_gamelog_{_type_tag(season_type)}_{ss}.parquet")

    def _build():
        from nba_api.stats.endpoints import leaguegamelog

        s = get_settings()

        def _call():
            polite_sleep()
            lg = leaguegamelog.LeagueGameLog(
                season=ss,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
                timeout=int(s.request_timeout_s),
            )
            return lg.get_data_frames()[0]

        df = with_retries(_call, what=f"LeagueGameLog {ss} {season_type}")
        df["__start_year"] = start_year
        return df

    df = cached_parquet(path, _build, force_refresh=force_refresh)
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("No team game log rows for %s (%s)", ss, season_type)
    return df


def pull_team_logs(seasons: list[int], force_refresh: bool = False, season_type: str = "Playoffs"):
    import pandas as pd

    frames = [_season_team_log(yr, force_refresh, season_type) for yr in seasons]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError(
            f"No {season_type} team logs pulled. Check nba_api connectivity, or use "
            "`python -m refball.data.synthetic` for demo mode."
        )
    return pd.concat(frames, ignore_index=True)


def pivot_to_games(team_log):
    """Collapse two team rows per game into one home/away row."""
    import pandas as pd

    def side(matchup: str) -> str:
        return "home" if " vs. " in str(matchup) else "away"

    team_log = team_log.copy()
    team_log["side"] = team_log["MATCHUP"].map(side)

    rows = []
    for gid, grp in team_log.groupby("GAME_ID"):
        h = grp[grp["side"] == "home"]
        a = grp[grp["side"] == "away"]
        if len(h) != 1 or len(a) != 1:
            logger.warning(
                "Skipping game %s: expected 1 home + 1 away, got %d/%d", gid, len(h), len(a)
            )
            continue
        h = h.iloc[0]
        a = a.iloc[0]
        start_year = int(h["__start_year"])
        rows.append(
            {
                "game_id": str(gid),
                "season": start_year,
                "season_str": season_str(start_year),
                "playoff_round": decode_playoff_round(gid),
                "game_date": _parse_date(h["GAME_DATE"]),
                "home_tricode": normalize_team(h["TEAM_ABBREVIATION"]) or h["TEAM_ABBREVIATION"],
                "away_tricode": normalize_team(a["TEAM_ABBREVIATION"]) or a["TEAM_ABBREVIATION"],
                "home_team": CANONICAL.get(
                    normalize_team(h["TEAM_ABBREVIATION"]) or "", h["TEAM_NAME"]
                ),
                "away_team": CANONICAL.get(
                    normalize_team(a["TEAM_ABBREVIATION"]) or "", a["TEAM_NAME"]
                ),
                "home_score": int(h["PTS"]),
                "away_score": int(a["PTS"]),
                "home_pf": int(h["PF"]),
                "away_pf": int(a["PF"]),
                "home_fta": int(h["FTA"]),
                "away_fta": int(a["FTA"]),
                "home_ftm": int(h["FTM"]),
                "away_ftm": int(a["FTM"]),
                "home_fga": int(h["FGA"]),
                "away_fga": int(a["FGA"]),
                "home_oreb": int(h["OREB"]),
                "away_oreb": int(a["OREB"]),
                "home_tov": int(h["TOV"]),
                "away_tov": int(a["TOV"]),
            }
        )
    return pd.DataFrame(rows)


# --- 2. Officials -------------------------------------------------------------
def _officials_for_game(game_id: str, force_refresh: bool) -> list[dict]:
    path = raw_path("nba", "officials", f"{game_id}.json")

    def _build() -> list[dict]:
        from nba_api.stats.endpoints import boxscoresummaryv2

        s = get_settings()

        def _call():
            polite_sleep()
            bs = boxscoresummaryv2.BoxScoreSummaryV2(
                game_id=game_id, timeout=int(s.request_timeout_s)
            )
            return bs.officials.get_data_frame()

        df = with_retries(_call, what=f"Officials {game_id}")
        recs = []
        for _, r in df.iterrows():
            name = f"{str(r.get('FIRST_NAME', '')).strip()} {str(r.get('LAST_NAME', '')).strip()}".strip()
            recs.append({"official_id": int(r["OFFICIAL_ID"]), "name": name})
        return recs

    return cached_json(path, _build, force_refresh=force_refresh)


def pull_officials(game_ids: list[str], force_refresh: bool = False):
    import pandas as pd

    rows = []
    n = len(game_ids)
    for i, gid in enumerate(game_ids, 1):
        try:
            officials = _officials_for_game(gid, force_refresh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Officials pull failed for %s: %s", gid, exc)
            officials = []
        officials = officials[:3] + [{}] * (3 - len(officials))
        rows.append(
            {
                "game_id": gid,
                "official_1": officials[0].get("name") or None,
                "official_2": officials[1].get("name") or None,
                "official_3": officials[2].get("name") or None,
                "ref_1_id": officials[0].get("official_id"),
                "ref_2_id": officials[1].get("official_id"),
                "ref_3_id": officials[2].get("official_id"),
            }
        )
        if i % 25 == 0 or i == n:
            logger.info("officials: %d/%d games", i, n)
    return pd.DataFrame(rows)


# --- 3. Assemble --------------------------------------------------------------
def assemble_games(seasons: list[int], force_refresh: bool = False, season_type: str = "Playoffs"):
    """Build + cache the canonical games table.

    Playoffs -> ``data/interim/games.parquet`` (the default modeling input);
    Regular Season -> ``data/interim/games_regular.parquet`` (kept separate).
    """
    s = get_settings()
    team_log = pull_team_logs(seasons, force_refresh, season_type)
    games = pivot_to_games(team_log)
    officials = pull_officials(games["game_id"].tolist(), force_refresh)
    games = games.merge(officials, on="game_id", how="left")

    for col in GAME_COLUMNS:
        if col not in games.columns:
            games[col] = np.nan
    games = games[GAME_COLUMNS]

    n_missing_off = int(games["official_1"].isna().sum())
    logger.info(
        "Assembled %d %s games; %d missing officials", len(games), season_type, n_missing_off
    )

    s.paths.ensure()
    out = (
        s.paths.games_regular_interim if _type_tag(season_type) == "reg" else s.paths.games_interim
    )
    games.to_parquet(out, index=False)
    log_source(
        "nba_api",
        f"LeagueGameLog({season_type},T) + BoxScoreSummaryV2.officials",
        season_start=seasons[0],
        season_end=seasons[-1],
        rows=len(games),
        note=f"{season_type}; {n_missing_off} games missing officials",
    )
    return games, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull NBA games + officials (nba_api).")
    parser.add_argument("--season-start", type=int, default=None)
    parser.add_argument("--season-end", type=int, default=None)
    parser.add_argument("--season-type", default="Playoffs", choices=["Playoffs", "Regular Season"])
    parser.add_argument("--force-refresh", action="store_true")
    ns = parser.parse_args(argv)

    s = get_settings()
    seasons = (
        list(range(ns.season_start, ns.season_end + 1))
        if ns.season_start and ns.season_end
        else s.seasons
    )
    logger.info(
        "Pulling %s seasons %s (force_refresh=%s)", ns.season_type, seasons, ns.force_refresh
    )
    games, out = assemble_games(seasons, force_refresh=ns.force_refresh, season_type=ns.season_type)
    print(f"[pull] assembled {len(games)} {ns.season_type} games -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
