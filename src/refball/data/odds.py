"""Swappable odds adapter.

Odds coverage is the messiest part of this project, so it is isolated behind one loader
that always emits the canonical :data:`refball.schema.ODDS_COLUMNS` contract:

    game_date, home_tricode, away_tricode, spread_home, total_market, odds_source

Spread convention (enforced/declared here): ``spread_home`` is signed from the **home**
team's perspective. Negative => home favored; positive => home underdog.

Supported inputs:
* a local CSV path you pass explicitly (``--odds path.csv``),
* the synthetic odds file written by :mod:`refball.data.synthetic` (demo default),
* any user-provided CSV, as long as columns can be mapped via the alias table below.

The loader **validates** columns, normalizes team names, coerces numerics, and reports how
many rows it kept/dropped. It never invents odds and never joins on nba_api game_id.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from refball.config import get_settings
from refball.schema import ODDS_COLUMNS, validate_columns
from refball.utils.logging import get_logger
from refball.utils.teams import normalize_team

logger = get_logger(__name__)

# Canonical column -> candidate source headers (case-insensitive, normalized).
_ALIASES: dict[str, list[str]] = {
    "game_date": ["game_date", "date", "gamedate", "commence_date", "datetime", "game date"],
    "home_tricode": ["home_tricode", "home_team", "home", "hometeam", "home_abbr", "home team"],
    "away_tricode": [
        "away_tricode",
        "away_team",
        "away",
        "awayteam",
        "visitor",
        "road",
        "away team",
    ],
    "spread_home": ["spread_home", "home_spread", "spread", "home_line", "line", "handicap"],
    "total_market": ["total_market", "total", "over_under", "ou", "game_total", "totals"],
    "odds_source": ["odds_source", "source", "book", "sportsbook"],
}


def _resolve_columns(df) -> dict[str, str]:
    """Map normalized source headers to canonical names; return {canonical: source_col}."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    resolved: dict[str, str] = {}
    for canon, candidates in _ALIASES.items():
        for cand in candidates:
            if cand in lowered:
                resolved[canon] = lowered[cand]
                break
    return resolved


def load_odds_csv(path: str | Path) -> object:
    """Load + normalize one odds CSV into the canonical contract."""
    import pandas as pd

    path = Path(path)
    raw = pd.read_csv(path)
    resolved = _resolve_columns(raw)

    required = ["game_date", "home_tricode", "away_tricode", "spread_home", "total_market"]
    missing = [c for c in required if c not in resolved]
    if missing:
        raise ValueError(
            f"Odds file {path} cannot be mapped to required fields {missing}. "
            f"Found columns: {list(raw.columns)}. Extend _ALIASES in data/odds.py if needed."
        )

    out = pd.DataFrame()
    out["game_date"] = pd.to_datetime(raw[resolved["game_date"]], errors="coerce").dt.date
    out["home_tricode"] = raw[resolved["home_tricode"]].map(lambda x: normalize_team(x))
    out["away_tricode"] = raw[resolved["away_tricode"]].map(lambda x: normalize_team(x))
    out["spread_home"] = pd.to_numeric(raw[resolved["spread_home"]], errors="coerce")
    out["total_market"] = pd.to_numeric(raw[resolved["total_market"]], errors="coerce")
    out["odds_source"] = raw[resolved["odds_source"]] if "odds_source" in resolved else path.stem

    n0 = len(out)
    out = out.dropna(
        subset=["game_date", "home_tricode", "away_tricode", "spread_home", "total_market"]
    )
    out = out.drop_duplicates(subset=["game_date", "home_tricode", "away_tricode"])
    dropped = n0 - len(out)
    if dropped:
        logger.warning(
            "Odds loader dropped %d/%d rows (missing/dup/unmappable teams).", dropped, n0
        )

    validate_columns(out, ODDS_COLUMNS, "odds table")
    return out[ODDS_COLUMNS]


# Public 10-year games+odds archive (MIT-licensed scrape of sportsbookreview); has
# home_close_spread + close_over_under per game. Covers seasons ~2011-2021.
ODDS_ARCHIVE_URL = "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/nba_archive_10Y.json"


def download_odds_archive(force_refresh: bool = False) -> Path:
    """Cache the public MIT-licensed NBA closing-odds archive locally."""
    path = get_settings().paths.external / "nba_odds_archive.json"
    if path.exists() and not force_refresh:
        return path
    import requests

    logger.info("Downloading NBA odds archive: %s", ODDS_ARCHIVE_URL)
    r = requests.get(ODDS_ARCHIVE_URL, timeout=60, headers={"User-Agent": "refball-research"})
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def load_odds_archive_json(path: str | Path) -> object:
    """Parse the sportsbookreview 10Y archive JSON into the canonical odds contract.

    Uses the **closing** line (``home_close_spread`` is already home-perspective: negative =
    home favored) and ``close_over_under`` as the total.
    """
    import json

    import pandas as pd

    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for r in recs:
        try:
            gd = pd.to_datetime(str(int(float(r["date"]))), format="%Y%m%d", errors="coerce")
        except (ValueError, TypeError, KeyError):
            continue
        rows.append(
            {
                "game_date": gd.date() if pd.notna(gd) else None,
                "home_tricode": normalize_team(r.get("home_team")),
                "away_tricode": normalize_team(r.get("away_team")),
                "spread_home": pd.to_numeric(r.get("home_close_spread"), errors="coerce"),
                "total_market": pd.to_numeric(r.get("close_over_under"), errors="coerce"),
                "odds_source": "sportsbookreview_archive",
            }
        )
    out = pd.DataFrame(rows)
    n0 = len(out)
    out = out.dropna(
        subset=["game_date", "home_tricode", "away_tricode", "spread_home", "total_market"]
    )
    out = out.drop_duplicates(subset=["game_date", "home_tricode", "away_tricode"])
    logger.info("Odds archive: %d usable rows (from %d records)", len(out), n0)
    validate_columns(out, ODDS_COLUMNS, "odds archive")
    return out[ODDS_COLUMNS]


def load_odds(path: str | Path | None = None) -> object:
    """Resolve an odds source. Returns an empty (but correctly-typed) frame if none found.

    Precedence: explicit ``path`` > synthetic demo file > empty. ``.json`` paths are parsed as
    the sportsbookreview archive; ``.csv`` via the alias-mapping loader.
    Callers decide whether to run the with-odds or no-odds model based on row count.
    """
    import pandas as pd

    s = get_settings()
    if path is not None:
        logger.info("Loading odds from explicit path: %s", path)
        return load_odds_archive_json(path) if str(path).endswith(".json") else load_odds_csv(path)
    if s.paths.odds_synthetic.exists():
        logger.info("Loading synthetic demo odds: %s", s.paths.odds_synthetic)
        return load_odds_csv(s.paths.odds_synthetic)
    logger.warning(
        "No odds source found. The pipeline will run a NO-ODDS model and flag it as weaker. "
        "Provide one with --odds <file.csv> or generate demo data with refball.data.synthetic."
    )
    return pd.DataFrame(columns=ODDS_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/normalize an odds CSV.")
    parser.add_argument("--odds", type=str, default=None, help="path to odds CSV")
    ns = parser.parse_args(argv)
    df = load_odds(ns.odds)
    print(f"[odds] rows={len(df)} sources={sorted(df['odds_source'].unique()) if len(df) else []}")
    if len(df):
        print(df.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
