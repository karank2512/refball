"""Team-name normalization.

Different sources spell teams differently: nba_api uses 'Golden State Warriors',
odds feeds use 'GS Warriors' / 'LA Clippers' / 'Brooklyn', box scores use tricodes.
We normalize everything to a canonical 3-letter tricode so joins on
(date, home_tricode, away_tricode) are reliable.

Coverage window is 2016-17 .. 2024-25; no franchise relocations occur in that window,
so a static map is sufficient. Add aliases here rather than special-casing at call sites.
"""

from __future__ import annotations

import re

from refball.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical tricode -> full display name.
CANONICAL: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def _norm(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[._'`]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Extra aliases beyond {tricode, full name, city, nickname}, keyed by normalized form.
_EXTRA_ALIASES: dict[str, str] = {
    "gs warriors": "GSW",
    "gs": "GSW",
    "golden state": "GSW",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "clippers": "LAC",
    "la lakers": "LAL",
    "los angeles lakers": "LAL",
    "lakers": "LAL",
    "ny knicks": "NYK",
    "new york": "NYK",
    "knicks": "NYK",
    "brooklyn": "BKN",
    "nets": "BKN",
    "ny nets": "BKN",
    "sa spurs": "SAS",
    "san antonio": "SAS",
    "spurs": "SAS",
    "no pelicans": "NOP",
    "new orleans": "NOP",
    "pelicans": "NOP",
    "okc thunder": "OKC",
    "oklahoma city": "OKC",
    "thunder": "OKC",
    "utah": "UTA",
    "jazz": "UTA",
    "phoenix": "PHX",
    "phx suns": "PHX",
    "suns": "PHX",
    "portland": "POR",
    "blazers": "POR",
    "trail blazers": "POR",
    "philadelphia": "PHI",
    "philadelphia sixers": "PHI",
    "sixers": "PHI",
    "76ers": "PHI",
    "phila": "PHI",
    "charlotte": "CHA",
    "hornets": "CHA",
    "washington": "WAS",
    "wizards": "WAS",
    "wsh": "WAS",
    "indiana": "IND",
    "pacers": "IND",
    "milwaukee": "MIL",
    "bucks": "MIL",
    "minnesota": "MIN",
    "timberwolves": "MIN",
    "wolves": "MIN",
    "memphis": "MEM",
    "grizzlies": "MEM",
    "denver": "DEN",
    "nuggets": "DEN",
    "sacramento": "SAC",
    "kings": "SAC",
    "dallas": "DAL",
    "mavericks": "DAL",
    "mavs": "DAL",
    "houston": "HOU",
    "rockets": "HOU",
    "atlanta": "ATL",
    "hawks": "ATL",
    "boston": "BOS",
    "celtics": "BOS",
    "chicago": "CHI",
    "bulls": "CHI",
    "cleveland": "CLE",
    "cavaliers": "CLE",
    "cavs": "CLE",
    "detroit": "DET",
    "pistons": "DET",
    "miami": "MIA",
    "heat": "MIA",
    "orlando": "ORL",
    "magic": "ORL",
    "toronto": "TOR",
    "raptors": "TOR",
}


def _build_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for tri, full in CANONICAL.items():
        idx[_norm(tri)] = tri
        idx[_norm(full)] = tri
        # city = everything but last word(s) for nickname; add full city token chains
        parts = full.split()
        nickname = parts[-1]
        city = " ".join(parts[:-1])
        idx.setdefault(_norm(nickname), tri)
        idx.setdefault(_norm(city), tri)
    idx.update(_EXTRA_ALIASES)
    return idx


_INDEX = _build_index()


def normalize_team(name: str | None, *, strict: bool = False) -> str | None:
    """Map any reasonable team spelling to a canonical tricode.

    Returns ``None`` (and logs a warning) for unknown names unless ``strict``,
    in which case a ``KeyError`` is raised. Callers that join across sources
    should treat ``None`` as an unmatched row, not silently drop it.
    """
    if name is None:
        return None
    key = _norm(str(name))
    if key in _INDEX:
        return _INDEX[key]
    # Last resort: a contained nickname (e.g. "Los Angeles Lakers (LAL)").
    for alias, tri in _INDEX.items():
        if len(alias) >= 4 and alias in key:
            return tri
    if strict:
        raise KeyError(f"Unknown team name: {name!r}")
    logger.warning("Could not normalize team name: %r", name)
    return None


def is_known(tricode: str) -> bool:
    return tricode in CANONICAL
