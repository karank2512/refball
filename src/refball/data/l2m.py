"""NBA Last Two Minute (L2M) officiating reports — direct, graded clutch-call evidence.

The NBA publicly grades every officiated event in the final ~2 minutes of close games
(margin within 3 since 2017-18) as Correct Call (CC), Correct Non-Call (CNC), Incorrect
Call (IC), or Incorrect Non-Call (INC). This is the most *direct* public measurement of
"did the crew make an error, and which team did it hurt" in exactly the moments that decide
close games.

We use the maintained, MIT-licensed tidy mirror (atlhawksfanatic/L2M). The raw
``errorInFavor`` column is empty in the dump, so we derive the beneficiary ourselves: for an
**error** (IC or INC), the ``disadvantaged_side`` is the team that was *hurt*, so an error
hurting the away team is an error that *favored* the home team.

Honest caveats (surfaced in the app + README):
* L2M only exists for games that were *close and late* — selection on the outcome we study.
* Grades attribute to the **crew**, not an individual official, and are the **NBA grading
  its own calls** (not an independent referee).
* It covers the last ~2 minutes only, not whole-game officiating.
"""

from __future__ import annotations

import argparse

from refball.config import get_settings
from refball.utils.cache import with_retries
from refball.utils.logging import get_logger
from refball.utils.provenance import log_source

logger = get_logger(__name__)

RAW_URL = (
    "https://raw.githubusercontent.com/atlhawksfanatic/L2M/master/1-tidy/L2M/L2M_stats_nba.csv"
)
ERROR_DECISIONS = {"IC", "INC"}  # incorrect call / incorrect non-call


def _raw_path():
    return get_settings().paths.external / "l2m_raw.csv"


def download_l2m(force_refresh: bool = False):
    """Ensure the raw L2M CSV is cached locally; return its path."""
    path = _raw_path()
    if path.exists() and not force_refresh:
        logger.info("L2M raw cache hit: %s", path)
        return path
    import requests

    s = get_settings()
    logger.info("Downloading L2M CSV from %s", RAW_URL)

    def _get():
        r = requests.get(
            RAW_URL, timeout=s.request_timeout_s, headers={"User-Agent": "refball-research"}
        )
        r.raise_for_status()
        return r.content

    content = with_retries(_get, what="L2M download")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    logger.info("Wrote %s (%.1f MB)", path, len(content) / 1e6)
    return path


def _norm_gid(x) -> str | None:
    """L2M GAME_ID is a float like 21400883.0 -> '0021400883' (10-digit NBA id)."""
    try:
        return str(int(float(x))).zfill(10)
    except (ValueError, TypeError):
        return None


def load_events(force_refresh: bool = False):
    """Load relevant L2M event columns with a normalized game_id and error flags."""
    import pandas as pd

    path = download_l2m(force_refresh)
    want = [
        "GAME_ID",
        "decision",
        "call_type",
        "playoff",
        "season",
        "committing_side",
        "disadvantaged_side",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in want, low_memory=False)
    df["game_id"] = df["GAME_ID"].map(_norm_gid)
    df = df.dropna(subset=["game_id"])
    df["decision"] = df["decision"].astype("string").str.strip()
    df["is_error"] = df["decision"].isin(ERROR_DECISIONS)
    df["disadvantaged_side"] = df["disadvantaged_side"].astype("string").str.lower().str.strip()
    return df


def build_game_table(force_refresh: bool = False):
    """Aggregate L2M events to one row per game, joinable on ``game_id``.

    Produces signed clutch-error counts. ``net_home_error_adv`` > 0 means the crew's
    incorrect calls/non-calls in the clutch *net-favored the home team*.
    """
    import pandas as pd

    ev = load_events(force_refresh)
    errors = ev[ev["is_error"]]

    def per_game(g: pd.DataFrame) -> pd.Series:
        err = g[g["is_error"]]
        against_home = int((err["disadvantaged_side"] == "home").sum())
        against_away = int((err["disadvantaged_side"] == "away").sum())
        return pd.Series(
            {
                "l2m_graded": int(len(g)),
                "l2m_errors": int(len(err)),
                "err_against_home": against_home,
                "err_against_away": against_away,
                # errors hurting away == errors favoring home
                "net_home_error_adv": against_away - against_home,
                "playoff": bool(
                    g["playoff"].astype("string").isin(["True", "TRUE", "1", "yes"]).any()
                ),
            }
        )

    table = ev.groupby("game_id", group_keys=False).apply(per_game).reset_index()
    s = get_settings()
    out = s.paths.processed / "l2m_game_table.parquet"
    table.to_parquet(out, index=False)

    n_err = int(errors["is_error"].sum())
    log_source(
        "L2M (atlhawksfanatic mirror)",
        RAW_URL,
        rows=len(table),
        note=f"{n_err} graded errors (IC+INC) across {len(table)} L2M games; MIT-licensed mirror",
    )
    logger.info("Wrote L2M game table: %s (%d games, %d total errors)", out, len(table), n_err)
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the L2M per-game clutch-error table.")
    p.add_argument("--force-refresh", action="store_true")
    ns = p.parse_args(argv)
    table = build_game_table(force_refresh=ns.force_refresh)

    import pandas as pd

    mt_path = get_settings().paths.modeling_table
    if mt_path.exists():
        mt = pd.read_parquet(mt_path)
        merged = mt.merge(table, on="game_id", how="inner")
        print("\n=== L2M COVERAGE QC ===")
        print(f"L2M games total:            {len(table)}")
        print(f"our modeling games:         {len(mt)}")
        print(
            f"covered by L2M (merged):    {len(merged)} ({100 * len(merged) / max(len(mt), 1):.1f}%)"
        )
        if len(merged):
            print(
                f"mean net home-error adv:    {merged['net_home_error_adv'].mean():+.3f} "
                f"(>0 => clutch errors net-favor home)"
            )
            print(f"total clutch errors:        {int(merged['l2m_errors'].sum())}")
            print(
                f"  against home / against away: {int(merged['err_against_home'].sum())} / "
                f"{int(merged['err_against_away'].sum())}"
            )
        print("========================")
    print(
        f"[l2m] wrote {len(table)} games -> {get_settings().paths.processed / 'l2m_game_table.parquet'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
