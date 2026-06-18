"""Synthetic playoff-game generator for demo / CI / fast app startup.

This fabricates a *plausible* dataset with **known planted structure** so that:

* the full pipeline (build-table -> models -> mediation -> app) runs offline in seconds,
* unit tests can assert the models recover the planted signal in the right direction,
* the website has something to render without an hours-long real fit.

It is **not** real data and must never be presented as such. The planted referee effects
are random draws, not statements about any real official. The generator also writes the
ground-truth parameters to ``data/external/synthetic_truth.json`` for validation.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import numpy as np

from refball.config import get_settings
from refball.schema import GAME_COLUMNS, ODDS_COLUMNS
from refball.utils.logging import get_logger
from refball.utils.provenance import log_source
from refball.utils.teams import CANONICAL

logger = get_logger(__name__)

# A pool of fictional officials. Names are invented; any resemblance is coincidental.
_REF_FIRST = [
    "Alex",
    "Blake",
    "Casey",
    "Dana",
    "Eli",
    "Frankie",
    "Gail",
    "Harper",
    "Ira",
    "Jules",
    "Kim",
    "Lee",
    "Morgan",
    "Nas",
    "Oak",
    "Park",
    "Quinn",
    "Reese",
    "Sage",
    "Toni",
    "Umi",
    "Val",
    "Wren",
    "Xan",
    "Yuki",
    "Zane",
    "Ari",
    "Bex",
    "Cory",
    "Drew",
    "Emer",
    "Finn",
    "Glen",
    "Hollis",
    "Indy",
    "Jaye",
]
_REF_LAST = [
    "Adams",
    "Boone",
    "Cruz",
    "Diaz",
    "Ellis",
    "Ford",
    "Greer",
    "Hahn",
    "Ito",
    "Jain",
    "Koch",
    "Long",
    "Mora",
    "Nash",
    "Ocampo",
    "Pratt",
    "Quill",
    "Ruiz",
    "Singh",
    "Tran",
    "Udall",
    "Vega",
    "Walsh",
    "Xiong",
    "York",
    "Zhao",
    "Abara",
    "Best",
    "Choi",
    "Dent",
    "Estes",
    "Fritz",
    "Gomez",
    "Hwang",
    "Imani",
    "Jotto",
]


def _make_referees(rng: np.random.Generator, n_refs: int) -> dict:
    names = [f"{_REF_FIRST[i]} {_REF_LAST[i]}" for i in range(n_refs)]
    ids = [9000 + i for i in range(n_refs)]
    # Planted *true* per-ref effects (log scale). Most refs ~0; a few are mild outliers.
    total_eff = rng.normal(0.0, 0.05, size=n_refs)
    lean_eff = rng.normal(0.0, 0.04, size=n_refs)
    # Plant 3 deliberate outliers so models have a clear target to recover.
    total_eff[0] += 0.18  # ref 0: more total fouls
    lean_eff[1] += 0.14  # ref 1: more home fouls (positive lean)
    lean_eff[2] -= 0.14  # ref 2: fewer home fouls (negative lean)
    return {"names": names, "ids": ids, "total_eff": total_eff, "lean_eff": lean_eff}


def generate(
    *,
    seasons: list[int] | None = None,
    games_per_season: int = 80,
    n_refs: int = 36,
    seed: int = 7,
    true_gamma_foul_diff: float = -0.45,
) -> tuple[object, object, dict]:
    """Return (games_df, odds_df, truth_dict).

    ``true_gamma_foul_diff`` is the planted association between home foul margin and home
    point differential: each extra home foul (relative to away) costs the home team about
    0.45 points on average, so Stage 2 has a real, signed signal to recover.
    """
    import pandas as pd

    s = get_settings()
    seasons = seasons or s.seasons
    rng = np.random.default_rng(seed)

    refs = _make_referees(rng, n_refs)
    tris = list(CANONICAL.keys())

    # Team latent strengths (point-differential scale) and foul tendencies.
    team_strength = {t: rng.normal(0.0, 4.0) for t in tris}
    team_commit = {t: rng.normal(0.0, 0.06) for t in tris}  # propensity to commit fouls
    team_draw = {t: rng.normal(0.0, 0.06) for t in tris}  # propensity to draw fouls
    season_eff = {yr: rng.normal(0.0, 0.04) for yr in seasons}

    base_pf = 21.0  # mean personal fouls per team per game
    home_adv = 2.6  # points
    home_foul_edge = -0.03  # home teams whistled slightly less on average (log scale)

    rows = []
    odds_rows = []
    gid_counter = 1
    for yr in seasons:
        # Round structure: more first-round games than later rounds.
        round_weights = np.array([0.5, 0.27, 0.15, 0.08])
        rounds = rng.choice([1, 2, 3, 4], size=games_per_season, p=round_weights)
        for g in range(games_per_season):
            home, away = rng.choice(tris, size=2, replace=False)
            crew = rng.choice(n_refs, size=s.officials_per_game, replace=False)
            crew_total = float(refs["total_eff"][crew].mean())
            crew_lean = float(refs["lean_eff"][crew].mean())

            poss = float(rng.normal(95.0, 6.0))
            log_poss = np.log(max(poss, 60.0))

            # --- Fouls (Poisson with log-linear mean) ---
            mu_home_pf = base_pf * np.exp(
                (log_poss - np.log(95.0))
                + crew_total
                + crew_lean
                + home_foul_edge
                + team_commit[home]
                + team_draw[away]
                + season_eff[yr]
            )
            mu_away_pf = base_pf * np.exp(
                (log_poss - np.log(95.0))
                + crew_total
                - crew_lean
                + team_commit[away]
                + team_draw[home]
                + season_eff[yr]
            )
            home_pf = int(rng.poisson(max(mu_home_pf, 1.0)))
            away_pf = int(rng.poisson(max(mu_away_pf, 1.0)))

            # --- Free throws: opponent fouls drive your attempts ---
            home_fta = int(rng.poisson(max(0.95 * away_pf, 1.0)))
            away_fta = int(rng.poisson(max(0.95 * home_pf, 1.0)))
            home_ftm = int(rng.binomial(home_fta, 0.77)) if home_fta else 0
            away_ftm = int(rng.binomial(away_fta, 0.77)) if away_fta else 0

            foul_diff_home = home_pf - away_pf

            # --- Scores: strength + home edge + foul-margin channel + noise ---
            base_diff = (
                team_strength[home]
                - team_strength[away]
                + home_adv
                + true_gamma_foul_diff * foul_diff_home
                + 0.55 * (home_ftm - away_ftm)
                + rng.normal(0.0, 9.0)
            )
            total_points = float(rng.normal(212.0, 14.0)) + 0.1 * poss
            home_score = int(round(total_points / 2 + base_diff / 2))
            away_score = int(round(total_points / 2 - base_diff / 2))

            # --- Box-score components for possession formula ---
            home_fga = int(rng.normal(86, 6))
            away_fga = int(rng.normal(86, 6))
            home_oreb = int(rng.normal(10, 3))
            away_oreb = int(rng.normal(10, 3))
            home_tov = int(rng.normal(13, 3))
            away_tov = int(rng.normal(13, 3))

            gdate = date(yr + 1, 4, 15) + timedelta(days=int(rng.integers(0, 60)))
            gid = f"00{4}{str(yr)[-2:]}{gid_counter:05d}"
            gid_counter += 1

            rows.append(
                {
                    "game_id": gid,
                    "season": yr,
                    "season_str": f"{yr}-{str(yr + 1)[-2:]}",
                    "playoff_round": int(rounds[g]),
                    "game_date": gdate,
                    "home_tricode": home,
                    "away_tricode": away,
                    "home_team": CANONICAL[home],
                    "away_team": CANONICAL[away],
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_pf": home_pf,
                    "away_pf": away_pf,
                    "home_fta": home_fta,
                    "away_fta": away_fta,
                    "home_ftm": home_ftm,
                    "away_ftm": away_ftm,
                    "home_fga": home_fga,
                    "away_fga": away_fga,
                    "home_oreb": home_oreb,
                    "away_oreb": away_oreb,
                    "home_tov": home_tov,
                    "away_tov": away_tov,
                    "official_1": refs["names"][crew[0]],
                    "official_2": refs["names"][crew[1]],
                    "official_3": refs["names"][crew[2]],
                    "ref_1_id": refs["ids"][crew[0]],
                    "ref_2_id": refs["ids"][crew[1]],
                    "ref_3_id": refs["ids"][crew[2]],
                }
            )

            # --- Odds: noisy reflection of true expected diff/total (home perspective) ---
            exp_diff = team_strength[home] - team_strength[away] + home_adv
            spread_home = -float(np.round((exp_diff + rng.normal(0, 1.5)) * 2) / 2)
            total_market = float(np.round((212.0 + rng.normal(0, 4)) * 2) / 2)
            odds_rows.append(
                {
                    "game_date": gdate,
                    "home_tricode": home,
                    "away_tricode": away,
                    "spread_home": spread_home,
                    "total_market": total_market,
                    "odds_source": "synthetic",
                }
            )

    games_df = pd.DataFrame(rows)[GAME_COLUMNS]
    odds_df = pd.DataFrame(odds_rows)[ODDS_COLUMNS]
    truth = {
        "true_gamma_foul_diff": true_gamma_foul_diff,
        "ref_names": refs["names"],
        "ref_ids": refs["ids"],
        "true_total_eff": refs["total_eff"].tolist(),
        "true_lean_eff": refs["lean_eff"].tolist(),
        "planted_outliers": {
            "high_total_fouls_ref": refs["names"][0],
            "positive_lean_ref": refs["names"][1],
            "negative_lean_ref": refs["names"][2],
        },
        "n_games": int(len(games_df)),
        "seasons": seasons,
    }
    return games_df, odds_df, truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic playoff data for demo mode.")
    parser.add_argument("--season-start", type=int, default=None)
    parser.add_argument("--season-end", type=int, default=None)
    parser.add_argument("--games-per-season", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    ns = parser.parse_args(argv)

    s = get_settings()
    seasons = (
        list(range(ns.season_start, ns.season_end + 1))
        if ns.season_start and ns.season_end
        else s.seasons
    )
    games_df, odds_df, truth = generate(
        seasons=seasons, games_per_season=ns.games_per_season, seed=ns.seed
    )

    s.paths.ensure()
    games_df.to_parquet(s.paths.games_interim, index=False)
    odds_df.to_csv(s.paths.odds_synthetic, index=False)
    s.paths.synthetic_truth.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    log_source(
        "synthetic",
        "refball.data.synthetic.generate",
        season_start=seasons[0],
        season_end=seasons[-1],
        rows=len(games_df),
        note="Fabricated demo data with planted referee structure. NOT real.",
    )
    logger.info(
        "Synthetic data written: %d games, %d odds rows -> %s",
        len(games_df),
        len(odds_df),
        s.paths.games_interim,
    )
    print(f"[synthetic] games={len(games_df)} odds={len(odds_df)} -> {s.paths.games_interim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
