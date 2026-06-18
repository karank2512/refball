"""Turn the modeling table into model-ready arrays, indices, and coords.

Centralizes the bookkeeping shared by Stage 1 (foul models) and Stage 2 (point-diff):
team indices (home/away), season index, the multi-membership referee matrix, the
log-possessions offset, and standardized market controls. Subsetting (officials-only,
odds-only) happens here so the model code stays declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from refball.config import get_settings
from refball.features.multimembership import RefIndex, build_membership_matrix, build_ref_index
from refball.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelData:
    table: object  # the (possibly subset) pandas DataFrame
    n_games: int
    team_codes: list[str]
    season_values: list[int]
    home_team_idx: np.ndarray
    away_team_idx: np.ndarray
    season_idx: np.ndarray
    R: np.ndarray  # [n_games, n_refs] multi-membership weights
    k: np.ndarray
    ref_index: RefIndex
    log_possessions: np.ndarray
    use_odds: bool
    z_spread_home: np.ndarray | None = None
    z_total_market: np.ndarray | None = None
    # observed
    total_fouls: np.ndarray = field(default=None)
    home_pf: np.ndarray = field(default=None)
    away_pf: np.ndarray = field(default=None)
    point_diff_home: np.ndarray = field(default=None)
    foul_diff_home: np.ndarray = field(default=None)
    fta_diff_home: np.ndarray = field(default=None)
    ftm_diff_home: np.ndarray = field(default=None)

    @property
    def n_refs(self) -> int:
        return len(self.ref_index.ref_ids)

    @property
    def coords(self) -> dict:
        return {
            "team": self.team_codes,
            "season": [str(s) for s in self.season_values],
            "ref": [str(r) for r in self.ref_index.ref_ids],
            "game": np.arange(self.n_games),
        }


def load_table():
    import pandas as pd

    s = get_settings()
    if not s.paths.modeling_table.exists():
        raise FileNotFoundError(
            "modeling_table.parquet not found. Run build_table (after pull/synthetic)."
        )
    return pd.read_parquet(s.paths.modeling_table)


def prepare(
    *,
    use_odds: bool = True,
    require_officials: bool = True,
    high_conf_odds_only: bool = False,
    table=None,
) -> ModelData:
    """Build :class:`ModelData`.

    * ``require_officials`` keeps only games with at least one usable official id (needed
      for referee models). Non-referee EDA can ignore this.
    * ``use_odds`` keeps only games with odds and includes the market controls. With
      ``use_odds=False`` the no-odds (weaker) model uses all rows and no spread/total terms.
    * ``high_conf_odds_only`` restricts further to high-confidence odds matches
      (sensitivity model C).
    """

    df = load_table() if table is None else table.copy()
    df = df.reset_index(drop=True)

    if require_officials:
        df = df[df["has_officials"]].reset_index(drop=True)
    if use_odds:
        df = df[df["has_odds"]].reset_index(drop=True)
        if high_conf_odds_only and "odds_high_conf" in df.columns:
            df = df[df["odds_high_conf"]].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows left after subsetting; check officials/odds coverage.")

    team_codes = sorted(set(df["home_tricode"]) | set(df["away_tricode"]))
    team_to_idx = {t: i for i, t in enumerate(team_codes)}
    season_values = sorted(df["season"].unique().tolist())
    season_to_idx = {s: i for i, s in enumerate(season_values)}

    ref_index = build_ref_index(df)
    R, k = build_membership_matrix(df, ref_index)

    md = ModelData(
        table=df,
        n_games=len(df),
        team_codes=team_codes,
        season_values=season_values,
        home_team_idx=df["home_tricode"].map(team_to_idx).to_numpy(),
        away_team_idx=df["away_tricode"].map(team_to_idx).to_numpy(),
        season_idx=df["season"].map(season_to_idx).to_numpy(),
        R=R,
        k=k,
        ref_index=ref_index,
        log_possessions=df["log_possessions"].to_numpy(dtype=float),
        use_odds=use_odds,
        z_spread_home=df["z_spread_home"].to_numpy(dtype=float) if use_odds else None,
        z_total_market=df["z_total_market"].to_numpy(dtype=float) if use_odds else None,
        total_fouls=df["total_fouls"].to_numpy(dtype=int),
        home_pf=df["home_pf"].to_numpy(dtype=int),
        away_pf=df["away_pf"].to_numpy(dtype=int),
        point_diff_home=df["point_diff_home"].to_numpy(dtype=float),
        foul_diff_home=df["foul_diff_home"].to_numpy(dtype=float),
        fta_diff_home=df["fta_diff_home"].to_numpy(dtype=float),
        ftm_diff_home=df["ftm_diff_home"].to_numpy(dtype=float),
    )
    logger.info(
        "prepared ModelData: games=%d teams=%d seasons=%d refs=%d use_odds=%s",
        md.n_games,
        len(team_codes),
        len(season_values),
        md.n_refs,
        use_odds,
    )
    return md
