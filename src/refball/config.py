"""Central configuration for the Ref Ball pipeline.

All paths, season coverage, network politeness knobs, QC thresholds, and MCMC sampler
settings live here so the rest of the codebase stays free of magic constants. Values can
be overridden with environment variables prefixed ``REFBALL_`` (e.g. ``REFBALL_SEASON_START``)
or via a local ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Repo root = two levels up from this file (src/refball/config.py -> repo)."""
    return Path(__file__).resolve().parents[2]


class Paths:
    """Filesystem layout. Created on demand via :meth:`ensure`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data = root / "data"
        self.raw = self.data / "raw"
        self.interim = self.data / "interim"
        self.processed = self.data / "processed"
        self.external = self.data / "external"
        self.models = root / "models"
        self.reports = root / "reports"
        self.figures = self.reports / "figures"

    # Assembled (pre-join) games + odds inputs ----------------------------
    @property
    def games_interim(self) -> Path:
        """Assembled one-row-per-game table; written by either the real pull or synthetic."""
        return self.interim / "games.parquet"

    @property
    def games_regular_interim(self) -> Path:
        """Regular-season assembled games (kept separate from the playoff table)."""
        return self.interim / "games_regular.parquet"

    @property
    def odds_synthetic(self) -> Path:
        return self.external / "odds_synthetic.csv"

    @property
    def synthetic_truth(self) -> Path:
        return self.external / "synthetic_truth.json"

    # Canonical processed artifacts ---------------------------------------
    @property
    def modeling_table(self) -> Path:
        return self.processed / "modeling_table.parquet"

    @property
    def provenance_log(self) -> Path:
        return self.processed / "source_provenance.jsonl"

    @property
    def stage1_total_nc(self) -> Path:
        return self.models / "stage1_total.nc"

    @property
    def stage1_lean_nc(self) -> Path:
        return self.models / "stage1_lean.nc"

    @property
    def stage2_nc(self) -> Path:
        return self.models / "stage2_pointdiff.nc"

    @property
    def referee_stage1_effects(self) -> Path:
        return self.processed / "referee_stage1_effects.parquet"

    @property
    def stage2_coefficients(self) -> Path:
        return self.processed / "stage2_coefficients.parquet"

    @property
    def referee_mediated_effects(self) -> Path:
        return self.processed / "referee_mediated_effects.parquet"

    @property
    def robustness_dir(self) -> Path:
        return self.processed / "robustness"

    def ensure(self) -> None:
        for p in (
            self.raw,
            self.interim,
            self.processed,
            self.external,
            self.models,
            self.figures,
            self.robustness_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


class SamplerConfig(BaseSettings):
    """NUTS sampler knobs. ``quick`` swaps in fast-but-rough settings for demos/CI."""

    model_config = SettingsConfigDict(env_prefix="REFBALL_MCMC_", extra="ignore")

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    cores: int = 4
    target_accept: float = 0.95
    random_seed: int = 20240101

    # Quick mode (demo / smoke / CI): tiny but still produces a valid InferenceData.
    quick_draws: int = 300
    quick_tune: int = 300
    quick_chains: int = 2
    quick_cores: int = 2

    def resolve(self, quick: bool) -> dict:
        """Return kwargs suitable for ``pm.sample``."""
        if quick:
            return dict(
                draws=self.quick_draws,
                tune=self.quick_tune,
                chains=self.quick_chains,
                cores=self.quick_cores,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
            )
        return dict(
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            cores=self.cores,
            target_accept=self.target_accept,
            random_seed=self.random_seed,
        )


class Settings(BaseSettings):
    """Top-level project settings."""

    model_config = SettingsConfigDict(
        env_prefix="REFBALL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Coverage --------------------------------------------------------------
    # season_start/season_end are the *starting calendar year* of each NBA season.
    # 2017 => 2016-17 season .. 2024 => 2024-25 season (playoffs played in 2025).
    season_start: int = 2017
    season_end: int = 2024

    # Network politeness for nba_api / scrapers ----------------------------
    request_timeout_s: float = 30.0
    request_min_sleep_s: float = 0.6
    request_max_retries: int = 5
    request_backoff_base_s: float = 1.5

    # QC gates --------------------------------------------------------------
    min_join_match_rate: float = 0.95  # below this, build_table warns loudly
    hdi_prob: float = 0.94

    # Modeling --------------------------------------------------------------
    # Officials per game used for multi-membership weighting. NBA crews are 3.
    officials_per_game: int = 3
    # Drop referees with fewer than this many games from the *ranking display*
    # (they are still in the model and shrunk toward zero by partial pooling).
    min_games_for_display: int = 3

    # Priors (documented in methodology). Hard regularization on purpose.
    sigma_ref_total_prior: float = 0.25  # HalfNormal sd for total-foul ref effects
    sigma_ref_lean_prior: float = 0.20  # HalfNormal sd for directional lean
    sigma_team_prior: float = 0.25
    sigma_season_prior: float = 0.20

    sampler: SamplerConfig = Field(default_factory=SamplerConfig)

    @property
    def seasons(self) -> list[int]:
        return list(range(self.season_start, self.season_end + 1))

    @property
    def paths(self) -> Paths:
        return Paths(_project_root())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    s = Settings()
    s.paths.ensure()
    return s


def season_str(start_year: int) -> str:
    """Convert a starting calendar year to the NBA 'YYYY-YY' string (2017 -> '2017-18')."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"
