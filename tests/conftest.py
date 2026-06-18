"""Shared test fixtures: small in-memory synthetic dataset + assembled modeling table."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def synth():
    from refball.data.synthetic import generate

    games, odds, truth = generate(seasons=[2020, 2021], games_per_season=20, seed=11)
    return games, odds, truth


@pytest.fixture(scope="session")
def modeling_table(synth):
    from refball.features.build_table import assemble_modeling_table

    games, odds, _ = synth
    return assemble_modeling_table(games, odds)
