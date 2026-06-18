"""Multi-membership referee structure.

A playoff game has up to three officials. We do **not** assign the whole game to a single
"crew chief", and we do **not** duplicate the game three times as if officials were
independent observations. Instead each game contributes a row of weights to a
``[n_games, n_refs]`` membership matrix ``R`` where each assigned official gets weight
``1/k`` (k = officials actually known for that game, normally 3). A crew-level effect is
then ``crew_effect = R @ ref_effect`` — the *average* of the assigned officials' effects.

This is the standard multi-membership construction (Browne et al.); it keeps small-sample
referees shrunk toward zero via the shared ``sigma`` prior in the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from refball.utils.logging import get_logger

logger = get_logger(__name__)

_ID_COLS = ["ref_1_id", "ref_2_id", "ref_3_id"]
_NAME_COLS = ["official_1", "official_2", "official_3"]


@dataclass(slots=True)
class RefIndex:
    ref_ids: list[int]  # canonical ordering of referee ids
    id_to_idx: dict[int, int]  # ref_id -> column index in R
    names: dict[int, str]  # ref_id -> display name
    games_count: dict[int, int]  # ref_id -> number of games worked (in this table)


def _clean_id(val) -> int | None:
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
    except TypeError:
        pass
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def build_ref_index(df) -> RefIndex:
    """Build the referee catalogue from the modeling table's official id/name columns."""
    counts: dict[int, int] = {}
    names: dict[int, str] = {}
    for _, row in df.iterrows():
        for id_col, name_col in zip(_ID_COLS, _NAME_COLS, strict=True):
            rid = _clean_id(row.get(id_col))
            if rid is None:
                continue
            counts[rid] = counts.get(rid, 0) + 1
            nm = row.get(name_col)
            if isinstance(nm, str) and nm.strip():
                names.setdefault(rid, nm.strip())
    ref_ids = sorted(counts)
    id_to_idx = {rid: i for i, rid in enumerate(ref_ids)}
    logger.info("Referee index: %d officials across %d games", len(ref_ids), len(df))
    return RefIndex(
        ref_ids=ref_ids,
        id_to_idx=id_to_idx,
        names={rid: names.get(rid, f"Ref {rid}") for rid in ref_ids},
        games_count=counts,
    )


def build_membership_matrix(df, index: RefIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, k)`` where R is ``[n_games, n_refs]`` weights and k is officials-per-game.

    Each known official in a game gets weight ``1/k`` for that game.
    """
    n_games = len(df)
    n_refs = len(index.ref_ids)
    R = np.zeros((n_games, n_refs), dtype=float)
    k = np.zeros(n_games, dtype=float)

    for gi, (_, row) in enumerate(df.iterrows()):
        present = [
            index.id_to_idx[rid]
            for rid in (_clean_id(row.get(c)) for c in _ID_COLS)
            if rid is not None and rid in index.id_to_idx
        ]
        if not present:
            continue
        w = 1.0 / len(present)
        for col in present:
            R[gi, col] = w
        k[gi] = len(present)

    if (k == 0).any():
        logger.warning(
            "%d games have no usable officials and contribute no crew effect.", int((k == 0).sum())
        )
    return R, k


def games_with_officials_mask(df) -> np.ndarray:
    """Boolean mask of games with at least one usable official id."""

    def has_any(row) -> bool:
        return any(_clean_id(row.get(c)) is not None for c in _ID_COLS)

    return df.apply(has_any, axis=1).to_numpy()
