"""L2M parsing helpers (the game-id normalization is the easy thing to get wrong)."""

from __future__ import annotations

from refball.data.l2m import ERROR_DECISIONS, _norm_gid


def test_norm_gid_pads_to_10_digits():
    # L2M stores GAME_ID as a float (e.g. 21400883.0) -> 10-digit NBA id string.
    assert _norm_gid(21400883.0) == "0021400883"
    assert _norm_gid("41700405") == "0041700405"  # a playoff id
    assert _norm_gid(41700405) == "0041700405"


def test_norm_gid_handles_bad_input():
    assert _norm_gid(None) is None
    assert _norm_gid("not-a-number") is None
    assert _norm_gid(float("nan")) is None


def test_error_decisions_are_incorrect_only():
    # Errors are incorrect calls + incorrect non-calls; correct ones are not errors.
    assert ERROR_DECISIONS == {"IC", "INC"}
    assert "CC" not in ERROR_DECISIONS
    assert "CNC" not in ERROR_DECISIONS
