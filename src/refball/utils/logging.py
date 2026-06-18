"""Project logging. One configured root logger; modules call :func:`get_logger`."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("REFBALL_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("refball")
    root.setLevel(level)
    root.handlers[:] = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the ``refball`` logger."""
    _configure_root()
    if not name.startswith("refball"):
        name = f"refball.{name}"
    return logging.getLogger(name)
