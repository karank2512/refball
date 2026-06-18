"""Caching + polite-HTTP helpers shared by the data layer.

Raw pulls are cached as parquet (tabular) or JSON (anything else) under ``data/raw``.
Re-running the pipeline is cheap and offline unless ``force_refresh=True``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from refball.config import get_settings
from refball.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def raw_path(*parts: str) -> Path:
    """Build a path under ``data/raw`` (parents created)."""
    p = get_settings().paths.raw.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cached_parquet(path: Path, builder: Callable[[], Any], *, force_refresh: bool = False):
    """Return a DataFrame from ``path``, building + caching it on a miss.

    ``builder`` must return a pandas DataFrame. Imported lazily so non-data code paths
    don't require pandas.
    """
    import pandas as pd

    if path.exists() and not force_refresh:
        logger.debug("cache hit (parquet): %s", path)
        return pd.read_parquet(path)
    logger.info("cache miss (parquet): %s -> building", path)
    df = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def cached_json(path: Path, builder: Callable[[], T], *, force_refresh: bool = False) -> T:
    """Return JSON-serialisable data from ``path``, building + caching it on a miss."""
    if path.exists() and not force_refresh:
        logger.debug("cache hit (json): %s", path)
        return json.loads(path.read_text(encoding="utf-8"))
    logger.info("cache miss (json): %s -> building", path)
    data = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    return data


def polite_sleep() -> None:
    """Minimum inter-request pause for public endpoints."""
    time.sleep(get_settings().request_min_sleep_s)


def _has_tenacity() -> bool:
    try:
        import tenacity  # noqa: F401
    except ImportError:
        return False
    return True


def with_retries(fn: Callable[[], T], *, what: str = "request") -> T:
    """Call ``fn`` with exponential backoff. Uses tenacity if present, else a manual loop."""
    s = get_settings()
    if _has_tenacity():
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        @retry(
            reraise=True,
            stop=stop_after_attempt(s.request_max_retries),
            wait=wait_exponential(multiplier=s.request_backoff_base_s, min=1, max=30),
            retry=retry_if_exception_type(Exception),
        )
        def _runner() -> T:
            return fn()

        return _runner()

    # Manual fallback (not nested in an except clause, so a bare re-raise is clean).
    for attempt in range(1, s.request_max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad for network calls
            if attempt >= s.request_max_retries:
                raise
            wait = s.request_backoff_base_s**attempt
            logger.warning(
                "%s failed (attempt %d): %s; retrying in %.1fs", what, attempt, exc, wait
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover
