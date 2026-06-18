"""Source provenance logging.

Every raw pull appends one record so the dataset is auditable: what source, which
endpoint/URL, when it was accessed, the season range, and the row count. The Data
Coverage page in the app renders this table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from refball.config import get_settings
from refball.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SourceRecord:
    source: str
    endpoint: str
    access_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    season_start: int | None = None
    season_end: int | None = None
    rows: int | None = None
    note: str = ""


def log_source(
    source: str,
    endpoint: str,
    *,
    season_start: int | None = None,
    season_end: int | None = None,
    rows: int | None = None,
    note: str = "",
    path: Path | None = None,
) -> SourceRecord:
    """Append one provenance record to the JSONL log and return it."""
    rec = SourceRecord(
        source=source,
        endpoint=endpoint,
        season_start=season_start,
        season_end=season_end,
        rows=rows,
        note=note,
    )
    log_path = path or get_settings().paths.provenance_log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec)) + "\n")
    logger.info("provenance: %s | %s | rows=%s", source, endpoint, rows)
    return rec


def read_provenance(path: Path | None = None) -> list[dict]:
    """Read all provenance records (newest last). Empty list if none yet."""
    log_path = path or get_settings().paths.provenance_log
    if not log_path.exists():
        return []
    out: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
