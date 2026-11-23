"""Append-only JSON Lines journal for paper-trading sessions.

One JSON object per line: ``{"ts": <UTC ISO time written>, "event": ..., ...}``.
Append-only so a crash never corrupts earlier records, and easy to load with
``pandas.read_json(path, lines=True)``.
"""

from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record


def read_journal(path: str | Path) -> list[dict[str, Any]]:
    """Load every record, skipping a partially written final line if present."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
