from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from radar import config as cfg
from radar import __version__

def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def update(stage_results: dict[str, Any]) -> dict:
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    registry = {
        "version": __version__,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "ref_date": cfg.REF_DATE,
        "data_hash": _file_hash(cfg.CURATED_PATH),
        "xlsx_hash": _file_hash(cfg.XLSX_PATH),
        "stages": stage_results,
        "artifacts": {
            p.name: _file_hash(p)
            for p in [
                cfg.CURATED_PATH,
                cfg.DAILY_SERIES_PATH,
                cfg.FORECAST_PATH,
                cfg.ALERT_SCORES_PATH,
                cfg.RECOMMENDATIONS_PATH,
                cfg.KPI_PROJECTION_PATH,
            ]
            if p.exists()
        },
    }
    cfg.REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"registry: → {cfg.REGISTRY_PATH}")
    return registry
