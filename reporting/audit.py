"""Append-only JSON audit ledger.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Every inference call appends one line (JSONL) recording timestamp (IST),
user, model version, and dataset provenance tag. Append-only: we never
rewrite prior lines, and each entry chains the previous line's SHA-256 so
tampering is detectable.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
LEDGER_PATH = Path("audit/ledger.jsonl")


def _last_hash(path: Path) -> str:
    if not path.exists():
        return "GENESIS"
    lines = path.read_text().strip().splitlines()
    if not lines:
        return "GENESIS"
    return hashlib.sha256(lines[-1].encode()).hexdigest()


def append_entry(user: str, model_version: str, provenance: str,
                 analysis_id: str, path: Path = LEDGER_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_ist": datetime.now(IST).isoformat(),
        "user": user,
        "model_version": model_version,
        "provenance": provenance,
        "analysis_id": analysis_id,
        "prev_hash": _last_hash(path),
    }
    with path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def verify_chain(path: Path = LEDGER_PATH) -> bool:
    if not path.exists():
        return True
    prev = "GENESIS"
    for line in path.read_text().strip().splitlines():
        entry = json.loads(line)
        if entry["prev_hash"] != prev:
            return False
        prev = hashlib.sha256(line.encode()).hexdigest()
    return True
