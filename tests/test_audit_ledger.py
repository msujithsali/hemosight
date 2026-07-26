"""Append-only ledger chains correctly and detects tampering."""
import json
from pathlib import Path

from reporting.audit import append_entry, verify_chain


def test_chain_valid(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    for i in range(3):
        append_entry(f"user{i}", "v1", "BOOTSTRAP", f"id{i}", path=ledger)
    assert verify_chain(ledger) is True


def test_tamper_detected(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_entry("u", "v1", "BOOTSTRAP", "id0", path=ledger)
    append_entry("u", "v1", "BOOTSTRAP", "id1", path=ledger)
    lines = ledger.read_text().splitlines()
    entry = json.loads(lines[0]); entry["user"] = "attacker"
    lines[0] = json.dumps(entry, sort_keys=True)
    ledger.write_text("\n".join(lines) + "\n")
    assert verify_chain(ledger) is False
