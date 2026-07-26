"""Verify an Ed25519-signed HemoSight PDF report.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Usage: python -m reporting.verify_report report.pdf [--pub keys/ed25519_pub.pem]
Exit code 0 = valid signature, 1 = invalid/missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization


def verify(pdf_path: Path, pub_path: Path) -> bool:
    sig_path = pdf_path.with_suffix(".pdf.sig")
    if not sig_path.exists():
        return False
    pub = serialization.load_pem_public_key(pub_path.read_bytes())
    try:
        pub.verify(sig_path.read_bytes(), pdf_path.read_bytes())
        return True
    except InvalidSignature:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("--pub", type=Path, default=Path("keys/ed25519_pub.pem"))
    args = p.parse_args()
    ok = verify(args.pdf, args.pub)
    print("VALID" if ok else "INVALID")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
