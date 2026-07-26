"""Ed25519-signed PDF verifies; tampering breaks verification.
Skips gracefully if WeasyPrint's native libs are unavailable in the runner."""
from pathlib import Path

import pytest

from reporting.report import default_context, ensure_keys, generate_report
from reporting.verify_report import verify


def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _weasyprint_available(), reason="WeasyPrint native libs absent")
def test_signed_report_verifies(tmp_path):
    _, pub = ensure_keys()
    out = tmp_path / "report.pdf"
    pdf, sig = generate_report(default_context(), out)
    assert pdf.exists() and sig.exists()
    assert verify(pdf, pub) is True
    pdf.write_bytes(pdf.read_bytes() + b"tampered")
    assert verify(pdf, pub) is False
