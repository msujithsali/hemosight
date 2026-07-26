"""Signed clinical PDF report generation (WeasyPrint + Ed25519).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Renders an HTML template to PDF, then appends a detached Ed25519 signature
sidecar (.sig) verifiable by reporting/verify_report.py. The disclaimer
footer is non-negotiable and pulled from common.disclaimer. No PII by default
— only a sample ID.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

from common.disclaimer import DISCLAIMER

IST = timezone(timedelta(hours=5, minutes=30))
KEY_DIR = Path("keys")


def ensure_keys() -> tuple[Path, Path]:
    KEY_DIR.mkdir(exist_ok=True)
    priv_path, pub_path = KEY_DIR / "ed25519_priv.pem", KEY_DIR / "ed25519_pub.pem"
    if not priv_path.exists():
        priv = Ed25519PrivateKey.generate()
        priv_path.write_bytes(priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        pub_path.write_bytes(priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ))
    return priv_path, pub_path


def render_html(context: dict) -> str:
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
body {{ font-family: DejaVu Sans, sans-serif; color:#111; }}
.metrics {{ font-family: 'DejaVu Sans Mono', monospace; }}
.footer {{ position: fixed; bottom: 0; font-size: 8pt; color:#b00; }}
table {{ border-collapse: collapse; width:100%; }}
td, th {{ border:1px solid #999; padding:4px; font-size:9pt; }}
</style></head><body>
<h2>HemoSight Screening Report</h2>
<p class='metrics'>Sample ID: {context['sample_id']} &nbsp; Time (IST): {context['timestamp']}</p>
<p class='metrics'>Model: {context['model_version']} &nbsp; MLflow: {context['mlflow_run_id']}
&nbsp; git: {context['git_sha']} &nbsp; Provenance: [{context['provenance']}]</p>
<h3>WBC differential</h3>
<table><tr><th>Class</th><th>Count</th></tr>
{''.join(f"<tr><td>{k}</td><td class='metrics'>{v}</td></tr>" for k, v in context['wbc_differential'].items())}
</table>
<p class='metrics'>Malaria abnormality flag: {context['parasite_flag']} &nbsp;
Calibrated confidence: {context['confidence']:.3f} &nbsp;
Uncertainty (std): {context['uncertainty']:.3f}</p>
<p class='metrics'>Needs review: {context['needs_review']}</p>
<div class='footer'>{DISCLAIMER}</div>
</body></html>"""


def generate_report(context: dict, out_pdf: Path) -> tuple[Path, Path]:
    from weasyprint import HTML

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=render_html(context)).write_pdf(str(out_pdf))

    priv_path, _ = ensure_keys()
    priv = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
    signature = priv.sign(out_pdf.read_bytes())
    sig_path = out_pdf.with_suffix(".pdf.sig")
    sig_path.write_bytes(signature)
    return out_pdf, sig_path


def default_context(**overrides) -> dict:
    ctx = {
        "sample_id": "PHC-DEMO-0001",
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "model_version": "PENDING",
        "mlflow_run_id": "PENDING",
        "git_sha": "PENDING",
        "provenance": "BOOTSTRAP",
        "wbc_differential": {"Neutrophil": 0, "Lymphocyte": 0, "Monocyte": 0,
                             "Eosinophil": 0, "Basophil": 0},
        "parasite_flag": False,
        "confidence": 0.0,
        "uncertainty": 0.0,
        "needs_review": True,
    }
    ctx.update(overrides)
    return ctx
