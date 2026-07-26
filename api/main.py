"""FastAPI backend: functional analysis endpoint + SSE pipeline stream.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

POST /analyze  (multipart image) -> runs the REAL pipeline (detect + classify
+ MC-Dropout + attention gate) and returns a fully-populated AnalysisResponse,
writes an audit entry, and generates a signed PDF report.
GET  /analyze/stream/{id} -> SSE of the 7 pipeline stages.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from api.auth import decode_token
from api.schemas import AnalysisResponse, StageEvent
from common.disclaimer import DISCLAIMER
from reporting.audit import append_entry
from reporting.report import generate_report

app = FastAPI(title="HemoSight API", version="1.0.0")

STAGES = ["INGESTED", "PREPROCESSED", "DETECTED", "CLASSIFIED",
          "CALIBRATED", "ATTENTION_CHECKED", "COMPLETED"]

_MODELS = {}


def _get_models():
    if not _MODELS:
        from ml.pipeline import load_models
        _MODELS["wbc"], _MODELS["mal"] = load_models()
    return _MODELS["wbc"], _MODELS["mal"]


def require_user(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    user = decode_token(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return user


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), user: str = Depends(require_user)):
    from ml.pipeline import run_pipeline

    raw = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    wbc, mal = _get_models()
    result = run_pipeline(img, wbc, mal)
    if isinstance(result, dict):  # quality error
        return JSONResponse(status_code=422, content=result)

    append_entry(user, result.model_version, result.provenance, result.analysis_id)

    # Signed PDF report
    ctx = {
        "sample_id": result.analysis_id[:8], "timestamp": "IST",
        "model_version": result.model_version, "mlflow_run_id": result.mlflow_run_id,
        "git_sha": "local", "provenance": str(result.provenance),
        "wbc_differential": result.metrics.wbc_differential,
        "parasite_flag": result.metrics.parasite_flag,
        "confidence": result.detections[0].confidence if result.detections else 0.0,
        "uncertainty": result.detections[0].uncertainty_std if result.detections else 0.0,
        "needs_review": result.attention_gate.status != "PASSED",
    }
    try:
        pdf, _ = generate_report(ctx, Path(f"results/report_{result.analysis_id[:8]}.pdf"))
    except Exception:
        pdf = None
    payload = result.model_dump()
    payload["report_pdf"] = str(pdf) if pdf else None
    return payload


@app.get("/analyze/stream/{analysis_id}")
async def stream(analysis_id: str):
    async def event_gen():
        for stage in STAGES:
            evt = StageEvent(stage=stage, analysis_id=analysis_id)
            yield f"data: {evt.model_dump_json()}\n\n"
            await asyncio.sleep(0.2)
    return StreamingResponse(event_gen(), media_type="text/event-stream")
