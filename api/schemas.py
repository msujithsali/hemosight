"""Pydantic v2 API contract for HemoSight.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

These models mirror the TypeScript interfaces in ``web/src/types/index.ts``
one-to-one. The ``AnalysisResponse`` shape is frozen by the project spec —
do not reorder or rename fields without updating both sides and the contract
test in ``tests/test_schema_contract.py``.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from common.disclaimer import DISCLAIMER


class Provenance(str, Enum):
    """Data-provenance tag. Never blend BOOTSTRAP and TARGET silently."""

    BOOTSTRAP = "BOOTSTRAP"  # public dataset, off-domain from rural PHC
    TARGET = "TARGET"        # real PHC-captured samples (not yet collected)


class AttentionStatus(str, Enum):
    PASSED = "PASSED"
    ATTENTION_MISALIGNMENT = "ATTENTION_MISALIGNMENT"


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    class_name: str
    bbox: list[int] = Field(min_length=4, max_length=4, description="[x1,y1,x2,y2]")
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_std: float = Field(ge=0.0, description="MC-Dropout std over N passes")


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_counts: int = Field(ge=0)
    wbc_differential: dict[str, int]
    parasite_flag: bool
    parasitemia_estimate_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class AttentionGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iou_score: float = Field(ge=0.0, le=1.0)
    status: AttentionStatus


class AnalysisResponse(BaseModel):
    """Frozen contract returned by POST /analyze."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    analysis_id: str
    provenance: Provenance
    model_version: str
    mlflow_run_id: str
    metrics: Metrics
    detections: list[Detection]
    attention_gate: AttentionGate
    disclaimer: str = DISCLAIMER


class StageEvent(BaseModel):
    """SSE payload for pipeline progress."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "INGESTED",
        "PREPROCESSED",
        "DETECTED",
        "CLASSIFIED",
        "CALIBRATED",
        "ATTENTION_CHECKED",
        "COMPLETED",
    ]
    analysis_id: str
    detail: Optional[str] = None


class QualityError(BaseModel):
    """Explicit, coded rejection from the quality gate — never a bare exception."""

    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "BLUR_REJECT",
        "BRIGHTNESS_REJECT",
        "RESOLUTION_REJECT",
        "DECODE_ERROR",
    ]
    message: str
    measured_value: Optional[float] = None
    threshold: Optional[float] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
