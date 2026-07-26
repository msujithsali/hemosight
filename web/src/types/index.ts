// HemoSight API contract — mirror of api/schemas.py
// Author: M Sujith Sali, ISE Dept, VTU Karnataka.
// Keep in lockstep with the Python side; tests/test_schema_contract.py checks parity.

export type Provenance = "BOOTSTRAP" | "TARGET";
export type AttentionStatus = "PASSED" | "ATTENTION_MISALIGNMENT";

export interface Detection {
  id: number;
  class_name: string;
  bbox: [number, number, number, number]; // [x1,y1,x2,y2]
  confidence: number;
  uncertainty_std: number;
}

export interface Metrics {
  total_counts: number;
  wbc_differential: Record<string, number>;
  parasite_flag: boolean;
  parasitemia_estimate_pct: number | null;
}

export interface AttentionGate {
  iou_score: number;
  status: AttentionStatus;
}

export interface AnalysisResponse {
  analysis_id: string;
  provenance: Provenance;
  model_version: string;
  mlflow_run_id: string;
  metrics: Metrics;
  detections: Detection[];
  attention_gate: AttentionGate;
  disclaimer: string;
}

export type PipelineStage =
  | "INGESTED"
  | "PREPROCESSED"
  | "DETECTED"
  | "CLASSIFIED"
  | "CALIBRATED"
  | "ATTENTION_CHECKED"
  | "COMPLETED";

export interface StageEvent {
  stage: PipelineStage;
  analysis_id: string;
  detail?: string | null;
}

export type QualityErrorCode =
  | "BLUR_REJECT"
  | "BRIGHTNESS_REJECT"
  | "RESOLUTION_REJECT"
  | "DECODE_ERROR";

export interface QualityError {
  code: QualityErrorCode;
  message: string;
  measured_value?: number | null;
  threshold?: number | null;
}
