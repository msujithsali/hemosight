"""AnalysisResponse contract stays exactly as specified."""
from api.schemas import AnalysisResponse, AttentionGate, AttentionStatus, Metrics, Provenance


def test_analysis_response_shape():
    r = AnalysisResponse(
        analysis_id="a1", provenance=Provenance.BOOTSTRAP, model_version="v1",
        mlflow_run_id="run1",
        metrics=Metrics(total_counts=3, wbc_differential={"Neutrophil": 3},
                        parasite_flag=False, parasitemia_estimate_pct=None),
        detections=[], attention_gate=AttentionGate(iou_score=0.9,
                                                    status=AttentionStatus.PASSED),
    )
    data = r.model_dump()
    expected = {"analysis_id", "provenance", "model_version", "mlflow_run_id",
                "metrics", "detections", "attention_gate", "disclaimer"}
    assert set(data.keys()) == expected
    assert "SCREENING AID ONLY" in data["disclaimer"]


def test_provenance_never_blends():
    assert {p.value for p in Provenance} == {"BOOTSTRAP", "TARGET"}
