import pytest
from fastapi.testclient import TestClient

from backend.app.deps import get_llm_client
from backend.app.main import app

SINGLE_CELL_ANALYSIS = {
    "predicted_class": 0,
    "prediction": "Parasitized (Infected)",
    "confidence": 91.9,
    "probabilities": {"parasitized": 91.9, "uninfected": 8.1},
    "model": "PlasmoSENet",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup: creates tables, loads the CNN
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def use_llm(client):
    def install(llm):
        app.dependency_overrides[get_llm_client] = lambda: llm
        return llm

    yield install
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database_ok"] is True
    assert body["clinical_validation"] is False


def test_predict_matches_pre_refactor_baseline(client, cell_image_bytes):
    r = client.post("/api/predict", files={"file": ("c.png", cell_image_bytes, "image/png")})
    body = r.json()
    assert body["predicted_class"] == 0
    assert body["confidence"] == 91.9
    assert body["probabilities"] == {"parasitized": 91.9, "uninfected": 8.1}
    assert "not a clinical diagnosis" in body["disclaimer"]


def test_predict_rejects_wrong_type(client):
    r = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_rejects_non_image(client):
    r = client.post("/api/predict", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_analyze_smear(client, cell_image_bytes):
    r = client.post("/api/analyze-smear", files={"file": ("c.png", cell_image_bytes, "image/png")})
    body = r.json()
    # cell_image.png is one pre-cropped NIH cell.
    assert body["candidate_count"] == len(body["cells"]) == 1
    assert body["annotated_image_mime_type"] == "image/png"
    assert "not a clinical diagnosis" in body["disclaimer"]


def test_treatment_recommend(client):
    r = client.post("/api/treatment/recommend", json={"severity": "severe"})
    assert r.json()["status"] == "urgent_escalation"


def test_explain_agent_with_fake_provider(client, use_llm, fake_llm_factory):
    use_llm(
        fake_llm_factory(
            [
                ("", [("inspect_analysis_evidence", {}), ("check_analysis_uncertainty", {})]),
                ("Report: one isolated cell classified as parasitized. Not a diagnosis.", []),
            ]
        )
    )
    body = client.post(
        "/api/agent/explain", json={"mode": "single_cell", "analysis": SINGLE_CELL_ANALYSIS}
    ).json()

    assert body["tools_executed"] == ["inspect_analysis_evidence", "check_analysis_uncertainty"]
    assert body["agent_runtime"] == "Fake provider"
    assert body["tool_results"]["check_analysis_uncertainty"]["checks"]["clinical_validation"] is False


def test_treatment_agent_with_fake_provider(client, use_llm, fake_llm_factory):
    use_llm(fake_llm_factory([("", [("assess_treatment", {})]), ("Urgent hospital assessment.", [])]))
    body = client.post("/api/agent/treatment-report", json={"severity": "severe"}).json()

    assert body["assessment"]["status"] == "urgent_escalation"
    assert body["explanation"] == "Urgent hospital assessment."


def test_agent_error_maps_to_http_status(client, use_llm, fake_llm_factory):
    use_llm(fake_llm_factory([("", [("rm_rf", {})])]))
    r = client.post(
        "/api/agent/explain", json={"mode": "single_cell", "analysis": SINGLE_CELL_ANALYSIS}
    )
    assert r.status_code == 502
    assert "unapproved tool" in r.json()["detail"]


def test_agent_reports_missing_provider(client, monkeypatch):
    from backend.app import deps
    from backend.app.config import LLMSettings, get_settings

    real = get_settings()
    no_llm = type(real)(
        database_url=real.database_url,
        cors_origins=real.cors_origins,
        lab_model_path=real.lab_model_path,
        llm=LLMSettings(provider="none"),
    )
    monkeypatch.setattr(deps, "get_settings", lambda: no_llm)

    r = client.post("/api/agent/explain", json={"mode": "single_cell", "analysis": {}})
    assert r.status_code == 503
    assert "No AI provider is configured" in r.json()["detail"]
