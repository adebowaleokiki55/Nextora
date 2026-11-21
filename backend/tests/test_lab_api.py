import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        c.post("/api/demo/seed")
        yield c


@pytest.fixture(scope="module")
def smear_png():
    img = np.full((500, 700, 3), (232, 226, 238), np.uint8)
    for centre in [(100, 100), (300, 120), (360, 120), (500, 300), (150, 350)]:
        cv2.circle(img, centre, 30, (182, 140, 205), -1)
        cv2.circle(img, centre, 30, (125, 85, 155), 2)
    return cv2.imencode(".png", img)[1].tobytes()


def ikorodu_lab(client):
    locations = client.get("/api/locations").json()["locations"]
    ikorodu = next(loc for loc in locations if loc["name"] == "Ikorodu")
    labs = client.get(
        "/api/facilities", params={"location_id": ikorodu["id"], "has_laboratory": True}
    ).json()["facilities"]
    return labs[0]


def submit(client, png, **form):
    data = {"facility_id": ikorodu_lab(client)["id"], "sample_date": "2026-11-20",
            "submitted_by": "Lab tech"}
    data.update(form)
    return client.post("/api/lab/submissions", files={"file": ("s.png", png, "image/png")}, data=data)


def test_places(client):
    assert len(client.get("/api/locations").json()["locations"]) == 20
    lab = ikorodu_lab(client)
    assert lab["has_laboratory"] and lab["data_origin"] == "SYNTHETIC"


def test_submit_review_flow(client, smear_png):
    r = submit(client, smear_png, sample_reference="SL-001")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "PENDING_REVIEW"
    assert body["data_origin"] == "SYNTHETIC"  # default unless marked REAL
    assert body["location"]["name"] == "Ikorodu"
    assert body["candidate_count"] == len(body["cells"]) == 5
    assert body["model_version"].startswith("sha256:")
    assert body["annotated_image_mime_type"] == "image/png"
    sub_id = body["id"]

    pending = client.get("/api/lab/submissions", params={"status": "PENDING_REVIEW"}).json()
    assert sub_id in [s["id"] for s in pending["submissions"]]

    png = client.get(f"/api/lab/submissions/{sub_id}/annotated.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"

    r = client.post(f"/api/lab/submissions/{sub_id}/review",
                    json={"decision": "VERIFIED", "reviewer_name": "Dr Ade", "reviewer_result": "POSITIVE"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "VERIFIED"

    again = client.post(f"/api/lab/submissions/{sub_id}/review",
                        json={"decision": "REJECTED", "reviewer_name": "X", "notes": "n"})
    assert again.status_code == 409


def test_real_origin_must_be_explicit(client, smear_png):
    body = submit(client, smear_png, data_origin="REAL").json()
    assert body["data_origin"] == "REAL"
    assert "demo/test" not in body["source"]


def test_submission_validation(client, smear_png):
    assert submit(client, smear_png, facility_id=99999).status_code == 404
    assert submit(client, smear_png, sample_date="2999-01-01").status_code == 422
    assert submit(client, smear_png, data_origin="MADE_UP").status_code == 422
    r = client.post("/api/lab/submissions", files={"file": ("a.txt", b"x", "text/plain")},
                    data={"facility_id": 1, "sample_date": "2026-11-20", "submitted_by": "t"})
    assert r.status_code == 415


def test_review_validation(client, smear_png):
    sub_id = submit(client, smear_png).json()["id"]
    r = client.post(f"/api/lab/submissions/{sub_id}/review",
                    json={"decision": "VERIFIED", "reviewer_name": "Dr Ade"})
    assert r.status_code == 422
    assert client.post("/api/lab/submissions/99999/review",
                       json={"decision": "REJECTED", "reviewer_name": "R", "notes": "n"}).status_code == 404


def test_model_card_is_honest(client):
    card = client.get("/api/lab/model-card").json()
    assert "NIH Malaria Cell Images" in card["training_data"]
    assert "Not a diagnostic tool" in card["intended_use"]
    assert card["metrics"] is None
    assert "_how_to_add_metrics" not in card
