from pathlib import Path

from fastapi.testclient import TestClient

from inspection.api.main import app

SAMPLE_IMAGE = Path("data/raw/mvtec_ad/bottle/train/good/000.png")


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_similar_returns_k_results():
    with TestClient(app) as client:
        with open(SAMPLE_IMAGE, "rb") as f:
            response = client.post(
                "/similar", files={"file": ("000.png", f, "image/png")}, params={"k": 3}
            )

    assert response.status_code == 200
    body = response.json()
    assert body["query_filename"] == "000.png"
    assert len(body["results"]) == 3


def test_similar_rejects_non_image_file():
    with TestClient(app) as client:
        response = client.post(
            "/similar", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
    assert response.status_code == 400