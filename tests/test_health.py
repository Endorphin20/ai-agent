from fastapi.testclient import TestClient

from ai_agent.main import app


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_check():
    client = TestClient(app)
    response = client.get("/api/health/check")
    assert response.status_code == 200
    assert response.json() == "ok"
