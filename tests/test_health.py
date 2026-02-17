from fastapi.testclient import TestClient
"""is a special testing tool that:

Simulates HTTP requests

Runs your FastAPI app internally

Does NOT require running uvicorn"""
from app.main import app

client= TestClient(app)
#This creates a fake HTTP client that can call

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    #Converts response body to Python dictionary.
    assert data["status"] == "ok"
    assert "service" in data