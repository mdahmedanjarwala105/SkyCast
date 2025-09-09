import os, pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENWEATHER_KEY", "test")
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)
