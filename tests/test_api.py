import base64
from fastapi.testclient import TestClient
import pytest

from backend.main import app

client = TestClient(app)

DUMMY_FORECAST = {
    "current": {"temp": 20, "weather": [{"main": "Clouds"}]},
    "hourly": [{"temp": 20, "dt": 0, "pop": 0.1}] * 6,
    "daily": [{"temp": {"min": 15, "max": 24}, "pop": 0.2}],
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch):
    # Replace fetch_forecast with a dummy so no network calls happen
    import backend.main as main_mod

    async def fake_fetch(lat: float, lon: float, units: str):
        return DUMMY_FORECAST

    monkeypatch.setattr(main_mod, "fetch_forecast", fake_fetch)


def test_forecast_endpoint():
    r = client.post("/api/forecast", json={})
    assert r.status_code == 200
    j = r.json()
    assert j["forecast"]["current"]["temp"] == 20


def test_ask_endpoint(monkeypatch: pytest.MonkeyPatch):
    import backend.main as main_mod

    async def fake_answer(q: str, f: dict):
        return "Light clouds, no umbrella needed."

    monkeypatch.setattr(main_mod, "ask_text_llm", fake_answer)

    r = client.post("/api/ask", json={"question": "Umbrella?"})
    assert r.status_code == 200
    assert "umbrella" in r.json()["answer"].lower()


def test_nowcast_endpoint(monkeypatch: pytest.MonkeyPatch):
    import backend.vision_providers as vp

    async def fake_describe(b64: str):
        return "Overcast skies, 70% confidence."

    monkeypatch.setattr(vp, "describe_image_with_openai", fake_describe)

    fake_img = base64.b64encode(b"fake").decode()
    files = {"image": ("x.png", base64.b64decode(fake_img), "image/png")}
    r = client.post("/api/nowcast", files=files)
    assert r.status_code == 200
    assert "overcast" in r.json()["vision_nowcast"].lower()
