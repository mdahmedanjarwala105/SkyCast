import base64, respx, httpx
from backend.main import OPENWEATHER_URL

DUMMY = {
    "current": {"temp": 20},
    "hourly": [{"temp": 20}],
    "daily": [{"temp": {"min": 15, "max": 25}}],
}


@respx.mock
def test_forecast(client: httpx.Client):
    respx.get(OPENWEATHER_URL).mock(return_value=httpx.Response(200, json=DUMMY))
    assert client.post("/api/forecast", json={}).status_code == 200
