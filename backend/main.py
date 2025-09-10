# backend/main.py
from __future__ import annotations
import os, json, base64
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from .prompts import SYSTEM_TEXT_ASSISTANT, TEXT_QA_PROMPT, PLAN_MY_DAY_PROMPT
from .vision_providers import describe_image  # <-- Gemini provider
from openai import OpenAI, RateLimitError

# --- Always load env from backend/.env (works no matter where you run from)
HERE = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(HERE, ".env"), override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", 40.7357))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", -74.1724))
DEFAULT_UNITS = os.getenv("DEFAULT_UNITS", "metric")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in backend/.env")

# FastAPI app ---------------------------------------------------------
app = FastAPI(title="SkyCast API (Open-Meteo + OpenAI text + Gemini vision)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Friendly home so '/' doesn’t 404
@app.get("/")
def root():
    return {
        "status": "ok",
        "try": [
            "/api/health",
            "/docs",
            "/api/forecast (POST)",
            "/api/ask (POST)",
            "/api/plan (POST)",
            "/api/nowcast (POST)",
        ],
    }


class WxRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    units: Optional[str] = None  # "metric" or "imperial"


class QARequest(WxRequest):
    question: str


# -------------------- Forecast via Open-Meteo (no key) --------------------
async def fetch_forecast(lat: float, lon: float, units: str):
    """
    Map Open-Meteo response into our simple schema used by prompts:
      current: temp, weather.main
      hourly:  dt (index), temp, pop [0-1]
      daily:   temp.min/max, pop [0-1]
    """
    temp_unit = "fahrenheit" if units == "imperial" else "celsius"
    wind_unit = "mph" if units == "imperial" else "kmh"

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,cloud_cover",
        "hourly": "temperature_2m,precipitation_probability",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "temperature_unit": temp_unit,
        "windspeed_unit": wind_unit,
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        om = r.json()

    cur_temp = om.get("current", {}).get("temperature_2m")
    cloud = om.get("current", {}).get("cloud_cover") or 0
    precip = om.get("current", {}).get("precipitation") or 0.0

    if precip > 0.1:
        main = "Rain"
    elif cloud > 60:
        main = "Clouds"
    elif cloud > 20:
        main = "Partly Cloudy"
    else:
        main = "Clear"

    current = {"temp": cur_temp, "weather": [{"main": main}]}

    hourly = []
    times = om.get("hourly", {}).get("time", []) or []
    temps = om.get("hourly", {}).get("temperature_2m", []) or []
    pops = om.get("hourly", {}).get("precipitation_probability", []) or []
    for i in range(min(len(times), len(temps))):
        hourly.append(
            {
                "dt": i,
                "temp": temps[i],
                "pop": (pops[i] or 0) / 100.0,
            }
        )

    daily = []
    tmax = om.get("daily", {}).get("temperature_2m_max", []) or []
    tmin = om.get("daily", {}).get("temperature_2m_min", []) or []
    popd = om.get("daily", {}).get("precipitation_probability_max", []) or []
    for i in range(min(len(tmax), len(tmin))):
        daily.append(
            {
                "temp": {"min": tmin[i], "max": tmax[i]},
                "pop": (popd[i] or 0) / 100.0,
            }
        )

    return {"current": current, "hourly": hourly, "daily": daily}


# -------------------- OpenAI text models w/ safe fallbacks --------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def _fallback_text_answer(question: str, forecast: dict) -> str:
    cur = forecast.get("current", {}) or {}
    hourly = forecast.get("hourly", []) or []
    t = cur.get("temp")
    pops = [h.get("pop", 0.0) or 0.0 for h in hourly[:6]]
    max_pop = max(pops) if pops else 0.0

    rain_msg = (
        "Likely rain in the next few hours—carry an umbrella. ☔"
        if max_pop >= 0.5
        else (
            "There’s a small chance of light showers—consider a compact umbrella. 🌦️"
            if max_pop >= 0.2
            else "Rain is unlikely in the next few hours. 🙂"
        )
    )
    temp_msg = (
        f" Current temp: {t}°{'F' if DEFAULT_UNITS=='imperial' else 'C'}."
        if t is not None
        else ""
    )
    return f"{rain_msg}{temp_msg}"


def _fallback_plan(forecast: dict) -> str:
    hourly = forecast.get("hourly", []) or []
    daily = forecast.get("daily", []) or []
    temps = [h.get("temp") for h in hourly[:12] if h.get("temp") is not None]
    if not temps and daily:
        temps = [
            daily[0].get("temp", {}).get("min"),
            daily[0].get("temp", {}).get("max"),
        ]
        temps = [t for t in temps if t is not None]
    if temps:
        tmin, tmax = min(temps), max(temps)
    else:
        tmin = tmax = None
    pops = [h.get("pop", 0.0) or 0.0 for h in hourly[:12]]
    rain = max(pops) if pops else 0.0

    tip_m = "Light layer and sunglasses." if (tmax or 20) >= 20 else "Warm layer."
    tip_a = (
        "Carry a compact umbrella." if rain >= 0.3 else "Hydrate and take shade breaks."
    )
    tip_e = (
        "Light jacket if it cools down."
        if (tmin or 16) <= 18
        else "Evening should be mild."
    )

    rng = (
        f"{int(round(tmin))}–{int(round(tmax))}°"
        if tmin is not None and tmax is not None
        else "—"
    )
    return (
        f"Morning: Start easy. {tip_m}\n"
        f"Afternoon: Peak around {rng}. {tip_a}\n"
        f"Evening: {tip_e}"
    )


async def ask_text_llm(question: str, forecast: dict) -> str:
    try:
        snippet = json.dumps(
            {
                "current": forecast.get("current", {}),
                "hourly": forecast.get("hourly", [])[:6],
                "daily": forecast.get("daily", [])[:1],
            },
            indent=2,
        )
        prompt = TEXT_QA_PROMPT.format(question=question, snippet=snippet)
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_TEXT_ASSISTANT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        return _fallback_text_answer(question, forecast)
    except Exception:
        return _fallback_text_answer(question, forecast)


async def plan_my_day_llm(forecast: dict) -> str:
    try:
        snippet = json.dumps(
            {
                "hourly": forecast.get("hourly", [])[:12],
                "daily": forecast.get("daily", [])[:1],
            },
            indent=2,
        )
        prompt = f"{PLAN_MY_DAY_PROMPT}\n\nJSON:\n{snippet}"
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_TEXT_ASSISTANT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        return _fallback_plan(forecast)
    except Exception:
        return _fallback_plan(forecast)


# -------------------- Routes --------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


@app.post("/api/forecast")
async def api_forecast(req: WxRequest):
    lat = req.lat or DEFAULT_LAT
    lon = req.lon or DEFAULT_LON
    units = req.units or DEFAULT_UNITS
    try:
        data = await fetch_forecast(lat, lon, units)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    return {"lat": lat, "lon": lon, "units": units, "forecast": data}


@app.post("/api/ask")
async def api_ask(req: QARequest):
    lat = req.lat or DEFAULT_LAT
    lon = req.lon or DEFAULT_LON
    units = req.units or DEFAULT_UNITS
    forecast = await fetch_forecast(lat, lon, units)
    answer = await ask_text_llm(req.question, forecast)
    return {"answer": answer}


@app.post("/api/plan")
async def api_plan(req: Optional[WxRequest] = Body(default=None)):
    # Allow calling without a body
    if req is None:
        req = WxRequest()
    lat = req.lat or DEFAULT_LAT
    lon = req.lon or DEFAULT_LON
    units = req.units or DEFAULT_UNITS
    forecast = await fetch_forecast(lat, lon, units)
    plan = await plan_my_day_llm(forecast)
    return {"plan": plan}


@app.post("/api/nowcast")
async def api_nowcast(image: UploadFile = File(...)):
    raw = await image.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    description = await describe_image(b64)  # <-- Gemini provider
    return {"vision_nowcast": description}
