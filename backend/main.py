import os, json, re
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI

from .prompts import SYSTEM_TEXT_ASSISTANT, TEXT_QA_PROMPT, PLAN_MY_DAY_PROMPT
from .vision_providers import describe_image

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_PLACE = os.getenv("DEFAULT_PLACE", "New York, NY")
DEFAULT_UNITS = os.getenv("DEFAULT_UNITS", "metric")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in backend/.env")

app = FastAPI(title="SkyCast API (Open-Meteo + OpenAI text + Gemini vision)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Models ----------------
class WxRequest(BaseModel):
    units = None  # "metric" or "imperial"
    place = None  # plain-language place name


class QARequest(WxRequest):
    question = None


# ---------------- Geocoding helpers ----------------
async def geocode_place(place: str):
    """
    Resolve a place name to (lat, lon) using Open-Meteo geocoding.
    Returns None if not found/errors.
    """
    if not place:
        return None
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": place, "count": 1, "language": "en", "format": "json"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        if data.get("results"):
            res = data["results"][0]
            return float(res["latitude"]), float(res["longitude"]), res["name"]
    except Exception:
        return None
    return None


def extract_place_from_question(q: str):
    """
    Grab trailing 'in <place>' from the question.
    E.g.: 'Should I take umbrella at 7PM in Mumbai?'
          'Rain in New York tonight'
    """
    if not q:
        return None
    m = re.search(
        r"\bin\s+([A-Za-z][A-Za-z\s\.\-\,]+?)\s*[\?\.\!]*$", q.strip(), re.IGNORECASE
    )
    return m.group(1).strip() if m else None


# ---------------- Forecast via Open-Meteo ----------------
async def fetch_forecast(lat: float, lon: float, units: str):
    temp_unit = "fahrenheit" if units == "imperial" else "celsius"
    wind_unit = "mph" if units == "imperial" else "kmh"

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
    url = "https://api.open-meteo.com/v1/forecast"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        om = r.json()

    cur_temp = om.get("current", {}).get("temperature_2m")

    current = {"temp": cur_temp}

    hourly = []
    times = om.get("hourly", {}).get("time", []) or []
    temps = om.get("hourly", {}).get("temperature_2m", []) or []
    pops = om.get("hourly", {}).get("precipitation_probability", []) or []
    for i in range(min(len(times), len(temps))):
        hourly.append({"temp": temps[i], "pop": (pops[i] or 0) / 100.0})

    daily = []
    tmax = om.get("daily", {}).get("temperature_2m_max", []) or []
    tmin = om.get("daily", {}).get("temperature_2m_min", []) or []
    for i in range(min(len(tmax), len(tmin))):
        daily.append({"temp": {"min": tmin[i], "max": tmax[i]}})

    return {"current": current, "hourly": hourly, "daily": daily}


# ---------------- OpenAI text LLM (with fallbacks) ----------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def _fallback_text_answer(forecast: dict) -> str:
    cur = forecast.get("current", {}) or {}
    hourly = forecast.get("hourly", []) or []
    t = cur.get("temp")
    pops = [h.get("pop", 0.0) or 0.0 for h in hourly[:6]]
    max_pop = max(pops) if pops else 0.0
    if max_pop >= 0.5:
        rain_msg = "Likely rain in the next few hours—carry an umbrella. ☔"
    elif max_pop >= 0.2:
        rain_msg = (
            "There's a small chance of light showers—consider a compact umbrella. 🌦️"
        )
    else:
        rain_msg = "Rain is unlikely in the next few hours. 🙂"
    unit = "F" if os.getenv("DEFAULT_UNITS") == "imperial" else "C"
    temp_msg = f" Current temp: {t}°{unit}." if t is not None else ""
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
        if tmin == tmax:
            range = f"{int(round(tmin))}°"
        elif tmin != tmax:
            range = f"{int(round(tmin))}-{int(round(tmax))}°"
        else:
            range = "-"
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
    return f"Morning: Start easy. {tip_m}\nAfternoon: Peak around {range}. {tip_a}\nEvening: {tip_e}"


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
        return (resp.choices[0].message.content or "").strip()
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
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return _fallback_plan(forecast)


# ---------------- Routes ----------------
@app.get("/api/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.post("/api/ask")
async def api_ask(req: QARequest):
    place = req.place or extract_place_from_question(req.question) or DEFAULT_PLACE
    coords = await geocode_place(place)
    if coords:
        lat, lon, resolved_name = coords
    else:
        raise HTTPException(
            status_code=404, detail=f"Could not resolve place '{place}'"
        )

    units = req.units or DEFAULT_UNITS
    try:
        forecast = await fetch_forecast(lat, lon, units)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    answer = await ask_text_llm(req.question, forecast)
    return {"answer": answer, "resolved_place": resolved_name}


@app.post("/api/plan")
async def api_plan(req: WxRequest = Body(default=None)):
    if req is None:
        req = WxRequest()

    place = req.place or DEFAULT_PLACE
    coords = await geocode_place(place)
    if coords:
        lat, lon, resolved_name = coords
    else:
        raise HTTPException(
            status_code=404, detail=f"Could not resolve place '{place}'"
        )

    units = req.units or DEFAULT_UNITS
    try:
        forecast = await fetch_forecast(lat, lon, units)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    plan = await plan_my_day_llm(forecast)
    return {"plan": plan, "resolved_place": resolved_name}


@app.post("/api/nowcast")
async def api_nowcast(image: UploadFile = File(...)):
    raw = await image.read()
    description = await describe_image(raw)
    return {"vision_nowcast": description}
