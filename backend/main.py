from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, base64, os, json
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime


from .prompts import SYSTEM_TEXT_ASSISTANT, TEXT_QA_PROMPT, PLAN_MY_DAY_PROMPT
from .vision_providers import describe_image_with_openai


# Load environment
load_dotenv()


OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPEN_WEATHER_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", 40.7357))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", -74.1724))
DEFAULT_UNITS = os.getenv("DEFAULT_UNITS", "metric")


if not OPEN_AI_API_KEY or not OPEN_WEATHER_API_KEY:
    raise RuntimeError(
        "Missing keys in backend/.env — set OPENAI_API_KEY and OPENWEATHER_KEY"
    )

# FastAPI app
app = FastAPI(title="SkyCast API (OpenAI-only)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENWEATHER_URL = "https://api.openweathermap.org/data/3.0/onecall"


class WxRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    units: Optional[str] = None  # metric or imperial


class QARequest(WxRequest):
    question: str


async def fetch_forecast(lat: float, lon: float, units: str):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPEN_WEATHER_API_KEY,
        "units": units,
        "exclude": "minutely,alerts",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(OPENWEATHER_URL, params=params)
        r.raise_for_status()
        return r.json()


# --- Text LLM (OpenAI SDK v1) ---
from openai import OpenAI

openai_client = OpenAI(api_key=OPEN_AI_API_KEY)


async def ask_text_llm(question: str, forecast: dict):
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


async def plan_my_day_llm(forecast: dict) -> str:
    snippet = json.dumps(
        {
            "hourly": forecast.get("hourly", [])[:12],
            "daily": forecast.get("daily", [])[:1],
        },
        indent=2,
    )
    prompt = f"{PLAN_MY_DAY_PROMPT} JSON: {snippet}"
    resp = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_TEXT_ASSISTANT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()

@app.get("/api/health")
async def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.post("/api/forecast")
async def api_forecast(req: WxRequest):
    lat, lon, units = req.lat or DEFAULT_LAT, req.lon or DEFAULT_LON, req.units or DEFAULT_UNITS
    return {"forecast": await fetch_forecast(lat, lon, units)}

@app.post("/api/ask")
async def api_ask(req: QARequest):
    forecast = await fetch_forecast(req.lat or DEFAULT_LAT, req.lon or DEFAULT_LON, req.units or DEFAULT_UNITS)
    return {"answer": await ask_text_llm(req.question, forecast)}

@app.post("/api/plan")
async def api_plan(req: WxRequest):
    forecast = await fetch_forecast(req.lat or DEFAULT_LAT, req.lon or DEFAULT_LON, req.units or DEFAULT_UNITS)
    return {"plan": await plan_my_day_llm(forecast)}

@app.post("/api/nowcast")
async def api_nowcast(image: UploadFile = File(...)):
    b64 = base64.b64encode(await image.read()).decode("utf-8")
    return {"vision_nowcast": await describe_image_with_openai(b64)}