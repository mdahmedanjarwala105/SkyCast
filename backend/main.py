import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .prompts import SYSTEM_TEXT_ASSISTANT
from .vision_providers import describe_image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_PLACE = os.getenv("DEFAULT_PLACE", "New York, NY")
DEFAULT_UNITS = os.getenv("DEFAULT_UNITS", "metric")

if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in backend/.env")

app = FastAPI(title="SkyCast AI Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Models ----------------
class WxRequest(BaseModel):
    place: Optional[str] = None


class QARequest(BaseModel):
    question: Optional[str] = None


# ---------------- Core Tool Definitions ----------------


def geocode_place(place: str) -> Optional[Dict[str, Any]]:
    """Get the latitude and longitude coordinates for a given city or place name."""
    if not place:
        return None
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": place, "count": 1, "language": "en", "format": "json"}
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        if data.get("results"):
            res = data["results"][0]
            return {
                "latitude": float(res["latitude"]),
                "longitude": float(res["longitude"]),
                "name": res["name"],
            }
    except Exception:
        return None
    return None


def fetch_forecast(
    lat: float, lon: float, units: str = DEFAULT_UNITS
) -> Dict[str, Any]:
    """Fetch the weather forecast for specific latitude and longitude coordinates."""
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
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            om = r.json()

        return {
            "current": {"temp": om.get("current", {}).get("temperature_2m")},
            "hourly": [
                {"temp": t, "pop": (p or 0) / 100.0}
                for t, p in zip(
                    om.get("hourly", {}).get("temperature_2m", []),
                    om.get("hourly", {}).get("precipitation_probability", []),
                )
            ][:12],
            "daily": [
                {"temp": {"min": mn, "max": mx}}
                for mn, mx in zip(
                    om.get("daily", {}).get("temperature_2m_min", []),
                    om.get("daily", {}).get("temperature_2m_max", []),
                )
            ][:3],
        }
    except Exception as e:
        return {"error": f"Failed to retrieve weather: {str(e)}"}


# ---------------- Tool Registry & Mapping ----------------

# Dictionary mapping tool names directly to function objects
TOOL_MAP = {
    "geocode_place": geocode_place,
    "fetch_forecast": fetch_forecast,
}

# Provide the list of tools to Gemini natively
AI_TOOLS = list(TOOL_MAP.values())

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------- Helper Logic ----------------


async def send_message_with_backoff(chat, payload, retries=3, initial_delay=2.0):
    """Sends a message to Gemini and automatically handles 429 rate limit errors."""
    delay = initial_delay
    for attempt in range(retries):
        try:
            return chat.send_message(payload)
        except APIError as e:
            if e.code == 429 and attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise e


async def run_weather_agent(user_question: str) -> str:
    """Executes a manual tool-calling chat loop with resilient error recovery."""
    system_instruction = f"{SYSTEM_TEXT_ASSISTANT}. The default unit is {DEFAULT_UNITS}. The default fallback place is {DEFAULT_PLACE} if no location can be inferred."

    chat = gemini_client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=AI_TOOLS,
            temperature=0.3,
        ),
    )

    response = await send_message_with_backoff(chat, user_question)

    while response.function_calls:
        function_responses = []

        for function_call in response.function_calls:
            name = function_call.name
            args = function_call.args

            # Refactored: Dynamic Dictionary Lookup mapping instead of if/elif chain
            if name in TOOL_MAP:
                try:
                    if name == "geocode_place":
                        place_arg = args.get("place")
                        result = TOOL_MAP[name](place_arg)
                        tool_output = (
                            result if result else {"error": "Location not found"}
                        )

                    elif name == "fetch_forecast":
                        # Explicitly enforce clean type extraction out of the LLM dictionary
                        lat = float(args.get("lat", 0))
                        lon = float(args.get("lon", 0))
                        units = args.get("units", DEFAULT_UNITS)
                        tool_output = TOOL_MAP[name](lat=lat, lon=lon, units=units)

                except Exception as e:
                    tool_output = {"error": f"Failed to execute {name}: {str(e)}"}
            else:
                tool_output = {"error": f"Unknown function invocation: {name}"}

            function_responses.append(
                types.Part.from_function_response(
                    name=name, response={"result": tool_output}
                )
            )

        response = await send_message_with_backoff(chat, function_responses)

    return response.text.strip()


# ---------------- Agentic Routes ----------------


@app.get("/api/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.post("/api/ask")
async def api_ask(req: QARequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Missing question parameter.")
    try:
        answer = await run_weather_agent(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")


@app.post("/api/plan")
async def api_plan(req: WxRequest = Body(default=None)):
    place = (req.place if req else None) or DEFAULT_PLACE
    prompt = f"Generate a precise daily schedule/plan for the weather at: {place} using the plan template standard format."
    try:
        plan = await run_weather_agent(prompt)
        return {"plan": plan, "resolved_place": place}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")


@app.post("/api/nowcast")
async def api_nowcast(image: UploadFile = File(...)):
    raw = await image.read()
    description = await describe_image(raw)
    return {"vision_nowcast": description}
