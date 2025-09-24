🌤️ SkyCast

AI-powered Weather Assistant (Open-Meteo + OpenAI + Gemini Vision + FastAPI + Vanilla JS Frontend)

SkyCast is a smart weather assistant that combines live forecast data with AI to give you:

✅ Quick answers to natural language weather questions (e.g., “Umbrella at 6 PM in Mumbai?”)

✅ A personalized day plan (Morning, Afternoon, Evening) with tips on clothing/commute

✅ A visual nowcast: upload a sky photo and get AI-based analysis of clouds, brightness, and rain/storm signs

✨ Features

Natural Language Q&A – Ask questions like “Will it rain tonight in New York?”

Plan My Day – Get a structured plan with temperature ranges, rain chances, and clothing suggestions

Visual Nowcast – Upload a photo of the sky and let Gemini Vision describe current conditions

Fallback Logic – If AI fails, simple rule-based answers still provide guidance

Modern Stack – FastAPI backend, OpenAI & Google Gemini APIs, and a clean HTML/JS frontend

🛠️ Tech Stack

Backend: Python 3, FastAPI, httpx, Pydantic

AI APIs:

OpenAI (gpt-4.1-mini) → Q&A + day plans

Google Gemini (gemini-1.5-flash) → Sky photo analysis

Weather Data: Open-Meteo API

Frontend: HTML, CSS, Vanilla JavaScript (fetch API)
