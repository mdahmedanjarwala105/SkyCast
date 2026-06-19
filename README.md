# 🌤️ SkyCast  

[![CI](https://github.com/mdahmedanjarwala105/SkyCast/actions/workflows/ci.yml/badge.svg)](https://github.com/mdahmedanjarwala105/SkyCast/actions/workflows/ci.yml)

**AI-powered Weather Assistant**  
*(Open-Meteo + OpenAI + Gemini Vision + FastAPI + Vanilla JS Frontend)*  

SkyCast is a smart weather assistant that combines **live forecast data** with **AI intelligence** to give you:  

✅ **Quick answers** to natural language weather questions  
  *(e.g., “Umbrella at 6 PM in Mumbai?”)*  

✅ **Personalized day plans** (Morning, Afternoon, Evening)  
  with temp ranges, rain chances & clothing tips 👕☂️  

✅ **Visual nowcast** – Upload a sky photo & get AI-based analysis  
  of clouds ☁️, brightness ☀️, and storm signs 🌩️  

---

## ✨ Features  

- 🌍 **Natural Language Q&A** – Ask things like *“Will it rain tonight in New York?”*  
- 📅 **Plan My Day** – Structured plan with temperatures, rain chances & tips  
- 📸 **Visual Nowcast** – AI analysis of sky photos  
- 🔄 **Fallback Logic** – Rule-based answers if AI/API fails  
- ⚡ **Modern Stack** – FastAPI backend, OpenAI & Gemini APIs, clean HTML/JS frontend  

---

## 🛠️ Tech Stack  

**Backend**: Python 3, FastAPI, httpx, Pydantic  

**AI APIs**:  
- 🤖 **OpenAI (`gpt-4.1-mini`)** → Q&A + Day Plans  
- 🌤️ **Google Gemini (`gemini-1.5-flash`)** → Sky photo analysis  

**Weather Data**: [Open-Meteo API](https://open-meteo.com/)  

**Frontend**: HTML, CSS, Vanilla JavaScript (Fetch API)  

---

## 🚀 Installation

### Prerequisites
- Python 3.13+
- pipenv

### Setup
```bash
git clone https://github.com/mdahmedanjarwala105/SkyCast.git
cd SkyCast
pipenv install
pipenv run uvicorn backend.main:app --reload
```

### Docker
```bash
docker compose up --build
```

---

## 📄 License
This project is licensed under the MIT License.
