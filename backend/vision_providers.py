"""Gemini-based VLM helper (uses gemini-1.5-flash for images)."""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from .prompts import SYSTEM_VISION_ASSISTANT, VISION_PROMPT

load_dotenv()


def _setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in backend/.env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


async def describe_image(image_bytes: bytes) -> str:
    """Analyze a sky photo with Google Gemini. Returns a concise paragraph or a friendly error."""
    model = _setup_gemini()

    content = [
        {"text": SYSTEM_VISION_ASSISTANT},
        {"text": VISION_PROMPT},
        {
            "inline_data": {
                "mime_type": "image/jpeg",  # png also works
                "data": image_bytes,
            }
        },
    ]

    try:
        resp = model.generate_content(content)
        return (resp.text or "").strip() or "No description returned by the model."
    except ResourceExhausted:
        return "⚠️ Vision model quota exceeded — please try again later."
    except Exception as e:
        return f"⚠️ Vision analysis failed: {e}"
