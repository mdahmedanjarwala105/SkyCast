"""Gemini-based VLM helper (uses gemini-1.5-flash for images)."""

import os, base64
from dotenv import load_dotenv

# Prompts reused from your project
from .prompts import SYSTEM_VISION_ASSISTANT, VISION_PROMPT

# Google Gemini SDK
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

load_dotenv()


def _setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in backend/.env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def _b64_to_bytes(b64_image: str) -> bytes:
    # Accepts base64 *without* data URL prefix
    return base64.b64decode(b64_image.split(",")[-1])


async def describe_image(b64_image: str) -> str:
    """
    Analyze a sky photo with Google Gemini.
    Returns a friendly, concise paragraph (or a friendly error message).
    """
    model = _setup_gemini()
    img_bytes = _b64_to_bytes(b64_image)

    # Gemini accepts multimodal "content parts"
    content = [
        {"text": SYSTEM_VISION_ASSISTANT},  # system-style guidance
        {"text": VISION_PROMPT},
        {
            "inline_data": {
                "mime_type": "image/jpeg",  # ok for png too; jpeg keeps size small
                "data": img_bytes,
            }
        },
    ]

    try:
        resp = model.generate_content(content)
        return (resp.text or "").strip() or "No description returned by the model."
    except ResourceExhausted:
        # Quota / rate limit
        return "⚠️ Vision model quota exceeded — please try again later."
    except Exception as e:
        return f"⚠️ Vision analysis failed: {e}"
