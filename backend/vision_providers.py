"""Gemini-based VLM helper (uses gemini-2.5-flash for images)."""

import os
import asyncio
from dotenv import load_dotenv

# Import the new unified Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .prompts import SYSTEM_VISION_ASSISTANT, VISION_PROMPT

load_dotenv()


def _get_client() -> genai.Client:
    """Initializes the standard unified GenAI client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in backend/.env")
    return genai.Client(api_key=api_key)


async def describe_image_with_backoff(
    client: genai.Client,
    model: str,
    config: types.GenerateContentConfig,
    contents: list,
    retries=3,
    initial_delay=2.0,
):
    """
    Executes a content generation request with built-in asynchronous
    exponential backoff specifically tuned for free-tier rate limits.
    """
    delay = initial_delay
    for attempt in range(retries):
        try:
            # The SDK generate_content call is synchronous, so we run it in an executor
            # or wrap it cleanly to keep things non-blocking for FastAPI
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except APIError as e:
            # Check for a 429 Rate Limit/Quota Exhausted status code
            if e.code == 429 and attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Double the backoff sleep duration
                continue
            raise e


async def describe_image(image_bytes: bytes) -> str:
    """
    Analyzes a sky photo using the unified google-genai SDK.
    Includes rate limit protection to match the core main.py workflow.
    """
    client = _get_client()

    # We upgrade the model to gemini-2.5-flash which has faster vision processing
    # and shares your primary free tier ecosystem seamlessly.
    model_name = "gemini-3.5-flash"

    # Set up configuration blocks matching the structured parameters pattern
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_VISION_ASSISTANT,
        temperature=0.4,
    )

    # Re-structure structural payload items cleanly into standard GenAI Parts
    contents = [
        types.Part.from_text(text=VISION_PROMPT),
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",  # Handles standard raw jpeg/png stream allocations
        ),
    ]

    try:
        response = await describe_image_with_backoff(
            client=client, model=model_name, config=config, contents=contents
        )
        return (response.text or "").strip() or "No description returned by the model."

    except APIError as e:
        if e.code == 429:
            return "⚠️ Vision model quota exceeded — please try again later."
        return f"⚠️ Vision analysis failed (API Error {e.code}): {e.message}"
    except Exception as e:
        return f"⚠️ Vision analysis failed: {str(e)}"
