"""OpenAI-based VLM helper (uses gpt-4o-mini for images)."""

from __future__ import annotations
import os
from openai import OpenAI
from .prompts import SYSTEM_VISION_ASSISTANT, VISION_PROMPT


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


async def describe_image_with_openai(b64_image: str) -> str:
    # Use Chat Completions multimodal with a data URL
    data_url = f"data:image/png;base64,{b64_image}"
    messages = [
        {"role": "system", "content": SYSTEM_VISION_ASSISTANT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
