"""Claude wrapper with retries and JSON parsing.

We deliberately don't use `instructor` here to keep deps light. The pattern:
- Always request JSON in the prompt
- Parse with json.loads, retry once on failure with the error fed back
- Fail loudly if it still doesn't parse — better than silently wrong data
"""
import json
import re
import time
from typing import Any
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import ANTHROPIC_API_KEY, MODEL_SONNET

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _strip_code_fences(text: str) -> str:
    """Claude sometimes wraps JSON in ```json ... ``` despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_json(text: str) -> Any:
    """Try hard to parse JSON from a model response."""
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find the largest {...} or [...] substring
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APITimeoutError)),
    reraise=True,
)
def call_claude(
    system: str,
    user: str,
    model: str = MODEL_SONNET,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Single Claude call returning raw text."""
    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_claude_json(
    system: str,
    user: str,
    model: str = MODEL_SONNET,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> Any:
    """Claude call that must return JSON. One retry on parse failure with the error fed back."""
    raw = call_claude(system, user, model, max_tokens, temperature)
    try:
        return _extract_json(raw)
    except json.JSONDecodeError as e:
        # Retry with the error in the prompt
        retry_user = (
            f"{user}\n\n---\n"
            f"Your previous response failed to parse as JSON: {e}\n"
            f"Previous response (truncated): {raw[:500]}\n"
            f"Return ONLY valid JSON, no commentary or code fences."
        )
        raw2 = call_claude(system, retry_user, model, max_tokens, temperature)
        return _extract_json(raw2)
