"""
LLM Clients — Shared Configuration
=====================================
Single source of truth for ALL AI client initialisation.

Exports:
    gemini_client          — raw google.genai Client
    call_gemini(prompt)    — simple one-shot Gemini call
    call_gemini_with_retry(prompt)  — retry on 429 with exponential backoff
    get_biollm_client()    — fresh InferenceClient (validates token first)
    HF_TOKEN               — HuggingFace token string
    GEMINI_MODEL           — active Gemini model name

File: app/services/llm_clients.py
"""

from __future__ import annotations

import logging
import time
import os

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Gemini  (google.genai — new SDK)
# ─────────────────────────────────────────────────────────────
_GEMINI_KEY: str = os.environ.get(
    "GEMINI_API_KEY"   # local dev fallback
)

try:
    from google import genai as _genai
    gemini_client = _genai.Client(api_key=_GEMINI_KEY)
    GEMINI_MODEL  = "gemini-2.5-flash"
    log.info("Gemini client initialised (google.genai / %s)", GEMINI_MODEL)
except ImportError as exc:
    gemini_client = None
    GEMINI_MODEL  = ""
    log.error("google-genai package not installed: %s", exc)


def call_gemini(prompt: str) -> str:
    if gemini_client is None:
        raise RuntimeError("Gemini client not available.")

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if hasattr(response, "text") and response.text:
        return response.text

    try:
        return response.candidates[0].content.parts[0].text
    except Exception:
        raise RuntimeError(f"Gemini returned empty/invalid response: {response}")


def call_gemini_with_retry(
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> str:
    """
    Gemini call with exponential backoff on 429 rate-limit errors.

    Retry schedule (default):
        attempt 1 — immediate
        attempt 2 — wait  5s
        attempt 3 — wait 10s
        attempt 4 — wait 20s  (then raises)

    Raises:
        RuntimeError — if all retries exhausted or non-retryable error
    """
    if gemini_client is None:
        raise RuntimeError("Gemini client not available.")

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text

        except Exception as exc:
            last_error = exc
            err_str = str(exc)

            # Only retry on 429 rate-limit
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    log.warning("429 rate-limit hit. Retrying in %.1fs (attempt %d/%d)...",
                                delay, attempt + 1, max_retries)
                    time.sleep(delay)
                    continue
                else:
                    log.error("Gemini 429: all %d retries exhausted.", max_retries)
                    raise RuntimeError(
                        f"Gemini quota exhausted after {max_retries} retries. "
                        "You have hit the free-tier daily limit (20 req/day). "
                        "Wait 24h or upgrade your Gemini plan."
                    ) from exc
            else:
                # Non-retryable error — fail immediately
                log.error("Gemini non-retryable error: %s", exc)
                raise RuntimeError(f"Gemini call failed: {exc}") from exc

    raise RuntimeError(f"Gemini call failed after retries: {last_error}")


# ─────────────────────────────────────────────────────────────
# 2. BioLLM  (HuggingFace InferenceClient)
# ─────────────────────────────────────────────────────────────
HF_TOKEN: str = os.environ.get(
    "HF_TOKEN"    # local dev fallback
)

BIOLLM_MODEL = "aaditya/Llama3-OpenBioLLM-8B"


def get_biollm_client():
    """
    Returns a fresh InferenceClient.
    Raises ValueError if HF_TOKEN is empty — prevents
    the 'Illegal header value b"Bearer "' error.
    """
    from huggingface_hub import InferenceClient

    if not HF_TOKEN or not HF_TOKEN.strip():
        raise ValueError(
            "HF_TOKEN is empty. Set the HF_TOKEN environment variable "
            "or update the fallback in app/services/llm_clients.py"
        )
    return InferenceClient(model=BIOLLM_MODEL, token=HF_TOKEN)