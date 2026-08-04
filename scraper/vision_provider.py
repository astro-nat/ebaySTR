"""Provider-agnostic vision call — Claude or Google Gemini.

Two places in the pipeline read a lot's photo with an LLM:
  - `pass2.Phase2Scraper._classify_by_image_api` — condition audit
    ("is this item broken?").
  - `vision_enrich.EbayImageEnricher` — identity ("what IS this?").

Both default to Claude Haiku. Gemini's free Google-AI-Studio tier is the
zero-marginal-cost option for small auctions — grab a key at
https://aistudio.google.com/apikey (no billing required) and drop it in
`config.json` under `gemini.api_key`.

Only the GEMINI path lives here. The Claude calls stay inline in their
own modules so their tuned prompts / schemas are byte-for-byte untouched
when the user picks Claude. This module is a thin, dependency-free REST
client (raw httpx) so it shares the same truststore SSL context the rest
of the app uses to survive Norton / corp-MITM TLS inspection — the exact
issue that killed every Anthropic call until we passed the context
explicitly (see scraper/_ssl_compat.py).
"""
from __future__ import annotations

import base64
import json
from typing import Optional

import httpx

from scraper._ssl_compat import make_ssl_context

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

# One module-level client. Building the SSL context isn't free and the
# call sites are hot (one request per audited photo). httpx.Client is safe
# for concurrent use, which matters — pass2 audits photos in a thread pool.
_CLIENT: Optional[httpx.Client] = None


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(timeout=45.0, verify=make_ssl_context())
    return _CLIENT


def gemini_vision_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    image_bytes: bytes,
    media_type: str,
    user_text: str,
    response_schema: dict,
) -> Optional[dict]:
    """Send one image + prompt to Gemini; return the parsed JSON dict.

    Returns None on ANY failure (no key, HTTP error, safety block, empty
    or unparseable body) so callers fall through exactly as they already
    do when a Claude call returns None — no new error handling needed.

    `response_schema` must be a Gemini-compatible schema: a subset of
    OpenAPI 3.0 (type/properties/required/enum/propertyOrdering). Gemini
    rejects `additionalProperties`, so pass a schema without it.
    """
    if not api_key or not image_bytes:
        return None
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": media_type, "data": b64}},
                {"text": user_text},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "maxOutputTokens": 700,
            "temperature": 0,
            # NB: no `thinkingConfig`. The Gemini 3.x flash models 400 on
            # `thinkingBudget` (that knob was 2.5-era, now retired), and a
            # 700-token cap is plenty for the short structured output even
            # with their default brief thinking — verified finishReason=STOP
            # on real auction photos across flash-lite-latest / 3.1-lite.
        },
    }
    try:
        r = _client().post(
            _GEMINI_ENDPOINT.format(model=model),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if r.status_code != 200:
            return None
        body = r.json()
        cands = body.get("candidates") or []
        if not cands:
            return None
        parts = ((cands[0].get("content") or {}).get("parts")) or []
        text = ""
        for p in parts:
            if isinstance(p, dict) and p.get("text"):
                text = p["text"]
                break
        if not text:
            return None
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def gemini_text_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    response_schema: dict,
) -> Optional[dict]:
    """Text-only Gemini call (no image) → parsed JSON dict or None.

    Lets the condition auditor run its TEXT tier on Gemini too, so a user
    with only a free Google key gets a complete audit with zero Claude
    spend. Same failure semantics as `gemini_vision_json`.
    """
    if not api_key:
        return None
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "maxOutputTokens": 500,
            "temperature": 0,
            # No `thinkingConfig` — see note in gemini_vision_json.
        },
    }
    try:
        r = _client().post(
            _GEMINI_ENDPOINT.format(model=model),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if r.status_code != 200:
            return None
        body = r.json()
        cands = body.get("candidates") or []
        if not cands:
            return None
        parts = ((cands[0].get("content") or {}).get("parts")) or []
        text = ""
        for p in parts:
            if isinstance(p, dict) and p.get("text"):
                text = p["text"]
                break
        if not text:
            return None
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
