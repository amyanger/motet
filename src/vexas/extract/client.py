"""Anthropic API wrapper.

Kept deliberately thin: one call, one JSON object out. Retries handle
transient API errors only -- a malformed response is a data-quality signal
and is surfaced, not silently retried into existence.
"""

import json
import time

from anthropic import Anthropic, APIError

from vexas.config import settings

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and fill it in."
            )
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


def extract_json_block(raw: str) -> dict:
    """Pull the first complete JSON object out of a model response.

    Handles bare JSON, fenced blocks, and leading prose by scanning for a
    balanced brace span.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : index + 1])

    raise ValueError(f"no JSON object in response: {raw[:200]!r}")


def complete_json(prompt: str | list[dict], max_tokens: int = 4096,
                  max_retries: int = 3) -> dict:
    """Send one prompt, return the parsed JSON object.

    `prompt` is either a plain string or a list of content blocks. The block
    form carries the `cache_control` breakpoint that makes repeated calls
    about the same report cost 0.1x -- see extract/prompts.build_prompt.
    """
    client = _get_client()
    content = prompt if isinstance(prompt, list) else [
        {"type": "text", "text": prompt}
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=settings.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            _log_cache_usage(message)
            return extract_json_block(message.content[0].text)
        except APIError as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"API failed after {max_retries} attempts") from last_error


#: Running totals, so a silently broken cache is visible rather than
#: just expensive. If cache_read stays at zero across a corpus run, the
#: prefix is being invalidated somewhere -- check block ordering first.
CACHE_STATS = {"write": 0, "read": 0, "uncached": 0}


def _log_cache_usage(message) -> None:
    usage = message.usage
    CACHE_STATS["write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    CACHE_STATS["read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    CACHE_STATS["uncached"] += usage.input_tokens
