import json
import os
import re
import time
from typing import Any

from anthropic import Anthropic, APIConnectionError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/anthropic"

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = Anthropic(api_key=api_key, base_url=BASE_URL, timeout=300.0, max_retries=0)
    return _client


def chat(system: str, user: str, max_tokens: int = 4096, temperature: float = 0.7, max_retries: int = 8) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _get_client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if getattr(resp, "stop_reason", None) == "max_tokens":
                print(f"[llm] WARNING: response truncated at max_tokens={max_tokens}")
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return "".join(parts).strip()
        except (APIConnectionError, APITimeoutError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[llm] connection error ({type(e).__name__}); retrying in {wait}s…")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (429, 500, 502, 503, 504):
                last_err = e
                wait = 2 ** attempt
                print(f"[llm] HTTP {e.status_code}; retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {max_retries} retries") from last_err


def chat_json(system: str, user: str, max_tokens: int = 4096, temperature: float = 0.4) -> dict[str, Any]:
    """Call the model and parse a JSON object from the response.

    Tolerates fenced code blocks (```json ... ```) and leading prose by extracting
    the first balanced {...} block.
    """
    raw = chat(system=system, user=user, max_tokens=max_tokens, temperature=temperature)
    return _extract_json(raw)


def _extract_json(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model response:\n{text}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(candidate)):
        c = candidate[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(candidate[start : i + 1])
    raise ValueError(f"Could not parse JSON from response:\n{text}")
