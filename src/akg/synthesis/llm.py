
import json
import re

import requests

from akg.config import Settings

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(Exception):
    """Transport failure or unparseable output. Never a valid observation."""


def parse_json(raw: str) -> tuple[dict, bool]:
    try:
        return json.loads(raw), False
    except (json.JSONDecodeError, TypeError):
        pass
    match = _JSON_BLOCK.search(raw or "")
    if match:
        try:
            return json.loads(match.group(0)), True
        except json.JSONDecodeError:
            pass
    raise LLMError(f"model output could not be parsed as JSON: {(raw or '')[:200]!r}")


def _post(
    settings: Settings,
    model: str,
    system: str,
    user: str,
    temperature: float,
    schema: dict | None,
) -> str:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }
    if schema is not None:
        payload["format"] = schema
    try:
        resp = requests.post(
            f"{settings.ollama_url}/api/chat", json=payload, timeout=600
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(
            f"Ollama request to {settings.ollama_url} failed for model "
            f"'{model}': {e}. Is Ollama running, and has '{model}' been pulled?"
        ) from e
    return resp.json().get("message", {}).get("content", "")


def chat(
    settings: Settings,
    model: str,
    system: str,
    user: str,
    schema: dict,
    temperature: float = 0.0,
) -> tuple[dict, bool]:
    return parse_json(_post(settings, model, system, user, temperature, schema))


def chat_text(
    settings: Settings,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
) -> str:
    return _post(settings, model, system, user, temperature, schema=None)
