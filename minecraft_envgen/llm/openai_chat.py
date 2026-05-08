from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

from minecraft_envgen.core.contracts import CurriculumDraft


def _load_dotenv(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _resolve_api_settings(dotenv_path: str | None = None) -> Tuple[str, str]:
    dotenv = _load_dotenv(dotenv_path)
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or dotenv.get("OPENAI_API_KEY")
        or dotenv.get("API_KEY")
    )
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or API_KEY, or pass --dotenv.")

    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("API_BASE")
        or dotenv.get("OPENAI_BASE_URL")
        or dotenv.get("API_BASE")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    return api_key, base_url


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    for index in range(start, len(stripped)):
        char = stripped[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
                raise ValueError("LLM response JSON root must be an object.")
    raise ValueError("Failed to locate a complete JSON object in LLM response.")


def generate_draft_with_openai(
    *,
    prompt: str,
    model: str = "gpt-5-nano",
    dotenv_path: str | None = None,
    timeout: int = 180,
) -> Tuple[CurriculumDraft, str]:
    api_key, base_url = _resolve_api_settings(dotenv_path)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed with HTTP {exc.code}: {detail}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response did not contain choices.")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("OpenAI response content was not a string.")

    draft_payload = _extract_json_object(content)
    return CurriculumDraft.from_dict(draft_payload), content
