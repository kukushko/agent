"""Stateless structured-output client for delegated language-model calls."""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from jsonschema import ValidationError, validators

from tools.llm import LanguageModelFailure, LanguageModelResponse


THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class OpenAIJsonService:
    """Call an OpenAI-compatible endpoint without history or tool access."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self._opener = build_opener(ProxyHandler({}))

    def ask(
        self,
        instruction: str,
        data: dict[str, object],
        response_schema: dict[str, object],
        max_output_tokens: int,
    ) -> LanguageModelResponse:
        """Return schema-validated JSON, retrying malformed output once."""
        started = time.monotonic()
        validator_class = validators.validator_for(response_schema)
        validator_class.check_schema(response_schema)
        validator = validator_class(response_schema)
        usage = {"llm_input_tokens": 0, "llm_output_tokens": 0}
        raw_responses: list[dict[str, object]] = []
        messages = [
            {
                "role": "system",
                "content": (
                    "Perform the requested semantic transformation. Return only JSON "
                    "that satisfies the supplied JSON Schema. Do not use tools, assume "
                    "missing facts, or add commentary."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": instruction,
                        "data": data,
                        "response_schema": response_schema,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        last_error = "invalid structured response"
        for attempt in range(2):
            try:
                raw_response = self._complete(
                    messages, response_schema, max_output_tokens
                )
            except RuntimeError as exc:
                raise model_failure(
                    str(exc), started, usage, attempt + 1, raw_responses
                ) from exc
            raw_responses.append(raw_response)
            add_usage(usage, raw_response.get("usage"))
            content = ""
            try:
                content = response_content(raw_response)
                value = parse_json_content(content)
                validator.validate(value)
                return LanguageModelResponse(
                    value=value,
                    usage=usage,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    attempts=attempt + 1,
                    raw_responses=tuple(raw_responses),
                )
            except (RuntimeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                if attempt == 0:
                    if content:
                        messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous attempt failed: " + last_error + ". "
                                "Do not deliberate about unavailable facts. Express uncertainty "
                                "inside the requested schema when necessary, and return the "
                                "corrected JSON value immediately."
                            ),
                        }
                    )
        raise model_failure(
            f"LLM did not return schema-valid JSON after one repair: {last_error}",
            started,
            usage,
            2,
            raw_responses,
        )

    def _complete(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object],
        max_output_tokens: int,
    ) -> dict[str, object]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_output_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "delegated_response",
                    "schema": response_schema,
                    "strict": True,
                },
            },
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM returned invalid transport JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("LLM response must be a JSON object")
        return value


def response_content(response: dict[str, object]) -> str:
    try:
        choices = response["choices"]
        message = choices[0]["message"]  # type: ignore[index]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM response has no assistant content") from exc
    if not isinstance(content, str) or not content.strip():
        finish_reason = choices[0].get("finish_reason")  # type: ignore[index]
        if finish_reason == "length":
            raise RuntimeError(
                "LLM exhausted its output budget during reasoning without producing JSON"
            )
        raise RuntimeError(
            f"LLM returned empty assistant content (finish_reason={finish_reason!r})"
        )
    return content


def parse_json_content(content: str) -> object:
    cleaned = THINK_RE.sub("", content).strip()
    cleaned = FENCE_RE.sub("", cleaned).strip()
    return json.loads(cleaned)


def add_usage(total: dict[str, int], raw: object) -> None:
    if not isinstance(raw, dict):
        return
    for source, target in (
        ("prompt_tokens", "llm_input_tokens"),
        ("completion_tokens", "llm_output_tokens"),
    ):
        value = raw.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            total[target] += value


def model_failure(
    message: str,
    started: float,
    usage: dict[str, int],
    attempts: int,
    raw_responses: list[dict[str, object]],
) -> LanguageModelFailure:
    return LanguageModelFailure(
        message,
        usage=dict(usage),
        duration_ms=int((time.monotonic() - started) * 1000),
        attempts=attempts,
        raw_responses=tuple(raw_responses),
    )
