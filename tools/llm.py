"""Delegated structured language-model interpretation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .base import (
    Availability,
    Delegation,
    ToolEnvironment,
    ToolExposure,
    ToolPackage,
    tool,
    with_tool_resource_usage,
    with_tool_success,
)


@dataclass(frozen=True)
class LanguageModelResponse:
    """Validated result and accounting returned by a model service."""

    value: object
    usage: dict[str, int]
    duration_ms: int
    attempts: int
    raw_responses: tuple[dict[str, object], ...]


class LanguageModelFailure(RuntimeError):
    """Failure carrying measured usage and provider responses for diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, int],
        duration_ms: int,
        attempts: int,
        raw_responses: tuple[dict[str, object], ...],
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.duration_ms = duration_ms
        self.attempts = attempts
        self.raw_responses = raw_responses


class LanguageModelService(Protocol):
    """Service contract consumed by the delegated LLM tool package."""

    def ask(
        self,
        instruction: str,
        data: dict[str, object],
        response_schema: dict[str, object],
        max_output_tokens: int,
    ) -> LanguageModelResponse: ...


class LlmTools(ToolPackage):
    """Apply bounded semantic transformations to job data."""

    namespace = "llm"
    parameter_defaults = {
        "max_output_tokens": "4096",
        "max_input_chars": "131072",
    }

    @classmethod
    def check_availability(cls, environment: ToolEnvironment) -> Availability:
        if environment.services.optional(LanguageModelService) is None:
            return Availability(False, "language-model service is not configured")
        return Availability(True)

    @tool(
        exposure=ToolExposure(direct=False, delegated=True),
        delegation=Delegation(
            costs={"tool_calls": 1, "llm_calls": 1},
            quota_defaults={
                "tool_calls": 200,
                "llm_calls": 2,
                "llm_input_tokens": 131072,
                "llm_output_tokens": 8192,
            },
            instructions=(
                "Use only for semantic interpretation that ordinary Python cannot perform. "
                "Pass compact data, state a precise instruction, and provide a strict JSON Schema.",
            ),
        ),
    )
    def ask(
        self,
        instruction: str,
        data: dict[str, object],
        response_schema: dict[str, object],
        max_output_tokens: int = 4096,
    ) -> dict[str, object]:
        """Ask a stateless model to transform compact data into schema-validated JSON."""
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        input_chars = len(
            json.dumps(
                {
                    "instruction": instruction,
                    "data": data,
                    "response_schema": response_schema,
                },
                ensure_ascii=False,
            )
        )
        configured_input_limit = self._positive_parameter("max_input_chars")
        if input_chars > configured_input_limit:
            raise ValueError(
                f"LLM input exceeds the configured {configured_input_limit}-character limit"
            )
        configured_limit = self._positive_parameter("max_output_tokens")
        if max_output_tokens < 1 or max_output_tokens > configured_limit:
            raise ValueError(
                f"max_output_tokens must be between 1 and {configured_limit}"
            )
        job_dir = self._job_directory()
        service = self.environment.services.require(LanguageModelService)
        try:
            response = service.ask(
                instruction, data, response_schema, max_output_tokens
            )
        except LanguageModelFailure as exc:
            artifact = self._write_artifact(
                job_dir,
                instruction,
                data,
                response_schema,
                max_output_tokens,
                value=None,
                usage=exc.usage,
                duration_ms=exc.duration_ms,
                attempts=exc.attempts,
                raw_responses=exc.raw_responses,
                error=str(exc),
            )
            failure: dict[str, object] = {
                "error": str(exc),
                "usage": exc.usage,
                "duration_ms": exc.duration_ms,
                "attempts": exc.attempts,
                "artifact": artifact,
            }
            return with_tool_resource_usage(
                with_tool_success(failure, False), exc.usage
            )
        artifact = self._write_artifact(
            job_dir,
            instruction,
            data,
            response_schema,
            max_output_tokens,
            value=response.value,
            usage=response.usage,
            duration_ms=response.duration_ms,
            attempts=response.attempts,
            raw_responses=response.raw_responses,
            error=None,
        )
        result: dict[str, object] = {
            "value": response.value,
            "usage": response.usage,
            "duration_ms": response.duration_ms,
            "attempts": response.attempts,
            "artifact": artifact,
        }
        return with_tool_resource_usage(result, response.usage)

    def _positive_parameter(self, name: str) -> int:
        try:
            value = int(self.parameter(name))
        except ValueError as exc:
            raise ValueError(f"llm:{name} must be an integer") from exc
        if value < 1:
            raise ValueError(f"llm:{name} must be positive")
        return value

    def _job_directory(self) -> Path:
        value = self.environment.context.state.get("job_dir")
        if not isinstance(value, Path):
            raise RuntimeError("llm.ask is available only inside a delegated job")
        return value

    @staticmethod
    def _write_artifact(
        job_dir: Path,
        instruction: str,
        data: dict[str, object],
        response_schema: dict[str, object],
        max_output_tokens: int,
        *,
        value: object,
        usage: dict[str, int],
        duration_ms: int,
        attempts: int,
        raw_responses: tuple[dict[str, object], ...],
        error: str | None,
    ) -> str:
        directory = job_dir / "llm"
        directory.mkdir(mode=0o700, exist_ok=True)
        name = f"call-{uuid4().hex}.json"
        path = directory / name
        path.write_text(
            json.dumps(
                {
                    "instruction": instruction,
                    "data": data,
                    "response_schema": response_schema,
                    "max_output_tokens": max_output_tokens,
                    "ok": error is None,
                    "value": value,
                    "error": error,
                    "usage": usage,
                    "duration_ms": duration_ms,
                    "attempts": attempts,
                    "raw_responses": raw_responses,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return f"llm/{name}"


TOOL_PACKAGES = (LlmTools,)
