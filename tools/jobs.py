"""Persistent sandboxed jobs that can orchestrate other tools."""

from __future__ import annotations

from typing import Protocol

from .base import (
    Availability,
    Delegation,
    ToolEnvironment,
    ToolExposure,
    ToolPackage,
    tool,
    with_tool_success,
)


JOB_PROMPT_INSTRUCTIONS = (
    "Python jobs:",
    "- Use the job tool when loops, conditions, filtering, aggregation, or many tool calls should execute without returning every intermediate value to the model context.",
    "- Design one end-to-end job for the user's task before calling it. Put predictable discovery, selection, fetching, managed-file reading, parsing, calculation, and semantic interpretation steps in the same program; return only the compact final evidence or answer in result.",
    "- Do not use repeated jobs as wrappers around one delegated call each. Run another job only when the previous job genuinely failed or returned evidence that could not have been anticipated when writing it.",
    "- Pass a concise Python program in its code argument without Markdown fences.",
    "- Every delegated tool call must begin with the literal tools. prefix: catalog name package.method becomes tools.package.method(...). Never call package.method(...) without tools.",
    "- The tools proxy is already available. Importing tools is supported but unnecessary; do not install or implement a tools module.",
    "- A tool with exactly one required parameter accepts that value positionally; otherwise use keyword arguments.",
    "- Tool results support both value.key and value['key']; arrays are ordinary Python lists.",
    "- Failed calls raise ToolError. Always assign the final JSON-compatible value to the global variable result; printing it is not a substitute.",
    "- Never use Python open(), pathlib, subprocess, shell commands, os.walk(), or similar local filesystem APIs to find or read files named or requested by the user. They see only the job's private directory, not managed work files; use delegated file tools instead.",
    "- After a successful job result, answer the user from that result. Do not run another job unless the result reports a failure or lacks required data.",
)


class JobRunner(Protocol):
    """Service contract consumed by the jobs tool package."""

    @property
    def is_available(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str | None: ...

    def configure(
        self,
        *,
        wall_time_seconds: int,
        quota_overrides: dict[str, int],
    ) -> None: ...

    def run(self, code: str) -> dict[str, object]: ...


class JobTools(ToolPackage):
    """Run concise Python programs with brokered access to authorized tools."""

    namespace = "jobs"
    parameter_defaults = {
        "wall_time": "600",
        "quota_tool_calls": "200",
    }

    @classmethod
    def accepts_parameter(cls, name: str) -> bool:
        return super().accepts_parameter(name) or name.startswith("quota_")

    def __init__(self, environment: ToolEnvironment) -> None:
        super().__init__(environment)
        service = environment.services.require(JobRunner)
        service.configure(
            wall_time_seconds=self._positive_integer("wall_time"),
            quota_overrides=self._quota_overrides(),
        )

    @classmethod
    def check_availability(cls, environment: ToolEnvironment) -> Availability:
        service = environment.services.optional(JobRunner)
        if service is None:
            return Availability(False, "job service is not configured")
        if service.is_available:
            return Availability(True)
        return Availability(False, service.unavailable_reason or "job sandbox is unavailable")

    @tool(
        exposure=ToolExposure(
            direct=True,
            delegated=False,
            orchestration_entrypoint=True,
        ),
        delegation=Delegation(
            allowed=False,
            costs={},
            quota_defaults={},
            reason="Recursive jobs are not supported",
        ),
        prompt_instructions=JOB_PROMPT_INSTRUCTIONS,
        epistemic_roles=("orchestration",),
    )
    def run(self, code: str) -> dict[str, object]:
        """Run sandboxed Python that orchestrates delegated tools and returns its JSON-compatible result."""
        service = self.environment.services.require(JobRunner)
        outcome = service.run(normalize_generated_code(code))
        if outcome.get("status") == "completed":
            outcome["next_action"] = (
                "The job completed successfully. Answer the user directly from "
                "the result; do not repeat the operation or call another tool."
            )
        return with_tool_success(outcome, outcome.get("status") == "completed")

    def _positive_integer(self, name: str) -> int:
        try:
            value = int(self.parameter(name))
        except ValueError as exc:
            raise ValueError(f"jobs:{name} must be an integer") from exc
        if value < 1:
            raise ValueError(f"jobs:{name} must be positive")
        return value

    def _quota_overrides(self) -> dict[str, int]:
        configured = {
            **self.parameter_defaults,
            **self.environment.tool_parameters.get(self.namespace, {}),
        }
        return {
            name.removeprefix("quota_"): self._parse_positive(name, raw_value)
            for name, raw_value in configured.items()
            if name.startswith("quota_")
        }

    @staticmethod
    def _parse_positive(name: str, raw_value: str) -> int:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"jobs:{name} must be an integer") from exc
        if value < 1:
            raise ValueError(f"jobs:{name} must be positive")
        return value


TOOL_PACKAGES = (JobTools,)


def normalize_generated_code(code: str) -> str:
    """Repair one extra JSON-escaping layer only when the original is invalid."""
    try:
        compile(code, "job.py", "exec")
        return code
    except SyntaxError:
        candidate = (
            code.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
        )
        try:
            compile(candidate, "job.py", "exec")
        except SyntaxError:
            return code
        return candidate
