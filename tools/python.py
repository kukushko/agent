"""Sandboxed Python 3.12 tools."""

from __future__ import annotations

from pathlib import Path

from sandbox import SandboxIO, SandboxPolicy
from workpaths import WorkPathResolver

from .base import (
    Availability,
    Delegation,
    ToolEnvironment,
    ToolPackage,
    tool,
    with_tool_success,
)


PYTHON_POLICY = SandboxPolicy(
    timeout_seconds=15.0,
    network=False,
    memory="256m",
    cpus=1.0,
    pids_limit=64,
    max_stdout_bytes=1_048_576,
    max_stderr_bytes=1_048_576,
)
SYNTAX_POLICY = SandboxPolicy(
    timeout_seconds=5.0,
    network=False,
    memory="128m",
    cpus=1.0,
    pids_limit=32,
    max_stdout_bytes=65_536,
    max_stderr_bytes=65_536,
)
EVAL_POLICY = SandboxPolicy(
    timeout_seconds=5.0,
    network=False,
    memory="128m",
    cpus=1.0,
    pids_limit=32,
    max_stdout_bytes=262_144,
    max_stderr_bytes=65_536,
)
MAX_EXPRESSION_CHARS = 32_768
MAX_INLINE_EVAL_FILE_BYTES = 4_096
CHECKER_PATH = "/opt/terminal-agent/bin/check_python.py"
EVALUATOR_PATH = "/opt/terminal-agent/bin/eval_python.py"


class PythonTools(ToolPackage):
    """Python 3.12 syntax checking and execution in isolated containers."""

    namespace = "python"

    def __init__(self, environment: ToolEnvironment) -> None:
        super().__init__(environment)
        self.paths = WorkPathResolver(environment.work_dir)

    @classmethod
    def check_availability(cls, environment: ToolEnvironment) -> Availability:
        sandbox = environment.sandbox
        if sandbox.is_available:
            return Availability(True)
        return Availability(
            False, sandbox.unavailable_reason or "sandbox is unavailable"
        )

    @tool(epistemic_roles=("code_verification",))
    def check_syntax(
        self,
        path: str,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
    ) -> dict[str, object]:
        """Check a work file with Python 3.12; optional output paths keep diagnostics out of the context."""
        script = self._require_file(path)
        self._protect_inputs([script], stdout_path, stderr_path)
        result = self.environment.sandbox.run(
            ["python", CHECKER_PATH, self.paths.relative(path)],
            SYNTAX_POLICY,
            SandboxIO(stdout_path=stdout_path, stderr_path=stderr_path),
        )
        return with_tool_success(
            {
                "path": self.paths.relative(path),
                "valid": result.succeeded,
                **result.as_dict(),
            },
            result.succeeded,
        )

    @tool(
        delegation=Delegation(
            allowed=False,
            costs={},
            quota_defaults={},
            reason="Nested program execution is not supported",
        ),
        epistemic_roles=("execution",),
    )
    def run(
        self,
        path: str,
        arguments: list[str] | None = None,
        stdin_path: str | None = None,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
    ) -> dict[str, object]:
        """Run a Python 3.12 work file without network access; streams may use relative work-file paths."""
        script = self._require_file(path)
        self._protect_inputs([script], stdout_path, stderr_path)
        argv = ["python", f"./{self.paths.relative(path)}", *(arguments or [])]
        result = self.environment.sandbox.run(
            argv,
            PYTHON_POLICY,
            SandboxIO(
                stdin_path=stdin_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ),
        )
        return with_tool_success(
            {"path": self.paths.relative(path), **result.as_dict()},
            result.succeeded,
        )

    @tool(
        delegation=Delegation(
            allowed=False,
            costs={},
            quota_defaults={},
            reason="Nested program execution is not supported",
        ),
        epistemic_roles=("computation",),
        reliability_guidance=(
            "Use an available computation tool for non-trivial numerical calculations, multi-step arithmetic, or unit conversions instead of relying only on mental arithmetic. Combine related calculations into one expression returning a dictionary with descriptive keys when this avoids redundant calls and preserves each value's meaning.",
        ),
    )
    def eval(
        self,
        expression: str,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
    ) -> dict[str, object]:
        """Evaluate a Python 3.12 expression or short snippet with math preloaded; snippets must assign `result`, and comparisons should return a descriptively keyed dictionary."""
        if not expression.strip():
            raise ValueError("expression must not be empty")
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise ValueError(
                f"expression exceeds {MAX_EXPRESSION_CHARS} characters"
            )
        result = self.environment.sandbox.run(
            ["python", EVALUATOR_PATH, expression],
            EVAL_POLICY,
            SandboxIO(stdout_path=stdout_path, stderr_path=stderr_path),
        )
        payload = result.as_dict()
        if result.succeeded and result.stdout.content is not None:
            payload["value"] = result.stdout.content.rstrip("\n")
        elif (
            result.succeeded
            and stdout_path is not None
            and not result.stdout.truncated
            and result.stdout.bytes_written <= MAX_INLINE_EVAL_FILE_BYTES
        ):
            output_path = self.paths.resolve(stdout_path)
            if output_path.is_file():
                payload["value"] = output_path.read_text(encoding="utf-8").rstrip("\n")
        return with_tool_success(payload, result.succeeded)

    def _require_file(self, raw_path: str) -> Path:
        path = self.paths.resolve(raw_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"Python file does not exist or is not a file: {raw_path}")
        return path

    def _protect_inputs(
        self,
        protected: list[Path],
        stdout_path: str | None,
        stderr_path: str | None,
    ) -> None:
        outputs = [
            self.paths.resolve(raw_path)
            for raw_path in (stdout_path, stderr_path)
            if raw_path is not None
        ]
        if any(path in outputs for path in protected):
            raise ValueError("output paths must not overwrite the executed source file")


TOOL_PACKAGES = (PythonTools,)
