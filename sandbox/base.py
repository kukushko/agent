"""Types shared by sandbox implementations and tool packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SandboxError(RuntimeError):
    """Report an unavailable or failed sandbox operation."""


@dataclass(frozen=True)
class SandboxPolicy:
    """Host-controlled limits for one sandbox invocation."""

    timeout_seconds: float = 10.0
    network: bool = False
    memory: str = "256m"
    cpus: float = 1.0
    pids_limit: int = 64
    max_stdout_bytes: int = 1_048_576
    max_stderr_bytes: int = 1_048_576


@dataclass(frozen=True)
class SandboxIO:
    """Optional workspace files connected to standard process streams."""

    stdin_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None


@dataclass(frozen=True)
class SandboxStreamResult:
    """Captured text or file metadata for one output stream."""

    content: str | None
    path: str | None
    bytes_written: int
    truncated: bool

    def as_dict(self) -> dict[str, object]:
        if self.path is not None:
            return {
                "path": self.path,
                "bytes_written": self.bytes_written,
                "truncated": self.truncated,
            }
        return {
            "content": self.content or "",
            "bytes_written": self.bytes_written,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class SandboxResult:
    """Normalized outcome of one sandbox container invocation."""

    exit_code: int | None
    stdout: SandboxStreamResult
    stderr: SandboxStreamResult
    timed_out: bool
    output_limit_exceeded: bool
    duration_ms: int

    @property
    def succeeded(self) -> bool:
        """Return whether the sandboxed process completed successfully."""
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.output_limit_exceeded
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout.as_dict(),
            "stderr": self.stderr.as_dict(),
            "timed_out": self.timed_out,
            "output_limit_exceeded": self.output_limit_exceeded,
            "duration_ms": self.duration_ms,
        }


class Sandbox(Protocol):
    """Execution environment exposed to tool packages."""

    @property
    def is_available(self) -> bool:
        """Return whether the sandbox runtime can currently be used."""

    @property
    def unavailable_reason(self) -> str | None:
        """Return a diagnostic when the sandbox runtime is unavailable."""

    def refresh_availability(self) -> bool:
        """Probe the runtime again and return its availability."""

    def run(
        self,
        argv: list[str],
        policy: SandboxPolicy,
        io: SandboxIO = SandboxIO(),
    ) -> SandboxResult:
        """Run one command in a fresh sandbox."""


class UnavailableSandbox:
    """Sandbox placeholder used when no runtime was configured."""

    def __init__(self, reason: str = "sandbox is not configured") -> None:
        self._reason = reason

    @property
    def is_available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def refresh_availability(self) -> bool:
        return False

    def run(
        self,
        argv: list[str],
        policy: SandboxPolicy,
        io: SandboxIO = SandboxIO(),
    ) -> SandboxResult:
        raise SandboxError(self._reason)
