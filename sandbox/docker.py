"""Ephemeral Docker-based sandbox implementation."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from workpaths import WorkPathResolver

from .base import (
    SandboxError,
    SandboxIO,
    SandboxPolicy,
    SandboxResult,
    SandboxStreamResult,
)


FINGERPRINT_LABEL = "org.terminal-agent.sandbox.context-sha256"
PROBE_TIMEOUT_SECONDS = 3.0
BUILD_TIMEOUT_SECONDS = 600.0
KILL_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.02


@dataclass
class _StreamState:
    path: str | None
    limit: int
    output: BinaryIO | None
    captured: bytearray
    bytes_written: int = 0
    truncated: bool = False


class DockerSandbox:
    """Run one isolated Docker container for every sandbox invocation."""

    def __init__(self, work_dir: Path, image: str, build_context: Path) -> None:
        self.work_dir = work_dir.resolve()
        self.build_context = build_context.resolve()
        self.image = image
        self.paths = WorkPathResolver(self.work_dir)
        self._available: bool | None = None
        self._unavailable_reason: str | None = None
        self._image_lock = threading.Lock()
        self._verified_fingerprint: str | None = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self.refresh_availability()
        return bool(self._available)

    @property
    def unavailable_reason(self) -> str | None:
        if self._available is None:
            self.refresh_availability()
        return self._unavailable_reason

    def refresh_availability(self) -> bool:
        executable = shutil.which("docker")
        if executable is None:
            self._set_unavailable("Docker executable was not found")
            return False
        try:
            result = subprocess.run(
                [executable, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._set_unavailable(f"Docker availability probe failed: {exc}")
            return False
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            self._set_unavailable(f"Docker daemon is not accessible: {detail}")
            return False
        self._available = True
        self._unavailable_reason = None
        return True

    def ensure_image(self) -> None:
        if not self.is_available:
            raise SandboxError(self.unavailable_reason or "Docker is unavailable")
        fingerprint = build_context_fingerprint(self.build_context)
        if self._verified_fingerprint == fingerprint:
            return
        with self._image_lock:
            if self._verified_fingerprint == fingerprint:
                return
            current = self._image_fingerprint()
            if current != fingerprint:
                self._build_image(fingerprint)
            self._verified_fingerprint = fingerprint

    def run(
        self,
        argv: list[str],
        policy: SandboxPolicy,
        io: SandboxIO = SandboxIO(),
    ) -> SandboxResult:
        if not argv or not all(isinstance(value, str) and value for value in argv):
            raise SandboxError("sandbox argv must contain non-empty strings")
        validate_policy(policy)
        self.ensure_image()

        stdin_path = self._input_path(io.stdin_path)
        stdout_path = self._output_path(io.stdout_path)
        stderr_path = self._output_path(io.stderr_path)
        self._validate_io_collisions(stdin_path, stdout_path, stderr_path)

        container_name = f"terminal-agent-sandbox-{uuid.uuid4().hex}"
        docker_argv = self._docker_argv(container_name, argv, policy)
        stdin_stream: BinaryIO | int
        stdin_file: BinaryIO | None = None
        if stdin_path is None:
            stdin_stream = subprocess.DEVNULL
        else:
            stdin_file = stdin_path.open("rb")
            stdin_stream = stdin_file

        stdout_state = self._stream_state(
            io.stdout_path, stdout_path, policy.max_stdout_bytes
        )
        stderr_state = self._stream_state(
            io.stderr_path, stderr_path, policy.max_stderr_bytes
        )
        limit_event = threading.Event()
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        timed_out = False
        exit_code: int | None = None
        try:
            try:
                process = subprocess.Popen(
                    docker_argv,
                    stdin=stdin_stream,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                raise SandboxError(f"failed to start Docker sandbox: {exc}") from exc
            assert process.stdout is not None
            assert process.stderr is not None
            threads = [
                threading.Thread(
                    target=pump_stream,
                    args=(process.stdout, stdout_state, limit_event),
                    daemon=True,
                ),
                threading.Thread(
                    target=pump_stream,
                    args=(process.stderr, stderr_state, limit_event),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()

            while process.poll() is None:
                if limit_event.is_set():
                    self._kill_container(container_name)
                    break
                if time.monotonic() - started >= policy.timeout_seconds:
                    timed_out = True
                    self._kill_container(container_name)
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
            try:
                process.wait(timeout=KILL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=KILL_TIMEOUT_SECONDS)
            for thread in threads:
                thread.join(timeout=KILL_TIMEOUT_SECONDS)
            exit_code = None if timed_out or limit_event.is_set() else process.returncode
        finally:
            if process is not None and process.poll() is None:
                process.kill()
            self._remove_container(container_name)
            if stdin_file is not None:
                stdin_file.close()
            close_stream_state(stdout_state)
            close_stream_state(stderr_state)

        duration_ms = int((time.monotonic() - started) * 1000)
        result = SandboxResult(
            exit_code=exit_code,
            stdout=stream_result(stdout_state),
            stderr=stream_result(stderr_state),
            timed_out=timed_out,
            output_limit_exceeded=limit_event.is_set(),
            duration_ms=duration_ms,
        )
        if process is not None and process.returncode == 125:
            detail = result.stderr.content or "Docker failed to start the sandbox"
            self._available = None
            self._verified_fingerprint = None
            raise SandboxError(detail.strip())
        return result

    def _docker_argv(
        self, container_name: str, argv: list[str], policy: SandboxPolicy
    ) -> list[str]:
        network = "bridge" if policy.network else "none"
        command = [
            "docker",
            "run",
            "--name",
            container_name,
            "--rm",
            "--interactive",
            "--network",
            network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(policy.pids_limit),
            "--memory",
            policy.memory,
            "--cpus",
            str(policy.cpus),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
        ]
        for group_id in os.getgroups():
            if group_id != os.getgid():
                command.extend(["--group-add", str(group_id)])
        command.extend(
            [
                "--workdir",
                "/workspace",
                "--mount",
                f"type=bind,src={self.work_dir},dst=/workspace",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=64m",
                self.image,
                *argv,
            ]
        )
        return command

    def _image_fingerprint(self) -> str | None:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    f'{{{{ index .Config.Labels "{FINGERPRINT_LABEL}" }}}}',
                    self.image,
                ],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"failed to inspect sandbox image: {exc}") from exc
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value if value and value != "<no value>" else None

    def _build_image(self, fingerprint: str) -> None:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "--label",
                    f"{FINGERPRINT_LABEL}={fingerprint}",
                    "--tag",
                    self.image,
                    str(self.build_context),
                ],
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"failed to build sandbox image: {exc}") from exc
        if result.returncode != 0:
            detail = tail_text(result.stderr or result.stdout, 8_192)
            raise SandboxError(f"failed to build sandbox image: {detail}")

    def _set_unavailable(self, reason: str) -> None:
        self._available = False
        self._unavailable_reason = reason

    def _input_path(self, raw_path: str | None) -> Path | None:
        if raw_path is None:
            return None
        path = self.paths.resolve(raw_path)
        if not path.exists() or not path.is_file():
            raise SandboxError(f"stdin file does not exist or is not a file: {raw_path}")
        return path

    def _output_path(self, raw_path: str | None) -> Path | None:
        if raw_path is None:
            return None
        path = self.paths.resolve(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_file():
            raise SandboxError(f"output path is not a file: {raw_path}")
        return path

    def _stream_state(
        self, raw_path: str | None, path: Path | None, limit: int
    ) -> _StreamState:
        output = path.open("wb") if path is not None else None
        return _StreamState(raw_path, limit, output, bytearray())

    def _validate_io_collisions(
        self, stdin: Path | None, stdout: Path | None, stderr: Path | None
    ) -> None:
        if stdin is not None and stdin in (stdout, stderr):
            raise SandboxError("stdin path must differ from output paths")
        if stdout is not None and stdout == stderr:
            raise SandboxError("stdout and stderr paths must differ")

    def _kill_container(self, name: str) -> None:
        try:
            subprocess.run(
                ["docker", "kill", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=KILL_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _remove_container(self, name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=KILL_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def validate_policy(policy: SandboxPolicy) -> None:
    if policy.timeout_seconds <= 0:
        raise SandboxError("sandbox timeout must be positive")
    if policy.cpus <= 0 or policy.pids_limit < 1:
        raise SandboxError("sandbox CPU and process limits must be positive")
    if policy.max_stdout_bytes < 1 or policy.max_stderr_bytes < 1:
        raise SandboxError("sandbox output limits must be positive")


def build_context_fingerprint(build_context: Path) -> str:
    if not build_context.is_dir():
        raise SandboxError(f"sandbox build context does not exist: {build_context}")
    files = sorted(
        path
        for path in build_context.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not files:
        raise SandboxError(f"sandbox build context is empty: {build_context}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(build_context).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"x" if os.access(path, os.X_OK) else b"-")
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def pump_stream(
    stream: BinaryIO, state: _StreamState, limit_event: threading.Event
) -> None:
    try:
        while True:
            chunk = stream.read(65_536)
            if not chunk:
                return
            remaining = max(0, state.limit - state.bytes_written)
            accepted = chunk[:remaining]
            if accepted:
                if state.output is not None:
                    state.output.write(accepted)
                    state.output.flush()
                else:
                    state.captured.extend(accepted)
                state.bytes_written += len(accepted)
            if len(accepted) < len(chunk):
                state.truncated = True
                limit_event.set()
    finally:
        stream.close()


def close_stream_state(state: _StreamState) -> None:
    if state.output is not None:
        state.output.close()


def stream_result(state: _StreamState) -> SandboxStreamResult:
    content = None
    if state.output is None:
        content = state.captured.decode("utf-8", errors="replace")
    return SandboxStreamResult(
        content=content,
        path=state.path,
        bytes_written=state.bytes_written,
        truncated=state.truncated,
    )


def tail_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[-limit:]
