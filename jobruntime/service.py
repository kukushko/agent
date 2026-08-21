"""Host-side execution and filesystem broker for sandboxed jobs."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from mcpbridge.client import McpToolCollection
from sandbox import DockerSandbox, SandboxError
from tools.base import Delegation
from tools.runtime import ToolRegistry


POLL_INTERVAL_SECONDS = 0.02
DOCKER_STOP_TIMEOUT_SECONDS = 5.0
MAX_REQUEST_BYTES = 2_097_152
MAX_CODE_CHARS = 131_072


class ToolExecutor(Protocol):
    def execute(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class JobTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    delegation: Delegation


@dataclass(frozen=True)
class JobPolicy:
    """Host-controlled resource limits for one job."""

    wall_time_seconds: int = 600
    quotas: dict[str, int] = field(default_factory=dict)
    result_bytes: int = 1_048_576
    stdout_bytes: int = 1_048_576
    stderr_bytes: int = 1_048_576
    memory: str = "512m"
    cpus: float = 1.0
    pids_limit: int = 64


@dataclass
class JobUsage:
    resources: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.resources.items()))


class JobService:
    """Run persisted jobs and broker their calls to local and external tools."""

    def __init__(
        self,
        root: Path,
        sandbox: DockerSandbox,
        policy: JobPolicy = JobPolicy(),
    ) -> None:
        self.root = root.resolve()
        self.sandbox = sandbox
        self.policy = policy
        self.local_registry: ToolRegistry | None = None
        self.external_tools: McpToolCollection | None = None
        self._tools: dict[str, JobTool] = {}
        self._quota_overrides: dict[str, int] = {}

    @property
    def is_available(self) -> bool:
        return self.sandbox.is_available

    @property
    def unavailable_reason(self) -> str | None:
        return self.sandbox.unavailable_reason

    def bind(
        self,
        local_registry: ToolRegistry,
        external_tools: McpToolCollection,
    ) -> None:
        """Bind the complete internal catalog after package discovery."""
        self.local_registry = local_registry
        self.external_tools = external_tools
        tools: dict[str, JobTool] = {}
        for definition in local_registry.definitions():
            delegation = definition.metadata.delegation
            if delegation.allowed:
                tools[definition.name] = JobTool(
                    definition.name,
                    definition.description,
                    {**definition.parameters, "additionalProperties": False},
                    delegation,
                )
        for tool in external_tools.tool_descriptors():
            delegation = delegation_from_metadata(tool.metadata)
            if delegation.allowed:
                if tool.name in tools:
                    raise ValueError(f"duplicate job tool name: {tool.name}")
                tools[tool.name] = JobTool(
                    tool.name, tool.description, tool.input_schema, delegation
                )
        quota_defaults: dict[str, int] = {}
        for tool in tools.values():
            for resource, limit in tool.delegation.quota_defaults.items():
                previous = quota_defaults.setdefault(resource, limit)
                if previous != limit:
                    raise ValueError(
                        f"conflicting default quota for resource {resource}: "
                        f"{previous} and {limit}"
                    )
        unknown_overrides = sorted(set(self._quota_overrides) - set(quota_defaults))
        if unknown_overrides:
            raise ValueError(
                f"job quota override has no declared resource: {unknown_overrides[0]}"
            )
        quota_defaults.update(self._quota_overrides)
        self.policy = replace(self.policy, quotas=quota_defaults)
        self._tools = tools

    def configure(
        self,
        *,
        wall_time_seconds: int,
        quota_overrides: dict[str, int],
    ) -> None:
        """Apply package-owned policy values before the service is bound."""
        if self._tools:
            raise RuntimeError("job service cannot be configured after binding")
        if wall_time_seconds < 1 or any(
            value < 1 for value in quota_overrides.values()
        ):
            raise ValueError("job policy values must be positive")
        self.policy = replace(self.policy, wall_time_seconds=wall_time_seconds)
        self._quota_overrides = dict(quota_overrides)

    def tool_descriptors(self) -> tuple[JobTool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def run(self, code: str) -> dict[str, object]:
        if not self._tools or self.local_registry is None or self.external_tools is None:
            raise RuntimeError("job service has not been bound to a tool catalog")
        if not code.strip():
            raise ValueError("job code must not be empty")
        if len(code) > MAX_CODE_CHARS:
            raise ValueError(f"job code exceeds {MAX_CODE_CHARS} characters")
        self.sandbox.ensure_image()
        job_id = create_job_id()
        job_dir = self.root / job_id
        create_job_directory(job_dir)
        write_json_atomic(job_dir / "manifest.json", {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "created",
        })
        write_json_atomic(job_dir / "policy.json", asdict(self.policy))
        write_json_atomic(
            job_dir / "tools.json",
            {
                tool.name: {
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in self.tool_descriptors()
            },
        )
        (job_dir / "job.py").write_text(code, encoding="utf-8")
        usage = JobUsage()
        outcome = self._execute(job_id, job_dir, usage)
        return outcome

    def _execute(
        self, job_id: str, job_dir: Path, usage: JobUsage
    ) -> dict[str, object]:
        container_name = f"terminal-agent-job-{uuid.uuid4().hex}"
        command = docker_command(
            self.sandbox,
            container_name,
            job_dir,
            self.policy,
        )
        stdout_path = job_dir / "stdout.txt"
        stderr_path = job_dir / "stderr.txt"
        stdout_file = stdout_path.open("wb")
        stderr_file = stderr_path.open("wb")
        limit_event = threading.Event()
        process: subprocess.Popen[bytes] | None = None
        started = time.monotonic()
        timed_out = False
        output_limit_exceeded = False
        broker_error: str | None = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdout is not None and process.stderr is not None
            threads = [
                threading.Thread(
                    target=pump_bounded,
                    args=(process.stdout, stdout_file, self.policy.stdout_bytes, limit_event),
                    daemon=True,
                ),
                threading.Thread(
                    target=pump_bounded,
                    args=(process.stderr, stderr_file, self.policy.stderr_bytes, limit_event),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            update_state(job_dir, "running", usage, started)
            while process.poll() is None:
                try:
                    self._service_requests(job_dir, usage)
                except (OSError, RuntimeError, ValueError) as exc:
                    broker_error = f"job broker failed: {exc}"
                    kill_container(container_name)
                    break
                if limit_event.is_set():
                    output_limit_exceeded = True
                    kill_container(container_name)
                    break
                if time.monotonic() - started >= self.policy.wall_time_seconds:
                    timed_out = True
                    kill_container(container_name)
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
            process.wait(timeout=DOCKER_STOP_TIMEOUT_SECONDS)
            if broker_error is None:
                self._service_requests(job_dir, usage)
            for thread in threads:
                thread.join(timeout=DOCKER_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.wait(timeout=DOCKER_STOP_TIMEOUT_SECONDS)
        except OSError as exc:
            raise SandboxError(f"failed to start job container: {exc}") from exc
        finally:
            if process is not None and process.poll() is None:
                process.kill()
            remove_container(container_name)
            stdout_file.close()
            stderr_file.close()

        duration_ms = int((time.monotonic() - started) * 1000)
        exit_code = process.returncode if process is not None else None
        result_file = job_dir / "result.json"
        error_file = job_dir / "error.json"
        if broker_error is not None:
            status, error = "failed", broker_error
        elif timed_out:
            status, error = "timed_out", "job exceeded its wall-time limit"
        elif output_limit_exceeded:
            status, error = "failed", "job exceeded its output limit"
        elif exit_code != 0:
            status = "failed"
            error = read_error(error_file) or f"job container exited with code {exit_code}"
        elif not result_file.exists():
            status, error = "failed", "job did not produce result.json"
        else:
            status, error = "completed", None

        result: Any = None
        if status == "completed":
            try:
                result = read_json_nofollow(result_file, self.policy.result_bytes)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                status, error = "failed", f"job returned invalid JSON: {exc}"
        update_state(job_dir, status, usage, started, duration_ms, error)
        response: dict[str, object] = {
            "job_id": job_id,
            "status": status,
            "result": result,
            "usage": usage.as_dict(),
            "duration_ms": duration_ms,
        }
        if error is not None:
            response["error"] = error
        return response

    def _service_requests(self, job_dir: Path, usage: JobUsage) -> None:
        pending = job_dir / "requests" / "pending"
        processing = job_dir / "requests" / "processing"
        completed = job_dir / "requests" / "completed"
        responses = job_dir / "responses"
        for directory in (pending, processing, completed, responses):
            require_safe_directory(job_dir, directory)
        for request_path in sorted(pending.glob("*.json")):
            if request_path.is_symlink() or not request_path.is_file():
                request_path.unlink(missing_ok=True)
                continue
            claimed = processing / request_path.name
            try:
                request_path.replace(claimed)
            except FileNotFoundError:
                continue
            response = self._handle_request(claimed, usage)
            write_json_atomic_in_directory(responses, claimed.name, response)
            append_json_line(job_dir, job_dir / "calls.jsonl", response)
            claimed.replace(completed / claimed.name)
            update_state(job_dir, "running", usage, None)

    def _handle_request(
        self, path: Path, usage: JobUsage
    ) -> dict[str, object]:
        request_id: object = path.stem
        try:
            request = read_json_nofollow(path, MAX_REQUEST_BYTES)
            if not isinstance(request, dict):
                raise ValueError("tool request must be an object")
            request_id = request.get("id", request_id)
            name, arguments = request.get("tool"), request.get("arguments", {})
            if not isinstance(name, str) or name not in self._tools:
                raise ValueError(f"tool is not available to jobs: {name}")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            self._charge(name, usage)
            result = self._execute_tool(name, arguments)
            return {
                "id": request_id,
                "tool": name,
                "ok": bool(result.ok),
                "result": result.result,
                "usage": usage.as_dict(),
            }
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "id": request_id,
                "ok": False,
                "result": {"error": str(exc)},
                "usage": usage.as_dict(),
            }

    def _charge(self, name: str, usage: JobUsage) -> None:
        costs = self._tools[name].delegation.costs
        for resource, amount in costs.items():
            limit = self.policy.quotas.get(resource)
            used = usage.resources.get(resource, 0)
            if limit is not None and used + amount > limit:
                raise RuntimeError(f"job resource budget is exhausted: {resource}")
        for resource, amount in costs.items():
            usage.resources[resource] = usage.resources.get(resource, 0) + amount

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self.local_registry is not None and self.external_tools is not None
        if any(definition.name == name for definition in self.local_registry.definitions()):
            return self.local_registry.execute(name, arguments)
        return self.external_tools.execute(name, arguments)


def delegation_from_metadata(metadata: dict[str, Any] | None) -> Delegation:
    """Decode delegation metadata, defaulting compatible MCP tools to one call."""
    raw = (metadata or {}).get("delegation", {})
    if not isinstance(raw, dict):
        return Delegation()
    allowed = raw.get("allowed", True)
    raw_costs = raw.get("costs", {"tool_calls": 1})
    raw_quotas = raw.get("quota_defaults", {"tool_calls": 200})
    raw_instructions = raw.get("instructions", [])
    reason = raw.get("reason")
    if (
        not isinstance(allowed, bool)
        or not isinstance(raw_costs, dict)
        or not isinstance(raw_quotas, dict)
        or not isinstance(raw_instructions, list)
    ):
        return Delegation()
    costs = {
        resource: amount
        for resource, amount in raw_costs.items()
        if isinstance(resource, str)
        and resource
        and isinstance(amount, int)
        and not isinstance(amount, bool)
        and amount > 0
    }
    quota_defaults = {
        resource: limit
        for resource, limit in raw_quotas.items()
        if isinstance(resource, str)
        and resource
        and isinstance(limit, int)
        and not isinstance(limit, bool)
        and limit > 0
    }
    return Delegation(
        allowed=allowed,
        costs=costs,
        quota_defaults=quota_defaults,
        instructions=tuple(
            value for value in raw_instructions if isinstance(value, str) and value
        ),
        reason=reason if isinstance(reason, str) else None,
    )


def create_job_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def create_job_directory(job_dir: Path) -> None:
    for relative in (
        "requests/pending",
        "requests/processing",
        "requests/completed",
        "responses",
        "artifacts",
    ):
        (job_dir / relative).mkdir(parents=True, exist_ok=False)


def docker_command(
    sandbox: DockerSandbox,
    container_name: str,
    job_dir: Path,
    policy: JobPolicy,
) -> list[str]:
    command = [
        "docker", "run", "--name", container_name, "--rm", "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", str(policy.pids_limit), "--memory", policy.memory,
        "--cpus", str(policy.cpus), "--user", f"{os.getuid()}:{os.getgid()}",
    ]
    for group_id in os.getgroups():
        if group_id != os.getgid():
            command.extend(("--group-add", str(group_id)))
    command.extend(
        (
            "--workdir", "/job", "--mount",
            f"type=bind,src={job_dir},dst=/job",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--env", f"JOB_CALL_TIMEOUT={policy.wall_time_seconds}",
            sandbox.image, "python", "/opt/terminal-agent/bin/run_job.py",
        )
    )
    return command


def pump_bounded(
    source: BinaryIO,
    destination: BinaryIO,
    limit: int,
    limit_event: threading.Event,
) -> None:
    written = 0
    try:
        while True:
            chunk = source.read(65_536)
            if not chunk:
                return
            remaining = max(0, limit - written)
            accepted = chunk[:remaining]
            if accepted:
                destination.write(accepted)
                destination.flush()
                written += len(accepted)
            if len(accepted) != len(chunk):
                limit_event.set()
                return
    finally:
        source.close()


def kill_container(name: str) -> None:
    try:
        subprocess.run(
            ["docker", "kill", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DOCKER_STOP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def remove_container(name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "--force", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DOCKER_STOP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def append_json_line(job_dir: Path, path: Path, value: Any) -> None:
    if path.is_symlink() or path.parent.resolve() != job_dir.resolve():
        raise RuntimeError("job call journal path is unsafe")
    directory_fd = os.open(
        job_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    finally:
        os.close(directory_fd)


def update_state(
    job_dir: Path,
    status: str,
    usage: JobUsage,
    started: float | None,
    duration_ms: int | None = None,
    error: str | None = None,
) -> None:
    state: dict[str, Any] = {"status": status, "usage": usage.as_dict()}
    if duration_ms is not None:
        state["duration_ms"] = duration_ms
    elif started is not None:
        state["duration_ms"] = int((time.monotonic() - started) * 1000)
    if error is not None:
        state["error"] = error
    write_json_atomic(job_dir / "state.json", state)


def read_error(path: Path) -> str | None:
    if not is_safe_regular_file(path.parent, path):
        return None
    try:
        value = read_json_nofollow(path, MAX_REQUEST_BYTES)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value.get("error") if isinstance(value, dict) and isinstance(value.get("error"), str) else None


def require_safe_directory(job_dir: Path, path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"job IPC directory is unsafe: {path.name}")
    try:
        path.resolve().relative_to(job_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"job IPC directory escapes its job: {path.name}") from exc


def is_safe_regular_file(job_dir: Path, path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.resolve().relative_to(job_dir.resolve())
    except ValueError:
        return False
    return True


def read_json_nofollow(path: Path, limit: int) -> Any:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("JSON path is not a regular file")
        if metadata.st_size > limit:
            raise ValueError(f"JSON file exceeds {limit} bytes")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            return json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_json_atomic_in_directory(
    directory: Path, name: str, value: Any
) -> None:
    directory_fd = os.open(
        directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
