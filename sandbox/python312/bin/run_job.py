#!/usr/bin/env python3
"""Execute one persisted job with a dynamic filesystem-backed tool proxy."""

from __future__ import annotations

import builtins
import json
import os
import sys
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Any


JOB_DIR = Path("/job")
TOOLS_FILE = JOB_DIR / "tools.json"
PENDING_DIR = JOB_DIR / "requests" / "pending"
RESPONSES_DIR = JOB_DIR / "responses"
CALL_TIMEOUT = float(os.environ.get("JOB_CALL_TIMEOUT", "600"))


def job_open(*args: Any, **kwargs: Any) -> Any:
    """Open a private artifact and explain the work-file boundary on misses."""
    try:
        return builtins.open(*args, **kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc.filename!r} is not a private job artifact. This does not mean "
            "the file is absent from the agent work directory; access managed "
            "files through an available delegated tool instead of open()."
        ) from exc


class ToolError(RuntimeError):
    """A brokered tool call failed."""

    def __init__(self, tool: str, details: dict[str, Any]) -> None:
        self.tool = tool
        self.details = details
        message = details.get("error", "tool call failed")
        super().__init__(f"{tool}: {message}")


class JobObject(dict):
    """Dictionary with convenient read access through attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ToolNode:
    """A lazily resolved namespace or callable tool path."""

    def __init__(self, client: "ToolClient", path: tuple[str, ...]) -> None:
        self._client = client
        self._path = path

    def __getattr__(self, name: str) -> "ToolNode":
        if name.startswith("_"):
            raise AttributeError(name)
        return ToolNode(self._client, (*self._path, name))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.call(".".join(self._path), args, kwargs)


class ToolClient:
    """Convert Python calls into atomic filesystem broker requests."""

    def __init__(self, schemas: dict[str, Any]) -> None:
        self.schemas = schemas
        self._counter = 0

    def __getattr__(self, name: str) -> ToolNode:
        if name.startswith("_"):
            raise AttributeError(name)
        return ToolNode(self, (name,))

    def call(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        metadata = self.schemas.get(name)
        if not isinstance(metadata, dict):
            raise AttributeError(f"unknown job tool: {name}")
        arguments = bind_arguments(name, metadata.get("input_schema"), args, kwargs)
        self._counter += 1
        request_id = f"{self._counter:06d}-{uuid.uuid4().hex[:8]}"
        request = {"id": request_id, "tool": name, "arguments": arguments}
        atomic_json(PENDING_DIR / f"{request_id}.json", request)
        response_path = RESPONSES_DIR / f"{request_id}.json"
        deadline = time.monotonic() + CALL_TIMEOUT
        while not response_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"tool call timed out: {name}")
            time.sleep(0.02)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict) or not response.get("ok"):
            details = response.get("result", {}) if isinstance(response, dict) else {}
            if not isinstance(details, dict):
                details = {"error": str(details)}
            raise ToolError(name, details)
        return wrap(response.get("result"))


def bind_arguments(
    name: str,
    raw_schema: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw_schema, dict):
        raise TypeError(f"tool {name} has no object input schema")
    properties = raw_schema.get("properties", {})
    required = raw_schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise TypeError(f"tool {name} has an invalid input schema")
    if len(args) > 1:
        raise TypeError(f"tool {name} accepts at most one positional argument")
    arguments = dict(kwargs)
    if args:
        required_names = [value for value in required if isinstance(value, str)]
        if len(required_names) != 1:
            raise TypeError(
                f"tool {name} requires keyword arguments because it does not have "
                "exactly one required parameter"
            )
        positional_name = required_names[0]
        if positional_name in arguments:
            raise TypeError(f"tool {name} received {positional_name!r} twice")
        arguments[positional_name] = args[0]
    unknown = sorted(set(arguments) - set(properties))
    if unknown and raw_schema.get("additionalProperties") is False:
        raise TypeError(f"tool {name} got an unexpected argument: {unknown[0]}")
    return arguments


def wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return JobObject({key: wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [wrap(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def create_tools_module(client: ToolClient) -> types.ModuleType:
    """Expose the dynamic proxy as both an injected value and importable modules."""
    root = types.ModuleType("tools")
    root.__path__ = []
    modules: dict[tuple[str, ...], types.ModuleType] = {(): root}
    sys.modules["tools"] = root
    for tool_name in sorted(client.schemas):
        parts = tool_name.split(".")
        parent = root
        prefix: tuple[str, ...] = ()
        for part in parts[:-1]:
            prefix = (*prefix, part)
            module = modules.get(prefix)
            if module is None:
                qualified = "tools." + ".".join(prefix)
                module = types.ModuleType(qualified)
                module.__path__ = []
                modules[prefix] = module
                sys.modules[qualified] = module
                setattr(parent, part, module)
            parent = module
        setattr(parent, parts[-1], ToolNode(client, tuple(parts)))
    return root


def main() -> int:
    schemas = json.loads(TOOLS_FILE.read_text(encoding="utf-8"))
    client = ToolClient(schemas)
    namespace: dict[str, Any] = {
        "tools": create_tools_module(client),
        "ToolError": ToolError,
        "__name__": "__job__",
        "__builtins__": {**vars(builtins), "open": job_open},
    }
    try:
        code = (JOB_DIR / "job.py").read_text(encoding="utf-8")
        os.chdir(JOB_DIR / "artifacts")
        exec(compile(code, "job.py", "exec"), namespace)
        if "result" not in namespace:
            raise RuntimeError("job code must assign a JSON-compatible value to result")
        serialized = json.dumps(namespace["result"], ensure_ascii=False)
        atomic_json(JOB_DIR / "result.json", json.loads(serialized))
        return 0
    except BaseException as exc:
        traceback.print_exc()
        atomic_json(
            JOB_DIR / "error.json",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
