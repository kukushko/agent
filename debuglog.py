"""Persistent structured debug tracing for the agent process."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DebugLog:
    """Append process events to a dedicated JSON Lines file."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        unique = uuid.uuid4().hex[:8]
        self.path = directory / f"agent-{started}-{os.getpid()}-{unique}.jsonl"
        self._lock = threading.Lock()

    def record(self, event: str, **fields: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False, default=repr, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


_DEFAULT_LOG: DebugLog | None = None
_DEFAULT_LOCK = threading.Lock()


def default_debug_log() -> DebugLog:
    """Return the process-wide log rooted beside the agent executable."""
    global _DEFAULT_LOG
    with _DEFAULT_LOCK:
        if _DEFAULT_LOG is None:
            _DEFAULT_LOG = DebugLog(Path(__file__).resolve().parent / "log")
        return _DEFAULT_LOG
