"""Shared validation for paths inside the agent work directory."""

from __future__ import annotations

from pathlib import Path


class WorkPathResolver:
    """Resolve model-provided relative paths without allowing workspace escapes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, raw_path: str) -> Path:
        """Resolve a relative path and ensure it remains below the work root."""
        if not isinstance(raw_path, str):
            raise ValueError("path must be a string")
        path_text = raw_path.strip()
        if not path_text:
            raise ValueError("path must not be empty")
        path = Path(path_text)
        if path.is_absolute():
            raise ValueError("absolute paths are not allowed")
        resolved = (self.root / path).resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path escapes the work directory") from exc
        return resolved

    def relative(self, raw_path: str) -> str:
        """Return a normalized relative path suitable for sandbox commands."""
        return self.resolve(raw_path).relative_to(self.root).as_posix()
