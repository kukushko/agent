"""Provider-neutral webpage fetch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class FetchPolicy:
    """Host-controlled limits for one proxied webpage request."""

    timeout_seconds: float = 20.0
    max_response_bytes: int = 2_097_152
    max_decoded_bytes: int = 8_388_608
    max_redirects: int = 5


@dataclass(frozen=True)
class FetchResponse:
    """Bounded response returned by a page-fetch transport."""

    requested_url: str
    final_url: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    wire_bytes: int | None = None


class PageFetcher(Protocol):
    """Transport contract consumed by offline document storage."""

    def fetch(self, url: str, timeout_seconds: float | None = None) -> FetchResponse:
        """Fetch one public HTTP(S) URL through the configured transport."""
