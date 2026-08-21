"""SearXNG implementation of the provider-neutral search interface."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from .base import SearchResponse, SearchResult


MAX_RESPONSE_BYTES = 2_097_152


class SearxngSearchProvider:
    """Search through a fixed SearXNG JSON endpoint."""

    def __init__(self, url: str, timeout: float) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("web:url must use http:// or https://")
        if timeout <= 0:
            raise ValueError("web:timeout must be positive")
        self.url = url
        self.timeout = timeout

    def search(
        self,
        query: str,
        max_results: int,
        language: str | None,
        time_range: str | None,
    ) -> SearchResponse:
        parameters = {"q": query, "format": "json"}
        if language is not None:
            parameters["language"] = language
        if time_range is not None:
            parameters["time_range"] = time_range
        separator = "&" if "?" in self.url else "?"
        request = Request(
            f"{self.url}{separator}{urlencode(parameters)}",
            headers={"Accept": "application/json", "User-Agent": "terminal-agent/1"},
        )
        try:
            # A self-hosted SearXNG endpoint is contacted directly. Inherited
            # process proxies commonly reject private-network addresses.
            opener = build_opener(ProxyHandler({}))
            with opener.open(request, timeout=self.timeout) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise RuntimeError(f"search provider returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"search provider request failed: {exc.reason}") from exc
        if len(payload) > MAX_RESPONSE_BYTES:
            raise RuntimeError("search provider response exceeds 2097152 bytes")
        try:
            data = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("search provider returned invalid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RuntimeError("search provider returned an unexpected response shape")

        results: list[SearchResult] = []
        for item in data["results"]:
            if not isinstance(item, dict):
                continue
            title, url = item.get("title"), item.get("url")
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            snippet = item.get("content") if isinstance(item.get("content"), str) else ""
            source = item.get("engine") if isinstance(item.get("engine"), str) else None
            published_at = (
                item.get("publishedDate")
                if isinstance(item.get("publishedDate"), str)
                else None
            )
            results.append(SearchResult(title, url, snippet, source, published_at))
            if len(results) >= max_results:
                break

        warnings = tuple(
            f"{entry[0]}: {entry[1]}"
            for entry in data.get("unresponsive_engines", [])
            if isinstance(entry, list)
            and len(entry) >= 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], str)
        )
        return SearchResponse(tuple(results), warnings)
