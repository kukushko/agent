"""Proxy-only HTTP transport with bounded redirects and response bodies."""

from __future__ import annotations

import ipaddress
import zlib
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from .base import FetchPolicy, FetchResponse


ALLOWED_SCHEMES = frozenset(("http", "https"))
ALLOWED_PORTS = frozenset((80, 443))
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal")
USER_AGENT = "terminal-agent/1"


class ForcedProxyHandler(ProxyHandler):
    """Route matching requests through a fixed proxy without NO_PROXY bypasses."""

    def __init__(self, proxy_url: str) -> None:
        parsed = urlsplit(proxy_url)
        if parsed.scheme != "http" or parsed.hostname is None:
            raise ValueError("web:fetch_proxy_url must be an http:// URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("web:fetch_proxy_url must not contain credentials")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("web:fetch_proxy_url must not contain a path, query, or fragment")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("web:fetch_proxy_url contains an invalid port") from exc
        self.proxy_host = parsed.hostname
        self.proxy_port = port or 80
        self.proxy_netloc = f"{self.proxy_host}:{self.proxy_port}"
        super().__init__({})

    def proxy_open(self, request: Request, proxy: str, request_type: str):
        request.set_proxy(self.proxy_netloc, "http")
        return None

    def http_open(self, request: Request):
        self.proxy_open(request, "", "http")
        return None

    https_open = http_open


class ValidatingRedirectHandler(HTTPRedirectHandler):
    """Validate every redirect target and enforce a small redirect budget."""

    def __init__(self, max_redirects: int) -> None:
        self.max_redirections = max_redirects
        self.max_repeats = max_redirects

    def redirect_request(self, request, response, code, message, headers, new_url):
        validated = validate_public_url(urljoin(request.full_url, new_url))
        return super().redirect_request(
            request, response, code, message, headers, validated
        )


class ProxyPageFetcher:
    """Fetch public pages exclusively through one explicitly configured proxy."""

    def __init__(self, proxy_url: str, policy: FetchPolicy = FetchPolicy()) -> None:
        if policy.timeout_seconds <= 0:
            raise ValueError("web:fetch_timeout must be positive")
        if policy.max_response_bytes < 1:
            raise ValueError("web:fetch_max_response_bytes must be positive")
        if policy.max_decoded_bytes < 1:
            raise ValueError("web:fetch_max_decoded_bytes must be positive")
        if policy.max_redirects < 0:
            raise ValueError("web:fetch_max_redirects must be non-negative")
        self.policy = policy
        self.opener = build_opener(
            ForcedProxyHandler(proxy_url),
            ValidatingRedirectHandler(policy.max_redirects),
        )

    def fetch(self, url: str, timeout_seconds: float | None = None) -> FetchResponse:
        requested_url = validate_public_url(url)
        timeout = self.policy.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise RuntimeError("web fetch batch timeout exhausted")
        request = Request(
            requested_url,
            headers={
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Encoding": "identity",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                wire_body = response.read(self.policy.max_response_bytes + 1)
                final_url = validate_public_url(response.geturl())
                status = int(response.status)
                headers = {name.lower(): value for name, value in response.headers.items()}
        except HTTPError as exc:
            raise RuntimeError(f"page returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"page request failed: {exc.reason}") from exc
        if len(wire_body) > self.policy.max_response_bytes:
            raise RuntimeError(
                f"page response exceeds {self.policy.max_response_bytes} bytes"
            )
        body = decode_content(wire_body, headers.get("content-encoding"), self.policy.max_decoded_bytes)
        return FetchResponse(
            requested_url, final_url, status, headers, body, len(wire_body)
        )


def decode_content(body: bytes, encoding: str | None, limit: int) -> bytes:
    """Decode supported HTTP content encodings without exceeding the output limit."""
    normalized = (encoding or "identity").strip().lower()
    if normalized in ("", "identity"):
        if len(body) > limit:
            raise RuntimeError(f"decoded page exceeds {limit} bytes")
        return body
    window_bits = {
        "gzip": zlib.MAX_WBITS | 16,
        "deflate": zlib.MAX_WBITS,
    }.get(normalized)
    if window_bits is None:
        raise RuntimeError(f"unsupported page content encoding: {normalized}")
    decoder = zlib.decompressobj(window_bits)
    try:
        decoded = decoder.decompress(body, limit + 1)
        if len(decoded) <= limit:
            decoded += decoder.flush(limit + 1 - len(decoded))
    except zlib.error as exc:
        raise RuntimeError(f"invalid {normalized} page response") from exc
    if len(decoded) > limit or decoder.unconsumed_tail:
        raise RuntimeError(f"decoded page exceeds {limit} bytes")
    return decoded


def validate_public_url(raw_url: str) -> str:
    """Validate and normalize a model- or server-provided destination URL."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("URL must be a non-empty string")
    if any(ord(character) < 32 for character in raw_url):
        raise ValueError("URL must not contain control characters")
    normalized, _ = urldefrag(raw_url.strip())
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("URL must use http:// or https://")
    if parsed.hostname is None:
        raise ValueError("URL must contain a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc
    if port is not None and port not in ALLOWED_PORTS:
        raise ValueError("URL port must be 80 or 443")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise ValueError("URL hostname is not a public Internet destination")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("URL IP address is not globally routable")
    return normalized
