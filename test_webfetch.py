import json
import gzip
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from webfetch import FetchPolicy, OfflineBatchStore, ProxyPageFetcher
from webfetch.base import FetchResponse
from webfetch.converter import html_to_markdown
from webfetch.proxy import validate_public_url


class RecordingProxyHandler(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        type(self).requests.append(self.path)
        content_encoding = None
        if self.path.endswith("/redirect-private"):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1/secret")
            self.end_headers()
            return
        if self.path.endswith("/large"):
            body = b"x" * 100
            content_type = "text/plain; charset=utf-8"
        elif self.path.endswith("/compressed"):
            body = gzip.compress(b"x" * 100)
            content_type = "text/plain; charset=utf-8"
            content_encoding = "gzip"
        else:
            body = (
                b"<html><head><title>Proxy Article</title></head><body>"
                b"<article><h1>Useful heading</h1><p>Useful body text.</p></article>"
                b"</body></html>"
            )
            content_type = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if content_encoding:
            self.send_header("Content-Encoding", content_encoding)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class LocalProxy:
    def __enter__(self):
        RecordingProxyHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingProxyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}"
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class ProxyPageFetcherTest(unittest.TestCase):
    def test_forces_http_through_explicit_proxy(self):
        with LocalProxy() as proxy:
            fetcher = ProxyPageFetcher(proxy.url, FetchPolicy(timeout_seconds=2))
            response = fetcher.fetch("http://public.example/article")

        self.assertEqual(response.status, 200)
        self.assertIn(b"Useful body", response.body)
        self.assertEqual(
            RecordingProxyHandler.requests, ["http://public.example/article"]
        )

    def test_rejects_private_redirect_target(self):
        with LocalProxy() as proxy:
            fetcher = ProxyPageFetcher(proxy.url, FetchPolicy(timeout_seconds=2))
            with self.assertRaisesRegex(ValueError, "globally routable"):
                fetcher.fetch("http://public.example/redirect-private")

    def test_enforces_response_limit(self):
        with LocalProxy() as proxy:
            fetcher = ProxyPageFetcher(
                proxy.url,
                FetchPolicy(timeout_seconds=2, max_response_bytes=20),
            )
            with self.assertRaisesRegex(RuntimeError, "exceeds 20 bytes"):
                fetcher.fetch("http://public.example/large")

    def test_enforces_decoded_response_limit(self):
        with LocalProxy() as proxy:
            fetcher = ProxyPageFetcher(
                proxy.url,
                FetchPolicy(
                    timeout_seconds=2,
                    max_response_bytes=1_000,
                    max_decoded_bytes=20,
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "decoded page exceeds 20 bytes"):
                fetcher.fetch("http://public.example/compressed")

    def test_rejects_unsafe_destination_urls(self):
        for url in (
            "file:///etc/passwd",
            "http://localhost/",
            "http://127.0.0.1/",
            "http://user:secret@example.com/",
            "https://example.com:8443/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_public_url(url)


class HtmlConversionTest(unittest.TestCase):
    def test_extracts_html_with_trafilatura(self):
        converted = html_to_markdown(
            "<html><head><title>Title</title></head><body>"
            "<article><h1>Heading</h1><p>Body text.</p></article></body></html>",
            "https://example.com/article",
        )

        self.assertEqual(converted.title, "Title")
        self.assertIn("Body text", converted.markdown)
        self.assertEqual(converted.converter, "trafilatura")

    def test_uses_markdownify_when_extraction_returns_nothing(self):
        with patch("webfetch.converter.trafilatura.extract", return_value=None):
            converted = html_to_markdown(
                "<html><body><h1>Fallback</h1><p><strong>Text</strong></p></body></html>",
                "https://example.com/fallback",
            )

        self.assertTrue(converted.fallback_used)
        self.assertEqual(converted.converter, "markdownify")
        self.assertIn("# Fallback", converted.markdown)
        self.assertIn("**Text**", converted.markdown)


class OfflineBatchStoreTest(unittest.TestCase):
    def test_stores_documents_sidecars_and_manifest_with_safe_names(self):
        class FakeFetcher:
            def fetch(self, url, timeout_seconds=None):
                if "failed" in url:
                    raise RuntimeError("simulated failure")
                return FetchResponse(
                    url,
                    url,
                    200,
                    {"content-type": "text/html; charset=utf-8"},
                    b"<html><head><title>Saved</title></head><body>"
                    b"<article><p>Downloaded content.</p></article></body></html>",
                )

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            store = OfflineBatchStore(work, FakeFetcher())
            result = store.fetch_batch(
                [
                    "https://example.com/a%20page?x=1",
                    "https://failed.example/document",
                ],
                100_000,
            )
            manifest = json.loads((work / result["manifest"]).read_text())
            successful = result["documents"][0]
            markdown_path = work / successful["markdown_path"]
            sidecar_path = work / successful["metadata_path"]

            self.assertEqual(result["directory"].split("/")[0], "offline")
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(result["failed"], 1)
            self.assertRegex(markdown_path.name, r"^001-example-com-a-20page-[0-9a-f]{12}\.md$")
            self.assertIn("untrusted external web content", markdown_path.read_text())
            self.assertTrue(sidecar_path.is_file())
            self.assertEqual(manifest["documents"][0]["requested_url"], "https://example.com/a%20page?x=1")
            self.assertIn("metadata_path", json.loads(sidecar_path.read_text()))
            self.assertEqual(manifest["documents"][1]["error"]["message"], "simulated failure")


if __name__ == "__main__":
    unittest.main()
