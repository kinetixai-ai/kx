"""Every test runs offline in its own config/data dirs.

Network access goes through kinetixai.net.client(); tests that need HTTP
install an httpx.MockTransport with the `mock_http` fixture. Ollama is
reported as not running unless a test fakes it.
"""
from __future__ import annotations

import httpx
import pytest

from kinetixai import net
from kinetixai.rag import ollama


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("KX_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KX_NO_KEYRING", "1")
    for var in ("KX_TOKEN", "KINETIXAI_ENDPOINT", "OLLAMA_HOST", "GITHUB_TOKEN", "GH_TOKEN",
                "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(net, "_sleep", lambda s: None)
    monkeypatch.setattr(ollama, "models", lambda url, client=None: None)

    def no_network(*a, **k):
        def refuse(request):
            raise httpx.ConnectError(f"test tried to reach {request.url}")
        return httpx.Client(transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(net, "client", no_network)
    return tmp_path


@pytest.fixture
def mock_http(monkeypatch):
    """mock_http(handler) routes every kx HTTP call to handler(request) and
    records the requests in the returned list."""
    def install(handler):
        seen: list[httpx.Request] = []

        def wrapped(request):
            seen.append(request)
            return handler(request)

        def factory(headers=None, timeout=None):
            h = {"User-Agent": net.UA}
            h.update(headers or {})
            return httpx.Client(transport=httpx.MockTransport(wrapped), headers=h)
        monkeypatch.setattr(net, "client", factory)
        return seen
    return install


def make_pdf(path, text: str) -> None:
    """A minimal one-page PDF with real extractable text (no extra deps)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
