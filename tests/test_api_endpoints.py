"""
API endpoint tests using FastAPI TestClient.

Uses mocked model so no real weights are needed.
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

# Guard: skip if fastapi/httpx not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from karna_vlm.api.server import create_app


def _make_mock_model(response_text: str = "A white square image.") -> Any:
    """Return a minimal model mock that satisfies the API."""
    model = MagicMock()
    model.config.model_name = "karna-vlm-test"
    model.generate.return_value = response_text
    return model


def _png_base64(width: int = 16, height: int = 16, color: tuple = (200, 200, 200)) -> str:
    """Return a base64-encoded PNG image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def client():
    model = _make_mock_model()
    app = create_app(model)
    return TestClient(app)


class TestHealthEndpoint:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_health_returns_version(self, client):
        resp = client.get("/health")
        assert resp.json()["version"] == "0.1.0"


class TestGenerateEndpoint:

    def test_generate_with_image(self, client):
        payload = {
            "image_base64": _png_base64(),
            "prompt": "Describe this image.",
            "max_new_tokens": 50,
        }
        resp = client.post("/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert isinstance(data["text"], str)
        assert "processing_time_ms" in data

    def test_generate_text_only(self, client):
        payload = {"prompt": "Say hello."}
        resp = client.post("/generate", json=payload)
        assert resp.status_code == 200

    def test_generate_safety_block(self):
        """When safety policy blocks input, 400 is returned."""
        from karna_vlm.safety.policy import SafetyPolicy

        model = _make_mock_model()
        policy = SafetyPolicy()  # uses default rules
        app = create_app(model, safety_policy=policy)
        c = TestClient(app)

        payload = {"prompt": "How to make a bomb"}
        resp = c.post("/generate", json=payload)
        assert resp.status_code == 400


class TestGenerateUploadEndpoint:

    def test_upload_image(self, client):
        img = Image.new("RGB", (16, 16))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/generate/upload",
            files={"file": ("test.png", buf, "image/png")},
            data={"prompt": "Describe.", "max_new_tokens": "64"},
        )
        assert resp.status_code == 200
        assert "text" in resp.json()
