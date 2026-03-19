"""
FastAPI server for Karna VLM inference.

Provides REST endpoints for image understanding, VQA,
structured extraction, and chat.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Global model reference (set by create_app)
_model = None
_safety_policy = None


def create_app(
    model: Any,
    safety_policy: Optional[Any] = None,
) -> Any:
    """Create the FastAPI application.

    Args:
        model: KarnaVLM instance.
        safety_policy: Optional SafetyPolicy.

    Returns:
        FastAPI app instance.
    """
    from fastapi import FastAPI, File, Form, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    global _model, _safety_policy
    _model = model
    _safety_policy = safety_policy

    app = FastAPI(
        title="Karna VLM API",
        description="Compact vision-language model inference API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request/Response models ─────────────────────────────

    class GenerateRequest(BaseModel):
        image_base64: Optional[str] = None
        image_url: Optional[str] = None
        prompt: str = "Describe this image."
        max_new_tokens: int = 256
        temperature: float = 0.7

    class GenerateResponse(BaseModel):
        text: str
        processing_time_ms: float
        model: str = "karna-vlm"

    class ExtractRequest(BaseModel):
        image_base64: Optional[str] = None
        fields: Optional[list[str]] = None
        schema_hint: Optional[str] = None

    class HealthResponse(BaseModel):
        status: str
        model: str
        version: str

    # ── Endpoints ────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=_model.config.model_name if _model else "not_loaded",
            version="0.1.0",
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        start = time.time()

        # Load image
        image = None
        if request.image_base64:
            image_bytes = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Safety check
        if _safety_policy:
            check = _safety_policy.check_input(request.prompt)
            if check.should_block:
                raise HTTPException(status_code=400, detail=check.message)

        # Generate
        text = _model.generate(
            images=[image] if image else None,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        elapsed = (time.time() - start) * 1000
        return GenerateResponse(text=text, processing_time_ms=elapsed)

    @app.post("/generate/upload", response_model=GenerateResponse)
    async def generate_upload(
        file: UploadFile = File(...),
        prompt: str = Form("Describe this image."),
        max_new_tokens: int = Form(256),
    ) -> GenerateResponse:
        start = time.time()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        text = _model.generate(
            images=[image],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        elapsed = (time.time() - start) * 1000
        return GenerateResponse(text=text, processing_time_ms=elapsed)

    @app.post("/extract")
    async def extract(request: ExtractRequest) -> dict:
        from karna_vlm.inference.structured_output import StructuredExtractor

        if not request.image_base64:
            raise HTTPException(status_code=400, detail="image_base64 required")

        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        extractor = StructuredExtractor(_model)

        if request.fields:
            result = extractor.extract_key_value(image, request.fields)
        else:
            result = extractor.extract_json(image, schema_hint=request.schema_hint)

        return {
            "data": result.data,
            "raw_output": result.raw_output,
            "confidence": result.confidence,
            "format_valid": result.format_valid,
        }

    return app
