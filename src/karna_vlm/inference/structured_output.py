"""
Structured output extraction from VLM.

Extracts JSON, key-value pairs, tables, and other structured
formats from model outputs. Builds on Karna's OCR heritage.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of structured extraction."""

    data: dict[str, Any]
    raw_output: str
    confidence: float = 0.0
    format_valid: bool = False
    errors: list[str] = field(default_factory=list)


class StructuredExtractor:
    """Extract structured data from VLM outputs.

    Provides methods for extracting JSON, key-value pairs,
    tables, and custom schemas from model-generated text.

    Args:
        model: KarnaVLM instance.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def extract_json(
        self,
        image: Image.Image,
        schema_hint: Optional[str] = None,
        prompt: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> ExtractionResult:
        """Extract structured JSON from an image.

        Args:
            image: Input image.
            schema_hint: Optional JSON schema or field list.
            prompt: Custom extraction prompt.
            max_new_tokens: Max tokens for generation.

        Returns:
            ExtractionResult with parsed data.
        """
        if prompt is None:
            prompt = "Extract all structured information from this image as JSON."
            if schema_hint:
                prompt += f"\n\nExpected fields: {schema_hint}"

        raw_output = self.model.generate(
            images=[image],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for structured output
            do_sample=False,
        )

        return self._parse_json_output(raw_output)

    def extract_key_value(
        self,
        image: Image.Image,
        keys: list[str],
        max_new_tokens: int = 512,
    ) -> ExtractionResult:
        """Extract specific key-value pairs from an image.

        Args:
            image: Input image.
            keys: List of field names to extract.
            max_new_tokens: Max tokens.

        Returns:
            ExtractionResult.
        """
        keys_str = ", ".join(keys)
        prompt = (
            f"Extract the following fields from this image: {keys_str}\n"
            f"Return as JSON with these exact keys."
        )

        raw_output = self.model.generate(
            images=[image],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )

        result = self._parse_json_output(raw_output)

        # Validate required keys
        missing = [k for k in keys if k not in result.data]
        if missing:
            result.errors.append(f"Missing keys: {missing}")

        return result

    def extract_table(
        self,
        image: Image.Image,
        max_new_tokens: int = 1024,
    ) -> ExtractionResult:
        """Extract tabular data from an image.

        Args:
            image: Input image containing a table.
            max_new_tokens: Max tokens.

        Returns:
            ExtractionResult with table as list of row dicts.
        """
        prompt = (
            "Extract the table from this image. Return as JSON array where "
            "each row is an object with column headers as keys."
        )

        raw_output = self.model.generate(
            images=[image],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )

        return self._parse_json_output(raw_output)

    def _parse_json_output(self, raw: str) -> ExtractionResult:
        """Parse JSON from model output, handling common formatting issues."""
        errors = []

        # Try direct parse
        try:
            data = json.loads(raw)
            return ExtractionResult(data=data, raw_output=raw, format_valid=True, confidence=0.9)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return ExtractionResult(data=data, raw_output=raw, format_valid=True, confidence=0.8)
            except json.JSONDecodeError:
                errors.append("Found code block but couldn't parse JSON")

        # Try finding any JSON-like structure
        brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return ExtractionResult(data=data, raw_output=raw, format_valid=True, confidence=0.6)
            except json.JSONDecodeError:
                errors.append("Found braces but couldn't parse JSON")

        # Fallback: return raw as single field
        errors.append("Could not parse structured output")
        return ExtractionResult(
            data={"raw_text": raw},
            raw_output=raw,
            format_valid=False,
            confidence=0.1,
            errors=errors,
        )
