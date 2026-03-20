"""
Prompt templates for different VLM tasks.

Templates control how the prompt is formatted before being packed
with image tokens and fed to the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from karna_vlm.data.schemas import TaskType


@dataclass
class PromptTemplate:
    """A prompt template for a specific task or instruction format.

    Attributes:
        name: Template identifier.
        system_prompt: Optional system-level instruction.
        user_template: Template for user messages ({prompt} will be replaced).
        assistant_prefix: Prefix for the assistant response.
        image_token: Where to insert image tokens.
    """

    name: str
    system_prompt: str = ""
    user_template: str = "<image>\n{prompt}"
    assistant_prefix: str = ""
    image_token: str = "<image>"

    def format_prompt(
        self,
        prompt: str,
        task_type: Optional[TaskType] = None,
    ) -> str:
        """Format a prompt using this template.

        Args:
            prompt: The raw user prompt/question.
            task_type: Optional task type for task-specific formatting.

        Returns:
            Formatted prompt string with image placeholder.
        """
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.append(self.user_template.format(prompt=prompt))
        if self.assistant_prefix:
            parts.append(self.assistant_prefix)
        return "\n".join(parts)


# ── Built-in templates ────────────────────────────────────────────

TEMPLATES: dict[str, PromptTemplate] = {
    "default": PromptTemplate(
        name="default",
        user_template="<image>\n{prompt}",
    ),

    "caption": PromptTemplate(
        name="caption",
        system_prompt="You are a helpful image captioning assistant.",
        user_template="<image>\nDescribe this image in detail.",
        assistant_prefix="",
    ),

    "vqa": PromptTemplate(
        name="vqa",
        system_prompt="Answer the question about the image concisely.",
        user_template="<image>\nQuestion: {prompt}\nAnswer:",
    ),

    "instruction": PromptTemplate(
        name="instruction",
        system_prompt="You are a helpful multimodal assistant. Follow the user's instructions carefully.",
        user_template="<image>\n{prompt}",
    ),

    "chat": PromptTemplate(
        name="chat",
        system_prompt="You are a helpful assistant that can see and discuss images.",
        user_template="<image>\nUser: {prompt}\nAssistant:",
    ),

    "ocr": PromptTemplate(
        name="ocr",
        system_prompt="Extract all text visible in the image. Preserve formatting where possible.",
        user_template="<image>\n{prompt}",
    ),

    "grounding": PromptTemplate(
        name="grounding",
        system_prompt="Locate objects in the image. Provide bounding boxes as [x1, y1, x2, y2] normalized to 0-1000.",
        user_template="<image>\nLocate: {prompt}",
    ),

    "structured_extraction": PromptTemplate(
        name="structured_extraction",
        system_prompt="Extract structured information from the image. Return valid JSON.",
        user_template="<image>\nExtract the following information:\n{prompt}\nOutput JSON:",
    ),

    # Llama-style chat template
    "llama_chat": PromptTemplate(
        name="llama_chat",
        system_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>",
        user_template="<|start_header_id|>user<|end_header_id|>\n<image>\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n",
    ),
}


def get_template(name: str) -> PromptTemplate:
    """Get a template by name, falling back to default."""
    return TEMPLATES.get(name, TEMPLATES["default"])
