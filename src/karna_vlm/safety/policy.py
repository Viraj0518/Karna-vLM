"""
Safety policy system for Karna VLM.

Configurable content policies, refusal hooks, and safety
checks that can be applied at input, output, or both.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SafetyAction(IntEnum):
    """Actions to take when safety policy is triggered.

    Uses IntEnum so severity comparisons are numeric (not lexicographic).
    Higher values = more restrictive actions.
    """

    ALLOW = 0
    WARN = 1
    REDACT = 2
    REFUSE = 3

    # Convenience string representation for logging
    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class SafetyResult:
    """Result of a safety check."""

    safe: bool
    action: SafetyAction = SafetyAction.ALLOW
    triggered_rules: list[str] = field(default_factory=list)
    message: str = ""
    modified_content: Optional[str] = None

    @property
    def should_block(self) -> bool:
        return self.action == SafetyAction.REFUSE


@dataclass
class SafetyRule:
    """A single safety rule."""

    name: str
    description: str
    check_fn: Optional[Callable[[str], bool]] = None
    patterns: list[str] = field(default_factory=list)
    action: SafetyAction = SafetyAction.REFUSE
    refusal_message: str = "I cannot process this request due to safety policies."


class SafetyPolicy:
    """Configurable safety policy for VLM inputs and outputs.

    Provides:
    - Pattern-based content filtering
    - Custom rule hooks
    - Configurable refusal messages
    - Audit logging

    Args:
        rules: List of safety rules to enforce.
        strict_mode: If True, refuse on any rule match.
    """

    def __init__(
        self,
        rules: Optional[list[SafetyRule]] = None,
        strict_mode: bool = False,
    ) -> None:
        self.rules = rules or self._default_rules()
        self.strict_mode = strict_mode
        self.audit_log: list[dict[str, Any]] = []

    @staticmethod
    def _default_rules() -> list[SafetyRule]:
        """Default safety rules."""
        return [
            SafetyRule(
                name="harmful_instructions",
                description="Block requests for harmful/dangerous instructions",
                patterns=[
                    r"how\s+to\s+make\s+a?\s*(bomb|weapon|explosive)",
                    r"how\s+to\s+(hack|exploit|break\s+into)",
                    r"instructions?\s+for\s+(killing|harming|poisoning)",
                ],
                action=SafetyAction.REFUSE,
                refusal_message="I cannot provide instructions that could cause harm.",
            ),
            SafetyRule(
                name="personal_info_extraction",
                description="Warn on attempts to extract personal information",
                patterns=[
                    r"(ssn|social\s+security|credit\s+card)\s+number",
                    r"extract\s+(personal|private)\s+information",
                ],
                action=SafetyAction.WARN,
            ),
            SafetyRule(
                name="deceptive_content",
                description="Block requests to generate deceptive content",
                patterns=[
                    r"(fake|forged?|counterfeit)\s+(id|document|passport|license)",
                    r"create\s+a?\s*fake",
                ],
                action=SafetyAction.REFUSE,
                refusal_message="I cannot help create deceptive or fraudulent content.",
            ),
        ]

    def check_input(self, text: str) -> SafetyResult:
        """Check input text against safety policies.

        Args:
            text: Input prompt text.

        Returns:
            SafetyResult with check outcome.
        """
        return self._check(text, stage="input")

    def check_output(self, text: str) -> SafetyResult:
        """Check output text against safety policies.

        Args:
            text: Generated output text.

        Returns:
            SafetyResult.
        """
        return self._check(text, stage="output")

    def _check(self, text: str, stage: str) -> SafetyResult:
        """Run all safety rules against text."""
        triggered = []
        action = SafetyAction.ALLOW
        messages = []

        for rule in self.rules:
            matched = False

            # Pattern matching
            for pattern in rule.patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matched = True
                    break

            # Custom check function
            if not matched and rule.check_fn is not None:
                matched = rule.check_fn(text)

            if matched:
                triggered.append(rule.name)
                # IntEnum comparison — numerically correct
                if rule.action > action or self.strict_mode:
                    action = rule.action
                messages.append(rule.refusal_message)

        # Audit log
        if triggered:
            self.audit_log.append({
                "stage": stage,
                "triggered_rules": triggered,
                "action": str(action),
                "text_preview": text[:200],
            })
            logger.warning(
                "Safety %s check: rules=%s, action=%s",
                stage, triggered, str(action),
            )

        return SafetyResult(
            safe=len(triggered) == 0,
            action=action,
            triggered_rules=triggered,
            message=" ".join(messages) if messages else "",
        )

    def add_rule(self, rule: SafetyRule) -> None:
        """Add a custom safety rule."""
        self.rules.append(rule)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the safety audit log."""
        return self.audit_log.copy()
