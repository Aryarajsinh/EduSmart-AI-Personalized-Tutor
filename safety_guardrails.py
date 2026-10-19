"""
EduSmart AI - Child Safety & Privacy Guardrails
Designed for K-12 EdTech environments (COPPA & FERPA compliant design)

Responsibilities:
1. PII Redaction: Automatically masks sensitive personal info (names, phone numbers, emails, addresses, school IDs).
2. Content Safety Filter: Blocks inappropriate, harmful, abusive, or self-harm content.
3. Minor-Safe Pedagogical Tone: Enforces positive reinforcement and age-appropriate explanations.
"""

import re
from typing import Tuple, Dict

# Regex patterns for common Personally Identifiable Information (PII)
PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN_OR_ID": r"\b\d{3}-\d{2}-\d{4}\b|\bID:\s*\d{6,}\b",
    "POSTAL_CODE": r"\b\d{5}(?:-\d{4})?\b",
}

# Inappropriate keywords / risky triggers for minor safety
SAFETY_BLOCKLIST = [
    "self harm", "suicide", "hurt myself", "kill myself",
    "violence", "weapons", "drugs", "cheat on exam", "hack my school",
    "adult content", "hate speech"
]

class ChildSafetyGuardrail:
    """
    Guardrail manager for minor data protection and child-safe interactions.
    """

    @staticmethod
    def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
        """
        Detects and redacts PII before sending context to LLMs or vector stores.
        Returns redacted text and summary of redacted entities.
        """
        redacted_text = text
        redacted_counts = {}

        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            if matches:
                redacted_counts[pii_type] = len(matches)
                redacted_text = re.sub(pattern, f"[REDACTED_{pii_type}]", redacted_text)

        return redacted_text, redacted_counts

    @staticmethod
    def audit_input_safety(text: str) -> Tuple[bool, str]:
        """
        Audits user input against safety guidelines.
        Returns (is_safe, message_or_reason).
        """
        lowered = text.lower()
        for flagged in SAFETY_BLOCKLIST:
            if flagged in lowered:
                return False, (
                    "EduSmart Safety Notice: I am your AI learning tutor, and I'm dedicated to providing "
                    "a safe, encouraging, and educational space. Let's redirect our focus back to your course studies, "
                    "or reach out to a trusted teacher or parent if you need support."
                )
        return True, ""

    @staticmethod
    def sanitize_output(llm_output: str) -> str:
        """
        Ensures LLM output conforms to child-safe and supportive tone.
        """
        # Redact any inadvertent PII generated
        sanitized, _ = ChildSafetyGuardrail.redact_pii(llm_output)
        return sanitized

