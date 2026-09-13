import re
from typing import Tuple

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt\s+override",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"reveal\s+(the\s+)?system\s+key"
]

def mask_fixed_pii(text: str) -> str:
    """Masks Indian phone numbers and payment-card last-4 digits."""
    # Phone numbers: +91 or standard 10 digits
    phone_pattern = r"(?:\+91[-\s]?)?[6-9]\d{9}\b"
    masked = re.sub(phone_pattern, "[PHONE_MASKED]", text)
    
    # Card numbers / last-4 mentions
    card_pattern = r"\b(?:\d{4}[-\s]?){3}(\d{4})\b|(?<=card\sending\sin\s)\d{4}\b"
    masked = re.sub(card_pattern, "[CARD_LAST4_MASKED]", masked, flags=re.IGNORECASE)
    return masked

def validate_input_guardrails(prompt: str) -> Tuple[bool, str, str]:
    """
    Validates input for prompt injections and applies PII masking.
    Returns: (is_safe, masked_text, reason)
    """
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False, prompt, "Prompt injection pattern detected."
    
    masked_prompt = mask_fixed_pii(prompt)
    return True, masked_prompt, "Accepted"