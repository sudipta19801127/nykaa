"""
governance/budget_guard.py
Runtime Layer AI Governance: Per-request capacity and cost budget enforcement.
Satisfies Part 4 Task 15 by rejecting deliberately oversized requests before
they reach the LLM or agent tool pipeline.
"""

import os
import re
from typing import Tuple, Optional

# Default budget thresholds
# Approx 1 token ~= 4 characters for English prose / code
DEFAULT_MAX_INPUT_CHARS = int(os.getenv("GOVERNANCE_MAX_INPUT_CHARS", "1200"))
DEFAULT_MAX_ESTIMATED_TOKENS = int(os.getenv("GOVERNANCE_MAX_TOKENS", "300"))
DEFAULT_MAX_QUERY_WORDS = int(os.getenv("GOVERNANCE_MAX_WORDS", "200"))


class BudgetExceededError(ValueError):
    """Raised when an incoming user request exceeds the runtime governance budget."""

    def __init__(self, message: str, estimated_tokens: int, limit: int):
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.limit = limit


def estimate_token_count(text: str) -> int:
    """
    Fast, keyless runtime token estimation without external network calls.
    Uses word boundaries and punctuation splits (~1.3 tokens per whitespace-delimited word).
    """
    if not text:
        return 0
    words = re.findall(r"\b\w+\b|[^\w\s]", text)
    return int(len(words) * 1.15)


def check_request_budget(
    prompt: str,
    max_chars: int = DEFAULT_MAX_INPUT_CHARS,
    max_tokens: int = DEFAULT_MAX_ESTIMATED_TOKENS,
    max_words: int = DEFAULT_MAX_QUERY_WORDS
) -> Tuple[bool, int]:
    """
    Validates that a request conforms to the runtime cost and capacity envelope.
    Raises BudgetExceededError if limits are breached.
    
    Returns:
        (is_within_budget: bool, estimated_tokens: int)
    """
    char_len = len(prompt)
    if char_len > max_chars:
        raise BudgetExceededError(
            f"Governance Budget Rejection: Request length ({char_len} chars) exceeds the "
            f"per-request safety limit of {max_chars} characters.",
            estimated_tokens=estimate_token_count(prompt),
            limit=max_chars
        )

    words_count = len(prompt.split())
    if words_count > max_words:
        raise BudgetExceededError(
            f"Governance Budget Rejection: Word count ({words_count} words) exceeds the "
            f"per-request cap of {max_words} words.",
            estimated_tokens=estimate_token_count(prompt),
            limit=max_words
        )

    estimated_tokens = estimate_token_count(prompt)
    if estimated_tokens > max_tokens:
        raise BudgetExceededError(
            f"Governance Budget Rejection: Estimated token cost ({estimated_tokens} tokens) "
            f"exceeds the per-request budget cap of {max_tokens} tokens.",
            estimated_tokens=estimated_tokens,
            limit=max_tokens
        )

    return True, estimated_tokens


def safe_budget_gate(prompt: str) -> Tuple[bool, Optional[str], int]:
    """
    Non-throwing wrapper suitable for API route validation.
    Returns: (allowed: bool, rejection_reason: Optional[str], token_count: int)
    """
    try:
        _, tokens = check_request_budget(prompt)
        return True, None, tokens
    except BudgetExceededError as err:
        return False, str(err), err.estimated_tokens


if __name__ == "__main__":
    print("--- Demonstrating Runtime Budget Guard (Part 4 Task 15) ---")

    # Case 1: Normal in-scope query (Within Budget)
    valid_query = "What is the return window for cosmetics and beauty products?"
    allowed, reason, tokens = safe_budget_gate(valid_query)
    print(f"\n[Test 1: Standard Query]")
    print(f"Query: {valid_query}")
    print(f"Allowed: {allowed} | Estimated Tokens: {tokens} | Reason: {reason}")

    # Case 2: Deliberately oversized query (Budget Cap Triggered)
    oversized_query = (
        "I need help with my order. " * 60 +
        "Please explain every single policy detail, shipping duration, and warranty clause."
    )
    print(f"\n[Test 2: Deliberately Oversized Query]")
    print(f"Oversized Query Length: {len(oversized_query)} chars, {len(oversized_query.split())} words")
    try:
        check_request_budget(oversized_query)
        print("Result: Unexpectedly Passed")
    except BudgetExceededError as exc:
        print(f"Result: Successfully Blocked by Governance Guardrail")
        print(f"Caught Exception: {exc}")