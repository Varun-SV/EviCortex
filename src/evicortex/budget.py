"""Deterministic token-budget utilities for compiled memory capsules.

Prototype A intentionally has no tokenizer dependency.  The estimator therefore uses
UTF-8 byte length, which is stable across platforms and deliberately more conservative
than the common ``characters / 4`` approximation used for English text.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BUDGET_TOKENS = 512


@dataclass(frozen=True, slots=True)
class BudgetTooSmallError(ValueError):
    """Raised when even an empty, valid capsule cannot fit in a budget."""

    budget_tokens: int
    minimum_tokens: int

    def __str__(self) -> str:
        return (
            f"budget_tokens={self.budget_tokens} is too small for a valid memory "
            f"capsule; at least {self.minimum_tokens} tokens are required"
        )


def validate_budget(budget_tokens: int) -> int:
    """Validate and return a positive integer token budget.

    ``bool`` is rejected explicitly even though it is an ``int`` subclass; accepting
    it here tends to hide calling-code mistakes.
    """

    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int):
        raise TypeError("budget_tokens must be a positive integer")
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be greater than zero")
    return budget_tokens


def estimate_tokens(text: str) -> int:
    """Return a conservative, deterministic, dependency-free token estimate.

    The estimate charges one token per three UTF-8 bytes, rounded up.  This is
    intentionally stricter than the usual four-characters-per-token rule while still
    leaving useful room in small research budgets.  Empty text costs zero tokens.

    The context compiler always measures the final serialized prompt with this exact
    function, so its advertised budget invariant is mechanically enforceable.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    byte_count = len(text.encode("utf-8"))
    return (byte_count + 2) // 3


def fits_budget(text: str, budget_tokens: int) -> bool:
    """Return whether *text* fits the validated estimated-token budget."""

    return estimate_tokens(text) <= validate_budget(budget_tokens)


__all__ = [
    "BudgetTooSmallError",
    "DEFAULT_BUDGET_TOKENS",
    "estimate_tokens",
    "fits_budget",
    "validate_budget",
]
