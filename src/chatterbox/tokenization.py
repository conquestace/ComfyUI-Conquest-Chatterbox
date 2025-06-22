"""Utility functions for text tokenization.

This module provides helpers to break large text prompts
into fixed length chunks for TTS generation."""

from typing import List


def tokenize_prompt(text: str, token_length: int = 300) -> List[str]:
    """Split ``text`` into chunks of ``token_length`` characters.

    Each chunk returned represents one generation token for the TTS
    pipeline. The final token may be shorter than ``token_length``.
    Empty strings are filtered out.
    """
    if token_length <= 0:
        raise ValueError("token_length must be positive")
    tokens = [text[i:i + token_length] for i in range(0, len(text), token_length)]
    return [t for t in tokens if t]
