"""Thin adapter from the agent graph to the existing fine-tuned model."""
from __future__ import annotations

from typing import Any


class LLMClient:
    """Lazy wrapper around app.inference.MODEL.

    Tests inject a stub with the same two methods, so the graph can be
    exercised without torch, an adapter or a GPU.
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from app.inference import MODEL

            self._model = MODEL
        return self._model

    def analyze(
        self,
        snippet: str,
        regulations=None,
        max_regulation_chars: int | None = None,
    ) -> dict:
        return self.model.analyze(
            snippet,
            regulations=regulations,
            max_regulation_chars=max_regulation_chars,
        )

    def generate(self, messages: list[dict], max_new_tokens: int = 128) -> str:
        return self.model.generate(messages, max_new_tokens=max_new_tokens)
