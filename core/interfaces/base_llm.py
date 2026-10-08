from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0) -> str:
        """Return the model's text completion for a single-turn prompt."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier used by this LLM."""
        ...
