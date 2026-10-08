from __future__ import annotations
from core.registry import LLMS
from core.interfaces.base_llm import BaseLLM

_DEFAULT_MODEL = "llama3.2"


@LLMS.register(provider="ollama", name="chat")
class OllamaChatLLM(BaseLLM):
    """Local chat model via Ollama.

    Requires: ollama running locally, e.g. `ollama pull llama3.2`
    Config: model (default llama3.2)
    """

    def __init__(self, model: str = _DEFAULT_MODEL, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0) -> str:
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        resp = ollama.chat(model=self._model, messages=messages,
                           options={"temperature": temperature})
        return resp["message"]["content"].strip()

    @property
    def model_name(self) -> str:
        return self._model
