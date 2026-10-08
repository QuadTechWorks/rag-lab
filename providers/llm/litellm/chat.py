from __future__ import annotations
from core.registry import LLMS
from core.interfaces.base_llm import BaseLLM

_DEFAULT_MODEL = "gpt-4o-mini"


@LLMS.register(provider="litellm", name="chat")
class LiteLLMChatLLM(BaseLLM):
    """Any hosted chat model via LiteLLM (OpenAI, Anthropic, Groq, Gemini, Mistral, ...).

    Requires: pip install litellm and the provider's API key env var
              (OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, ...).
    Config: model — a LiteLLM model string, e.g. "gpt-4o-mini",
            "anthropic/claude-haiku-4-5-20251001", "groq/llama-3.1-8b-instant"
    """

    def __init__(self, model: str = _DEFAULT_MODEL, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0) -> str:
        try:
            import litellm
        except ImportError:
            raise ImportError("pip install litellm")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        resp = litellm.completion(model=self._model, messages=messages,
                                  temperature=temperature)
        return resp.choices[0].message.content.strip()

    @property
    def model_name(self) -> str:
        return self._model
