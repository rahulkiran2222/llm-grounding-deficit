"""
Unified model interface for the grounding-deficit harness.

Each backend implements `complete()` (single best-effort answer) and
`sample()` (multiple stochastic samples, used to estimate empirical
confidence for facts where the provider does not expose token logprobs,
e.g. most chat-completion endpoints from closed providers).

Design note: we deliberately keep this interface minimal. Adding a new
provider means writing one small adapter class below and registering it
in `get_model()` -- nothing else in the package needs to change.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    """Normalized response from any backend."""
    text: str
    raw_confidence: float | None = None   # provider-reported confidence/logprob, if available
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)  # original provider payload, for debugging


class BaseModel(ABC):
    name: str

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, **kwargs) -> ModelResponse:
        """Single deterministic-ish completion (temperature=0 where supported)."""
        ...

    def sample(self, prompt: str, n: int = 10, temperature: float = 1.0,
               system: str | None = None, **kwargs) -> list[ModelResponse]:
        """
        Default implementation: call complete() n times at the given temperature.
        Providers with native batch/n-sample support should override this
        for efficiency and cost.
        """
        out = []
        for _ in range(n):
            out.append(self.complete(prompt, system=system, temperature=temperature, **kwargs))
        return out


class OpenAIModel(BaseModel):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI
        self.name = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 256, **kwargs) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=1,
        )
        latency = time.time() - t0

        choice = resp.choices[0]
        text = choice.message.content or ""

        # Average token logprob as a crude confidence proxy when available.
        raw_conf = None
        if choice.logprobs and choice.logprobs.content:
            import math
            lps = [t.logprob for t in choice.logprobs.content]
            if lps:
                raw_conf = math.exp(sum(lps) / len(lps))

        return ModelResponse(text=text, raw_confidence=raw_conf, latency_s=latency,
                              raw=resp.model_dump())

    def sample(self, prompt: str, n: int = 10, temperature: float = 1.0,
               system: str | None = None, **kwargs) -> list[ModelResponse]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 256),
            n=n,
        )
        latency = (time.time() - t0) / max(n, 1)
        return [
            ModelResponse(text=c.message.content or "", latency_s=latency, raw={})
            for c in resp.choices
        ]


class AnthropicModel(BaseModel):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        import anthropic
        self.name = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 256, **kwargs) -> ModelResponse:
        t0 = time.time()
        resp = self._client.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - t0
        text = "".join(block.text for block in resp.content if block.type == "text")
        # Anthropic's API does not expose token logprobs; raw_confidence stays None.
        # delta_s.py falls back to the sample-based empirical-frequency estimator
        # for any backend that returns raw_confidence=None.
        return ModelResponse(text=text, raw_confidence=None, latency_s=latency,
                              raw=resp.model_dump())


class TogetherModel(BaseModel):
    """Adapter for open-weight models (Llama, Gemma, Mixtral, ...) via Together AI's
    OpenAI-compatible endpoint. Swap base_url to point at any other OpenAI-compatible
    host (Groq, Fireworks, a local vLLM server, etc.) without touching calling code."""

    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 api_key: str | None = None, base_url: str = "https://api.together.xyz/v1"):
        from openai import OpenAI
        self.name = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("TOGETHER_API_KEY"),
            base_url=base_url,
        )

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 256, **kwargs) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.name, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        latency = time.time() - t0
        text = resp.choices[0].message.content or ""
        return ModelResponse(text=text, raw_confidence=None, latency_s=latency, raw={})


_REGISTRY = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    "together": TogetherModel,
}


def get_model(backend: str, model: str, **kwargs) -> BaseModel:
    """
    Factory. Example:
        get_model("openai", "gpt-4o-mini")
        get_model("anthropic", "claude-haiku-4-5-20251001")
        get_model("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    """
    if backend not in _REGISTRY:
        raise ValueError(f"Unknown backend '{backend}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[backend](model=model, **kwargs)
