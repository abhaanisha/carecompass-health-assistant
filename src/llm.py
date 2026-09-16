"""Provider-agnostic chat client built on plain HTTP.

No vendor SDKs. Two request shapes cover every provider in the registry: the
OpenAI-compatible ``/chat/completions`` body (Groq, OpenAI, OpenRouter and the
Hugging Face router all accept it) and the Anthropic Messages API.

The payoff is a four-line ``requirements.txt`` that does not break when a
vendor ships a major SDK release, and a Space that a reviewer can run against
whichever free API key they happen to have.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import PROVIDER_NONE, PROVIDER_REGISTRY, Settings, get_settings

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class LLMResponse:
    text: str
    ok: bool
    provider: str
    model: str
    latency_ms: int
    error: str | None = None
    usage: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @property
    def available(self) -> bool:
        return self.settings.has_model and bool(self.settings.api_key())

    def describe(self) -> dict:
        spec = self.settings.spec
        return {
            "provider": self.settings.provider,
            "label": spec.label if spec else "Retrieval only",
            "model": self.settings.model or None,
            "available": self.available,
        }

    # -- public ------------------------------------------------------------

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send a conversation and return the reply.

        Never raises. A failed call returns ``ok=False`` so the pipeline can
        fall back to retrieval-only output instead of showing a stack trace to
        someone asking about their child's fever.
        """
        started = time.perf_counter()
        if not self.available:
            return LLMResponse(
                text="",
                ok=False,
                provider=PROVIDER_NONE,
                model="",
                latency_ms=0,
                error="No model provider configured",
            )

        spec = self.settings.spec
        assert spec is not None
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = self.settings.temperature if temperature is None else temperature

        last_error = "unknown error"
        for attempt in range(2):
            try:
                if spec.api == "anthropic":
                    text, usage = self._anthropic(system, messages, max_tokens, temperature)
                else:
                    text, usage = self._openai(system, messages, max_tokens, temperature)
                return LLMResponse(
                    text=text.strip(),
                    ok=bool(text.strip()),
                    provider=spec.key,
                    model=self.settings.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    usage=usage,
                )
            except _RetryableError as exc:
                last_error = str(exc)
                if attempt == 0:
                    time.sleep(1.5)
                    continue
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                break

        return LLMResponse(
            text="",
            ok=False,
            provider=spec.key,
            model=self.settings.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=last_error,
        )

    # -- transports --------------------------------------------------------

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        # Imported lazily so the package loads in environments without it --
        # notably Pyodide, where the browser build has no model provider at all
        # and sockets do not exist.
        import requests

        response = requests.post(
            url, headers=headers, json=payload, timeout=self.settings.timeout_s
        )
        if response.status_code in RETRYABLE_STATUS:
            raise _RetryableError(f"HTTP {response.status_code} from provider")
        if response.status_code >= 400:
            detail = response.text[:200].replace("\n", " ")
            raise RuntimeError(f"HTTP {response.status_code}: {detail}")
        return response.json()

    def _openai(self, system, messages, max_tokens, temperature) -> tuple[str, dict]:
        spec = self.settings.spec
        data = self._post(
            f"{spec.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self.settings.api_key()}",
                "Content-Type": "application/json",
            },
            {
                "model": self.settings.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        choices = data.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        return text or "", data.get("usage") or {}

    def _anthropic(self, system, messages, max_tokens, temperature) -> tuple[str, dict]:
        spec = self.settings.spec
        data = self._post(
            f"{spec.base_url}/messages",
            {
                "x-api-key": self.settings.api_key(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            {
                "model": self.settings.model,
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text, data.get("usage") or {}


class _RetryableError(RuntimeError):
    pass


def available_providers() -> list[dict]:
    """Every provider the app knows about, and whether its key is present."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "env_var": spec.env_var,
            "default_model": spec.default_model,
            "configured": bool(get_settings().with_(provider=spec.key).api_key()),
            "signup_url": spec.signup_url,
        }
        for spec in PROVIDER_REGISTRY.values()
    ]
