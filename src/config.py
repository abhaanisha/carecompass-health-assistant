"""Central configuration.

Every tunable lives here so the rest of the package never reads ``os.environ``
directly. That matters for the evaluation harness, which needs to build a
:class:`Settings` with dense retrieval forced on or off without touching the
process environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

APP_NAME = "CareCompass"
APP_TAGLINE = "Grounded health guidance with a safety floor you can audit"
VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "knowledge"
INDEX_DIR = DATA_DIR / "index"
RED_FLAGS_PATH = DATA_DIR / "red_flags.yaml"
LEXICON_PATH = DATA_DIR / "lexicon.yaml"
TRANSLATIONS_PATH = DATA_DIR / "translations.yaml"
EVAL_DIR = ROOT / "eval"
LOG_DIR = Path(os.getenv("CARECOMPASS_LOG_DIR", str(ROOT / "logs")))
EVENT_LOG = LOG_DIR / "events.jsonl"


#: Model ids drift: vendors retire them, sometimes only for new accounts, and a
#: pinned default eventually 404s. ``python -m src.llm`` surfaces that in one
#: command, and every provider here exposes a ``/models`` endpoint listing what
#: a given key can actually reach. ``CARECOMPASS_MODEL`` overrides any default.
@dataclass(frozen=True)
class ProviderSpec:
    """How to talk to one model vendor.

    ``api`` is either ``openai`` (the OpenAI-compatible ``/chat/completions``
    shape, which Groq, OpenRouter and the Hugging Face router all speak) or
    ``anthropic`` (the Messages API). Two request shapes cover every provider
    below, which is why this package needs ``requests`` and no vendor SDKs.
    """

    key: str
    label: str
    env_var: str
    base_url: str
    default_model: str
    api: str = "openai"
    signup_url: str = ""
    #: Extra request-body fields this vendor needs. Applied by llm.py, which
    #: drops any that the selected model does not support.
    extra_body: dict = field(default_factory=dict)


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        key="groq",
        label="Groq",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-120b",
        signup_url="https://console.groq.com/keys",
        # Every model Groq currently serves is a reasoning model, and reasoning
        # tokens are charged against max_tokens. Left alone, a reply can spend
        # its entire budget thinking and come back empty.
        extra_body={"reasoning_effort": "low"},
    ),
    "cerebras": ProviderSpec(
        key="cerebras",
        label="Cerebras",
        env_var="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        signup_url="https://cloud.cerebras.ai/",
    ),
    "gemini": ProviderSpec(
        key="gemini",
        label="Google Gemini",
        env_var="GEMINI_API_KEY",
        # Gemini exposes an OpenAI-compatible surface, so it needs no new
        # transport -- only a base URL and a model name.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        # A rolling alias rather than a pinned version: Google retires specific
        # Gemini builds for new users fairly often, and a portfolio demo that
        # breaks silently six months from now is worse than one model behind.
        default_model="gemini-flash-latest",
        signup_url="https://aistudio.google.com/apikey",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        signup_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        label="Anthropic",
        env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-5",
        api="anthropic",
        signup_url="https://console.anthropic.com/settings/keys",
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        label="OpenRouter",
        env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct",
        signup_url="https://openrouter.ai/keys",
    ),
    "huggingface": ProviderSpec(
        key="huggingface",
        label="Hugging Face Inference",
        env_var="HF_TOKEN",
        base_url="https://router.huggingface.co/v1",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        signup_url="https://huggingface.co/settings/tokens",
    ),
}

# Checked in order. The first vendor with a key present wins. Vendors with a
# no-credit-card free tier come first, so the zero-cost path is the default one.
PROVIDER_PRIORITY = [
    "groq",
    "cerebras",
    "gemini",
    "openai",
    "anthropic",
    "openrouter",
    "huggingface",
]

#: Used when no vendor key is configured at all. The app still answers, from
#: retrieved passages only, and labels itself as running without a model.
PROVIDER_NONE = "retrieval-only"

SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
}


def load_dotenv(path: Path | None = None) -> list[str]:
    """Read a local ``.env`` into the environment. Returns the names it set.

    Hand-rolled rather than depending on ``python-dotenv``: it is twenty lines,
    and every dependency in this project has to earn its place in a 512 MB free
    instance.

    Real environment variables always win — the file only fills in what is
    missing — so a Space secret or a shell export is never silently overridden
    by a stale file someone left in their working copy.
    """
    target = Path(path or ROOT / ".env")
    if not target.exists():
        return []

    applied: list[str] = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name or not value:
            continue
        if name not in os.environ:
            os.environ[name] = value
            applied.append(name)
    return applied


#: Loaded once, at import, so every entry point picks it up without ceremony.
DOTENV_LOADED = load_dotenv()


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def detect_provider() -> str:
    """Pick a vendor from the environment.

    ``CARECOMPASS_PROVIDER`` forces a choice (including ``none`` to demo the
    retrieval-only path even when a key is present).
    """
    forced = (os.getenv("CARECOMPASS_PROVIDER") or "").strip().lower()
    if forced in {"none", "off", PROVIDER_NONE}:
        return PROVIDER_NONE
    if forced in PROVIDER_REGISTRY:
        return forced

    for key in PROVIDER_PRIORITY:
        spec = PROVIDER_REGISTRY[key]
        if os.getenv(spec.env_var, "").strip():
            return key
    return PROVIDER_NONE


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    embedding_model: str
    use_dense: bool
    top_k: int
    max_tokens: int
    temperature: float
    timeout_s: int
    log_events: bool
    show_debug: bool

    @property
    def spec(self) -> ProviderSpec | None:
        return PROVIDER_REGISTRY.get(self.provider)

    @property
    def has_model(self) -> bool:
        return self.provider != PROVIDER_NONE

    def api_key(self) -> str:
        spec = self.spec
        return os.getenv(spec.env_var, "").strip() if spec else ""

    def with_(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


def get_settings() -> Settings:
    provider = detect_provider()
    spec = PROVIDER_REGISTRY.get(provider)
    model = os.getenv("CARECOMPASS_MODEL", "").strip() or (
        spec.default_model if spec else ""
    )
    return Settings(
        provider=provider,
        model=model,
        embedding_model=os.getenv(
            "CARECOMPASS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        # Dense retrieval needs sentence-transformers and ~100 MB of model
        # download. Set CARECOMPASS_DISABLE_DENSE=1 for a lexical-only run.
        use_dense=not _flag("CARECOMPASS_DISABLE_DENSE", False),
        top_k=int(os.getenv("CARECOMPASS_TOP_K", "5")),
        max_tokens=int(os.getenv("CARECOMPASS_MAX_TOKENS", "900")),
        temperature=float(os.getenv("CARECOMPASS_TEMPERATURE", "0.2")),
        timeout_s=int(os.getenv("CARECOMPASS_TIMEOUT", "45")),
        log_events=not _flag("CARECOMPASS_DISABLE_LOGGING", False),
        show_debug=_flag("CARECOMPASS_DEBUG", False),
    )


MEDICAL_DISCLAIMER = (
    "CareCompass provides general health information from a curated public-health "
    "knowledge base. It does not diagnose, prescribe, or replace a qualified "
    "clinician. In an emergency call 112."
)
