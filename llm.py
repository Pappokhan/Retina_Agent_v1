from __future__ import annotations
import re
import config


class VLMUnavailable(RuntimeError):

class VLMError(RuntimeError):

class VLMRateLimited(VLMError):

PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic API",
        "keys": ("ANTHROPIC_API_KEY",),
        "package": "langchain-anthropic",
        "hint": "Add ANTHROPIC_API_KEY to .env.",
        "kwargs": {"timeout": 45, "max_retries": 2},
        "privacy": "Images sent for explanations go to the Anthropic API.",
    },
    "google_genai": {
        "label": "Google Gemini API",
        "keys": ("GOOGLE_API_KEY",),
        "package": "langchain-google-genai",
        "hint": "Create a free key in Google AI Studio and put it in .env as GOOGLE_API_KEY.",
        "kwargs": {"timeout": 45, "max_retries": 2},
        "privacy": ("Images sent for explanations go to Google. On the free tier Google may use them to "
                    "improve its products and human reviewers may read them, so do not send patient data."),
    },
    "ollama": {
        "label": "local Ollama",
        "keys": (),
        "package": "langchain-ollama",
        "hint": "Install Ollama, run it, and pull the model first (ollama pull <model>).",
        "kwargs": {},
        "privacy": "Explanations run on this machine; images are not sent to a third party.",
    },
}


_MODULES = {"anthropic": "langchain_anthropic", "google_genai": "langchain_google_genai", "ollama": "langchain_ollama"}


def install_hint() -> str:
    """Exact command for *this* interpreter (the usual cause is an IDE using another environment)."""
    import sys

    return (f'Install the requirements into the Python that runs the app: "{sys.executable}" -m pip install -r '
            "requirements.txt  (or switch your IDE's interpreter to this project's .venv).")


def missing_packages(*specs: str) -> list[str]:
    """pip names of packages the explanation features and the agent need but cannot import.

    Each is really imported (not just looked up), so an installed-but-incompatible version (for
    example an old langchain that fails on import) is reported too, marked "(outdated or broken)".
    """
    import importlib
    import importlib.util

    need = {"langchain.chat_models": "langchain", "langchain.agents": "langchain", "langgraph": "langgraph",
            "langchain_core": "langchain-core",
            "langchain_text_splitters": "langchain-text-splitters", "rank_bm25": "rank-bm25"}
    for spec in specs:
        provider, _ = parse(spec)
        if provider in _MODULES:
            need[_MODULES[provider]] = PROVIDERS[provider]["package"]
    out = []
    for module, pip_name in need.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            entry = pip_name
        else:
            try:
                importlib.import_module(module)
                continue
            except Exception:
                entry = f"{pip_name} (outdated or broken)"
        if entry not in out:
            out.append(entry)
    return out


def parse(spec: str) -> tuple[str, str]:
    spec = (spec or "").strip()
    provider, sep, model = spec.partition(":")
    if not sep:
        return "anthropic", spec
    return provider, model


def normalize(spec: str) -> str:
    provider, model = parse(spec)
    return f"{provider}:{model}"


def info(spec: str) -> dict:
    provider, model = parse(spec)
    meta = PROVIDERS.get(provider, {})
    return {
        "provider": provider,
        "model": model,
        "label": meta.get("label", provider),
        "privacy": meta.get("privacy", "Images sent for explanations go to this model provider."),
    }


def is_configured(spec: str) -> bool:
    import os

    provider, model = parse(spec)
    if not model:
        return False
    keys = PROVIDERS.get(provider, {}).get("keys")
    return True if not keys else any(os.getenv(k) for k in keys)


def check_provider(spec: str) -> None:
    """Raise VLMUnavailable with a message that says what to do."""
    provider, model = parse(spec)
    if not model:
        raise VLMUnavailable("No model is set. Put VLM_MODEL (and AGENT_MODEL) in .env, for example "
                             "google_genai:gemini-2.5-flash or ollama:gemma4.")
    meta = PROVIDERS.get(provider)
    if meta is None:
        raise VLMUnavailable(f"Unknown provider '{provider}'. Write the model as provider:model, for example "
                             "google_genai:gemini-2.5-flash, ollama:gemma4 or anthropic:claude-sonnet-5-5.")
    if not is_configured(spec):
        raise VLMUnavailable(f"{' or '.join(meta['keys'])} is not set. {meta['hint']}")


def model_kwargs(spec: str, json_mode: bool = False) -> dict:
    """Provider kwargs: timeouts, Gemini 3 thinking level (latency), native JSON output."""
    provider, model = parse(spec)
    kw = dict(PROVIDERS[provider]["kwargs"])
    if provider == "google_genai":
        if model.startswith("gemini-3") and config.VLM_THINKING:
            kw["thinking_level"] = config.VLM_THINKING      # "low" keeps flash-lite answers fast
        if json_mode:
            kw["response_mime_type"] = "application/json"
    elif provider == "ollama" and json_mode:
        kw["format"] = "json"
    return kw


def build(spec: str, json_mode: bool = False, fallback: str | None = None):
    """Create the LangChain chat model for a spec, optionally with a fallback model.

    The fallback (config.VLM_FALLBACK_MODEL) is tried when the main model fails, for example when a
    free-tier Gemini model is overloaded (503) or out of quota (429).
    """
    check_provider(spec)
    provider, model = parse(spec)
    meta = PROVIDERS[provider]
    try:
        from langchain.chat_models import init_chat_model
        main = init_chat_model(f"{provider}:{model}", **model_kwargs(spec, json_mode))
        if fallback and normalize(fallback) != normalize(spec) and is_configured(fallback):
            fp, fm = parse(fallback)
            backup = init_chat_model(f"{fp}:{fm}", **model_kwargs(fallback, json_mode))
            return main.with_fallbacks([backup])
        return main
    except ImportError as exc:
        raise VLMUnavailable(f"The {provider} integration ({meta['package']}) is not installed in this Python "
                             f"environment. {install_hint()}") from exc
    except ValueError as exc:
        raise VLMUnavailable(f"Could not create model '{spec}': {exc}") from exc


def text_of(content) -> str:
    """Message content is a string for some providers and a list of blocks for others."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                out.append(str(block.get("text", "")))
        return "".join(out)
    return "" if content is None else str(content)


def friendly(exc: Exception) -> VLMError:
    """Map a provider exception to a safe message (never echoes the raw error, which may hold keys)."""
    name, msg = type(exc).__name__.lower(), str(exc).lower()
    if (any(k in name for k in ("ratelimit", "resourceexhausted", "toomanyrequests"))
            or re.search(r"\b429\b", msg) or "quota" in msg or "rate limit" in msg):
        return VLMRateLimited("The model provider's rate or quota limit was reached (common on free tiers). "
                              "Wait a minute and try again.")
    if (any(k in name for k in ("authentication", "permissiondenied", "unauthorized"))
            or re.search(r"\b40[13]\b", msg) or "api key" in msg or "api_key" in msg):
        return VLMError("The provider rejected the API key. Check the key in .env.")
    if (any(k in name for k in ("serviceunavailable", "overloaded"))
            or re.search(r"\b50[03]\b", msg) or "unavailable" in msg or "high demand" in msg or "overloaded" in msg):
        return VLMError("The model provider is overloaded right now. Try again in a minute, or switch "
                        "VLM_MODEL / AGENT_MODEL in .env to a less busy model.")
    if any(k in name for k in ("connect", "timeout")):
        return VLMError("Could not reach the model provider. Check the network, or that Ollama is running, and try again.")
    return VLMError(f"The explanation request failed ({type(exc).__name__}). Try again in a moment.")
