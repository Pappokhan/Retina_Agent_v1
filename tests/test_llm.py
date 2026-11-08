import pytest

import config
import llm


def test_spec_parsing():
    assert llm.parse("google_genai:gemini-2.5-flash") == ("google_genai", "gemini-2.5-flash")
    assert llm.parse("ollama:qwen2.5vl:7b") == ("ollama", "qwen2.5vl:7b")     # tag after the second colon stays
    assert llm.normalize("claude-sonnet-5-5") == "anthropic:claude-sonnet-5-5"  # bare name = Anthropic


def test_missing_key_message_says_what_to_do(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(llm.VLMUnavailable) as e:
        llm.check_provider("google_genai:gemini-2.5-flash")
    assert "GOOGLE_API_KEY" in str(e.value) and "AI Studio" in str(e.value)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    llm.check_provider("google_genai:gemini-2.5-flash")


def test_ollama_needs_no_key_and_unknown_provider_is_rejected():
    assert llm.is_configured("ollama:gemma4") is True
    with pytest.raises(llm.VLMUnavailable):
        llm.check_provider("nope:model")
    with pytest.raises(llm.VLMUnavailable):
        llm.check_provider("ollama:")


def test_build_creates_real_langchain_models(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    names = {s: type(llm.build(s)).__name__ for s in
             ("google_genai:gemini-2.5-flash", "ollama:gemma4", "anthropic:claude-sonnet-5-5")}
    assert names == {"google_genai:gemini-2.5-flash": "ChatGoogleGenerativeAI",
                     "ollama:gemma4": "ChatOllama", "anthropic:claude-sonnet-5-5": "ChatAnthropic"}


def test_text_of_handles_string_and_blocks():
    assert llm.text_of("hi") == "hi"
    assert llm.text_of([{"type": "text", "text": "a"}, {"type": "thinking", "thinking": "x"}, "b"]) == "ab"
    assert llm.text_of(None) == ""


@pytest.mark.parametrize("msg,cls", [
    ("429 Too Many Requests", llm.VLMRateLimited),
    ("quota exceeded", llm.VLMRateLimited),
    ("403 API key not valid", llm.VLMError),
    ("weird", llm.VLMError),
])
def test_friendly_maps_errors_without_leaking(msg, cls):
    out = llm.friendly(RuntimeError(msg + " key=AIzaSECRET"))
    assert isinstance(out, cls) and "AIzaSECRET" not in str(out)


def test_friendly_says_when_provider_is_overloaded():
    out = llm.friendly(RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand."))
    assert "overloaded" in str(out) and not isinstance(out, llm.VLMRateLimited)


def test_agent_model_gets_provider_timeout(monkeypatch):
    monkeypatch.setattr(config, "AGENT_MODEL", "google_genai:gemini-2.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    import agent
    built = []
    monkeypatch.setattr(llm, "build", lambda spec: built.append(spec) or "MODEL")
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: kw)
    assert agent.build_agent()["model"] == "MODEL" and built == ["google_genai:gemini-2.5-flash"]


def test_agent_model_defaults_to_vlm_model():
    assert config.AGENT_MODEL  # falls back to VLM_MODEL when AGENT_MODEL is unset


def test_missing_packages_and_install_hint(monkeypatch):
    import importlib.util
    import sys
    assert llm.missing_packages("google_genai:gemini-3.5-flash-lite") == []        # this venv is complete
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "langgraph" else real(name, *a))
    assert llm.missing_packages("google_genai:x") == ["langgraph"]
    assert sys.executable in llm.install_hint() and "requirements.txt" in llm.install_hint()
