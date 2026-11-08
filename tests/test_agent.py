import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agent
import app_utils as U
import config
from conftest import png_bytes


class FakeToolModel(BaseChatModel):
    """Replays a script of AIMessages; the last one repeats if the script runs out."""
    script: list
    i: int = 0
    seen: list = []

    @property
    def _llm_type(self):
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        msg = self.script[min(self.i, len(self.script) - 1)]
        calls = [{**c, "id": f"call_{self.i}_{n}"} for n, c in enumerate(msg.tool_calls)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=msg.content, tool_calls=calls))])


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "x", "type": "tool_call"}])


@pytest.fixture(autouse=True)
def reset_agent():
    yield
    agent.set_agent(None)


def test_agent_runs_a_tool_and_reports_it(env):
    model = FakeToolModel(script=[call("list_models"), AIMessage(content="Dataset A has three variants.")], seen=[])
    agent.set_agent(agent.build_agent(model))
    out = agent.chat("which models are installed?", "t1")
    assert out == {"reply": "Dataset A has three variants.", "tools_used": ["list_models"], "sources": []}
    tool_msgs = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs and "Dataset A: best, standard, fedavg" in tool_msgs[0].content


def test_agent_tool_args_reach_real_segmentation(env):
    image_id, _, _ = U.save_upload(png_bytes())
    model = FakeToolModel(
        script=[call("segment_image", image_id=image_id, dataset="A", variant="best"), AIMessage(content="done")], seen=[])
    a = agent.build_agent(model)
    agent.set_agent(a)
    out = agent.chat(f"[ui_context image_id={image_id}]\nsegment it", "t2")
    assert out["tools_used"] == ["segment_image"]
    tool_msg = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
    assert "area_percent" in json.loads(tool_msg.content)


def test_system_prompt_and_context_are_sent(env):
    model = FakeToolModel(script=[AIMessage(content="ok")], seen=[])
    agent.set_agent(agent.build_agent(model))
    agent.chat("hello", "t3")
    first = model.seen[0]
    assert "not a medical device" in first[0].content and "[ui_context" in first[0].content
    assert isinstance(first[-1], HumanMessage) and first[-1].content == "hello"


def test_threads_keep_separate_memory(env):
    model = FakeToolModel(script=[AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="c")], seen=[])
    agent.set_agent(agent.build_agent(model))
    agent.chat("one", "thread-a")
    agent.chat("two", "thread-a")
    agent.chat("three", "thread-b")
    humans = lambda msgs: [m.content for m in msgs if isinstance(m, HumanMessage)]
    assert humans(model.seen[1]) == ["one", "two"]      # same thread remembers
    assert humans(model.seen[2]) == ["three"]           # other thread does not


def test_runaway_tool_loop_is_stopped(env, monkeypatch):
    monkeypatch.setattr(config, "AGENT_RECURSION_LIMIT", 4)
    model = FakeToolModel(script=[call("list_models")], seen=[])
    agent.set_agent(agent.build_agent(model))
    out = agent.chat("loop forever", "t4")
    assert "too many steps" in out["reply"]


def test_agent_reports_the_notes_it_read(env):
    model = FakeToolModel(script=[call("search_knowledge", query="FedAvg caveat"),
                                  AIMessage(content="It is sequential training (federated_setup.md).")], seen=[])
    agent.set_agent(agent.build_agent(model))
    out = agent.chat("is fedavg real?", "t7")
    assert out["tools_used"] == ["search_knowledge"]
    assert out["sources"] and out["sources"][0].startswith("federated_setup.md > ")


def test_provider_failure_becomes_friendly_error(env):
    import vlm

    class Boom(FakeToolModel):
        def _generate(self, *a, **k):
            raise RuntimeError("503 UNAVAILABLE high demand key=AIzaSECRET")

    agent.set_agent(agent.build_agent(Boom(script=[AIMessage(content="")], seen=[])))
    with pytest.raises(vlm.VLMError) as e:
        agent.chat("hi", "t6")
    assert "overloaded" in str(e.value) and "AIzaSECRET" not in str(e.value)


def test_agent_needs_a_key(monkeypatch):
    import vlm
    import config
    monkeypatch.setattr(config, "AGENT_MODEL", "anthropic:claude-sonnet-5-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent.set_agent(None)
    with pytest.raises(vlm.VLMUnavailable):
        agent.chat("hi", "t5")
