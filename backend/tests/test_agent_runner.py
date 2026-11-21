import pytest

from backend.app.agents.runner import AgentTool, run_tool_loop
from backend.app.core.llm import LLMError, ToolSpec


def make_tools(counter):
    def handler(_args):
        counter["n"] += 1
        return {"value": 42}

    return [AgentTool(ToolSpec("lookup", "d"), handler)]


def test_required_tool_is_enforced_and_result_cached(fake_llm_factory):
    counter = {"n": 0}
    llm = fake_llm_factory(
        [
            ("I think it is fine.", []),  # answers too early: gets reminded
            ("", [("lookup", {}), ("lookup", {})]),  # duplicate call: cached
            ("The value is 42.", []),
        ]
    )
    result = run_tool_loop(llm, "sys", "q", make_tools(counter), required_tools={"lookup"})

    assert result.answer == "The value is 42."
    assert result.tools_executed == ["lookup"]
    assert counter["n"] == 1
    assert "lookup" in llm.calls[1]["messages"][-1]["content"]  # the reminder


def test_unapproved_tool_is_blocked(fake_llm_factory):
    llm = fake_llm_factory([("", [("delete_everything", {})])])
    with pytest.raises(LLMError) as err:
        run_tool_loop(llm, "sys", "q", make_tools({"n": 0}))
    assert err.value.status_code == 502


def test_round_limit(fake_llm_factory):
    llm = fake_llm_factory([("", [("lookup", {})])] * 3)
    with pytest.raises(LLMError, match="allowed rounds"):
        run_tool_loop(llm, "sys", "q", make_tools({"n": 0}), max_rounds=3)


def test_empty_answer_is_an_error(fake_llm_factory):
    llm = fake_llm_factory([("", [])])
    with pytest.raises(LLMError, match="empty response"):
        run_tool_loop(llm, "sys", "q", make_tools({"n": 0}))


def test_events_are_emitted(fake_llm_factory):
    events = []
    llm = fake_llm_factory([("checking", [("lookup", {})]), ("done", [])])
    run_tool_loop(
        llm, "sys", "q", make_tools({"n": 0}), on_event=lambda kind, _p: events.append(kind)
    )
    assert events == ["thought", "tool_call", "tool_result", "final"]
