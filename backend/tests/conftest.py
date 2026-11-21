import os
import tempfile
from pathlib import Path

# Point the app at a throwaway database before anything imports it.
_TMP = Path(tempfile.mkdtemp(prefix="mmc-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"

import pytest  # noqa: E402

from backend.app.core.llm import LLMResponse, ToolCall  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeLLM:
    """Replays scripted responses; records every request it receives."""

    provider = "fake"
    model = "fake-model"
    runtime_label = "Fake provider"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, system, messages, tools=None, max_tokens=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.script:
            raise AssertionError("FakeLLM ran out of scripted responses")
        text, calls = self.script.pop(0)
        return LLMResponse(
            text=text,
            tool_calls=[
                ToolCall(id=f"c{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(calls)
            ],
            stop_reason="tool_use" if calls else "end_turn",
            provider="fake",
            model=self.model,
        )


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


@pytest.fixture
def cell_image_bytes():
    return (PROJECT_ROOT / "cell_image.png").read_bytes()
