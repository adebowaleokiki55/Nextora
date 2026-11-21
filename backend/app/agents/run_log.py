"""Writes every agent step to agent_messages so the dashboard can stream it."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models import AgentMessage, AgentMessageKind

MAX_CONTENT = 4000

LOOP_KINDS = {
    "thought": AgentMessageKind.THOUGHT,
    "tool_call": AgentMessageKind.TOOL_CALL,
    "tool_result": AgentMessageKind.TOOL_RESULT,
    "final": AgentMessageKind.FINAL,
    "error": AgentMessageKind.ERROR,
}


def _compact(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_CONTENT else text[: MAX_CONTENT - 20] + " … (truncated)"


class RunLogger:
    def __init__(self, session_factory: Callable[[], Session], run_id: int):
        self._factory = session_factory
        self.run_id = run_id
        self._lock = threading.Lock()
        with self._factory() as session:
            self._seq = session.scalar(
                select(func.coalesce(func.max(AgentMessage.seq), 0)).where(AgentMessage.run_id == run_id)
            )

    def log(
        self,
        agent: str,
        kind: AgentMessageKind,
        content: Any = "",
        tool_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._seq += 1
            with self._factory() as session:
                session.add(
                    AgentMessage(
                        run_id=self.run_id,
                        seq=self._seq,
                        agent=agent,
                        kind=kind,
                        content=_compact(content),
                        tool_name=tool_name,
                        payload=json.loads(json.dumps(payload, default=str)) if payload is not None else None,
                    )
                )
                session.commit()

    def loop_hook(self, agent: str):
        """Adapter for run_tool_loop's on_event(kind, payload)."""

        def hook(kind: str, payload: dict[str, Any]) -> None:
            mapped = LOOP_KINDS.get(kind, AgentMessageKind.THOUGHT)
            if kind == "tool_call":
                self.log(agent, mapped, payload.get("arguments") or {}, tool_name=payload.get("tool"))
            elif kind == "tool_result":
                self.log(agent, mapped, payload.get("result"), tool_name=payload.get("tool"))
            else:
                self.log(agent, mapped, payload.get("text", ""))

        return hook
