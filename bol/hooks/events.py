"""Typed views over Claude Code hook payloads + per-turn accumulation.

Payload shapes verified against Claude Code 2.1.252:
  Stop         → hook_event_name, session_id, transcript_path, prompt_id,
                 last_assistant_message, stop_hook_active, cwd, permission_mode
  PostToolUse  → tool_name, tool_input, tool_response, session_id, prompt_id
  Notification → notification_type (permission_prompt | idle_prompt |
                 agent_needs_input | agent_completed), message
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class ToolUse:
    tool_name: str
    detail: str  # short human-readable target (file path, command, ...)
    ok: bool


@dataclass
class StopEvent:
    session_id: str
    prompt_id: str
    last_assistant_message: str
    transcript_path: str
    cwd: str
    tools: list[ToolUse] = field(default_factory=list)


@dataclass
class NotificationEvent:
    notification_type: str
    message: str
    session_id: str


_DETAIL_KEYS = ("file_path", "command", "pattern", "url", "path", "prompt", "query")


def _tool_detail(tool_input: dict) -> str:
    for key in _DETAIL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= 120 else value[:117] + "..."
    return ""


def _tool_ok(tool_response: object) -> bool:
    if isinstance(tool_response, dict):
        if tool_response.get("is_error") or tool_response.get("isError"):
            return False
        if "success" in tool_response:
            return bool(tool_response["success"])
    if isinstance(tool_response, str):
        lowered = tool_response[:2000].lower()
        if lowered.startswith("error") or "command failed" in lowered:
            return False
    return True


# A turn whose Stop never arrives (interrupted, crashed, a session Bol isn't
# following) would otherwise keep its tool list for the life of the process.
MAX_TURNS = 32
MAX_TOOLS_PER_TURN = 200


class TurnTracker:
    """Accumulates PostToolUse events until the matching Stop arrives.

    Bounded on both axes: the oldest turn is evicted past MAX_TURNS, and each
    turn keeps a rolling window of its most recent tool calls.
    """

    def __init__(self) -> None:
        self._tools: OrderedDict[str, list[ToolUse]] = OrderedDict()

    def record_tool(self, payload: dict) -> None:
        key = payload.get("prompt_id") or payload.get("session_id") or "?"
        tools = self._tools.get(key)
        if tools is None:
            tools = self._tools[key] = []
            while len(self._tools) > MAX_TURNS:
                self._tools.popitem(last=False)  # the key just added is newest
        else:
            self._tools.move_to_end(key)
        if len(tools) >= MAX_TOOLS_PER_TURN:
            del tools[0]
        tool_input = payload.get("tool_input") or {}
        tools.append(
            ToolUse(
                tool_name=payload.get("tool_name", "?"),
                detail=_tool_detail(tool_input if isinstance(tool_input, dict) else {}),
                ok=_tool_ok(payload.get("tool_response")),
            )
        )

    def finish_turn(self, payload: dict) -> StopEvent:
        key = payload.get("prompt_id") or payload.get("session_id") or "?"
        tools = self._tools.pop(key, [])
        # A session-keyed fallback may have collected under a different key.
        if not tools and payload.get("session_id") in self._tools:
            tools = self._tools.pop(payload["session_id"], [])
        return StopEvent(
            session_id=payload.get("session_id", ""),
            prompt_id=payload.get("prompt_id", ""),
            last_assistant_message=payload.get("last_assistant_message", ""),
            transcript_path=payload.get("transcript_path", ""),
            cwd=payload.get("cwd", ""),
            tools=tools,
        )

    @staticmethod
    def notification(payload: dict) -> NotificationEvent:
        return NotificationEvent(
            notification_type=payload.get("notification_type", ""),
            message=payload.get("message", ""),
            session_id=payload.get("session_id", ""),
        )
