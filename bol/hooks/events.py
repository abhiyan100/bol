"""Typed views over Claude Code hook payloads + per-turn accumulation.

Payload shapes verified against Claude Code 2.1.252:
  Stop         → hook_event_name, session_id, transcript_path, prompt_id,
                 last_assistant_message, stop_hook_active, cwd, permission_mode
  PostToolUse  → tool_name, tool_input, tool_response, session_id, prompt_id
  Notification → notification_type (permission_prompt | idle_prompt |
                 agent_needs_input | agent_completed), message
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ToolUse:
    tool_name: str
    detail: str  # short human-readable target (file path, command, …)
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


class TurnTracker:
    """Accumulates PostToolUse events until the matching Stop arrives."""

    def __init__(self) -> None:
        self._tools: dict[str, list[ToolUse]] = defaultdict(list)

    def record_tool(self, payload: dict) -> None:
        key = payload.get("prompt_id") or payload.get("session_id") or "?"
        tool_input = payload.get("tool_input") or {}
        self._tools[key].append(
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
