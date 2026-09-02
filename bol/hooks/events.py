"""Typed views over coding-agent hook payloads + per-turn accumulation.

Claude Code payloads, verified against 2.1.252:
  Stop         → hook_event_name, session_id, transcript_path, prompt_id,
                 last_assistant_message, stop_hook_active, cwd, permission_mode
  PostToolUse  → tool_name, tool_input, tool_response, session_id, prompt_id
  Notification → notification_type (permission_prompt | idle_prompt |
                 agent_needs_input | agent_completed), message

Codex CLI payloads (developers.openai.com/codex/hooks) carry the same common
fields (session_id, transcript_path, cwd, hook_event_name) plus model, and
differ in three ways Bol cares about:
  Stop              → turn_id instead of prompt_id, same last_assistant_message
  PostToolUse       → tool_name is Bash, apply_patch or mcp__server__tool,
                      with tool_use_id and no turn_id
  PermissionRequest → tool_name + tool_input, and no Notification event at
                      all; this is where Codex is about to ask for approval

Every view therefore carries `agent`, so whatever speaks the event says the
right name. It is inferred from the payload: a PermissionRequest or a turn_id
is Codex, and anything else falls back to what the caller already knows.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

CLAUDE, CODEX = "claude", "codex"

_DISPLAY_NAMES = {CLAUDE: "Claude", CODEX: "Codex"}


def display_name(agent: str) -> str:
    """What Bol calls this agent out loud."""
    return _DISPLAY_NAMES.get(agent, _DISPLAY_NAMES[CLAUDE])


def agent_of(payload: dict, default: str = CLAUDE) -> str:
    """Which agent posted this payload.

    Two positive signals, both Codex-only: the PermissionRequest event, which
    Claude Code does not have, and turn_id, which rides every Codex Stop. A
    Codex PostToolUse carries neither, so the caller's default is what keeps a
    mid-turn tool event attributed to the agent already being narrated.
    """
    if payload.get("hook_event_name") == "PermissionRequest":
        return CODEX
    if payload.get("turn_id"):
        return CODEX
    if payload.get("prompt_id"):
        return CLAUDE
    return default


@dataclass
class ToolUse:
    tool_name: str
    detail: str  # short human-readable target (file path, command, ...)
    ok: bool
    agent: str = CLAUDE


@dataclass
class StopEvent:
    session_id: str
    prompt_id: str
    last_assistant_message: str
    transcript_path: str
    cwd: str
    tools: list[ToolUse] = field(default_factory=list)
    agent: str = CLAUDE


@dataclass
class NotificationEvent:
    notification_type: str
    message: str
    session_id: str
    agent: str = CLAUDE


_DETAIL_KEYS = ("file_path", "command", "pattern", "url", "path", "prompt", "query")

# Codex's apply_patch puts the whole patch in tool_input.command. Reading a
# diff aloud helps nobody; the file it touches is the detail worth keeping.
_PATCH_TARGET = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)


def _clip(value: str) -> str:
    return value if len(value) <= 120 else value[:117] + "..."


def _tool_detail(tool_input: dict, tool_name: str = "") -> str:
    if tool_name == "apply_patch":
        for key in ("file_path", "path"):
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                return _clip(value)
        found = _PATCH_TARGET.search(tool_input.get("command", "") or "")
        return _clip(found.group(1).strip()) if found else ""
    for key in _DETAIL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return _clip(value)
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

    @staticmethod
    def _key(payload: dict) -> str:
        # Claude Code keys a turn by prompt_id, Codex by turn_id. Codex's
        # PostToolUse carries neither, so it lands under session_id and
        # finish_turn's session fallback is what reunites it with its Stop.
        return (
            payload.get("prompt_id")
            or payload.get("turn_id")
            or payload.get("session_id")
            or "?"
        )

    def record_tool(self, payload: dict, agent: str = CLAUDE) -> ToolUse:
        key = self._key(payload)
        tools = self._tools.get(key)
        if tools is None:
            tools = self._tools[key] = []
            while len(self._tools) > MAX_TURNS:
                self._tools.popitem(last=False)  # the key just added is newest
        else:
            self._tools.move_to_end(key)
        if len(tools) >= MAX_TOOLS_PER_TURN:
            del tools[0]
        tool_name = payload.get("tool_name", "?")
        tool_input = payload.get("tool_input") or {}
        use = ToolUse(
            tool_name=tool_name,
            detail=_tool_detail(
                tool_input if isinstance(tool_input, dict) else {}, tool_name
            ),
            ok=_tool_ok(payload.get("tool_response")),
            agent=agent_of(payload, agent),
        )
        tools.append(use)
        # Returned so a caller can show what the agent just did without
        # parsing the payload a second time.
        return use

    def finish_turn(self, payload: dict, agent: str = CLAUDE) -> StopEvent:
        key = self._key(payload)
        tools = self._tools.pop(key, [])
        # A session-keyed fallback may have collected under a different key.
        if not tools and payload.get("session_id") in self._tools:
            tools = self._tools.pop(payload["session_id"], [])
        return StopEvent(
            session_id=payload.get("session_id", ""),
            prompt_id=payload.get("prompt_id") or payload.get("turn_id", ""),
            last_assistant_message=payload.get("last_assistant_message", ""),
            transcript_path=payload.get("transcript_path", ""),
            cwd=payload.get("cwd", ""),
            tools=tools,
            agent=agent_of(payload, agent),
        )

    @staticmethod
    def notification(payload: dict, agent: str = CLAUDE) -> NotificationEvent:
        return NotificationEvent(
            notification_type=payload.get("notification_type", ""),
            message=payload.get("message", ""),
            session_id=payload.get("session_id", ""),
            agent=agent_of(payload, agent),
        )

    @staticmethod
    def permission_request(payload: dict) -> NotificationEvent:
        """Codex's PermissionRequest, as the permission prompt Bol reads out.

        Codex has no Notification event, so this is the one moment it says
        "about to ask the user". The question is built here rather than in the
        daemon because only the payload knows whether there is a command to
        name: a shell escalation has one, an MCP tool usually does not.
        """
        tool_name = payload.get("tool_name", "") or "something"
        tool_input = payload.get("tool_input") or {}
        detail = _tool_detail(
            tool_input if isinstance(tool_input, dict) else {}, tool_name
        )
        name = display_name(CODEX)
        message = (
            f"{name} wants to run: {detail}" if detail else f"{name} asks: {tool_name}"
        )
        return NotificationEvent(
            notification_type="permission_prompt",
            message=message,
            session_id=payload.get("session_id", ""),
            agent=CODEX,
        )
