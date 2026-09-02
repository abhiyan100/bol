"""Installs Bol's hooks into a coding agent's settings, idempotently.

Two agents, one shape. Claude Code reads ~/.claude/settings.json (user scope)
or <project>/.claude/settings.json (project scope); Codex CLI reads
~/.codex/hooks.json. Only the events Bol needs are touched; existing hooks are
preserved. Entries are matched by their base URL, so a regenerated token or a
changed port never leaves a dead Bol hook behind.

Each hook is an async `command` that posts the event to Bol over loopback
with curl and always exits 0. Not an `http` hook: Claude Code prints a hook
error into the transcript every time an http hook cannot connect, and Bol is
a daemon the user starts and stops, so every session would nag whenever Bol
was off. An async command never blocks the agent, and `|| true` keeps a
refused connection silent. Older http entries from earlier Bol versions are
replaced.

Codex validates what a Stop hook writes: it "expects JSON on stdout when it
exits 0", and a bare curl to /dev/null writes nothing at all. So the Codex
command ends with `; echo '{}'`, an empty decision. Codex also asks the user
to review and trust a hooks file it has not seen before, once.

The file belongs to the agent, not to us: reads fail loudly with a friendly
message instead of a traceback, writes go through a temp file plus os.replace
(atomic, never a half-written settings file), and the first install keeps a
one-time copy alongside it with a .bol-backup suffix.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

EVENTS = ("Stop", "PostToolUse", "Notification")
# Codex has no Notification event. PermissionRequest fires when Codex is about
# to ask for approval (a shell escalation, a network call), which is the same
# moment Bol reads a prompt aloud.
CODEX_EVENTS = ("Stop", "PostToolUse", "PermissionRequest")

AGENTS = ("claude", "codex")
# What to look for on PATH to decide an agent is installed.
AGENT_CLI = {"claude": "claude", "codex": "codex"}
# How each agent is named in setup and doctor output.
AGENT_LABELS = {"claude": "Claude Code", "codex": "Codex CLI"}

_AGENT_EVENTS = {"claude": EVENTS, "codex": CODEX_EVENTS}
_AGENT_FILE = {"claude": (".claude", "settings.json"), "codex": (".codex", "hooks.json")}

CODEX_TRUST_NOTE = (
    "Codex asks you to review and trust new hooks once; accept it when it appears."
)

BACKUP_SUFFIX = ".bol-backup"

PROJECT_SCOPE_WARNING = (
    "bol: project scope writes your hook token into this repo's .claude/settings.json. "
    "Teams commit that file, so keep it out of git or use the default user scope."
)


def _check_agent(agent: str) -> str:
    if agent not in _AGENT_FILE:
        raise ValueError(f"unknown agent: {agent}")
    return agent


def events_for(agent: str = "claude") -> tuple[str, ...]:
    """The hook events Bol installs for this agent."""
    return _AGENT_EVENTS[_check_agent(agent)]


def _settings_path(
    scope: str, project_dir: Path | None = None, agent: str = "claude"
) -> Path:
    directory, name = _AGENT_FILE[_check_agent(agent)]
    if scope == "user":
        return Path.home() / directory / name
    if scope == "project":
        base = project_dir or Path.cwd()
        return base / directory / name
    raise ValueError(f"unknown scope: {scope}")


def settings_path(
    scope: str = "user", project_dir: Path | None = None, agent: str = "claude"
) -> Path:
    """Where hooks for this agent and scope are written (the CLI shows it)."""
    return _settings_path(scope, project_dir, agent)


def _base(url: str) -> str:
    """A hook URL without its ?token=... query."""
    return url.split("?", 1)[0]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text().strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Comments and trailing commas are the usual cause. A traceback here
        # helps nobody; say which file and what is wrong.
        raise SystemExit(
            f"bol: {path} isn't valid JSON ({exc}). Fix it, then rerun."
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(
            f"bol: {path} isn't a JSON object. Fix it, then rerun."
        )
    return data


def _save(path: Path, settings: dict) -> None:
    """Atomic write: temp file in the same directory, fsync, rename over.

    An interrupted write (Ctrl+C, full disk, laptop lid) leaves the old
    settings.json intact rather than a truncated one Claude Code cannot read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=".settings-", suffix=".bol-tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(settings, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _backup_once(path: Path) -> None:
    """Keep the user's pre-Bol settings.json, once, so a bad edit is recoverable."""
    if not path.exists():
        return
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def hook_command(url: str, agent: str = "claude") -> str:
    """The shell line the agent runs per event: post stdin to Bol, never fail."""
    line = (
        "curl -s -m 3 -o /dev/null -X POST -H 'Content-Type: application/json' "
        f"--data-binary @- '{url}' || true"
    )
    if _check_agent(agent) == "codex":
        # Codex reads a Stop hook's stdout and rejects anything that is not
        # JSON. curl writes nothing here, so say "no decision" explicitly.
        return line + "; echo '{}'"
    return line


def bol_hook(url: str, agent: str = "claude") -> dict:
    return {"type": "command", "command": hook_command(url, agent), "async": True}


def _hook_url(hook: dict) -> str:
    """The Bol URL inside a hook entry, for either the current command form
    or the http form earlier versions installed; empty for anything else."""
    if hook.get("type") == "http":
        return hook.get("url", "")
    command = hook.get("command", "")
    if "/hook" in command:
        start = command.find("'http")
        end = command.find("'", start + 1)
        if start >= 0 and end > start:
            return command[start + 1 : end]
    return ""


def _strip_bol(entries: list, base: str) -> list:
    """Every entry minus Bol's hooks (any token, either form), dropping
    entries left empty."""
    kept = []
    for entry in entries:
        hooks = [h for h in entry.get("hooks", []) if _base(_hook_url(h)) != base]
        if hooks:
            entry["hooks"] = hooks
            kept.append(entry)
    return kept


def install(
    url: str,
    scope: str = "user",
    project_dir: Path | None = None,
    agent: str = "claude",
) -> Path:
    path = _settings_path(scope, project_dir, agent)
    settings = _load(path)
    _backup_once(path)
    base = _base(url)
    hooks = settings.setdefault("hooks", {})
    for event in events_for(agent):
        entries = hooks.setdefault(event, [])
        # Replace rather than append: an entry from an older port or token is
        # dead weight that would post into nothing forever.
        entries[:] = _strip_bol(entries, base)
        entry: dict = {"hooks": [bol_hook(url, agent)]}
        # Claude Code matches PostToolUse by tool name; Codex's schema has no
        # matcher, and an unknown key is what a trust review would query.
        if agent == "claude" and event == "PostToolUse":
            entry["matcher"] = "*"
        entries.append(entry)
    _save(path, settings)
    return path


def uninstall(
    url: str,
    scope: str = "user",
    project_dir: Path | None = None,
    agent: str = "claude",
) -> Path:
    path = _settings_path(scope, project_dir, agent)
    if not path.exists():
        return path
    settings = _load(path)
    base = _base(url)
    before = json.dumps(settings, sort_keys=True)
    hooks = settings.get("hooks", {})
    for event in events_for(agent):
        entries = hooks.get(event)
        if not entries:
            continue
        kept = _strip_bol(entries, base)
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
    if json.dumps(settings, sort_keys=True) != before:
        _save(path, settings)
    return path


def installed(
    url: str,
    scope: str = "user",
    project_dir: Path | None = None,
    agent: str = "claude",
) -> bool:
    settings = _load(_settings_path(scope, project_dir, agent))
    hooks = settings.get("hooks", {})
    wanted = bol_hook(url, agent)
    return all(
        any(
            h.get("type") == "command"
            and h.get("command") == wanted["command"]
            and h.get("async") is True
            for entry in hooks.get(event, [])
            for h in entry.get("hooks", [])
        )
        for event in events_for(agent)
    )
