"""Installs Bol's HTTP hooks into Claude Code settings, idempotently.

Writes to ~/.claude/settings.json (user scope) or <project>/.claude/settings.json
(project scope). Only touches the three events Bol needs; existing hooks are
preserved. Uninstall removes exactly the entries whose URL points at Bol.
"""

from __future__ import annotations

import json
from pathlib import Path

EVENTS = ("Stop", "PostToolUse", "Notification")


def _settings_path(scope: str, project_dir: Path | None = None) -> Path:
    if scope == "user":
        return Path.home() / ".claude" / "settings.json"
    if scope == "project":
        base = project_dir or Path.cwd()
        return base / ".claude" / "settings.json"
    raise ValueError(f"unknown scope: {scope}")


def _load(path: Path) -> dict:
    if path.exists():
        text = path.read_text().strip()
        if text:
            return json.loads(text)
    return {}


def _bol_hook(url: str) -> dict:
    return {"type": "http", "url": url}


def install(url: str, scope: str = "user", project_dir: Path | None = None) -> Path:
    path = _settings_path(scope, project_dir)
    settings = _load(path)
    hooks = settings.setdefault("hooks", {})
    for event in EVENTS:
        entries = hooks.setdefault(event, [])
        already = any(
            h.get("url") == url
            for entry in entries
            for h in entry.get("hooks", [])
        )
        if already:
            continue
        entry: dict = {"hooks": [_bol_hook(url)]}
        if event == "PostToolUse":
            entry["matcher"] = "*"
        entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return path


def uninstall(url: str, scope: str = "user", project_dir: Path | None = None) -> Path:
    path = _settings_path(scope, project_dir)
    settings = _load(path)
    hooks = settings.get("hooks", {})
    for event in EVENTS:
        entries = hooks.get(event)
        if not entries:
            continue
        kept = []
        for entry in entries:
            entry_hooks = [h for h in entry.get("hooks", []) if h.get("url") != url]
            if entry_hooks:
                entry["hooks"] = entry_hooks
                kept.append(entry)
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
    if path.exists():
        path.write_text(json.dumps(settings, indent=2) + "\n")
    return path


def installed(url: str, scope: str = "user", project_dir: Path | None = None) -> bool:
    settings = _load(_settings_path(scope, project_dir))
    hooks = settings.get("hooks", {})
    return all(
        any(
            h.get("url") == url
            for entry in hooks.get(event, [])
            for h in entry.get("hooks", [])
        )
        for event in EVENTS
    )
