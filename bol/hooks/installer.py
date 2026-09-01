"""Installs Bol's HTTP hooks into Claude Code settings, idempotently.

Writes to ~/.claude/settings.json (user scope) or <project>/.claude/settings.json
(project scope). Only touches the three events Bol needs; existing hooks are
preserved. Entries are matched by their base URL, so a regenerated token or a
changed port never leaves a dead Bol hook behind.

The file belongs to Claude Code, not to us: reads fail loudly with a friendly
message instead of a traceback, writes go through a temp file plus os.replace
(atomic, never a half-written settings.json), and the first install keeps a
one-time copy at settings.json.bol-backup.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

EVENTS = ("Stop", "PostToolUse", "Notification")

BACKUP_SUFFIX = ".bol-backup"

PROJECT_SCOPE_WARNING = (
    "bol: project scope writes your hook token into this repo's .claude/settings.json. "
    "Teams commit that file, so keep it out of git or use the default user scope."
)


def _settings_path(scope: str, project_dir: Path | None = None) -> Path:
    if scope == "user":
        return Path.home() / ".claude" / "settings.json"
    if scope == "project":
        base = project_dir or Path.cwd()
        return base / ".claude" / "settings.json"
    raise ValueError(f"unknown scope: {scope}")


def settings_path(scope: str = "user", project_dir: Path | None = None) -> Path:
    """Where hooks for this scope are written (the CLI shows it to the user)."""
    return _settings_path(scope, project_dir)


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


def _bol_hook(url: str) -> dict:
    return {"type": "http", "url": url}


def _strip_bol(entries: list, base: str) -> list:
    """Every entry minus Bol's hooks (any token), dropping entries left empty."""
    kept = []
    for entry in entries:
        hooks = [h for h in entry.get("hooks", []) if _base(h.get("url", "")) != base]
        if hooks:
            entry["hooks"] = hooks
            kept.append(entry)
    return kept


def install(url: str, scope: str = "user", project_dir: Path | None = None) -> Path:
    path = _settings_path(scope, project_dir)
    settings = _load(path)
    _backup_once(path)
    base = _base(url)
    hooks = settings.setdefault("hooks", {})
    for event in EVENTS:
        entries = hooks.setdefault(event, [])
        # Replace rather than append: an entry from an older port or token is
        # dead weight that would post into nothing forever.
        entries[:] = _strip_bol(entries, base)
        entry: dict = {"hooks": [_bol_hook(url)]}
        if event == "PostToolUse":
            entry["matcher"] = "*"
        entries.append(entry)
    _save(path, settings)
    return path


def uninstall(url: str, scope: str = "user", project_dir: Path | None = None) -> Path:
    path = _settings_path(scope, project_dir)
    if not path.exists():
        return path
    settings = _load(path)
    base = _base(url)
    before = json.dumps(settings, sort_keys=True)
    hooks = settings.get("hooks", {})
    for event in EVENTS:
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
