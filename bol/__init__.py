"""Bol: speak to Claude Code, hear back what happened."""

__version__ = "0.6.9"

INSTALL_URL = "git+https://github.com/abhiyan100/bol"


def install_hint(extras: str) -> str:
    """One reinstall command that adds an optional extra, for error messages.

    `uv tool install` users cannot `uv sync`; the --force reinstall with the
    extras they want is the only path. The clone form is mentioned second.
    """
    last = extras.split(",")[-1]
    return (
        f'uv tool install --force "bol[{extras}] @ {INSTALL_URL}" '
        f"(from a clone: uv sync --extra {last})"
    )
