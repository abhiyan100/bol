"""bol CLI.

  bol run              start the voice daemon (hotkey + mic + talk-back)
  bol talk             text mode — same loop, typed instead of spoken
  bol launch [dir]     start claude inside tmux and attach to it
  bol hook install     add Bol's hooks to Claude Code settings (user scope)
  bol hook uninstall   remove them
  bol doctor           check environment, permissions, and wiring
  bol config           write the default config file and print its path
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import shutil
import sys

from . import __version__
from .config import CONFIG_PATH, load_config, write_default_config
from .hooks import installer


def _url(cfg) -> str:
    return f"http://{cfg.server.host}:{cfg.server.port}/hook"


def cmd_run(args: argparse.Namespace) -> int:
    from .daemon import Daemon

    cfg = load_config()
    if not installer.installed(_url(cfg)):
        print("bol: hooks not installed — running `bol hook install` for you.")
        installer.install(_url(cfg))
        print("bol: note — Claude Code sessions started before this need a restart to pick up hooks.")
    daemon = Daemon(cfg, text_mode=args.text)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nbol: bye.")
    return 0


def cmd_talk(args: argparse.Namespace) -> int:
    args.text = True
    return cmd_run(args)


def cmd_launch(args: argparse.Namespace) -> int:
    from .bridge import TmuxBridge

    async def _launch():
        pane = await TmuxBridge.launch(session=args.session, cwd=args.dir)
        print(f"bol: claude running in tmux session '{args.session}' (pane {pane.pane_id})")
        print(f"bol: attach with — tmux attach -t {args.session}")

    asyncio.run(_launch())
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    cfg = load_config()
    url = _url(cfg)
    if args.hook_cmd == "install":
        path = installer.install(url, scope=args.scope)
        print(f"bol: hooks installed in {path}")
        print("bol: restart running Claude Code sessions to pick them up.")
    elif args.hook_cmd == "uninstall":
        path = installer.uninstall(url, scope=args.scope)
        print(f"bol: hooks removed from {path}")
    return 0


def cmd_config(_args: argparse.Namespace) -> int:
    path = write_default_config()
    print(path)
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    cfg = load_config()
    ok = True

    def check(label: str, passed: bool, hint: str = "") -> None:
        nonlocal ok
        mark = "ok " if passed else "FAIL"
        print(f"  [{mark}] {label}" + (f" — {hint}" if hint and not passed else ""))
        ok = ok and passed

    print(f"bol {__version__} doctor\n")
    check("macOS", platform.system() == "Darwin", "Bol currently targets macOS")
    check("Apple Silicon", platform.machine() == "arm64",
          "Parakeet STT needs Apple Silicon; use text mode elsewhere")
    check("tmux installed", shutil.which("tmux") is not None, "brew install tmux")
    check("claude installed", shutil.which("claude") is not None,
          "https://code.claude.com — npm install -g @anthropic-ai/claude-code")
    check("say available", shutil.which("say") is not None)

    try:
        import parakeet_mlx  # noqa: F401
        check("parakeet-mlx (voice input)", True)
    except ImportError:
        check("parakeet-mlx (voice input)", False, "uv sync --extra stt")

    try:
        import mlx_audio  # noqa: F401
        check("mlx-audio (Kokoro voice)", True)
    except ImportError:
        print("  [ -- ] mlx-audio not installed — `say` voice will be used "
              "(optional: uv sync --extra kokoro)")

    check(
        "hooks installed (user scope)",
        installer.installed(_url(cfg)),
        "bol hook install",
    )
    check("config file", CONFIG_PATH.exists(), "bol config")
    if cfg.summarizer.engine == "openrouter":
        check("OpenRouter key", bool(cfg.openrouter_key),
              "set OPENROUTER_API_KEY or [summarizer] openrouter_api_key")

    async def _tmux_check() -> tuple[bool, str]:
        from .bridge import TmuxBridge, TmuxError

        try:
            panes = await TmuxBridge().discover()
        except TmuxError as exc:
            if "no server running" in str(exc):
                return False, "tmux isn't running — `bol launch` starts claude in a fresh session"
            return False, str(exc)
        if not panes:
            return False, "no Claude pane found — run `bol launch` or start claude inside tmux"
        return True, ", ".join(f"{p.pane_id} ({p.target})" for p in panes)

    found, detail = asyncio.run(_tmux_check())
    print(f"  [{'ok ' if found else ' -- '}] Claude pane: {detail}")

    print("\nPermissions reminders (macOS prompts on first use):")
    print("  - Microphone: granted to your terminal app on first recording")
    print("  - Input Monitoring: needed by the global hotkey — System Settings")
    print("    → Privacy & Security → Input Monitoring → enable your terminal")
    print("\n" + ("all good — run `bol run`" if ok else "fix the FAIL lines above, then `bol run`"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bol",
        description="Speak. Bol runs Claude Code hands-free.",
    )
    parser.add_argument("--version", action="version", version=f"bol {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="start the voice daemon")
    p_run.add_argument("--text", action="store_true", help="text mode (no mic)")
    p_run.set_defaults(func=cmd_run)

    p_talk = sub.add_parser("talk", help="text mode — typed loop, no mic")
    p_talk.set_defaults(func=cmd_talk)

    p_launch = sub.add_parser("launch", help="start claude in tmux")
    p_launch.add_argument("dir", nargs="?", default=os.getcwd())
    p_launch.add_argument("--session", default="bol")
    p_launch.set_defaults(func=cmd_launch)

    p_hook = sub.add_parser("hook", help="manage Claude Code hooks")
    p_hook.add_argument("hook_cmd", choices=["install", "uninstall"])
    p_hook.add_argument("--scope", choices=["user", "project"], default="user")
    p_hook.set_defaults(func=cmd_hook)

    p_doc = sub.add_parser("doctor", help="check environment and wiring")
    p_doc.set_defaults(func=cmd_doctor)

    p_cfg = sub.add_parser("config", help="write default config, print path")
    p_cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )
    if not args.cmd:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
