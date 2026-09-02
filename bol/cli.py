"""bol CLI.

  bol                  same as `bol run`
  bol run              start the voice daemon (hotkey + mic + talk-back)
  bol talk             text mode, the same loop typed instead of spoken
  bol setup            first run: download models, install hooks, check permissions
  bol launch [dir]     start claude inside tmux and attach to it
  bol hook install     add Bol's hooks to Claude Code settings (user scope)
  bol hook uninstall   remove them
  bol doctor           check environment, permissions, and wiring
  bol config           write the default config file and print its path
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys

from . import __version__, install_hint
from .config import CONFIG_PATH, hook_token, load_config, write_default_config
from .hooks import installer

# Probe results: (status, label, hint). The hint is printed only for failures,
# so an informational row carries everything it wants to say in its label.
OK = "ok"
BAD = "bad"
INFO = "info"

_MARKS = {OK: "ok ", BAD: "!! ", INFO: " --"}


def _base_url(cfg) -> str:
    return f"http://{cfg.server.host}:{cfg.server.port}/hook"


def _url(cfg) -> str:
    return f"{_base_url(cfg)}?token={hook_token()}"


def _print_rows(rows: list[tuple[str, str, str]]) -> bool:
    """Print probe rows, return True when nothing failed."""
    ok = True
    for status, label, hint in rows:
        print(f"  [{_MARKS.get(status, ' --')}] {label}")
        if status == BAD:
            ok = False
            if hint:
                print(f"         fix: {hint}")
    return ok


# --------------------------------------------------------------------- probes


def probe_system(cfg) -> list[tuple[str, str, str]]:
    system = platform.system()
    machine = platform.machine()
    rows = [
        (
            OK if system == "Darwin" else BAD,
            f"platform: {system} {platform.mac_ver()[0] or ''}".strip(),
            "Bol currently targets macOS",
        ),
        (
            OK if machine == "arm64" else BAD,
            f"cpu: {machine}",
            "Parakeet needs Apple Silicon; `bol talk` still works elsewhere",
        ),
        (
            OK if shutil.which("claude") else BAD,
            "claude CLI on PATH",
            "npm install -g @anthropic-ai/claude-code (https://code.claude.com)",
        ),
        (
            OK if shutil.which("say") else BAD,
            "say available",
            "macOS ships `say`; check your PATH",
        ),
    ]
    rows.append(probe_memory(cfg))
    return rows


# Rough resident set of the default local stack: Parakeet in the daemon plus
# the 1.2B summarizer and the cleanup model in the mlx_lm server, before
# Claude Code, a browser, and an IDE take their share.
_LOCAL_STACK_GB = 4
_COMFORTABLE_GB = 16


def probe_memory(cfg) -> tuple[str, str, str]:
    total = physical_memory_bytes()
    if total is None:
        return (INFO, "memory: could not be read", "")
    gb = total / (1024**3)
    label = f"memory: {gb:.0f} GB"
    if cfg.llm.provider == "local" and gb < _COMFORTABLE_GB:
        return (
            INFO,
            f"{label}; the local stack uses about {_LOCAL_STACK_GB} GB. If the "
            "Mac swaps, set [llm] local_model = "
            '"LiquidAI/LFM2.5-350M-MLX-4bit" or provider = "api"',
            "",
        )
    return (OK, label, "")


def physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def probe_packages(cfg) -> list[tuple[str, str, str]]:
    rows = []

    def importable(name: str) -> bool:
        try:
            __import__(name)
        except Exception:
            return False
        return True

    if cfg.stt.engine == "parakeet":
        rows.append((
            OK if importable("parakeet_mlx") else BAD,
            "parakeet-mlx (voice input)",
            install_hint("stt,llm"),
        ))
    else:
        rows.append((INFO, f"stt engine: {cfg.stt.engine} (no voice input)", ""))

    if importable("mlx_audio"):
        rows.append((OK, "mlx-audio (Kokoro voice)", ""))
    else:
        rows.append((
            INFO,
            "mlx-audio not installed, the `say` voice will be used "
            "(optional, reinstall with the kokoro extra)",
            "",
        ))

    if cfg.llm.provider == "local":
        if importable("mlx_lm"):
            rows.append((OK, "mlx-lm (local brain: cleanup and summaries)", ""))
        else:
            rows.append((
                INFO,
                "mlx-lm not installed, template summaries only "
                "(optional, reinstall with the llm extra)",
                "",
            ))
    elif cfg.llm.provider == "api":
        rows.append((
            OK if cfg.api_key else BAD,
            f"API key (${cfg.llm.api_key_env})",
            f"export {cfg.llm.api_key_env}=... for "
            f"{cfg.llm.base_url or 'your endpoint'}",
        ))
    else:
        rows.append((INFO, "llm provider: off (template summaries only)", ""))
    return rows


def probe_wiring(cfg) -> list[tuple[str, str, str]]:
    rows = []
    try:
        hooks_ok = installer.installed(_url(cfg))
    except SystemExit as exc:
        rows.append((BAD, "hooks: Claude Code settings unreadable", str(exc)))
    else:
        rows.append((
            OK if hooks_ok else BAD,
            "hooks installed (user scope)",
            "bol hook install (or just run `bol run`, it installs them)",
        ))
    # A missing config file is normal: every key has a default.
    if CONFIG_PATH.exists():
        rows.append((OK, f"config file: {CONFIG_PATH}", ""))
    else:
        rows.append((
            INFO,
            "config file: none yet, built-in defaults in use "
            "(`bol config` writes one)",
            "",
        ))
    rows.append(probe_port(cfg))
    return rows


def probe_port(cfg) -> tuple[str, str, str]:
    host, port = cfg.server.host, cfg.server.port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
        return (OK, f"hook port {port} free", "")
    except OSError:
        if _bol_on_port(host, port):
            return (INFO, f"hook port {port}: a Bol is already running", "")
        return (
            BAD,
            f"hook port {port} is busy",
            f"another process holds it; set [server] port in {CONFIG_PATH}",
        )


def _bol_on_port(host: str, port: int) -> bool:
    """Does the thing holding our port answer Bol's /health?"""
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=1.0
        ) as resp:
            return json.loads(resp.read() or b"{}").get("app") == "bol"
    except Exception:
        return False


def probe_permissions(cfg) -> list[tuple[str, str, str]]:
    return [
        probe_microphone(cfg),
        probe_input_monitoring(),
        probe_automation(),
        probe_say_voice(cfg),
    ]


def probe_microphone(cfg) -> tuple[str, str, str]:
    """Ask macOS for the real TCC answer, then open the device for 100 ms.

    A denied mic opens fine and returns silence, so the permission status is
    the only trustworthy signal; the stream open proves the device exists and
    raises the system prompt the first time.
    """
    hint = (
        "System Settings > Privacy & Security > Microphone > enable your "
        "terminal app, then restart it"
    )
    status = _mic_authorization()
    if status in (_MIC_DENIED, _MIC_RESTRICTED):
        return (BAD, "microphone: permission denied for this terminal app", hint)
    try:
        import sounddevice as sd
    except Exception as exc:
        return (BAD, f"microphone: sounddevice unavailable ({exc})", install_hint("stt,llm"))
    try:
        device = _input_device(cfg)
    except ValueError as exc:
        return (BAD, f"microphone: {exc}", "set [audio] input_device")
    try:
        with sd.InputStream(
            samplerate=cfg.audio.sample_rate,
            channels=1,
            device=device,
            blocksize=0,
        ) as stream:
            name = sd.query_devices(stream.device, "input")["name"]
            stream.read(int(cfg.audio.sample_rate * 0.1))
    except Exception as exc:
        return (BAD, f"microphone: cannot record ({exc})", hint)
    if status == _MIC_NOT_DETERMINED and _mic_authorization() == _MIC_NOT_DETERMINED:
        return (
            INFO,
            f"microphone: {name} (macOS is still asking for permission; "
            "approve it, then rerun `bol doctor`)",
            "",
        )
    return (OK, f"microphone: {name}", "")


# AVAuthorizationStatus
_MIC_NOT_DETERMINED, _MIC_RESTRICTED, _MIC_DENIED, _MIC_AUTHORIZED = 0, 1, 2, 3


def _mic_authorization() -> int | None:
    """AVCaptureDevice.authorizationStatusForMediaType(AVMediaTypeAudio).

    ctypes rather than pyobjc: Bol does not depend on the AVFoundation
    bindings, and this is the only question worth asking about a mic that
    opens happily while returning silence. None when the answer is unavailable.
    """
    if platform.system() != "Darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        av_path = ctypes.util.find_library("AVFoundation")
        objc_path = ctypes.util.find_library("objc")
        if not av_path or not objc_path:
            return None
        av = ctypes.cdll.LoadLibrary(av_path)
        objc = ctypes.cdll.LoadLibrary(objc_path)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        cls = objc.objc_getClass(b"AVCaptureDevice")
        sel = objc.sel_registerName(b"authorizationStatusForMediaType:")
        media = ctypes.c_void_p.in_dll(av, "AVMediaTypeAudio")
        if not cls or not sel:
            return None
        send = objc.objc_msgSend
        send.restype = ctypes.c_long
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        return int(send(ctypes.c_void_p(cls), ctypes.c_void_p(sel), media))
    except Exception:
        return None


def _input_device(cfg):
    """The configured input device as a sounddevice id, or None for default.
    Raises ValueError with the available names when the setting matches nothing."""
    spec = getattr(cfg.audio, "input_device", "")
    if not spec:
        return None
    from .audio.capture import _resolve_input_device

    return _resolve_input_device(spec)


def probe_input_monitoring() -> tuple[str, str, str]:
    trusted = _process_trusted()
    if trusted is None:
        return (INFO, "input monitoring: cannot be checked on this platform", "")
    if trusted:
        return (OK, "input monitoring: granted (the hotkey can fire)", "")
    try:
        from .hotkey.listener import NOT_TRUSTED as hint
    except Exception:
        hint = (
            "System Settings > Privacy & Security > Input Monitoring > enable "
            "your terminal app, then restart Bol"
        )
    return (BAD, "input monitoring: not granted, the hotkey will never fire", hint)


def _process_trusted() -> bool | None:
    """AXIsProcessTrusted, the same call pynput's macOS backend makes.

    Done through ctypes so the check needs no listener and no key press.
    """
    if platform.system() != "Darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("ApplicationServices")
        if not path:
            return None
        lib = ctypes.cdll.LoadLibrary(path)
        lib.AXIsProcessTrusted.argtypes = []
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except Exception:
        return None


def probe_automation() -> tuple[str, str, str]:
    if platform.system() != "Darwin":
        return (INFO, "automation: not macOS, skipped", "")
    hint = (
        "System Settings > Privacy & Security > Automation > allow your "
        "terminal app to control System Events"
    )
    try:
        proc = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get name of first process',
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return (BAD, "automation: osascript did not answer", hint)
    if proc.returncode == 0:
        return (OK, "automation: System Events reachable (Bol can press Enter)", "")
    return (BAD, "automation: System Events refused", hint)


def probe_say_voice(cfg) -> tuple[str, str, str]:
    if cfg.tts.engine != "say":
        return (INFO, f"tts engine: {cfg.tts.engine}", "")
    voice = cfg.tts.say_voice
    if not voice:
        return (OK, "say voice: system default", "")
    try:
        proc = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return (INFO, f"say voice: {voice} (could not list installed voices)", "")
    names = set()
    for line in proc.stdout.splitlines():
        match = re.match(r"^(.+?)\s{2,}[a-z]{2}[_-]", line)
        if match:
            names.add(match.group(1).strip().lower())
    if not names or voice.lower() in names:
        return (OK, f"say voice: {voice}", "")
    return (
        BAD,
        f"say voice: {voice} is not installed, Bol would stay silent",
        "run `say -v '?'` for the list, then set [tts] say_voice",
    )


def probe_weights(cfg) -> list[tuple[str, str, str]]:
    from .llm.engine import (
        hf_available,
        human_bytes,
        size_hint,
        weights_cached,
        weights_size_bytes,
    )

    if not hf_available():
        return [(INFO, "models: cannot check (huggingface_hub not installed)", "")]
    rows = []
    for role, repo in wanted_models(cfg):
        if weights_cached(repo):
            size = human_bytes(weights_size_bytes(repo))
            rows.append((OK, f"{role}: {repo} ({size} on disk)", ""))
        else:
            rows.append((
                BAD,
                f"{role}: {repo} not downloaded ({size_hint(repo)})",
                "run `bol setup` to fetch it now, or let first use pause for it",
            ))
    return rows or [(INFO, "models: none needed by this config", "")]


def probe_injection(cfg) -> list[tuple[str, str, str]]:
    return [probe_frontmost(cfg), probe_tmux(cfg)]


def probe_frontmost(cfg) -> tuple[str, str, str]:
    """Which app would receive a paste right now. Informational: the answer
    depends on which window happens to be in front while doctor runs."""
    try:
        from .bridge import DEFAULT_ALLOWED_APPS, frontmost_bundle_id
    except ImportError:
        return (INFO, "frontmost app: this build cannot report it", "")
    try:
        bundle = asyncio.run(frontmost_bundle_id())
    except Exception as exc:
        return (INFO, f"frontmost app: could not read it ({exc})", "")
    if not bundle:
        return (INFO, "frontmost app: unknown (no Automation permission?)", "")
    allowed = set(cfg.bridge.allowed_apps) or set(DEFAULT_ALLOWED_APPS)
    if bundle in allowed:
        return (OK, f"frontmost app: {bundle} (allowed to receive dictation)", "")
    return (
        INFO,
        f"frontmost app: {bundle} is not allowed; add it with "
        f'[bridge] allowed_apps = ["{bundle}"]',
        "",
    )


def probe_tmux(cfg) -> tuple[str, str, str]:
    if shutil.which("tmux") is None:
        return (
            INFO,
            "tmux: not installed (focused mode pastes into your front terminal)",
            "",
        )

    async def _panes():
        from .bridge import TmuxBridge, TmuxError

        try:
            return await TmuxBridge().discover()
        except TmuxError:
            return []

    try:
        panes = asyncio.run(_panes())
    except Exception:
        panes = []
    if panes:
        listing = ", ".join(f"{p.pane_id} ({p.target})" for p in panes)
        return (INFO, f"tmux: claude panes {listing}", "")
    return (INFO, "tmux: no claude pane, focused mode will be used", "")


# ---------------------------------------------------------------- model needs


def wanted_models(cfg) -> list[tuple[str, str]]:
    """(role, repo id) for every model this config will actually load."""
    models = []
    if cfg.stt.engine == "parakeet":
        models.append(("speech to text", cfg.stt.parakeet_model))
    if cfg.llm.provider == "local":
        models.append(("summaries", cfg.llm.local_model))
    if cfg.cleanup.model and cfg.cleanup.mode != "off":
        models.append(("cleanup", cfg.cleanup.model))
    if cfg.tts.engine == "kokoro":
        models.append(("voice", cfg.tts.kokoro_model))
    return models


def installed_models(cfg) -> list[tuple[str, str]]:
    """The wanted models whose runtime package is actually installed, so
    `bol setup` never downloads 2.3 GB for an extra the user skipped."""
    needs = {
        "speech to text": "parakeet_mlx",
        "summaries": "mlx_lm",
        "cleanup": "mlx_lm",
        "voice": "mlx_audio",
    }
    out = []
    for role, repo in wanted_models(cfg):
        module = needs.get(role)
        if module:
            try:
                __import__(module)
            except Exception:
                continue
        out.append((role, repo))
    return out


# -------------------------------------------------------------------commands


def cmd_run(args: argparse.Namespace) -> int:
    from .daemon import Daemon

    try:
        from .bridge import BridgeError
    except ImportError:  # pragma: no cover - bridge ships with Bol
        class BridgeError(Exception):
            pass

    cfg = load_config()
    _quiet_model_libraries(cfg)
    if not installer.installed(_url(cfg)):
        print("bol: hooks not installed, running `bol hook install` for you.")
        installer.uninstall(_base_url(cfg))  # drop stale token-less entries
        installer.install(_url(cfg))
        print(
            "bol: note, Claude Code sessions started before this need a "
            "restart to pick up hooks."
        )
    _warn_missing_weights(cfg)
    try:
        if not args.text:
            # Fail now with the device list, not at the first held hotkey.
            _input_device(cfg)
        daemon = Daemon(cfg, text_mode=args.text)
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nbol: bye.")
    except PermissionError as exc:
        # macOS permission, e.g. Input Monitoring off. The raiser owns the remedy.
        print(f"bol: {exc}")
        return 1
    except ValueError as exc:
        # Bad config value, e.g. an unknown hotkey name.
        print(f"bol: {exc}")
        return 1
    except BridgeError as exc:
        print(f"bol: {exc}")
        return 1
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"bol: port {cfg.server.port} is busy. Another Bol is probably "
                f"running, or set [server] port in {CONFIG_PATH}."
            )
        else:
            print(f"bol: {exc}")
        return 1
    return 0


# Third-party loggers that narrate every HTTP request at INFO. Bol's own
# `bol.*` loggers stay at INFO so the user still sees model loads and hooks.
NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "hf_xet")


def _quiet_model_libraries(cfg) -> None:
    """Keep the daemon's terminal readable and its startup offline.

    Progress bars belong in `bol setup`, not in a running daemon. And once
    every model this config needs is on disk there is no reason to ask the
    Hub about it on each start: offline mode skips the HEAD requests, the
    rate-limit warning, and works with the wifi off. Both are only defaults;
    an explicit HF_HUB_* in the environment wins. The env vars also reach the
    mlx_lm.server child, which inherits them.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from .llm.engine import hf_available, weights_cached
    except Exception:
        return
    if not hf_available():
        return
    models = installed_models(cfg)
    if models and all(weights_cached(repo) for _role, repo in models):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        # huggingface_hub read the environment when it was imported (the cache
        # check above imported it), so mirror the decision onto its constants.
        try:
            from huggingface_hub import constants

            constants.HF_HUB_OFFLINE = os.environ["HF_HUB_OFFLINE"] == "1"
            constants.HF_HUB_DISABLE_PROGRESS_BARS = (
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
            )
        except Exception:
            pass


def _warn_missing_weights(cfg) -> None:
    """One line, not a wall: first use still works, it just pauses to download."""
    try:
        from .llm.engine import hf_available, weights_cached
    except Exception:
        return
    if not hf_available():
        return
    missing = [repo for _role, repo in installed_models(cfg) if not weights_cached(repo)]
    if missing:
        noun = "model" if len(missing) == 1 else "models"
        print(
            f"bol: {len(missing)} {noun} not downloaded yet, so the first use "
            "will pause to fetch them. Run `bol setup` to do it now."
        )


def cmd_talk(args: argparse.Namespace) -> int:
    args.text = True
    return cmd_run(args)


def cmd_launch(args: argparse.Namespace) -> int:
    from .bridge import TmuxBridge

    async def _launch():
        pane = await TmuxBridge.launch(session=args.session, cwd=args.dir)
        print(f"bol: claude running in tmux session '{args.session}' (pane {pane.pane_id})")
        print(f"bol: attach with: tmux attach -t {args.session}")

    asyncio.run(_launch())
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    cfg = load_config()
    url = _url(cfg)
    if args.hook_cmd == "install":
        if args.scope == "project":
            print(installer.PROJECT_SCOPE_WARNING)
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


def cmd_setup(_args: argparse.Namespace) -> int:
    cfg = load_config()
    print(f"bol {__version__} setup\n")

    existed = CONFIG_PATH.exists()
    path = write_default_config()
    print(f"config: {'already at' if existed else 'written to'} {path}\n")

    ok = _setup_models(cfg)
    _setup_hooks(cfg)

    print("\nchecks (macOS asks for permission the first time):")
    rows = probe_permissions(cfg) + probe_wiring(cfg) + probe_injection(cfg)
    ok = _print_rows(rows) and ok

    if ok:
        print("\nsetup done. Next: run `bol run`, then tap or hold right Option and talk.")
    else:
        print(
            "\nsetup done, but fix the [!!] lines above first. "
            "Then run `bol run` and tap or hold right Option to talk."
        )
    return 0 if ok else 1


def _setup_models(cfg) -> bool:
    """Print the disk budget, then download whatever is missing."""
    from .llm.engine import (
        hf_available,
        human_bytes,
        human_mb,
        size_hint,
        weights_cached,
        weights_size_bytes,
        MODEL_SIZES_MB,
    )

    models = installed_models(cfg)
    if not models:
        print("models: nothing to download for this config.\n")
        return True
    if not hf_available():
        print(
            "models: huggingface_hub is not installed, so Bol cannot "
            "pre-download. Models will be fetched on first use.\n"
        )
        return True

    print("disk budget for this config:")
    missing = []
    to_download = 0.0
    on_disk = 0
    for role, repo in models:
        if weights_cached(repo):
            size = weights_size_bytes(repo)
            on_disk += size
            print(f"  have  {role}: {repo} ({human_bytes(size)})")
        else:
            print(f"  need  {role}: {repo} ({size_hint(repo)})")
            to_download += MODEL_SIZES_MB.get(repo, 0)
            missing.append((role, repo))
    summary = f"  total: {human_bytes(on_disk)} already cached"
    if to_download:
        summary += f", about {human_mb(to_download)} to download"
    print(summary)
    if not missing:
        print("every model this config needs is already on disk.\n")
        return True
    print(f"\ndownloading {len(missing)} of {len(models)} into your Hugging Face cache.\n")

    from huggingface_hub import snapshot_download

    ok = True
    for role, repo in missing:
        print(f"downloading {repo} ({role})")
        try:
            snapshot_download(repo)
        except Exception as exc:
            ok = False
            print(f"bol: could not download {repo} ({exc}). Bol will retry on first use.")
    print()
    return ok


def _setup_hooks(cfg) -> None:
    """Show the exact entry going into settings.json, then write it."""
    url = _url(cfg)
    shown = url.split("?", 1)[0] + "?token=<your local token>"
    entry = {"matcher": "*", "hooks": [installer.bol_hook(shown)]}
    path = installer.settings_path("user")
    print(f"hooks: adding one entry per event to {path}")
    print(f"  events: {', '.join(installer.EVENTS)} (matcher only on PostToolUse)")
    for line in json.dumps(entry, indent=2).splitlines():
        print(f"  {line}")
    print("  async, so Claude never waits on Bol; silent when Bol is not running.")
    installer.install(url)
    print("hooks installed. Restart running Claude Code sessions to pick them up.")


def cmd_doctor(_args: argparse.Namespace) -> int:
    cfg = load_config()
    print(f"bol {__version__} doctor\n")
    ok = True
    for title, rows in (
        ("system", probe_system(cfg)),
        ("packages", probe_packages(cfg)),
        ("wiring", probe_wiring(cfg)),
        ("permissions", probe_permissions(cfg)),
        ("models", probe_weights(cfg)),
        ("injection", probe_injection(cfg)),
    ):
        if not rows:
            continue
        print(title)
        ok = _print_rows(rows) and ok
    print(
        "\n"
        + (
            "all good, run `bol run`"
            if ok
            else "fix the [!!] lines above, then run `bol run`"
        )
    )
    return 0 if ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bol",
        description="Speak. Bol runs Claude Code hands-free.",
    )
    parser.add_argument("--version", action="version", version=f"bol {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="start the voice daemon (the default)")
    p_run.add_argument("--text", action="store_true", help="text mode (no mic)")
    p_run.set_defaults(func=cmd_run)

    p_talk = sub.add_parser("talk", help="text mode: typed loop, no mic")
    p_talk.set_defaults(func=cmd_talk)

    p_setup = sub.add_parser(
        "setup", help="download models, install hooks, check permissions"
    )
    p_setup.set_defaults(func=cmd_setup)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if not args.cmd:
            # Bare `bol` is `bol run`: the thing people want almost every time.
            args = parser.parse_args([*argv, "run"])
    except SystemExit as exc:  # --version, --help, or a usage error
        return int(exc.code or 0)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )
    if not args.verbose:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
