"""Configuration: TOML file at ~/.config/bol/config.toml, env overrides, sane defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("BOL_CONFIG_DIR", "~/.config/bol")).expanduser()
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    # Energy-gate endpointing: stop capture after this much trailing silence.
    silence_ms: int = 900
    # Utterances shorter than this are discarded as noise.
    min_speech_ms: int = 300
    max_utterance_s: int = 90
    # Hands-free reopen: give up if no speech starts within this window.
    listen_window_s: int = 8
    # RMS multiplier over measured noise floor that counts as speech.
    energy_threshold: float = 3.0


@dataclass
class SttConfig:
    engine: str = "parakeet"  # parakeet | none (text mode)
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v3"


@dataclass
class HotkeyConfig:
    # push_to_talk: hold key to speak; toggle: tap to start/stop.
    mode: str = "push_to_talk"
    key: str = "alt_r"  # pynput key name


@dataclass
class TtsConfig:
    engine: str = "say"  # say | kokoro | none
    say_voice: str = ""  # empty = system default
    say_rate: int = 200
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    kokoro_voice: str = "af_heart"
    kokoro_speed: float = 1.1


@dataclass
class LlmConfig:
    # local: Bol runs a small model on your Mac via mlx_lm.server (no key).
    # api: any OpenAI-compatible endpoint (OpenRouter, Groq, Ollama, LM
    #      Studio, OpenAI, Anthropic's compat endpoint) with your own key.
    # off: deterministic template summaries only, no cleanup.
    provider: str = "local"  # local | api | off
    local_model: str = "mlx-community/LFM2.5-1.2B-Instruct-4bit"
    # Budget alternative for low-RAM Macs: LiquidAI/LFM2.5-350M-MLX-4bit
    local_port: int = 8771
    base_url: str = ""  # api provider, e.g. https://openrouter.ai/api/v1
    api_key_env: str = "BOL_API_KEY"
    api_model: str = ""
    temperature: float = 0.3


@dataclass
class CleanupConfig:
    # Fixes the dictated prompt before injection. Local mode uses instant
    # deterministic rules (fillers, stutters, "auth dot py" -> "auth.py");
    # api mode adds an LLM grammar pass by the user's own model. Small local
    # models are never given the rewrite job (they drop clauses).
    # off: never. on_command: when you say "clean it up". always: every time.
    mode: str = "on_command"  # off | on_command | always
    deadline_s: float = 2.5
    # Bol's own tuned cleanup model (HF repo or local path), used after the
    # deterministic rules when mlx-lm is installed. Empty = rules only.
    model: str = "abhiyan10/bol-cleanup-350m-4bit"


@dataclass
class SummarizerConfig:
    # auto: LLM persona summary when an LLM is available, template otherwise.
    # template: free, deterministic, built from the tool log. Always the
    # fallback when the LLM errors or times out.
    engine: str = "auto"  # auto | template
    timeout_s: float = 10.0
    user_name: str = ""  # spoken name, e.g. "Abhiyan"


@dataclass
class BridgeConfig:
    # auto: tmux when a Claude pane exists, else focused-app paste.
    # focused: always paste into the frontmost terminal (FluidVoice-style).
    # tmux: always inject into a tmux pane (focus-independent).
    mode: str = "auto"
    # tmux pane id (e.g. "%3") to pin; empty = auto-discover.
    pane: str = ""
    enter_delay_s: float = 0.2
    # Focused mode: bundle ids allowed to receive injection (empty = built-in
    # terminal allowlist). Guard against dictating into the wrong app.
    allowed_apps: list = field(default_factory=list)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8770
    # Shared secret appended to the hook URL; auto-generated on install.
    token: str = ""


@dataclass
class Config:
    # Voice-command phrase overrides: {"send": [...], "type": [...],
    # "discard": [...], "sleep": [...], "interrupt": [...], "repeat": [...],
    # "clean": [...]}. Unset keys keep the defaults.
    commands: dict = field(default_factory=dict)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    # After Bol speaks a reply, automatically open the mic for the next instruction.
    hands_free: bool = True
    sound_cues: bool = True

    @property
    def api_key(self) -> str:
        return os.environ.get(self.llm.api_key_env, "") or os.environ.get(
            "OPENROUTER_API_KEY", ""
        )


def _apply(section: object, data: dict) -> None:
    for key, value in data.items():
        if hasattr(section, key) and not isinstance(getattr(section, key), (AudioConfig,)):
            setattr(section, key, value)


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    cfg = Config()
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for name in (
            "audio", "stt", "hotkey", "tts", "llm", "cleanup", "summarizer",
            "bridge", "server",
        ):
            if name in data and isinstance(data[name], dict):
                _apply(getattr(cfg, name), data[name])
        for name in ("hands_free", "sound_cues", "commands"):
            if name in data:
                setattr(cfg, name, data[name])
    return cfg


DEFAULT_CONFIG_TOML = """\
# Bol configuration. Every key is optional; these are the defaults.

hands_free = true    # reopen the mic automatically after Bol speaks
sound_cues = true    # audible blips when listening starts/stops

[hotkey]
mode = "push_to_talk"  # or "toggle"
key = "alt_r"          # right Option

# Remap any voice command to whatever you like. Unset keys keep defaults.
# [commands]
# send = ["ship it", "send it", "go"]
# discard = ["forget it"]
# interrupt = ["whoa whoa"]

[stt]
engine = "parakeet"    # or "none" for text-only mode

[tts]
engine = "say"         # or "kokoro" (pip install 'bol[kokoro]')

[llm]
provider = "local"     # local (no key needed) | api (your key) | off
local_model = "mlx-community/LFM2.5-1.2B-Instruct-4bit"
# low-RAM Macs: local_model = "LiquidAI/LFM2.5-350M-MLX-4bit"
# api example (works with OpenRouter, Groq, Ollama, LM Studio, OpenAI):
# provider = "api"
# base_url = "https://openrouter.ai/api/v1"
# api_model = "google/gemini-2.5-flash-lite"
# api_key_env = "BOL_API_KEY"

[cleanup]
mode = "on_command"    # say "clean it up" | "always" | "off"
model = "abhiyan10/bol-cleanup-350m-4bit"  # Bol's own 195MB model; "" = rules only

[summarizer]
engine = "auto"        # llm persona when available, template otherwise
user_name = ""         # your name, spoken in replies

[bridge]
mode = "auto"          # auto | focused (paste into front terminal) | tmux
pane = ""              # tmux mode: pane id like "%3"; empty auto-discovers

[server]
port = 8770
"""


def write_default_config() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML)
    return CONFIG_PATH


def hook_token() -> str:
    """Shared secret for the hook URL, generated once and persisted."""
    import secrets

    path = CONFIG_DIR / "hook_token"
    if path.exists():
        token = path.read_text().strip()
        if token:
            return token
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    path.write_text(token)
    path.chmod(0o600)
    return token
