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
class SummarizerConfig:
    # template: free, deterministic, built from the tool log.
    # openrouter: persona LLM pass over the template + Claude's last message.
    engine: str = "template"  # template | openrouter
    openrouter_model: str = "nvidia/nemotron-3.5-lightning:free"
    openrouter_api_key: str = ""  # falls back to $OPENROUTER_API_KEY
    timeout_s: float = 8.0
    user_name: str = ""  # spoken name, e.g. "Abhiyan"


@dataclass
class BridgeConfig:
    # tmux pane id (e.g. "%3") to pin; empty = auto-discover.
    pane: str = ""
    enter_delay_s: float = 0.2


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8770


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    # After Bol speaks a reply, automatically open the mic for the next instruction.
    hands_free: bool = True
    sound_cues: bool = True

    @property
    def openrouter_key(self) -> str:
        return self.summarizer.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")


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
            "audio", "stt", "hotkey", "tts", "summarizer", "bridge", "server",
        ):
            if name in data and isinstance(data[name], dict):
                _apply(getattr(cfg, name), data[name])
        for name in ("hands_free", "sound_cues"):
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

[stt]
engine = "parakeet"    # or "none" for text-only mode

[tts]
engine = "say"         # or "kokoro" (pip install 'bol[kokoro]')

[summarizer]
engine = "template"    # or "openrouter" for the persona voice
openrouter_model = "nvidia/nemotron-3.5-lightning:free"
user_name = ""         # your name, spoken in replies

[bridge]
pane = ""              # tmux pane id like "%3"; empty auto-discovers claude

[server]
port = 8770
"""


def write_default_config() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML)
    return CONFIG_PATH
