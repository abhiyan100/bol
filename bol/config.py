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
    # Hands-free only: utterances shorter than this are discarded as noise.
    # Push-to-talk returns everything captured while the key is held.
    min_speech_ms: int = 300
    max_utterance_s: int = 90
    # Hands-free reopen: give up if no speech starts within this window.
    listen_window_s: int = 8
    # RMS multiplier over measured noise floor that counts as speech.
    energy_threshold: float = 3.0
    # Mic to record from: device index or name substring; empty = system default.
    input_device: str = ""
    # Audio kept from just before the press, so the first syllable survives.
    # Only available while the mic stream is still warm.
    pre_roll_ms: int = 300
    # Keep the mic stream running this long after a recording, then release
    # the device. Bounded on purpose: a Bluetooth headset held open forever
    # stays in its tinny headset profile.
    warm_s: float = 120.0


@dataclass
class SttConfig:
    engine: str = "parakeet"  # parakeet | none (text mode)
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    # Show words in the pill while you speak. Display only: the text that
    # reaches Claude always comes from the full-buffer decode afterwards.
    live: bool = True
    # (left, right) attention context for the streaming decoder, in encoder
    # frames of 80 ms. The right half is the finalization lag, so parakeet-mlx's
    # documented default of (256, 256) holds every word back for 20 seconds.
    # (256, 16) commits after 1.3 s; (256, 8) is faster and slightly less sure.
    stream_context: list = field(default_factory=lambda: [256, 16])
    # Audio handed to the streaming decoder per step. Smaller updates the pill
    # more often and costs more decodes per second.
    stream_chunk_ms: int = 320


@dataclass
class HotkeyConfig:
    # auto: tap to start (Bol ends the turn when you stop talking) or hold to
    # speak, whichever you did. push_to_talk: hold only. toggle: tap on/off.
    mode: str = "auto"  # auto | push_to_talk | toggle
    key: str = "alt_r"  # pynput key name
    # auto mode: a press shorter than this is a tap, longer is a hold.
    tap_ms: int = 400
    # auto: plain dictation is submitted for you, like pressing Enter.
    # voice: nothing is submitted until you say "send it".
    submit: str = "auto"  # auto | voice
    # submit = "auto" only fires on this many words or more. A one-word
    # misfire ("yes") or a stray noise is pasted, never sent. Saying
    # "send it" still submits whatever it is riding on, however short.
    auto_send_min_words: int = 3


HOTKEY_MODES = ("auto", "push_to_talk", "toggle")
SUBMIT_MODES = ("auto", "voice")


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
    # Claude Code hooks are user-scoped, so every session on this machine
    # posts here. first: narrate only the first session Bol hears from.
    # all: narrate every session (replies will interleave).
    follow: str = "first"  # first | all
    # /hook types into your terminal, so it stays on loopback unless you
    # deliberately open it up.
    allow_remote: bool = False


@dataclass
class UiConfig:
    # The on-screen pill that says what Bol is doing (listening, thinking,
    # speaking, and what went wrong). Runs as its own tiny window process.
    pill: bool = True
    # Audible blips when listening starts and stops.
    sounds: bool = True
    # Which edge of the active screen the pill sits on.
    position: str = "top"  # top | bottom


UI_POSITIONS = ("top", "bottom")


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
    ui: UiConfig = field(default_factory=UiConfig)
    # After Bol speaks a reply, automatically open the mic for the next
    # instruction. Off by default: with [hotkey] submit = "auto" a mic that
    # reopens unasked can send whatever the room said next, and a tap is
    # instant anyway.
    hands_free: bool = False

    @property
    def sound_cues(self) -> bool:
        """Deprecated alias for [ui] sounds. Old config files still work."""
        return self.ui.sounds

    @sound_cues.setter
    def sound_cues(self, value: bool) -> None:
        self.ui.sounds = bool(value)

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
        # The old top-level sound_cues, applied first so an explicit
        # [ui] sounds in the same file still wins.
        if "sound_cues" in data:
            cfg.sound_cues = data["sound_cues"]
        for name in (
            "audio", "stt", "hotkey", "tts", "llm", "cleanup", "summarizer",
            "bridge", "server", "ui",
        ):
            if name in data and isinstance(data[name], dict):
                _apply(getattr(cfg, name), data[name])
        for name in ("hands_free", "commands"):
            if name in data:
                setattr(cfg, name, data[name])
    return cfg


def _one_of(label: str, value: object, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(
            f"unknown {label} {value!r}. Choose one of: " + ", ".join(choices) + "."
        )


def validate_config(cfg: Config) -> None:
    """Reject config values Bol cannot act on.

    Called at startup rather than at load, so a typo costs one clear line
    ("unknown [hotkey] mode ...") instead of a hotkey that quietly never fires.
    """
    _one_of("[hotkey] mode", cfg.hotkey.mode, HOTKEY_MODES)
    _one_of("[hotkey] submit", cfg.hotkey.submit, SUBMIT_MODES)
    _one_of("[ui] position", cfg.ui.position, UI_POSITIONS)


DEFAULT_CONFIG_TOML = """\
# Bol configuration. Every key is optional; these are the defaults.

# reopen the mic automatically after Bol speaks. Off by default: with
# submit = "auto" below, a mic that reopens unasked can send whatever the room
# said next, and tapping the hotkey is instant anyway.
hands_free = false

[ui]
pill = true            # the on-screen pill that shows what Bol is doing
sounds = true          # audible blips when listening starts/stops
position = "top"       # which edge the pill sits on: "top" | "bottom"

[hotkey]
mode = "auto"          # tap or hold, whichever you did | "push_to_talk" | "toggle"
key = "alt_r"          # right Option
tap_ms = 400           # a press shorter than this counts as a tap, not a hold
submit = "auto"        # auto: dictation submits itself | "voice": only "send it" submits
auto_send_min_words = 3  # shorter than this is pasted, not sent; "send it" still submits

# Remap any voice command to whatever you like. Unset keys keep defaults.
# [commands]
# send = ["ship it", "send it", "go"]
# discard = ["forget it"]
# interrupt = ["whoa whoa"]

[audio]
# input_device = "MacBook Pro Microphone"  # mic name substring or index; empty = system default
pre_roll_ms = 300      # audio kept from just before the press, so no clipped first word
warm_s = 120           # hold the mic stream open this long after a recording, then let it go

[stt]
engine = "parakeet"    # or "none" for text-only mode
live = true            # show words in the pill while you talk (display only)
# stream_context = [256, 16]  # (left, right) attention frames of 80 ms. The right
#                             # half is how long a word waits before it is committed,
#                             # so parakeet-mlx's own default of [256, 256] would
#                             # hold your text back for 20 seconds. [256, 8] is faster.
# stream_chunk_ms = 320       # audio per streaming decode; smaller redraws more often

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
# follow = "first"      # narrate one Claude Code session at a time; "all" narrates every session
# allow_remote = false  # true lets the hook server bind off loopback (your network can type into your terminal)
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
