"""Configuration: TOML file at ~/.config/bol/config.toml, env overrides, sane defaults."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("BOL_CONFIG_DIR", "~/.config/bol")).expanduser()
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    # Which speech gate decides that a recording nobody is holding a key
    # for is over.
    # silero: Silero v6, a real voice-activity model (ships with Bol).
    # energy: RMS against an adaptive noise floor. Cheaper, and fooled by a
    #         keyboard, a fan, or anyone else in the room.
    # silero falls back to energy on its own if it cannot be loaded.
    vad: str = "silero"  # silero | energy
    # Endpointing: stop capture after this much trailing silence.
    silence_ms: int = 900
    # Until-silence recordings only: utterances shorter than this are
    # discarded as noise. Holding the key returns everything captured.
    min_speech_ms: int = 300
    max_utterance_s: int = 90
    # Until-silence recordings: give up if no speech starts in this window.
    listen_window_s: int = 8
    # Energy gate only: RMS multiplier over the measured noise floor that
    # counts as speech. Ignored when vad = "silero".
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
    # frames of 80 ms. The right half is how many frames a step refuses to
    # finalize, and 256 of them is longer than a normal recording -- which is
    # deliberate. A small right context makes each step commit the frames it
    # just added, and those sit on parakeet-mlx's window seam, so the live
    # words come out misspelt. See bol/stt/parakeet.py for the measurements.
    stream_context: list = field(default_factory=lambda: [256, 256])
    # Audio handed to the streaming decoder per step. Every step re-decodes the
    # buffer, so this is the redraw rate and the CPU bill at once. Floored at
    # MIN_CHUNK_MS in bol/stt/parakeet.py.
    stream_chunk_ms: int = 640


@dataclass
class HotkeyConfig:
    # Hold to talk, and that is the whole gesture: press the key, speak, let
    # go, and what you said is pasted where the cursor is. The recording ends
    # on the release, not on a pause. Nothing is ever submitted by itself;
    # say "send it" (or your own send phrase) to press Enter.
    key: str = "alt_r"  # pynput key name


VAD_MODES = ("silero", "energy")


@dataclass
class WakeConfig:
    # The trigger words, listened for from the moment Bol starts. On by
    # default: this is how Bol is meant to be used, and a microphone you have
    # to switch on is one most people never switch on. false turns the
    # always-on microphone off and leaves the hotkey exactly as it was.
    enabled: bool = True
    # The trigger for everything. A recording that ends when you stop talking
    # and is pasted where the cursor is; with talk_back = true it is the
    # conversation flow, and one-way (the default) it is the dictation flow
    # "type" used to start, endpointed on pause_ms. Bol matches the spellings a
    # speech model actually produces for each phrase (for "hey bol": "hey
    # bowl", "hey ball", "hey bull", "a bol" and "babel" too).
    phrases: list = field(default_factory=lambda: ["hey bol"])
    # A short trigger word that starts dictation, off by default: one syllable
    # scores far below "hey bol" in a real room; ["type"] to turn it on.
    type_phrases: list = field(default_factory=list)
    # Press Enter on text Bol has already pasted. Nothing pasted, nothing
    # happens. Left at this default, a [commands] send list wins instead, so
    # remapping "send it" to "ship it" remaps the trigger word too.
    send_phrases: list = field(default_factory=lambda: ["send it", "send", "enter"])
    # Wipe the whole input box, when there is a pending paste to wipe. Said a
    # dozen ways, because a wrong guess here costs the dictation again.
    cancel_phrases: list = field(
        default_factory=lambda: [
            "scratch that", "close", "scratch", "clear it", "clear that",
            "clear this", "clear the box",
        ]
    )
    # Stop listening for trigger words until the next hotkey press. Same rule
    # as send_phrases: a [commands] sleep list wins over this default.
    sleep_phrases: list = field(default_factory=lambda: ["stop listening"])
    # Trigger probability. Lower hears more, including the TV. Measured
    # against two macOS voices: 0.12 hears every trigger word, and 51 seconds
    # of speech full of "ball", "bowl", "close" and "send" false-fires on none
    # of them except "type".
    threshold: float = 0.12
    # type_phrases only, and 0 means "use threshold". Measured: raising this does
    # not buy precision. At 0.30 the real "type add a login test" stops
    # firing for one of the two test voices while all eight false positives
    # inside "prototype"/"what type of file" survive, because a keyword
    # spotter scores the same sounds the same way wherever they sit in a
    # sentence. If false dictation bothers you, change type_phrases to
    # something longer ("bol type", "dictate") rather than raising this.
    type_threshold: float = 0.0
    # How long a pause ends a dictation and pastes it. Longer than the [audio]
    # silence_ms the conversation flow uses: dictating a prompt has thinking
    # pauses in it. Two seconds, and the paste is instant after them.
    pause_ms: int = 2000
    # How long the pill waits for you to start speaking after a trigger word
    # (or after Bol asked you something). Nothing said in that window and the
    # pill goes away again, so a trigger heard across the room costs five
    # seconds of a dark capsule and nothing else.
    speak_window_ms: int = 5000
    # After a paste, how long Bol listens for a bare command like "send it" or
    # "scratch that", with nothing on screen. 0 = the keyword ear alone.
    command_window_s: float = 10.0
    # After a trigger word or a hold, how long follow-up speech needs no
    # trigger. 0 = only trigger words and the key ever start listening
    # (the default; room noise cannot wake the pill). 60 = a free minute.
    awake_s: float = 0.0


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
    # off: never. on_command: when you say "clean it up". always: every time,
    # which is the default because raw dictation is not what anybody wants to
    # hand a coding agent.
    mode: str = "always"  # off | on_command | always
    deadline_s: float = 2.5
    # Bol's own tuned cleanup model (HF repo or local path), used after the
    # deterministic rules when mlx-lm is installed. Empty = rules only.
    model: str = "abhiyan10/bol-cleanup-350m-4bit"


@dataclass
class VocabularyConfig:
    # Words the transcriber gets close to but not quite right: your name,
    # your project, a library nobody outside your terminal has heard of.
    # A token (or a pair of tokens) within one edit of an entry is replaced
    # with the entry, spelled exactly as it is written here. Common English
    # words are never touched, and entries shorter than five characters only
    # ever fix capitalization, because a wrong correction is worse than none.
    words: list = field(default_factory=list)


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
    # There is one bridge: Bol types into the app you are looking at. (A tmux
    # bridge that injected into a pinned pane lived here until v0.5.)
    enter_delay_s: float = 0.2
    # Bundle ids allowed to receive injection (empty = built-in terminal
    # allowlist). Guard against dictating into the wrong app.
    allowed_apps: list = field(default_factory=list)
    # Let dictation land wherever the cursor is (Notes, Slack, a
    # browser box), not only in a terminal or an IDE. Enter is a separate
    # question and stays gated: Bol presses it by itself only in a window
    # running Claude, and anywhere else only when the user asked for it in
    # words ("send it", "go ahead"). false is the old behavior, where the
    # allowlist above applies to every keystroke and every paste.
    anywhere: bool = True


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8770
    # Shared secret appended to the hook URL; auto-generated on install.
    token: str = ""
    # Agent hooks are user-scoped, so every Claude Code and every Codex on
    # this machine posts here, including ones nobody is looking at.
    # first: narrate one session, the one you last sent to (see bind_window_s).
    # all: narrate every session (replies will interleave).
    follow: str = "first"  # first | all
    # follow = "first" only. Sending is the one signal that says which
    # session the user means, so after Bol presses Enter for you (a spoken
    # send, or a send trigger word) the first hook event from another session
    # inside this many seconds moves the narration there. 0 pins Bol to
    # whichever session it happened to hear from first.
    bind_window_s: float = 20.0
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
    # Whether the pill spells out what it is doing beside the dots. Off: the
    # capsule is a mark and five dots, which is the whole state at a glance
    # and nothing to read. Errors still reach you either way, in the terminal
    # and out loud.
    text: bool = False


UI_POSITIONS = ("top", "bottom")


@dataclass
class SetupConfig:
    # What the `bol setup` wizard was told, so a rerun knows which coding
    # agents this Mac dictates into without asking again. "auto" is nobody
    # having answered yet: look at PATH. A list is an answer, and the empty
    # list is the answer "no coding agent", which is why this cannot simply
    # default to [] -- the two have to stay tellable apart.
    agents: list | str = "auto"  # "auto" | ["claude"] | ["claude", "codex"] | []


@dataclass
class Config:
    # Voice-command phrase overrides: {"send": [...], "type": [...],
    # "discard": [...], "sleep": [...], "interrupt": [...], "repeat": [...],
    # "clean": [...]}. Unset keys keep the defaults.
    commands: dict = field(default_factory=dict)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
    # Speak what Claude did. Off by default, which makes Bol one-way: your
    # voice goes in, nothing comes back out loud, and no summarizer, no
    # voice model and no local LLM server are ever loaded. true (or
    # `bol run --talk-back`) turns the reply on: a spoken summary after every
    # turn, and the "hey Bol" conversation flow. Either way the cleanup model
    # loads, because that is part of getting the dictation right.
    talk_back: bool = False

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
            "audio", "stt", "hotkey", "wake", "tts", "llm", "cleanup",
            "vocabulary", "summarizer", "bridge", "server", "ui", "setup",
        ):
            if name in data and isinstance(data[name], dict):
                _apply(getattr(cfg, name), data[name])
        for name in ("talk_back", "commands"):
            if name in data:
                setattr(cfg, name, data[name])
    return cfg


def _one_of(label: str, value: object, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(
            f"unknown {label} {value!r}. Choose one of: " + ", ".join(choices) + "."
        )


def _flag(label: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false, not {value!r}.")


def validate_config(cfg: Config) -> None:
    """Reject config values Bol cannot act on.

    Called at startup rather than at load, so a typo costs one clear line
    ("unknown [ui] position ...") instead of a pill that never appears.
    """
    _one_of("[ui] position", cfg.ui.position, UI_POSITIONS)
    _one_of("[audio] vad", cfg.audio.vad, VAD_MODES)
    # anywhere = "false" is a string, and a non-empty string is truthy, so a
    # quoted value would silently mean the opposite of what it says.
    _flag("[bridge] anywhere", cfg.bridge.anywhere)
    # Same trap, and this one decides whether a summarizer, a voice and a
    # local LLM server are loaded at all.
    _flag("talk_back", cfg.talk_back)
    validate_wake(cfg.wake)


def validate_wake(wake: WakeConfig) -> None:
    """Reject a wake section that would arm a listener with nothing to hear.

    Only checked when wake is on: a nonsense threshold in a disabled section
    is a note someone left themselves, not a reason to refuse to start.
    """
    if not wake.enabled:
        return
    phrases = wake.phrases if isinstance(wake.phrases, list) else []
    cleaned = [p for p in phrases if isinstance(p, str) and p.strip()]
    if not cleaned:
        raise ValueError(
            '[wake] enabled = true needs at least one phrase, e.g. phrases = ["hey bol"].'
        )
    try:
        threshold = float(wake.threshold)
    except (TypeError, ValueError):
        raise ValueError(f"[wake] threshold must be a number, not {wake.threshold!r}.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"[wake] threshold must be above 0 and at most 1, not {threshold}. "
            "Lower hears more, including the TV."
        )
    # 0 is "use threshold", so this one is allowed to be zero and nothing else.
    try:
        type_threshold = float(wake.type_threshold)
    except (TypeError, ValueError):
        raise ValueError(
            f"[wake] type_threshold must be a number, not {wake.type_threshold!r}."
        )
    if not 0.0 <= type_threshold <= 1.0:
        raise ValueError(
            f"[wake] type_threshold must be between 0 and 1, not {type_threshold}. "
            "0 means use the shared threshold."
        )
    try:
        pause = float(wake.pause_ms)
    except (TypeError, ValueError):
        raise ValueError(f"[wake] pause_ms must be a number, not {wake.pause_ms!r}.")
    if pause <= 0:
        raise ValueError(
            f"[wake] pause_ms must be above 0, not {pause}. It is how long a "
            "pause ends a dictation and pastes it."
        )
    try:
        command_window = float(wake.command_window_s)
    except (TypeError, ValueError):
        raise ValueError(
            f"[wake] command_window_s must be a number, not {wake.command_window_s!r}."
        ) from None
    if command_window < 0:
        raise ValueError(
            f"[wake] command_window_s must be 0 or more, not {command_window}."
        )
    try:
        speak_window = float(wake.speak_window_ms)
    except (TypeError, ValueError):
        raise ValueError(
            f"[wake] speak_window_ms must be a number, not {wake.speak_window_ms!r}."
        )
    if speak_window <= 0:
        raise ValueError(
            f"[wake] speak_window_ms must be above 0, not {speak_window}. It is "
            "how long the pill waits for you to start speaking."
        )
    try:
        awake = float(wake.awake_s)
    except (TypeError, ValueError):
        raise ValueError(f"[wake] awake_s must be a number, not {wake.awake_s!r}.")
    if awake < 0:
        raise ValueError(f"[wake] awake_s cannot be negative, got {awake}.")


DEFAULT_CONFIG_TOML = """\
# Bol configuration. Every key is optional; these are the defaults.

# Speak what Claude did. false is one-way: your voice goes in, nothing comes
# back out loud, and no summarizer, no voice and no local LLM server are
# loaded. true adds the spoken summary after every turn and the "hey Bol"
# conversation flow. `bol run --talk-back` turns it on for one session.
talk_back = false

[ui]
pill = true            # the on-screen pill that shows what Bol is doing
sounds = true          # audible blips when listening starts/stops
position = "top"       # which edge the pill sits on: "top" | "bottom"
text = false           # also spell the state out beside the dots

[hotkey]
key = "alt_r"          # right Option. Hold it, talk, let go, and what you said is
                       # pasted where the cursor is. Nothing is ever sent by itself:
                       # say "send it" to press Enter.

[wake]
# The trigger words, listened for from the moment Bol starts, so there is no
# key to press. Say "hey Bol" and talk; pause two seconds and it is pasted.
# Say "send it" and it is sent. With talk_back = true above, "hey Bol" is the
# conversation flow instead, and there is someone to talk to.
# Here is what leaving this on actually does.
#   Wake mode keeps the microphone open and runs a small keyword model on
#   your Mac. Nothing is recorded or sent anywhere. Expect the occasional
#   false wake from TV or conversation; a false wake costs a Listening pill,
#   and nothing is sent unless you say a send phrase.
# Bol itself needs no internet (your coding agent does). `bol setup` downloads the
# 5 MB keyword model. Set enabled = false to close the microphone and keep
# the hotkey, which works the same either way.
enabled = true
phrases = ["hey bol"]              # also heard as "hey bowl", "hey ball", "babel"
type_phrases = []      # a short trigger word that starts dictation, off by default:
                       # one syllable scores far below "hey bol" in a real room;
                       # ["type"] to turn it on
send_phrases = ["send it", "send", "enter"]  # presses Enter on a pending paste
cancel_phrases = ["scratch that", "close", "scratch", "clear it", "clear that", "clear this", "clear the box"]  # wipes the box
sleep_phrases = ["stop listening"] # pause Bol; press the hotkey to resume
pause_ms = 2000        # a pause this long ends a dictation and pastes it
speak_window_ms = 5000 # how long the pill waits for you to start speaking after a
command_window_s = 10  # after a paste, seconds Bol listens for "send it" or "scratch that", nothing on screen; 0 = keyword ear only
                       # trigger word before it gives up and goes away again
threshold = 0.12       # trigger probability; lower hears more, including the TV
# type_threshold = 0.0 # type_phrases only; 0 = use threshold. Raising it costs the
#                      # real "type ..." before it costs the one inside "prototype",
#                      # so if false dictation bothers you, change type_phrases instead.
awake_s = 0            # after a trigger word or a hold, how long follow-up speech needs
                       # no trigger. 0 = only trigger words and the key ever start
                       # listening (default; room noise cannot wake the pill).
                       # 60 = a free minute.

# Remap any voice command to whatever you like. Unset keys keep defaults.
# [commands]
# send = ["ship it", "send it", "go"]
# discard = ["forget it"]
# interrupt = ["whoa whoa"]

[audio]
# input_device = "MacBook Pro Microphone"  # mic name substring or index; empty = system default
vad = "silero"         # what decides you stopped talking: Silero v6 | "energy" (RMS only)
pre_roll_ms = 300      # audio kept from just before the press, so no clipped first word
warm_s = 120           # hold the mic stream open this long after a recording, then let it go

[stt]
engine = "parakeet"    # or "none" for text-only mode
live = true            # show words in the pill while you talk (display only)
# stream_context = [256, 256] # (left, right) attention frames of 80 ms. A small
#                             # right half makes the live words misspelt, not faster:
#                             # it commits frames off parakeet-mlx's window seam.
# stream_chunk_ms = 640       # audio per streaming decode; each one re-reads the
#                             # buffer, so smaller redraws more often and costs more.

[tts]
engine = "say"         # "say" is instant; "kokoro" is a neural voice (pip install 'bol[kokoro]')

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
mode = "always"        # every dictation | "on_command" (say "clean it up") | "off"
model = "abhiyan10/bol-cleanup-350m-4bit"  # Bol's own 195MB model; "" = rules only

[vocabulary]
# Words Bol should spell your way: your name, your project, a library the
# transcriber has never met. A dictated word within one edit of an entry is
# replaced with the entry, exactly as written. Common English words are left
# alone. Bol always knows the usual tool names (Claude Code, GitHub, uv).
# Bol also learns this session's own words as it goes, from the front window's
# title and from your earlier pastes, and spells a word that sounds like one of
# them ("bowl" in a window titled Bol) that way.
words = []             # e.g. ["Abhiyan", "Poudel", "Parakeet", "Kokoro"]

[summarizer]
engine = "auto"        # llm persona when available, template otherwise
user_name = ""         # your name, spoken in replies

[bridge]
anywhere = true        # dictation lands wherever the cursor is: Notes, Slack, a browser
                       # box, and "send it" presses Enter wherever you are. Bol never
                       # presses Enter by itself. false = terminals and IDEs only.

[server]
port = 8770
bind_window_s = 20     # agent hooks are user-scoped, so every session on this Mac posts
                       # here. Bol narrates the one you last sent to: after it presses
                       # Enter for you, a hook event from another session within this
                       # many seconds moves the narration there. 0 = never move.
# follow = "first"      # narrate the session you last sent to; "all" narrates every session
# allow_remote = false  # true lets the hook server bind off loopback (your network can type into your terminal)

[setup]
# What you told the `bol setup` wizard, so a rerun installs hooks for the same
# coding agents without asking again. "auto" is nobody having answered yet:
# whichever of the two is on your PATH. agents = [] means no coding agent, and
# then Bol is pure dictation and touches no settings file at all.
agents = "auto"        # "auto" | ["claude"] | ["codex"] | ["claude", "codex"] | []
"""


# Keys an older Bol wrote that this one no longer reads. A config file from
# `bol setup` spells every value out, so a file written before these went away
# would otherwise sit there looking like it still decides something.
# (section or None for top level, key, what to do instead). One line each at
# startup; the key itself is ignored, never migrated.
REMOVED_KEYS = (
    ("hotkey", "mode", "[hotkey] mode is gone, hold the key"),
    (
        "hotkey",
        "submit",
        '[hotkey] submit is gone, nothing is ever sent by itself: say "send it"',
    ),
    ("hotkey", "tap_ms", "[hotkey] tap_ms is gone, a tap does nothing now"),
    (
        "hotkey",
        "auto_send_min_words",
        "[hotkey] auto_send_min_words is gone, no dictation submits itself",
    ),
    (
        None,
        "hands_free",
        "hands_free is gone, the awake window ([wake] awake_s) replaces it",
    ),
    ("bridge", "mode", "[bridge] mode is gone, Bol types into the app you are in"),
    ("bridge", "pane", "[bridge] pane is gone with the tmux bridge"),
    (None, "sound_cues", "sound_cues is gone, use [ui] sounds"),
)


def removed_keys(path: Path | None = None) -> list[str]:
    """Hints for keys in the config file that this Bol no longer reads."""
    path = path or CONFIG_PATH
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        # An unreadable config is the loader's problem to report, not a
        # reason for the hints to raise on the way past.
        return []
    hints = []
    for section, key, hint in REMOVED_KEYS:
        table = data if section is None else data.get(section)
        if isinstance(table, dict) and key in table:
            hints.append(f"{hint} ({path})")
    return hints


def write_default_config() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML)
    return CONFIG_PATH


# ------------------------------------------------- writing the wizard's answers
#
# Two ways in, and the difference is whose file it is. A machine with no
# config yet gets the documented template with the answers substituted into
# it, comments and all. A machine that already has one gets its own values
# edited and everything else left alone, which is a tomllib read and a
# tomli_w write, and that loses the comments. The wizard says so when it
# happens rather than quietly flattening a file somebody wrote by hand.

_ASSIGN = re.compile(r"^(\s*)(#\s*)?([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(.*)$")


def _toml_literal(value: object) -> str:
    """The few TOML scalars the wizard writes. Deliberately not a serializer."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    raise ValueError(f"cannot write {value!r} to the config file")


def render_default_config(values: dict | None = None) -> str:
    """DEFAULT_CONFIG_TOML with these values substituted.

    values is {(section or None, key): value}. A key the template ships
    commented out (input_device) is uncommented; the trailing comment on the
    line is kept, because it is the documentation the file exists for.
    """
    remaining = dict(values or {})
    section: str | None = None
    lines: list[str] = []
    for line in DEFAULT_CONFIG_TOML.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
        match = _ASSIGN.match(line)
        if match:
            indent, _commented, key, rest = match.groups()
            target = (section, key)
            if target in remaining:
                value = remaining.pop(target)
                cut = rest.find("  #")
                head = f"{indent}{key} = {_toml_literal(value)}"
                if cut < 0:
                    line = head
                else:
                    # Keep the comment in the column it was written in, so a
                    # substituted file still reads as the documented template.
                    column = len(line) - len(rest) + cut
                    line = head.ljust(max(column, len(head) + 2)) + rest[cut:].lstrip()
        lines.append(line)
    if remaining:
        raise ValueError(f"no such key in the default config: {sorted(remaining)}")
    return "\n".join(lines) + "\n"


def write_config_values(values: dict, path: Path | None = None) -> tuple[Path, bool]:
    """Save the wizard's answers. Returns (path, comments_were_lost)."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(render_default_config(values))
        return path, False
    import tomli_w

    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    for (section, key), value in values.items():
        table = data
        if section is not None:
            if not isinstance(data.get(section), dict):
                data[section] = {}
            table = data[section]
        table[key] = value
    with open(path, "wb") as handle:
        tomli_w.dump(data, handle)
    return path, True


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
