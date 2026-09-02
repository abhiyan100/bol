# Bol

**Hands-free dictation for Claude Code, Codex, and anywhere you type.**

[![ci](https://github.com/abhiyan100/bol/actions/workflows/ci.yml/badge.svg)](https://github.com/abhiyan100/bol/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![platform](https://img.shields.io/badge/macOS-Apple%20Silicon-black.svg)](#install)
[bol-voice.vercel.app](https://bol-voice.vercel.app)

Say "type" and talk. Pause, and the words land where your cursor is. Say
"send it" and Enter is pressed. Bol listens from the moment it starts, runs
every model on your Mac, and sends nothing anywhere. Turn on talk-back and
it also tells you out loud what your coding agent did when the turn ends.

```
you    "type, add a login test for the parser and run pytest"   (pause)
bol    pastes it into Claude Code, no Enter
you    "send it"
claude ...edits files, runs tests, finishes...
bol    (with --talk-back) "Login test's in and pytest passes. One warning in
        parser.py. Want Claude to commit it?"
```

## Install

```bash
uv tool install "bol[stt,llm] @ git+https://github.com/abhiyan100/bol"
bol setup     # permissions, models, hooks, checks
bol run
```

Requires macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/).
Setup downloads about 2.5 GB of models the first time (speech 2.3 GB,
cleanup 0.2 GB, keyword spotting 5 MB). Talk-back adds a 0.6 GB
summarizer; `,kokoro` in the extras adds a neural voice.

## Use

Two ways to start talking. Both end the same way: text where the cursor is,
nothing sent until you say so.

| Do this | Bol does |
|---|---|
| Say **"type"**, talk, pause 3 s | pastes what you said, no Enter |
| **Hold right Option**, talk, release | same, for noisy rooms |
| Say **"send it"** (or "send", "enter") | presses Enter, wherever you are |
| Say **"scratch that"** | clears what Bol pasted |
| Say **"stop listening"** | pauses the ear; press the key to resume |
| Say **"hey Bol ..."** | the conversation flow, with talk-back on |
| Say **"go ahead"** / **"no"** | answers a permission prompt |

Works in Claude Code, Codex, Cursor, Terminal, Notes, Slack, a browser
field: anywhere you can type. After any trigger, Bol stays awake for a
minute, so follow-ups need no trigger word (`awake_s = 0` makes trigger
words mandatory). A small pill at the top of the screen shows what Bol is
hearing and doing: a level meter while you talk, a bounce while the agent
works, blue when a paste is waiting for "send it", red with the fix when
something is wrong. It never takes focus.

Every phrase is yours to change under `[wake]` and `[commands]`.

**Talk-back.** `bol run --talk-back` (or `talk_back = true`) adds the
other direction: when Claude Code finishes a turn, Bol speaks a short
summary of what it did, flags failures, and reads permission prompts to
you. A local 1B model writes the summary; the voice is macOS `say` or
Kokoro.

Talk-back works with Claude Code and Codex CLI. `bol setup` wires whichever
of the two is installed, and Bol names the one it is narrating, so a Codex
turn is reported as Codex. Codex asks you once to review and trust the new
hooks; accept it when it appears.

## How it works

```
 "type" / hold key            your editor, terminal or agent (unchanged)
        |                              ^                  |
        v                              | paste, Enter     | Stop / PostToolUse /
 keyword spotter (5 MB) ---+           |                  | Notification hooks
 mic -> Silero VAD -> Parakeet STT     |                  v
        |                        +------------- bol daemon --------------+
        v                        |  focused-app bridge (paste + keys)     |
 cleanup + vocabulary -----------|  hook server on 127.0.0.1:8770         |
 "send it" / "scratch that"      |  pill (separate process)              |
                                 |  talk-back: summarizer + TTS (opt-in) |
                                 +---------------------------------------+
```

- **Nothing leaves your Mac.** Keyword spotting
  ([sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)), voice activity
  ([Silero](https://github.com/snakers4/silero-vad)), speech recognition
  ([parakeet-mlx](https://github.com/senstella/parakeet-mlx), NVIDIA Parakeet
  TDT v3), cleanup (Bol's own
  [bol-cleanup-350m](https://huggingface.co/abhiyan10/bol-cleanup-350m-4bit)),
  and, with talk-back, the summarizer
  ([LFM2.5-1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) via
  [mlx-lm](https://github.com/ml-explore/mlx-lm)) and the voice
  ([Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) via
  [mlx-audio](https://github.com/Blaizzy/mlx-audio)). No account, no key.
- **The mic is ready before you are.** One warm audio stream with 300 ms of
  pre-roll, so the first word is never clipped. Live words show in the pill
  as you speak; only the final full-accuracy decode is pasted.
- **Enter is never a guess.** Bol pastes; you say "send it". That is the
  whole safety model, and it is why Bol can type into any app.
- **Turn end is exact, not screen-scraped.** Claude Code's own
  [hooks](https://code.claude.com/docs/en/hooks) tell Bol when a turn ends
  and every tool it ran, over loopback, as background commands that stay
  silent when Bol is off.
- **Bring your own summarizer if you want.** One config block points
  talk-back at any OpenAI-compatible endpoint. Your key, your model.

## Configure

`bol config` writes `~/.config/bol/config.toml` with every option
documented. Highlights:

```toml
talk_back = false            # true: hear what the agent did after each turn

[wake]
enabled = true               # listen for trigger words from the start
type_phrases = ["type"]      # change to "start typing" if "type" fires too often
send_phrases = ["send it", "send", "enter"]
pause_ms = 3000              # a pause this long ends a "type" dictation
awake_s = 60                 # follow-ups need no trigger word for this long

[hotkey]
key = "alt_r"                # right Option. On AltGr layouts use "cmd_r" or "f13"

[ui]
pill = true
text = false                 # true: words and state spelled out beside the dots

[bridge]
anywhere = true              # false: terminals and IDEs only

[vocabulary]
words = ["Abhiyan", "Kokoro"]   # names Bol should spell your way

[cleanup]
mode = "on_command"          # "always" strips fillers from every dictation

[llm]                        # talk-back only
provider = "local"           # or "api" with base_url, api_model, api_key_env
```

Cleanup is deliberately conservative. Deterministic rules run first
(fillers, stutters, "auth dot py" to `auth.py`, "dash dash verbose" to
`--verbose`), then a spelling pass for tool names and your `[vocabulary]`
words, then Bol's own 350M model, trained for exactly this job because
generic small models silently drop clauses like "don't touch login.py".
Every step falls back to the text before it.

## Troubleshooting

`bol doctor` checks all of this and prints the fix next to anything red.

| Symptom | Cause | Fix |
|---|---|---|
| Nothing happens when you speak | Keyword model not downloaded, or ear paused | `bol setup`; press the key to resume |
| Hotkey does nothing | Input Monitoring not granted | System Settings > Privacy & Security > Input Monitoring > your terminal, restart Bol |
| "Mic lost" in the pill | Device gone or wrong `input_device` | `bol doctor` lists devices |
| "type" fires inside other words | Keyword spotting hears the sound | `type_phrases = ["start typing"]` |
| Text pasted, Enter never pressed | By design | say "send it" |
| Summaries sound robotic | Talk-back model still downloading | `bol setup` shows progress; `~/.config/bol/llm.log` |
| Right Option types symbols | AltGr layout | `[hotkey] key = "cmd_r"` or `"f13"` |

## Security and privacy

- The keyword model, VAD, speech recognition, cleanup, and (with talk-back)
  the summarizer and voice all run on your Mac. Turn wifi off and try it.
- The hook server binds `127.0.0.1` only, with a random 128-bit token in
  `~/.config/bol/hook_token` (mode 0600). Non-loopback hosts are refused
  unless you opt in.
- `bol setup` adds three hooks to `~/.claude/settings.json` and, when Codex
  CLI is installed, three more to `~/.codex/hooks.json`, each a background
  `curl` to loopback that never fails, so the agent never waits on Bol and
  stays quiet when Bol is off. Setup shows the entry first, writes
  atomically, keeps a one-time backup. `bol hook uninstall` removes them.
- Bol pastes; it never presses Enter on its own. Your clipboard is restored
  after every paste. `[bridge] anywhere = false` limits pastes to terminals
  and IDEs.
- `provider = "api"` (talk-back only) sends the tool log and the agent's
  last message to the endpoint you configure, with your key. Nothing else
  ever leaves the machine. No telemetry.

## Permissions (macOS)

- **Microphone**: prompted on first run, granted to your terminal app. The
  orange indicator stays on while Bol runs; `[wake] enabled = false` closes
  the mic between key presses.
- **Input Monitoring**: the hotkey and the click-to-cancel need it.
- **Automation**: macOS asks once when Bol first pastes via System Events.

## Contributing

```bash
git clone https://github.com/abhiyan100/bol && cd bol
uv sync --extra all
uv run pytest -q
```

Tests run without models or audio hardware. PRs welcome.

## Credits

- [FluidVoice](https://github.com/altic-dev/FluidVoice): its modular STT
  provider design inspired Bol's.

MIT licensed.

*Bol* (बोल) means "speak" in Nepali.
