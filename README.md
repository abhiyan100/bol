# Bol

**Talk to Claude Code. It talks back.**

[![ci](https://github.com/abhiyan100/bol/actions/workflows/ci.yml/badge.svg)](https://github.com/abhiyan100/bol/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![platform](https://img.shields.io/badge/macOS-Apple%20Silicon-black.svg)](#install)
[bol-voice.vercel.app](https://bol-voice.vercel.app)

Tap a key, say what you want, and when Claude finishes the turn Bol tells
you out loud what happened, flags what failed, and asks what Claude should do
next. Speech recognition, the summarizer, and the voice all run on your Mac.
No account, no key, nothing leaves the machine.

```
you   (tap right Option)   "refactor the auth module and run the tests"
claude                      ...edits files, runs tests, finishes...
bol   (out loud)            "Auth module's refactored and tests pass. One
                             deprecation warning in login.py though. Want
                             Claude to commit it?"
you   (tap)                 "yes, commit it"
```

## Install

```bash
uv tool install "bol[stt,llm] @ git+https://github.com/abhiyan100/bol"
bol setup     # permissions, model downloads with progress, hook install, self-test
bol run
```

Requires macOS on Apple Silicon, [Claude Code](https://code.claude.com), and
[uv](https://docs.astral.sh/uv/). Setup downloads about 3 GB of models
(speech 2.3 GB, summarizer 0.6 GB, cleanup 0.2 GB), cached after the first
time. Add `,kokoro` to the extras for a neural voice (0.3 GB more).

## Why Bol

Dictation into a coding agent exists. Reading the screen to find out what the
agent did is still on you. Bol closes that loop.

|  | Bol | Claude Code `/voice` | Dictation apps |
|---|---|---|---|
| Speaks into Claude Code | yes | yes | yes |
| Reads back what Claude did | **yes** | no | no |
| Answers permission prompts by voice | **yes** | no | no |
| Audio stays on your Mac | yes | no, [streamed to Anthropic](https://code.claude.com/docs/en/voice-dictation#requirements) | some |
| Works with API keys and Bedrock | yes | no, Claude.ai login only | yes |
| Open source | MIT | no | mostly no |

Dictation apps: Superwhisper, Wispr Flow, Infina, FluidVoice and friends.
Good tools; they stop at the input box.

## How it works

```
 hotkey / auto-listen           Claude Code (your normal TUI, in your terminal)
        |                              ^                |
        v                              | paste + Enter  | Stop / PostToolUse /
 mic -> Parakeet STT (local)           |                | Notification hooks
        |                              |                v
        v                        +------------- bol daemon -------------+
 command grammar ----------------|  bridge (focused terminal / tmux)     |
 "send it" / "type ..." / "close"|  hook server on 127.0.0.1:8770         |
                                 |  summarizer (local LLM, template)     |
                                 |  TTS (macOS say, or local Kokoro)     |
                                 +---------------------------------------+
```

- **You keep your normal Claude Code TUI.** Bol pastes into the terminal
  window you are looking at: Terminal, iTerm2, Ghostty, Warp, Kitty,
  Alacritty, WezTerm, and the integrated terminals in VS Code, Cursor, Zed
  and JetBrains IDEs. Text lands the moment you stop speaking. Bol refuses to
  type into anything that is not a terminal, and only presses Enter when the
  front window is running Claude.
- **Completion detection is exact**, not screen scraping. Claude Code's own
  [hooks](https://code.claude.com/docs/en/hooks) post to Bol over loopback
  when a turn ends, with the final message and every tool call made.
- **Everything runs locally**: STT with
  [parakeet-mlx](https://github.com/senstella/parakeet-mlx), TTS with macOS
  `say` or [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) via
  [mlx-audio](https://github.com/Blaizzy/mlx-audio), and a small local brain
  ([LFM2.5-1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) via
  [mlx-lm](https://github.com/ml-explore/mlx-lm)) that writes the spoken
  summaries. A deterministic template summarizer is the always-on fallback.
- **Bring your own inference if you want.** One config block points the brain
  at any OpenAI-compatible endpoint: OpenRouter, Groq, Ollama, LM Studio,
  OpenAI, Anthropic. Your key, your model, your call.

## Use

```bash
bol run
```

Open Claude Code in any terminal. **Tap right Option** and talk; tap again
or just stop talking, and the text lands in the window you are looking at
and is sent. **Hold** it instead for push-to-talk: release to send. The mic
is live on the keystroke, with 300 ms of pre-roll, so the first word is never
lost. Anything under three words is pasted without Enter, so a stray "yes"
never reaches Claude unread. A small pill at the top of the screen shows
what Bol is doing: Listening, Finalizing, Sent, Thinking with the tool Claude
is running, the permission question, what Bol is saying, and any error with
its fix. It never takes focus. Voice commands ride on your speech:

| You say | Bol does |
|---|---|
| "...fix the login bug" (3+ words) | pastes the prompt, presses Enter |
| "...fix the login bug **send it**" | same, in any mode and at any length |
| "**type** hello world" | inserts text, no Enter |
| "**send**" / "**go ahead**" | presses Enter (also answers permission prompts) |
| "**close**" / "**scratch that**" | clears the input box |
| "**interrupt**" / "**stop claude**" | Escape, stops the running turn |
| "...**clean it up and send it**" | strips fillers and stutters, fixes "auth dot py" to `auth.py`, then submits |
| "**say that again**" | re-speaks the last reply |
| "**stop listening**" | sleep until the next hotkey press |

Every phrase is yours to remap:

```toml
[commands]
send = ["ship it", "go"]
interrupt = ["whoa whoa"]
```

When Claude finishes, Bol speaks a summary. Tap to answer. Set
`hands_free = true` and Bol reopens the mic by itself after speaking, so the
whole session is a conversation with no keys at all. When Claude asks for
permission, Bol reads the request and waits for "go ahead" or "no".

Running several Claude Code sessions? Bol narrates the first one it hears
from and tells you when it ignores another. Set `[server] follow = "all"` to
hear every session.

No mic handy? `bol talk` gives the identical loop over typed text. Prefer
focus-independent injection? Run Claude inside tmux and Bol auto-switches to
pinned-pane mode: `bol launch ~/code/myproject`, then `tmux attach -t bol`.

## Configure

`bol config` writes `~/.config/bol/config.toml` with every option documented.
Highlights:

```toml
[hotkey]
key = "alt_r"                # right Option. On layouts where that is AltGr,
                             # use "cmd_r" or "f13".
mode = "auto"                # tap to talk, hold for push-to-talk;
                             # or "push_to_talk", "toggle"
submit = "auto"              # send when you stop; "voice" = only on "send it"
auto_send_min_words = 3      # shorter text is pasted, not sent

[audio]
input_device = ""            # name substring or index; empty = system default
pre_roll_ms = 300            # audio kept from before the keystroke
warm_s = 120                 # mic stays open this long after use (orange dot)

[ui]
pill = true                  # on-screen state pill; false = sound cues only
sounds = true
position = "top"             # or "bottom"

[tts]
engine = "kokoro"            # neural voice instead of `say`

[llm]
provider = "local"           # default: LFM2.5-1.2B on your Mac, no key
# local_model = "LiquidAI/LFM2.5-350M-MLX-4bit"   # low-RAM Macs

# or bring your own key, any OpenAI-compatible endpoint:
# provider = "api"
# base_url = "https://openrouter.ai/api/v1"
# api_model = "google/gemini-2.5-flash-lite"
# api_key_env = "BOL_API_KEY"

[cleanup]
mode = "on_command"          # "always" cleans every dictation; "off" never

[summarizer]
user_name = "Abhiyan"        # spoken in replies
```

Every AI layer degrades gracefully: if a model is still downloading, busy, or
broken, summaries come from the deterministic template and dictation goes in
raw. The loop never breaks because a model hiccuped.

Cleanup is deliberately conservative. Deterministic rules (fillers, stutters,
spoken tokens like "dash dash verbose") run first. Then
[bol-cleanup-350m](https://huggingface.co/abhiyan10/bol-cleanup-350m-4bit),
**Bol's own model**, polishes grammar in 40 to 100 ms: a 195 MB fine-tune
trained for exactly this job, because generic 1B-class local models silently
drop clauses like "don't touch login.py" (94% exact match, 100% negation and
file/flag preservation on held-out eval; full training pipeline in
[`training/`](training/)). Any failure falls back to the deterministic text.
In `api` mode your own model does the polish instead.

## Troubleshooting

`bol doctor` checks all of this and prints the fix next to anything red.

| Symptom | Cause | Fix |
|---|---|---|
| Hotkey does nothing | Input Monitoring not granted | System Settings > Privacy & Security > Input Monitoring > enable your terminal, restart Bol |
| "Lost the microphone" or silence | Mic permission or wrong input device | Grant Microphone to your terminal; set `[audio] input_device` |
| Text appears but never sends | Front window is not running Claude | Focus the Claude Code tab; check `bol doctor` for the frontmost app |
| "front app ... isn't a terminal" | Your terminal is not on the allowlist | `[bridge] allowed_apps = ["your.bundle.id"]`, `bol doctor` prints the id |
| Summaries sound robotic | Local model still downloading or failed | `bol setup` shows progress; `~/.config/bol/llm.log` has details |
| Right Option types symbols | AltGr layout | `[hotkey] key = "cmd_r"` or `"f13"` |

## Security and privacy

- The hook server binds `127.0.0.1` only, and every request carries a random
  128-bit token stored in `~/.config/bol/hook_token` (mode 0600). Non-loopback
  hosts are refused unless you opt in.
- `bol setup` adds three hooks (Stop, PostToolUse, Notification) to
  `~/.claude/settings.json`: each one a background `curl` to loopback that
  never fails, so Claude Code never waits on Bol and stays quiet when Bol is
  not running. Setup shows you the entry first, writes atomically, and keeps
  a one-time backup next to it. `bol hook uninstall` removes them.
- Injection only targets terminal apps, re-checks the front app right before
  pasting, and restores your clipboard afterwards.
- In the default configuration nothing leaves your Mac. `provider = "api"`
  sends the summary prompt (tool log and Claude's last message) to the
  endpoint you configure, with your key.
- No telemetry.

## Permissions (macOS)

- **Microphone**: prompted on first recording, granted to your terminal app.
- **Input Monitoring**: the global hotkey needs it. System Settings >
  Privacy & Security > Input Monitoring > enable your terminal.
- **Automation**: macOS asks once when Bol first pastes via System Events.
  Not needed in tmux mode.

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
