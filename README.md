# Bol

**Speak. Bol runs Claude Code hands-free.** · [bol-voice.vercel.app](https://bol-voice.vercel.app)

Hold a key, talk to Claude Code, and, unlike every dictation tool, **Bol
talks back**: when Claude finishes a turn, a summarizer tells you out loud
what happened, flags anything that failed, and asks what you want to do
next. Fully local by default, zero recurring cost.

```
you   (hold right-Option)  "refactor the auth module and run the tests… send it"
claude                      …edits files, runs tests, finishes…
bol   (out loud)            "Auth module's refactored and tests pass. One
                             deprecation warning in login.py though. Ship it?"
you                         "yes, commit it, send"
```

## Why

Voice input tools for coding agents exist (infina.so, Superwhisper, Claude
Code's own dictation). None of them close the loop: you still have to *read*
the screen to know what your agent did. Bol closes it. Dictation in, spoken
summary out, conversation continues. See [Anthropic's issue tracker](https://github.com/anthropics/claude-code/issues/50720)
for how much people want this.

## How it works

```
 hotkey / auto-listen           Claude Code (your normal TUI, any terminal)
        │                              ▲                │
        ▼                              │ paste + Enter  │ Stop / PostToolUse /
 mic → Parakeet STT (local)            │                │ Notification hooks
        │                              │                ▼
        ▼                        ┌───────────── bol daemon ─────────────┐
 command grammar ────────────────│  bridge (focused app / tmux)         │
 "send it" / "type …" / "close"  │  hook server :8770                   │
                                 │  summarizer (free template or LLM)   │
                                 │  TTS (macOS say, or local Kokoro)    │
                                 └──────────────────────────────────────┘
```

- **You keep your normal Claude Code TUI.** Bol pastes into the terminal
  you're looking at (any terminal app, no tmux needed); dictated text appears
  in the input box as you speak. A guard makes sure speech only ever lands in
  a terminal, never in your chat apps.
- **Completion detection is exact**, not screen-scraping: Claude Code's own
  [hooks](https://code.claude.com/docs/en/hooks) post to Bol over loopback
  when a turn ends, including the final message and every tool call made.
- **Everything runs locally**: STT ([parakeet-mlx](https://github.com/senstella/parakeet-mlx)),
  TTS (macOS `say`, or [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) via
  [mlx-audio](https://github.com/Blaizzy/mlx-audio)), and a small local brain
  ([LFM2.5-1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) via
  [mlx-lm](https://github.com/ml-explore/mlx-lm)) that writes the spoken
  summaries and cleans up your dictation on request. No account, no key.
  A deterministic template summarizer is the always-on fallback.
- **Bring your own inference if you want.** One config block points the brain
  at any OpenAI-compatible endpoint instead: OpenRouter, Groq, Ollama,
  LM Studio, OpenAI, Anthropic. Your key, your model, your call.

## Install

Requires macOS on Apple Silicon, [Claude Code](https://code.claude.com), and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/abhiyan100/bol && cd bol
uv sync --extra all          # or pick: --extra stt --extra llm --extra kokoro
uv run bol doctor            # checks everything, tells you what's missing
```

## Use

```bash
uv run bol run
```

That's it. Open Claude Code in any terminal (Terminal, iTerm, Ghostty, Warp,
VS Code), hold **right Option**, and speak. Bol pastes into the window you're
looking at.

Want focus-independent injection instead (talk while another app is focused)?
Run claude inside tmux and Bol auto-switches to pinned-pane mode:
`uv run bol launch ~/code/myproject`, then `tmux attach -t bol`.

Voice commands ride on your speech:

| You say | Bol does |
|---|---|
| "…fix the login bug **send it**" | pastes prompt, presses Enter |
| "**type** hello world" | inserts text, no Enter |
| "**send**" / "**go ahead**" | presses Enter (also answers permission prompts) |
| "**close**" / "**scratch that**" | clears the input box |
| "**interrupt**" / "**stop claude**" | Escape, stops the running turn |
| "…**clean it up and send it**" | strips fillers and stutters, fixes "auth dot py" to `auth.py`, then submits |
| "**say that again**" | re-speaks the last reply |
| "**stop listening**" | sleep until the next hotkey press |

When Claude finishes, Bol speaks a summary and reopens the mic (hands-free
mode, on by default), so the whole session is a conversation.

No mic handy? `uv run bol talk` gives the identical loop over typed text.

## Configure

`bol config` writes `~/.config/bol/config.toml` with every option documented.
Highlights:

```toml
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

[hotkey]
mode = "toggle"              # tap instead of hold
```

Every AI layer degrades gracefully: if the model is still downloading, busy,
or broken, summaries come from the free deterministic template and dictation
goes in raw. The loop never breaks because a model hiccuped.

Cleanup is deliberately conservative. Deterministic rules (fillers,
stutters, spoken tokens) run first and mechanically cannot change meaning.
Then [bol-cleanup-350m](https://huggingface.co/abhiyan10/bol-cleanup-350m-4bit),
**Bol's own model**, polishes grammar in 40-100ms: a 195MB fine-tune trained
for exactly this job because our testing showed generic 1B-class local
models silently drop clauses like "don't touch login.py" (94% exact match,
100% negation and file/flag preservation on held-out eval; full training
pipeline in [`training/`](training/)). Any failure falls back to the
deterministic text. In `api` mode your own big model does the polish
instead.

## Permissions (macOS)

- **Microphone**: prompted on first recording, granted to your terminal app.
- **Input Monitoring**: the global hotkey needs it. System Settings →
  Privacy & Security → Input Monitoring → enable your terminal.
- **Automation** (focused mode): macOS asks once when Bol first pastes via
  System Events. Not needed in tmux mode.

## Status & roadmap

Early but working end-to-end. Roadmap:

- [ ] Menu-bar app (signed .app; stable mic permission, modifier-only hotkeys)
- [ ] Streaming TTS (speak the first sentence while the rest generates)
- [ ] Silero VAD (replace the energy gate)
- [ ] Apple SpeechAnalyzer backend on macOS 26+
- [ ] Mobile companion (the daemon already speaks a clean protocol)

## Credits

- [FluidVoice](https://github.com/altic-dev/FluidVoice): its modular STT
  provider design inspired Bol's.

MIT licensed. PRs welcome.

*Bol* (बोल) means "speak" in Nepali.
