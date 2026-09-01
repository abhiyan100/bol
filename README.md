# Bol

**Speak. Bol runs Claude Code hands-free.** · [bol-voice.vercel.app](https://bol-voice.vercel.app)

*Bol* (बोल) means "speak" in Nepali. Hold a key, talk to Claude Code, and,
unlike every dictation tool, **Bol talks back**: when Claude finishes a turn,
a summarizer tells you out loud what happened, flags anything that failed,
and asks what you want to do next. Fully local by default, zero recurring cost.

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
  [mlx-audio](https://github.com/Blaizzy/mlx-audio)), and the default
  summarizer (a deterministic template over the tool log, free and instant).
  Optionally plug an OpenRouter model in for a wittier persona voice.

## Install

Requires macOS on Apple Silicon, [Claude Code](https://code.claude.com), and
[uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo-url> bol && cd bol
uv sync --extra stt          # add --extra kokoro for the neural voice
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

[summarizer]
engine = "openrouter"        # persona voice for summaries
openrouter_model = "nvidia/nemotron-3.5-lightning:free"
user_name = "Abhiyan"        # spoken in replies

[hotkey]
mode = "toggle"              # tap instead of hold
```

The persona summarizer needs `OPENROUTER_API_KEY`. Free-tier models are
rate-limited (~50 requests/day); Bol silently falls back to the free template
summarizer whenever the LLM is unavailable, so the loop never breaks.

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
