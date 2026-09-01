# Bol architecture

## Design goals

1. **The user keeps their normal Claude Code TUI.** Bol never wraps, replaces,
   or screen-scrapes it: it pastes into the same tmux pane the user watches.
2. **Exact completion detection, no heuristics.** Claude Code's own hook
   system reports when a turn ends and what tools ran.
3. **Local-first, zero recurring cost.** STT, TTS, and the default summarizer
   all run on-device. Cloud LLM polish is opt-in.
4. **Every capability is a protocol seam.** STT, TTS, summarizer, and the
   agent bridge are swappable: that's how a mobile companion, other agents
   (Codex, Cursor CLI), and better models land later without a rewrite.

## The loop

```
        ┌──────────────────────────── user ───────────────────────────┐
        │  holds hotkey / speaks after Bol's reply                    │
        ▼                                                             ▲
  Recorder (sounddevice + energy VAD)                       Speaker (say | Kokoro)
        │ float32 utterance                                           ▲ short reply
        ▼                                                             │
  Transcriber (parakeet-mlx)                                Summarizer (LFM2.5 persona | template)
        │ final text                                                  ▲ StopEvent
        ▼                                                             │
  grammar.parse_transcript ──► Bridge (focused|tmux) ─paste+Enter─► Claude ─► HookServer :8770
   SEND / TYPE / DICTATE /       (title-gated)              Code       Stop / PostToolUse /
   DISCARD / INTERRUPT / …                                             Notification (HTTP hooks)
```

### Key decisions and their evidence

**Injection is a swappable bridge.** Default `auto` picks between two
backends at startup: `focused` (paste into the frontmost app via pasteboard
swap + System Events Cmd+V, FluidVoice-style, guarded by a terminal-app
allowlist so speech can't land in a chat app) and `tmux` (focus-independent
injection into a pinned pane, used automatically when a Claude pane exists).
Hooks make completion detection bridge-independent, which is what lets tmux
be optional.

**tmux injection, never `send-keys` for text.** `send-keys` submits early on
embedded newlines and mangles `#`, `!`, `$`. Verified empirically: text goes
in via `load-buffer` + `paste-buffer`, then Enter as a separate `send-keys`
after a short delay (the TUI otherwise swallows it as part of the paste).

**`pane_current_command` lies.** Claude Code retitles its process to its bare
version string (`2.1.252`), so pane discovery confirms via `ps -t <pane_tty>`
looking for a `claude` process on the pane's TTY.

**HTTP hooks, not command hooks.** `{"type": "http", "url": …}` posts every
event to the loopback daemon with no per-event shell spawn. The `Stop` payload
carries `last_assistant_message` directly: no transcript JSONL parsing on the
hot path. `PostToolUse` events are accumulated per `prompt_id` and flushed
into the `StopEvent`, which is how the summarizer knows what Claude *did*
rather than just what it *said*. The hook server always answers `{}`
immediately and processes in the background: Bol observes, never blocks.

**Command words are grammar, not a model.** "send it", "type …", "close" are
matched by normalizing the utterance tail. A second intent model would add
latency and a failure mode for zero benefit at this vocabulary size.

**One local brain, carefully scoped.** LFM2.5-1.2B (4-bit, ~1GB) runs in a
supervised `mlx_lm.server` subprocess (OpenAI-compatible, crash-isolated: an
MLX crash can never take down the microphone) and writes the persona
summaries; measured at ~0.4s per summary warm. `provider = "api"` swaps it
for any OpenAI-compatible endpoint with the user's key over the identical
request path. The next call's system prefix is prewarmed into the KV cache
the moment the hotkey goes down (FluidVoice's trick).

**Transcript cleanup is deterministic in local mode, by evidence.** Live
testing showed 1B-class models silently dropping clauses ("but don't touch
login.py") and parroting few-shot examples at temperature 0. So local
cleanup is regex-tier only: fillers, stutters, doubled words, "auth dot py"
to auth.py, "dash dash verbose" to --verbose. Instant and mechanically
meaning-safe. The LLM grammar pass exists but only runs in api mode with the
user's own big model, deadline-bounded, with a size-sanity check and the
deterministic text as the unconditional fallback. Cleanup runs AFTER command
parsing, so "send it" can never be cleaned away.

**Focused-mode submit is title-gated.** The app allowlist proves the front
app is a terminal, not that Claude is in the active tab; pasting speech plus
Enter into a plain shell would execute it. Auto-Enter therefore also requires
"claude" in the front window title; otherwise Bol types the text, withholds
Enter, and says so.

**Summarizer always has a floor.** The deterministic template over the tool
log + Claude's final message is free, instant, and covers failure flagging.
The LLM persona (local or api) always falls back to it on error or timeout:
the loop never goes silent because a model hiccuped.

**Why not the Agent SDK?** It would kill the interactive TUI and force
API-key metered billing instead of the user's subscription. Bol drives the
session the user already has.

**Why not fork FluidVoice?** GPLv3 + ~30 upstream commits/week = permanent
merge debt, and its strength (dictation UX) is not Bol's differentiator (the
talk-back loop is). Bol credits it and mirrors its best ideas: the modular
STT provider protocol especially.

## Module map

| Module | Responsibility |
|---|---|
| `bol/daemon.py` | State machine: wires hotkey → record → transcribe → parse → act; hooks → summarize → speak → auto-listen |
| `bol/bridge/base.py` | Bridge protocol + auto-selection |
| `bol/bridge/focused.py` | Frontmost-app paste injection with terminal allowlist |
| `bol/bridge/tmux.py` | Pane discovery/pinning/verification, paste injection, key sends |
| `bol/hooks/server.py` | Loopback aiohttp receiver for hook events |
| `bol/hooks/events.py` | Typed payload views + per-turn tool accumulation |
| `bol/hooks/installer.py` | Idempotent settings.json hook install/uninstall |
| `bol/grammar/commands.py` | Voice-command grammar over final transcripts |
| `bol/audio/capture.py` | Mic capture; push-to-talk stop or energy-gate endpointing |
| `bol/stt/` | `Transcriber` protocol; parakeet-mlx implementation |
| `bol/speak/` | `Speaker` protocol; `say` and Kokoro implementations |
| `bol/summarize/` | `Summarizer` protocol; LLM persona + template floor |
| `bol/llm/` | OpenAI-compatible engine; supervised mlx_lm.server or user endpoint |
| `bol/cleanup.py` | Deterministic transcript rules + api-mode LLM grammar pass |
| `bol/config.py` | TOML config, env overrides |
| `bol/cli.py` | `run`, `talk`, `launch`, `hook`, `doctor`, `config` |

## Testing

- `tests/`: grammar, hook installer, summarizer, turn tracker (pure unit).
- Live loop verified end-to-end against a real Claude Code session: prompt
  injected by voice-equivalent text path, Stop hook received, summary spoken.
- STT verified by synthesizing speech with `say` and asserting the
  transcription parses to the right action.

## Mobile later

The daemon is already the product; macOS specifics (hotkey, mic, tmux) live
at the edges. A phone client becomes another front end for the same loop:
it ships audio in and receives summaries/audio out over an authenticated
channel, while Claude and the models stay on the Mac.
