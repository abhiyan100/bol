# Bol architecture

## Design goals

1. **The user keeps their normal Claude Code TUI.** Bol never wraps, replaces,
   or screen-scrapes it: it pastes into the terminal the user is looking at (or a pinned tmux pane).
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

**Async command hooks that post to loopback.** Each hook is
`curl ... --data-binary @- 'http://127.0.0.1:8770/hook?token=...' || true`
with `"async": true`. Bol first shipped `http` hooks (no per-event process),
but Claude Code prints a hook error into the transcript every time an http
hook cannot connect, and Bol is a daemon people start and stop; every
session nagged whenever Bol was off. The async command costs one short-lived
curl per event, never blocks Claude, and is silent when nothing listens. The
`Stop` payload carries `last_assistant_message` directly: no transcript JSONL
parsing on the hot path. `PostToolUse` events are accumulated per `prompt_id`
and flushed into the `StopEvent`, which is how the summarizer knows what
Claude *did* rather than just what it *said*. The hook server answers `{}`
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

**Transcript cleanup is rules first, then a model trained for the job.**
Live testing showed generic 1B-class models silently dropping clauses ("but
don't touch login.py") and parroting few-shot examples at temperature 0. So
the deterministic tier always runs first: fillers, stutters, doubled words,
"auth dot py" to auth.py, "dash dash verbose" to --verbose. On top of that,
local mode uses Bol's own fine-tune (`abhiyan10/bol-cleanup-350m-4bit`,
195 MB, pipeline in `training/`), which was trained specifically to preserve
negations, file names and flags; api mode uses the user's configured model
instead. Both are deadline-bounded with a size-sanity check and the
deterministic text as the unconditional fallback. Cleanup runs AFTER command
parsing, so "send it" can never be cleaned away.

**Focused-mode submit is gated twice.** The app allowlist (bundle ids of
known terminals and IDEs) proves the front app can host a shell, not that
Claude is in the active tab; pasting speech plus Enter into a plain shell
would execute it. So auto-Enter also requires evidence that Claude is
running there. For terminals: the window title contains the whole word
"claude" and is not a `claude-<slug>` project name. For IDEs, whose titles
name the file, not the terminal: a `claude` process must exist in the
frontmost app's process tree. Unreadable title or process list fails closed.
When the gate blocks, Bol still types the text, withholds Enter, and says
why (`SubmitBlocked.reason`). Paste itself re-checks the front app right
before Cmd+V and restores the clipboard in a `finally`, skipping the restore
when the clipboard held non-text content.

**One narrated session at a time.** Hook payloads carry `session_id` and
`cwd`, so the daemon binds to the first session it hears from and ignores
others (with one printed hint), including their permission prompts: a
spoken "go ahead" must never answer a prompt the user is not looking at.
`[server] follow = "all"` opts out. The hook server compares tokens in
constant time and refuses to bind a non-loopback host unless
`allow_remote` is set.

**The mic is ready before the key goes down.** The recorder builds one
`sounddevice` stream and keeps it; the callback always feeds a two-second
ring, so a recording can prepend 300 ms of pre-roll and the first syllable
survives even when the user speaks before pressing. Starting hardware I/O
measured 9 to 24 ms; the chime used to be awaited before recording and cost
300 ms, so cues are now fire-and-forget tasks. The stream stops after a warm
window (default two minutes) rather than running forever: Bluetooth headsets
drop to their low-quality route while any app holds the mic.

**Tap or hold, one key.** Key-down always starts recording; the release
decides the gesture (under 400 ms is a tap, FluidVoice's threshold). A tap
keeps listening until the next tap or trailing silence; a hold ends on
release. In `submit = "auto"` a dictation of three or more words is pasted and
sent in one motion, shorter text is pasted without Enter (Claude Code's own
`/voice` rule), "type ..." never sends, and a trailing "send it" always does.
Hands-free reopen after Bol speaks is opt-in: with auto-send, an unasked mic
could turn room noise into a prompt. Hands-free and tap-ended listening
endpoint with Silero VAD (pysilero-vad, a 2.4 MB wheel with no
dependencies, about 1 ms per 32 ms block): speech starts after two blocks
above 0.5, ends after `silence_ms` below 0.35. The old energy gate remains
as the explicit and automatic fallback; it provably cannot hear speech that
starts before it has measured any room tone, which is why it is no longer
the default. The hotkey listener checks pynput's
`IS_TRUSTED` after start and raises a clear Input Monitoring error instead
of silently never firing.

**Live words are display only.** While a recording runs, the same blocks
that fill the authoritative buffer are also tapped into parakeet-mlx's
streaming decoder (`transcribe_stream`, context (256, 16): the documented
default of (256, 256) holds text back for 20 s, since finalization lags by
`context_size[1] * depth` frames of 80 ms). One call on the MLX thread owns
the whole recording; partials go to the pill at 4 Hz, committed text solid
and draft text dim. They never reach the bridge: the final text is always
the full-buffer decode, then the grammar, then one paste. Measured on a
7 s clip: first partial after about 2.5 s, 2.6 updates per second, peak RSS
0.9 GB, and the streamed text missed the opening two seconds while the full
decode was exact. That blind spot is a known open item.

**The pill is a separate process.** State on screen (listening, finalizing,
thinking with the running tool, permission, speaking, error with its remedy)
is drawn by `bol/hud/app.py`, a PyObjC child fed JSON lines over stdin. Not
in-process AppKit: an app-policy NSApplication becomes the frontmost app,
and the focused bridge would then refuse to paste into "Python". The child
uses the accessory activation policy, a borderless non-activating panel that
ignores the mouse and joins every space, and never asks to become key. A
crash there can never take the microphone down; the daemon respawns it at
most once a minute and continues without it if AppKit is missing. The pill
is set on the keystroke, before any await, so it is the first thing that
happens when the key goes down.

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
| `bol/hud/` | On-screen pill: `Hud` client, AppKit child (`app.py`), pure render table |
| `bol/mlx_thread.py` | The one thread every in-process MLX model runs on |
| `bol/hooks/server.py` | Loopback aiohttp receiver for hook events |
| `bol/hooks/events.py` | Typed payload views + per-turn tool accumulation |
| `bol/hooks/installer.py` | Idempotent settings.json hook install/uninstall |
| `bol/grammar/commands.py` | Voice-command grammar over final transcripts |
| `bol/audio/capture.py` | One persistent mic stream, ring with pre-roll, per-recording sessions and taps |
| `bol/audio/vad.py` | `SpeechGate` protocol: Silero (default) and the energy fallback |
| `bol/stt/` | `Transcriber` protocol; parakeet-mlx implementation |
| `bol/speak/` | `Speaker` protocol; `say` and Kokoro implementations |
| `bol/summarize/` | `Summarizer` protocol; LLM persona + template floor |
| `bol/llm/` | OpenAI-compatible engine; supervised mlx_lm.server or user endpoint |
| `bol/cleanup.py` | Deterministic transcript rules + api-mode LLM grammar pass |
| `bol/config.py` | TOML config, env overrides |
| `bol/cli.py` | `run`, `setup`, `talk`, `launch`, `hook`, `doctor`, `config` |

## Testing

- `tests/`: grammar, hook installer and server, summarizer, turn tracker,
  bridge gate, capture (fake `sounddevice` stream), hotkey, daemon loop with
  fakes. No models, no audio hardware, no network; CI runs them on macOS.
- Live loop verified end-to-end against a real Claude Code session: prompt
  injected by voice-equivalent text path, Stop hook received, summary spoken.
- STT verified by synthesizing speech with `say` and asserting the
  transcription parses to the right action.

## Mobile later

The daemon is already the product; macOS specifics (hotkey, mic, tmux) live
at the edges. A phone client becomes another front end for the same loop:
it ships audio in and receives summaries/audio out over an authenticated
channel, while Claude and the models stay on the Mac.
