# Strata TUI

`strata-tui` is the operator-facing Ratatui application for Strata. It is a
separate Rust frontend around the C++ `strata-chat` runtime, so the interface can
be expressive without moving inference, tokenizer, routing, precision, or
admission logic out of the dependency-light runtime.

## Build

Ratatui 0.30.2 requires Rust 1.88 or newer.

```bash
make build
make tui
./target/release/strata-tui
```

`make tui-check` runs formatting, Clippy with warnings denied, and the Rust unit
and render tests. `make check-all` runs both the existing C++ gate and the TUI
gate. The normal CMake build remains C/C++-only; Rust is not a runtime
dependency.

## Launch

Running without arguments opens the launch form. If a known checkpoint exists
under `models/`, the form discovers it. It also discovers `build/strata-chat`, a
`strata-chat` next to the TUI executable, or `STRATA_CHAT_BIN`.

```bash
./target/release/strata-tui
```

A complete command bypasses the form:

```bash
./target/release/strata-tui \
  --model models/glm52 --model-type glm \
  --devices 0,1,2 --context-size 2048 --max-new 512 \
  --temperature 0 --vram-fraction 0.85
```

Add `--setup` to prefill the form instead of launching. Add
`--flash-attention` only when intentionally selecting the CUDA FlashAttention
candidate. The launch form labels temperature zero as exact greedy and any
nonzero temperature as seeded sampling.

## Interface

The session view is responsive. At 96 columns and wider it shows the
conversation beside a telemetry rail; smaller terminals keep the conversation
and surface the phase in the header. The chat view needs at least 48 by 15; the
one-time launch form needs 64 by 22. Smaller terminals receive a clear resize
prompt rather than a broken layout.

The telemetry rail reports:

- the exact greedy or seeded-sampling contract;
- runtime adapter, CUDA devices, context ceiling, and session uptime;
- separate load, prefill, and decode time;
- prompt/decode tokens and throughput;
- a live decode-speed sparkline and logical context usage.

The diagnostics overlay retains the latest 400 stderr records. Conversation
display is capped at 128 messages and 4 MiB; prompt history is capped at 64
entries. A single editor input is capped at 1 MiB and the C++ protocol accepts
records up to 16 MiB. These bounds keep frontend memory independent of model and
context size. The TUI never stores or duplicates model weights.

## Keyboard and mouse

| Key | Action |
|---|---|
| `Enter` | Send a prompt; advance the launch form |
| `Shift+Enter` | Insert a newline |
| `Up` / `Down` | Navigate single-line prompt history |
| `PageUp` / `PageDown` | Scroll the conversation |
| mouse wheel | Scroll the conversation |
| `Ctrl+W` | Delete the previous word |
| `Ctrl+U` | Clear the editor |
| `Ctrl+L` | Open runtime diagnostics |
| `F1` | Open the keyboard map |
| `Ctrl+Q` | Quit; requires a second press while the runtime is busy |

Bracketed paste and Unicode editing are supported. On shutdown, the TUI closes
the child it launched and restores raw mode, cursor visibility, mouse capture,
and the alternate screen, including through the panic hook.

## Protocol

`strata-tui` starts `strata-chat --protocol jsonl`. Every stdout record has the
following versioned envelope:

```json
{"protocol":"strata-chat","version":1,"event":"ready","load_seconds":22.1}
```

Prompts travel in the opposite direction as one JSON record, including
multiline and Unicode input:

```json
{"command":"prompt","text":"Explain sparse attention.\nUse two examples."}
```

The runtime emits `hello`, `status`, `ready`, `turn_start`, `token`,
`turn_done`, and `error`. Token records are flushed as soon as complete UTF-8 is
available. `turn_done` carries separate prompt/decode token counts and timing.
Unknown input fields are ignored for forward compatibility; malformed,
duplicate, empty, oversized, or unsupported requests produce explicit errors.
Protocol version mismatch is fatal in the frontend.

All runtime stderr remains diagnostics. No load output is allowed to contaminate
the JSON stdout stream. A nonzero runtime exit, invalid record, or explicit
fatal event is rendered as an error; there is no frontend fallback.
