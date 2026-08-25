# Experiment 0192: DeepSeek fast prefill page reaches `strata-chat`

Date: 2026-08-25  
Branch: `fix/dsv4-chat-fast-page`

## Decision

Accept `--prefill-page-tokens` in `strata-chat` and use 8,192 in the DeepSeek
chat runbook. This is production-path wiring for the already accepted
layer-major page, not a new engine throughput result.

Before this change, the server and runner could select the measured 8,192-row
upper bound but chat was locked to the runtime's conservative 64-row default.
That made the documented interactive path incapable of reproducing the prompt
shape used by every current DeepSeek prefill benchmark.

## Predeclared contract

- Hypothesis: chat's missing flag prevents users from selecting the accepted
  page shape.
- Primary gate: parse, report, and forward the exact positive integer to
  `RuntimeConfig::deepseek_prefill_page_tokens`.
- Correctness: CLI failure on zero/malformed values, placement dry-run at
  8,192, and `make check`; no numerical runtime code changes.
- Memory: unchanged 0.95 admission. The DeepSeek placement model already owns
  page workspace sizing and must fail closed if a requested page cannot fit.
- Rollback: any parser/config mismatch, admission failure at the documented
  16K operating point, or test failure.

## Changes and gates

`strata-chat` now:

- lists `--prefill-page-tokens N` in help;
- rejects zero and malformed values through the common positive-u32 parser;
- reports an explicit page in the human banner and JSONL hello event; and
- forwards the value without reinterpretation to the model runtime config.

The Release dry-run command from `docs/models/deepseek.md` completed with exit
zero, selected physical devices 1 and 2, and reported `fits with a host tier`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build-release/strata-chat \
  --model models/dsv4f --model-type deepseek \
  --devices 1,2 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048 \
  --decode-topology rank-local-tp2 \
  --prefill-page-tokens 8192 --dry-run --no-plan-cache
```

The flag is an upper bound: a shorter prompt allocates and executes only its
actual rows. This makes it safe to retain in one copy-paste command across
short and long conversations.

## Scope

No checkpoint, kernel, route, precision, KV layout, cache policy, or default
was changed. The remaining Strata/vLLM prefill gap measured in experiment 0190
is not attributed to or claimed closed by this CLI fix.
