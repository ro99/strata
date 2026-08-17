# Gemma 4 layer-hash-trace fixture

`layer-hash-trace.json` is the tier-1 (per-layer, per-operation BF16 hash) and
tier-2 (generated token IDs) equivalence-oracle fixture for Gemma 4, captured
at commit `5783d39` (branch `infra/00-equivalence-oracle`).

## How it was captured

```
./build/strata-gemma4-run \
  --model /home/rodrigo/Developer/strata/models/gemma4 \
  --prompt "The capital of France is" --max-new 8 --temperature 0 \
  --layer-hash-trace --json > layer-hash-trace.json
```

Greedy (`--temperature 0`), fixed seed (default `33377335`, unused at
temperature 0), fixed 18-token prompt. Devices default to `0,1,2`; the model
is fully VRAM-resident so device topology does not change the result.

## Determinism

Run twice, back to back, same command, same machine: **byte-identical**
(`diff` empty). This is the binding claim task 2 asked to verify before
anything downstream trusts this fixture — see the report for the second run's
artifact if it's still around, or just re-run the command above twice and diff.

## What's in it

- `answer` / `generated_token_ids`: tier 2. Greedy decode of the fixed prompt
  produced token `50429` ("Paris") and then stopped (hit an end token before
  reaching `--max-new 8`).
- `diagnostics.layer_hidden_hashes`: tier 1. 1080 entries = 18 prompt tokens x
  60 layers, one BF16 hash of the residual stream per (position, layer) pair,
  plus one rolling `trace_hash` aggregate over all of them.
- `diagnostics.operation_hashes`: 3240 entries = 18 x 60 x 3, one hash each
  for `attention_local`/`attention_global`, `attention_residual`, and `mlp`
  per layer. Finer-grained than task 2 strictly asked for, but it's what
  localizes a detected break to *which part* of a layer, not just which
  layer -- kept because task 3 needs it.

Only the prefill pass is covered: this prompt is short enough that every
generated token comes from the fused CUDA decode path, which has no
host-visible layer boundary to hash (see the commit message on `5783d39`).
That is not a gap in this fixture -- it is the actual coverage boundary of
what `--layer-hash-trace` can see today, and it is enough for task 3, since
prefill alone touches every layer once per prompt token.

## Re-verifying

```
diff <(./build/strata-gemma4-run --model .../gemma4 --prompt "The capital of France is" \
        --max-new 8 --temperature 0 --layer-hash-trace --json) \
     tests/fixtures/gemma4/layer-hash-trace.json
```

Empty diff means the oracle agrees with this fixture. Any diff should name the
first divergent `(position, layer, operation)` from `operation_hashes` before
falling back to eyeballing the full file.
