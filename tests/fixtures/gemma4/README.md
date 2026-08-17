# Gemma 4 layer-hash-trace fixture

`layer-hash-trace.json` is the tier-1 (per-layer, per-operation BF16 hash) and
tier-2 (generated token IDs) equivalence-oracle fixture for Gemma 4.

**Re-captured** after closing the P3 blind spot (brief 00c task 3 / brief 01
task 0): a fourth operation-hash checkpoint, `mlp_residual`, was added right
after the final scaled residual add, at the same point `record_layer_hash`
already fires. Before this, a fault landing between the `mlp` checkpoint and
the next layer's first checkpoint was invisible to `operation_hashes`
specifically (though still caught by `layer_hidden_hashes`, which is why the
gate passed anyway) -- see perturbation 3's commit (`84e41ee`) for the
mechanism. `layer_hidden_hashes`' `trace_hash` is unchanged by this fix
(`9fdbee6c3f216c01`, same as the original capture at `d3852a0`), because it
was never blind to begin with; only `operation_hashes` gained a checkpoint.

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

Run twice, back to back, same command, same machine, after the `mlp_residual`
change: **byte-identical** (`diff` empty). Re-checked, not assumed carried
over from the pre-fix capture.

## What's in it

- `answer` / `generated_token_ids`: tier 2. Greedy decode of the fixed prompt
  produced token `50429` ("Paris") and then stopped (hit an end token before
  reaching `--max-new 8`).
- `diagnostics.layer_hidden_hashes`: tier 1. 1080 entries = 18 prompt tokens x
  60 layers, one BF16 hash of the residual stream per (position, layer) pair,
  plus one rolling `trace_hash` aggregate over all of them.
- `diagnostics.operation_hashes`: **4320** entries = 18 x 60 x 4, one hash each
  for `attention_local`/`attention_global`, `attention_residual`, `mlp`, and
  now `mlp_residual` per layer. Finer-grained than task 2 of brief 00c
  strictly asked for, but it's what localizes a detected break to *which
  part* of a layer, not just which layer.

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
