# Experiment 0033: OpenAI server overhead

Date: 2026-07-26  
Branch: `feat/openai-server`  
Result: acceptance gate passed

## Question and gate

Does the C++ OpenAI-compatible serving layer add less than 5% of one batch-1
decode step? The mechanism changes no inference resource term. It adds bounded
HTTP parsing, JSON serialization, and socket writes to `Σ_serial`; GPU compute,
H2D/D2H, NVMe traffic, weights, and KV storage are unchanged. The rollback gate
was median serving overhead greater than or equal to 5%, any response mismatch,
or a non-deterministic seeded result.

The cheap protocol/parser tests ran first. The real-model arm then used the
production three-device order and 2,048-token context because a reduced context
would not describe the serving operating point. Model setup took about 29 s;
the four-request measured window took about 124 s, a fixed-setup/window ratio of
0.23. Expected total wall time was under three minutes and the arm met that
budget.

## Operating point

- Model: pinned `models/glm52` checkpoint, int4 runtime contract
- GPUs: RTX 5060 Ti 16 GiB plus two RTX 3090 24 GiB devices, `--devices 0,1,2`
- Context ceiling: 2,048 tokens
- Request: system plus user message, batch 1, three generated tokens
- Sampling: temperature 1, top-p 0.9, seed 17, identical across repetitions
- Command: `scripts/run_openai_server_smoke.sh`
- Ignored raw output: `results/openai-server-smoke/`

`serving_overhead_ms` is request parse and response construction time outside
the runtime's measured prefill and decode phases. Socket transmission after
response construction is not included; the 333-byte response makes that cost
negligible on loopback. `decode_step_ms` is the runtime's measured decode time
divided by decode steps at the same request operating point.

## Results

| Repetition | Serving overhead (ms) | Decode step (ms) | Overhead |
|---:|---:|---:|---:|
| 1 | 1.3387 | 3,605.32 | 0.0371% |
| 2 | 0.6615 | 2,720.35 | 0.0243% |
| 3 | 5.0496 | 2,688.02 | 0.1879% |
| **Median** | **1.3387** | **2,720.35** | **0.0371%** |

The decode range is consistent with the repository's recorded GLM baseline of
0.283925 tok/s (about 3.52 s/token), so the denominator is plausible rather
than a serialization defect. All three seeded responses returned the identical
text and usage. The separate SSE request emitted content deltas, a length finish
reason, and a final `data: [DONE]` record. `/v1/health` and `/v1/models` also
passed.

## Conclusion

The 0.0371% median is well below the 5% acceptance gate and below observed
run-to-run decode variance. This is a serving-cost result, not a throughput win:
no `argmax_r` resource term was reduced.
