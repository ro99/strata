# Experiment 0116 — the output projection on the SM86 FP8 tensor path

Status: **accepted.** Total GPU kernel time at 677 tokens falls 17.58 s to
12.47 s (1.41x). `native_fp8_matmul_kernel`, the largest single kernel in the
prefill, falls 5.685 s to 0.020 s.

Run by the orchestrator directly; the codex executor was unavailable.

## Premise

Experiment 0114's nsys profile of Strata contradicted the phase counters we had
been reasoning from. Categorised by kernel at 677 tokens:

| category | seconds | share |
| --- | ---: | ---: |
| dense projections | 8.28 | 47.1% |
| MoE experts | 7.92 | 45.1% |
| mHC | 0.77 | 4.4% |
| attention proper | 0.58 | 3.3% |

Attention is 3.3%, not the ~51% implied by
`maximum_device_dsv4_paged_attention_kernel_seconds`. That counter measures the
whole attention-to-mHC command, which also contains the output projection, the
router and the mHC transition. Comparing it against the reference's attention
kernel produced a 286x figure that was an artifact of comparing a subsystem to
a kernel.

The largest single kernel was `native_fp8_matmul_kernel` at 5.685 s over 215
instances, median 190 us and maximum 137 ms. Experiment 0105 replaced this
kernel with an SM86 tensor path but wired it into only three call sites —
`wq_a`, `wq_b`, `wkv` — through `linear_rows`. The output projection `wo_b`
inside the fused command still launched it directly with grid
`(kDsv4MhcHidden, rows)`: one block per (output row, batch row), so every batch
row re-reads the whole 4096-row weight.

## Changes

Three call sites, of which only the third mattered.

1. `ffn.gate` router via `linear_rows` — flag enabled. **No effect**, 215
   instances before and after.
2. Unfused `wo_b` via `linear_rows` — flag enabled. **No effect**; production
   takes the fused command.
3. `wo_b` inside `dsv4_paged_attention_to_mhc` — routed to
   `quantize_activation_e4m3_bytes_kernel` plus
   `dsv4_fp8_decode_bf16_tensor_kernel`. This is the whole result.

The tensor kernel writes whole 64-row tiles, so the branch workspace region is
sized to the padded row count. `branch_elements` keeps its exact meaning
because caller-facing contracts validate against it; a separate
`branch_capacity_elements` sizes the region. Compact E4M3 activation value and
scale regions were added to the layout. When the caller owns the destination —
the rank-local NCCL reduction buffer — the exact rows are copied out with a
device-to-device `cudaMemcpyAsync`.

## Result, 677 tokens, nsys per-kernel

| kernel | before | after | delta |
| --- | ---: | ---: | ---: |
| `native_fp8_matmul_kernel` | 5.685 s | 0.020 s | -5.664 |
| `dsv4_fp8_decode_bf16_tensor_kernel` | 0.371 s | 0.719 s | +0.348 |
| all other kernels | — | — | +0.06 |
| **total GPU kernel time** | **17.58 s** | **12.47 s** | **-5.10** |

The replacement costs 0.348 s to remove 5.664 s, a 16x net trade, inside the
1 s ceiling declared before the run. Instances fall 215 to 172: the 43
converted calls are the fused command's, and the remaining 172 are single-row
tail calls that correctly stay on the incumbent path.

## Correctness

Generated IDs `2107, 8777, 1277, 440` at both 677 and 2,612 tokens, identical
to every prior arm. Decode checkpoint reads 0. At 2,612 tokens
`prefill_max_workspace_bytes` is 171,114,496 and per-GPU VRAM is unchanged from
0113, so the padded-tile writes are contained. `make check` 2/2.

The 2,612 arm matters independently: it sub-chunks into roughly 475-row pieces
with a different remainder from 677's single 677-row chunk, so it exercises a
different padded-tile boundary.

## Not claimed

Wall-clock prefill moved 52.77 s to 46.94 s at 677 tokens and 192.91 s to
136.12 s at 2,612, but the MoE bucket swung 27.70 s to 42.95 s between two
arms of an unrelated change earlier in the day. Only the per-kernel nsys
figures are attributed here.

## Next

With this kernel gone the profile is dominated by the MoE experts, about 64% of
the remaining 12.47 s: `fp4_tiled_page_gate_up` 3.994 s,
`fp4_tiled_page_down` 1.869 s, `deepseek_fp8_page_gate_up` 1.309 s,
`deepseek_fp8_page_down` about 0.79 s. Experiment 0114 established the
reference's shape: dequantise MXFP4 to BF16 in a separate pass, 1.572 s, so the
GEMM itself is 0.218 s. Strata decodes FP4 inside the matmul instead.
