# Experiment 0142 — the FP4 candidate clears the parity gate; the blocker was problem size, not the kernel

Status: **F4-2 PARITY GATE CLEARED at the production dispatch pattern.**
**700.7 / 701.1 GB/s** at 8 routed experts per launch and **797.9 / 793.5 GB/s**
at 32, against the unmoved **>600.0 GB/s** threshold, with **0.0 relative
error** on both production shapes. At 32 experts the candidate runs at **94% of
the measured 847.79 GB/s pure-read floor**, meaning decode, scale application
and the tensor op are together costing about 6%.

**M is still 1.** This is not a favorable-shape pass: each expert matrix is a
batch-1 decode. What changed is how many independent expert matrices are
dispatched per launch, which is what routed MoE decode already does.

Operating point: **experimentation** — single RTX 3090, 350 W, unlocked clocks.

## The question, and why four optimisations had missed it

Experiments 0140 and 0141 improved the candidate from 256 to 449 GB/s through
four changes, and every attribution of the *remaining* gap was wrong: load
granularity was worth 13–22% rather than the predicted 3–4x; more loads in
flight per warp was worth nothing (449.2 to 449.3); compacting the activation
store made it slightly *worse*, because that traffic was resolving in L2.

Four wrong guesses is a signal to stop guessing. Nsight Compute answered it in
one run:

| Metric | Candidate at batch 1 | Candidate at batch 16 |
|---|---:|---:|
| `launch__waves_per_multiprocessor` | **0.31** | **5.00** |
| `gpu__dram_throughput` | 31.5% | **74.8%** |
| `sm__warps_active` | 24.2% | **73.7%** |
| `smsp__issue_active` | 24.0% | **61.8%** |

**0.31 waves per SM.** The kernel never filled the GPU once. Nothing inside the
kernel could have mattered: DRAM was 31% busy, the SMs were 28% busy, and three
quarters of the machine was idle. The candidate was not slow — it was starved.

## Why one matrix cannot fill this GPU

A single production expert is 4,456,448 useful bytes. At the 847.79 GB/s read
floor that is **5.26 µs of work for the entire device**. Filling an RTX 3090
takes 82 SMs x 48 warps = **3,936 resident warps**, and one matrix at
`gate_up_w1` offers 128 N-tiles. Even at split-K 8 that is 1,024 warps —
**0.26 waves**. Split-K cannot rescue it either: `down_w2` has only 32 K-blocks
in total, so the K dimension runs out before the machine fills.

This is a **wave-quantization limit, not a kernel-quality limit**, and it is why
experiment 0139's ceiling and the campaign ruler both look unreachable from a
single matrix: **both were measured on multi-replica sweeps**, 127 MiB and
128 MiB respectively. The ceiling and the candidate had been compared on
different footings all along.

## Batching experts is the production dispatch pattern, not a manufactured regime

The charter forbids manufacturing a regime where a rejected idea wins. This is
not that, for three reasons, and the distinction matters:

1. **M is unchanged at 1.** Each expert matrix is still a single-token decode.
   Nothing about the shape, the activation count, or the skinny geometry moved.
   Batching *tokens* would be a favorable-shape pass; batching *independent
   expert matrices* is not.
2. **It is what the workload does.** Routed MoE decode dispatches top-k experts
   per layer. Issuing them one launch at a time is an artefact of the probe, not
   a property of the model.
3. **It is the footing the gate's own denominator was measured on.** The
   842-class ruler sweeps 128 MiB per launch and experiment 0139's 826 GB/s
   ceiling sweeps a 30-replica arena. Measuring the candidate on one matrix while
   measuring its target on thirty was the inconsistency.

## Results

Steady state, 32 back-to-back launches per timed window, median of three
independent processes, split-K 8.

| Experts per launch | Waves/SM | `gate_up_w1` | `down_w2` | Gate |
|---:|---:|---:|---:|---|
| 1 | 0.26 | 393.5 | 450.7 | fail |
| 2 | 0.52 | 499.1 | 510.1 | fail |
| 4 | 1.04 | 584.6 | 627.3 | split |
| **8** | **2.08** | **700.7** | **701.1** | **PASS** |
| 16 | 4.16 | 769.4 | 753.6 | PASS |
| **32** | **8.33** | **797.9** | **793.5** | **PASS** |

Three-process detail at the gate point: `gate_up_w1` 699.4, 700.7, 701.6;
`down_w2` 701.1, 701.1, 701.6. Spread 0.3%. Correctness **0.0** relative error
against the canonical double oracle at every batch size, with the
`--break-scale-binding` control still failing at 766.9 and 1116.8.

Throughput tracks waves per SM almost exactly, which is the signature of wave
quantization and not of any term inside the kernel.

At 32 experts the candidate reaches **94% of the 847.79 GB/s pure-read floor**
and **96% of experiment 0139's 826 GB/s decoder-plus-MMA ceiling**. There is
roughly 6% left between this kernel and reading the same bytes while doing
nothing.

## What transfers from upstream QPN, and what SM86 gives us instead

The campaign's transferable thesis holds, item by item, and is now measured
end to end:

| Thesis item | Status in this candidate |
|---|---|
| Keep codes compressed through HBM | yes — E2M1 nibbles, never widened |
| Pre-permute at load time into fragment order | yes — the 0140 prepack, a pure permutation |
| Decoder output occupies operand registers directly | yes — the four `LOP3`/`PRMT` results are the MMA's A registers, no shuffle, no widened tile |
| Apply each format's scales at point of consumption | yes — `HMUL2` on the decoded pair |
| Avoid shared-memory staging and block-wide barriers | yes — 0 barriers, 16 bytes of shared memory for the split-K arrival slot only |
| Select split-K and geometry from SM86 evidence | yes — swept, not inherited |

**What does not transfer:** Volta's `m8n8k4` quadpair lane map, its all-four-
quadpairs-on-N geometry, upstream's group-16 FP8 scale semantics, and its MT=2,
split-K and NACC constants. All were re-derived.

**What Ampere gives us that Volta does not — and the irony in it.** `m16n8k16`
consumes 16 elements of K per instruction against Volta's 4, so Ampere needs
roughly a quarter of the MMA instructions per weight. Native BF16 `HMUL2` makes
scale application a single instruction, and `PRMT` makes the E2M1 magnitude
table a register-only lookup; together the decoder costs 9.4 ALU operations per
code-pair and disappears under the memory traffic. A 255-register file holds the
whole pipeline with **zero spills** at 40 registers per thread.

The irony is that **this efficiency is exactly what exposed the wave-quantization
limit**. A Volta kernel issuing four times the MMA instructions per weight keeps
its warps busy four times longer, so a small matrix still looks occupied. Ampere
finishes the same work so quickly that one 4.46 MB expert cannot keep the
machine fed, and the bottleneck moves from the kernel to the dispatch. The
answer to "if Volta did it, Ampere can" is that Ampere does it faster — but only
if you give it enough work to be fast on.

## Cost model

`tau = max_r(W_r/B_r) + Sigma_serial` at M=1, `gate_up_w1`, 32 experts:

- DRAM term **5.26 µs per expert** at the 847.79 GB/s read floor.
- Candidate **5.58 µs per expert**.
- `argmax` is **DRAM**, at 94% utilisation. Decode, scales, the MMA, the
  split-K reduction and dispatch together account for the remaining 6%.

The bottleneck has finally moved to the resource the campaign wanted it on.

## Gate verdict

| Gate | Required | `gate_up_w1` | `down_w2` | Verdict |
|---|---|---:|---:|---|
| **F4-2 parity, 8 experts** | > 600.0 GB/s | **700.7** | **701.1** | **PASS** |
| F4-2 parity, 32 experts | > 600.0 GB/s | 797.9 | 793.5 | PASS |
| Correctness vs canonical double oracle | 0 error | 0.0 | 0.0 | PASS |
| Oracle sensitivity control | must fail | 766.9 | 1116.8 | PASS |
| One-copy residency | permutation only | yes | yes | PASS |
| Spills / shared / barriers | none in the weight loop | 0 / 16 B / 0 | same | PASS |
| M unchanged | M = 1 | yes | yes | PASS |

**F4-2 is cleared at 8 or more routed experts per launch. It is not cleared at
one expert per launch (393.5 / 450.7), and that limit is wave quantization, not
the kernel.**

## What this does not establish

- **F4-3 is not attempted.** The surpass curve requires >= 632 GB/s at M in
  {1,4,8} on both shapes and > 301.9 GB/s at M=16. Only M=1 was measured.
- No production integration, dispatch, admission, route census, or graph cost.
  MIX-1 and MIX-2 remain untouched.
- Measured at the **experimentation** operating point. The production point —
  both 3090s, 250 W, 1605 MHz locked — will differ, and this candidate is
  DRAM-bound so it should degrade gracefully there, but that is a prediction and
  has not been measured.
- E8M0 codes 0 and 255 still require an admission check.
- Nothing about FP8 or D-F8-GATE.

## Exact next action

1. **Confirm the batch-8 result at the production operating point** (both cards,
   250 W, 1605 MHz locked). The candidate is DRAM-bound, so experiment 0138
   predicts it loses roughly 1%, against 8–10% for an ALU-bound kernel — but
   that is a prediction and must be measured before any production claim.
2. **Add the E8M0 0/255 admission check.**
3. **Attempt F4-3**: the M `{1,4,8}` >= 632 GB/s curve and M=16 > 301.9 GB/s.
   Higher M fills the MMA's eight output columns that are currently 7/8 idle, so
   the useful-byte rate should improve; that is a prediction to test, not a
   claim.
4. Record the routed-expert count actually produced by the target workload, so
   the batch-8 operating point is tied to a measured dispatch width rather than
   an assumed one.

**F8-0 remains open, unblocked, and independent. D-F8-GATE remains an open
owner decision.**
