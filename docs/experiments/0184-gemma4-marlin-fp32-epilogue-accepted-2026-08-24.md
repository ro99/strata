# Experiment 0184 — FP32 Marlin epilogue clears the page-kernel gate

**Date:** 2026-08-24  
**Branch:** `fix/gemma4-marlin-fp32-epilogue`  
**Origin:** experiment 0183's binding next action  
**Verdict:** **ACCEPTED as an isolated projection primitive**

## Hypothesis and predeclared gates

Changing only the accepted-speed Marlin core's result epilogue to preserve
FP32 accumulators through global reduction and publication can restore
Strata's declared compact-MXFP4 numerical contract while retaining at most
0.63 ms for each exact M=128 Gemma MLP shape.

The primary metric was the 32-launch median projection latency. Correctness
required maximum relative difference below 1e-4 against the canonical
FP32-accumulation oracle. Probe device memory was capped at 256 MiB; checkpoint
codes and E8M0 scales remained compact and single-copy in the candidate. Either
speed, correctness, or memory failure was the rollback condition.

## Cost model and mechanism

The production prefill `argmax_r` remains 2,419.841 ms of serial CUDA/host
handoffs. This isolated kernel does not reduce that term and earns no system
throughput claim. It establishes the primitive required by a later
device-owned executor, which must remove those handoffs rather than merely
replace projection calls.

The mechanism leaves compact weight and activation reads unchanged. It adds
one padded FP32 reorder tile per physical CTA:

`82 SMs * 64 rows * (256 + 8) columns * 4 bytes = 5,541,888 bytes`.

The final output grows from BF16 to the declared FP32 boundary, adding `M*N*2`
bytes relative to experiment 0183. Cross-CTA partials were already FP32. No
H2D/D2H, persistent weight, prepack, or host term changes in this probe. The
sign is therefore a bounded increase in device output traffic and transient
memory in exchange for removing an inadmissible rounding boundary.

Peak allocation, including simultaneous probe controls, was 233,697,608 bytes
for gate/up and 225,440,072 bytes for down. This is below 256 MiB; it is not a
runtime residency proposal.

## Cheap experiment

One correctness-first process took 3.5 seconds and passed all gates, after
which three fresh processes were run. Each process spends about 3.1 seconds in
allocation, deterministic prepack, oracle, and setup versus about 0.4 seconds
in aggregate measured launch windows, a fixed/measured ratio of about 7.8:1.
The rejected alternative was another roughly six-minute full checkpoint load:
it could not decide the primitive's numerical or speed gate. Planned total arm
time was below 15 seconds; the three repeated processes completed in 7.0
seconds after their shared build.

## Every run and median

All runs used `CUDA_DEVICE_ORDER=PCI_BUS_ID`, CUDA device 1, one RTX 3090 at
1605 MHz and 250 W, M=128, synthetic deterministic BF16-representable
activations and compact E2M1/E8M0 weights, and the canonical double-precision
oracle.

| Shape | Run 1 | Run 2 | Run 3 | Median | Worst max-relative | vLLM ruler |
|---|---:|---:|---:|---:|---:|---:|
| gate/up `[21504,5376]` | 0.563712 ms | 0.563904 ms | 0.563422 ms | **0.563712 ms** | **5.9e-8** | 0.490811 ms |
| down `[5376,21504]` | 0.523040 ms | 0.522779 ms | 0.522463 ms | **0.522779 ms** | **1.39e-7** | 0.506624 ms |

The largest spread is 0.000482 ms, far inside the 0.066288/0.107221 ms gate
margin. The medians are 1.149x and 1.032x the measured vLLM Marlin ruler and
both satisfy the predeclared 1.28x bound. The numerical residual is FP32
accumulation ordering, not an output-precision substitution.

## Correctness and verdict

The correctness-first process measured 5.9e-8 and 1.39e-7 maximum relative
error. The same maxima reproduced in every timed process. The canonical
compact code and scale byte counts remain respectively 57,802,752 and
3,612,672 for each 61,415,424-byte projection weight set; the Marlin layouts
are pure load-time permutations. No widened persistent weight exists.

**ACCEPTED as the M=128 MXFP4 projection primitive.** This does not close the
Gemma prefill defect: the current runtime still owns activations on the host,
dispatches eight projected-weight passes, and serializes thousands of CUDA
handoffs. The exact next experiment must integrate the accepted primitive
only as part of a bounded device-owned page executor and show that it reduces
the measured `sum_serial` term. W8A16 dispatch and Gemma's full numerical
oracles remain equal gates.

Raw results remain outside Git at
`/tmp/gemma4-marlin-fp32-run-{1,2,3}.json`; the correctness-first result is
`/tmp/gemma4-marlin-fp32-gate.json`.
