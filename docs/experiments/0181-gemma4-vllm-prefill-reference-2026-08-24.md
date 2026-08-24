# Experiment 0181 — Gemma 4 vLLM prefill and decode reference

**Date:** 2026-08-24  
**Runtime:** `/home/rodrigo/Developer/Lvllmds4-x` (`vllm 2.3.8`)  
**Checkpoint:** `models/gemma4/model.safetensors`, 19,531,513,296 bytes  
**Verdict:** **3000 tok/s is not reproduced; the Strata defect remains large**

## Question

Before continuing the Strata prefill work, directly test the owner's remembered
3000 tok/s vLLM figure using this exact Gemma 4 MXFP4 checkpoint and the same
capped RTX 3090s. This is a reference measurement, not a Strata throughput
claim and not a runtime change.

## Operating point and method

Both RTX 3090s were held at the task's production point: 1605 MHz SM clock and
250 W. `CUDA_DEVICE_ORDER=PCI_BUS_ID` was set. The server used compressed-
tensors MXFP4 through `MarlinMxFp4LinearKernel`, Triton attention, text-only
mode, prefix caching disabled, chunked prefill enabled, one sequence, and one
generated token. Each shape was warmed once; three subsequent measurements
were interleaved. The primary metric is the server-side delta of
`vllm:request_prefill_time_seconds_sum` divided into the delta of
`vllm:prompt_tokens_total`, rather than client TTFT.

TP=1 could not admit a 4096-token KV allocation at VRAM fraction 0.95: 3.45 GiB
was required and 1.99 GiB was available. It was therefore run with
`max_model_len=2200`, which admitted 2384 KV tokens. TP=2 admitted 4096. The
two 3090s are connected through PHB without NVLink; vLLM's peer/custom-allreduce
probe failed and TP=2 used PYNCCL.

## Results

Every value below is server-side prefill throughput. Parentheses contain all
three runs in tok/s.

| Actual prompt tokens | TP=1 median | TP=2 median |
|---:|---:|---:|
| 127/128 | **881.67** (790.99, 881.67, 897.71) | **987.78** (934.02, 987.78, 1042.83) |
| 706/692 | **948.04** (960.52, 948.04, 947.33) | **1121.55** (1121.55, 1115.01, 1121.65) |
| 1391/1402 | **921.31** (930.68, 920.81, 921.31) | **1109.29** (1106.38, 1109.29, 1111.98) |
| 2070/2065 | **912.17** (912.73, 911.80, 912.17) | **1101.69** (1100.11, 1103.71, 1101.69) |
| 2705 | — | **1094.56** (1094.20, 1094.56, 1095.07) |

The comparable TP=1 short page spent a median 0.141776 s in server-side
prefill for about 127 prompt tokens. Strata's measured M=128 page spends
6.354801 s and produces 20.14 tok/s, so the observed gap is approximately
**44.8x in page time** or **43.8x in reported token rate**. TP=2 improves the
long-prompt median only about 1.20x because its extra compute is offset by the
PHB/PYNCCL topology.

During active TP=1 work the selected GPU averaged 1603 MHz, 236.1 W, and 93.8%
utilization. The TP=2 GPUs averaged 1605 MHz, 192.1/192.9 W, and 97.3/95.5%
utilization. The runs therefore exercised the declared clock point rather
than comparing different boost states.

## Consequence for the Strata experiment

The 3000 tok/s recollection is not a defensible design constant on this
machine and software stack. It invalidates the external premise used to set
experiment 0180's 600 GB/s gate, but it does not make Strata's result plausible:
vLLM is still roughly 44x faster on the comparable page.

The mechanism contrast remains direct. vLLM chooses a load-time repacked,
page-tiled Ampere W4A16 Marlin kernel, while Strata reuses its M<=16
decode-oriented register-fed path eight times and repeatedly crosses the
host/device boundary for normalization, RoPE, attention, KV work, and
activation materialization. The next experiment must profile and gate an
isolated page-tiled projection against this measured reference before runtime
integration.

Raw server and GPU telemetry remained outside Git under `/tmp`:
`gemma4-vllm-tp1-server.log`, `gemma4-vllm-tp1-gpu.csv`,
`gemma4-vllm-tp2-server.log`, and `gemma4-vllm-tp2-gpu.csv`.

## Decode addendum requested by the owner

The prefill matrix intentionally generated one token and therefore carried no
valid steady decode result. After the owner made matching decode an equal goal,
TP=1 was relaunched at the same 1605 MHz / 250 W point with prefix caching off,
one sequence, and the same exact checkpoint. A short prompt was warmed once;
three subsequent requests forced 128 completion tokens with EOS ignored.

The primary metric was `(completion_tokens - 1)` divided by the server-side
delta of `vllm:request_decode_time_seconds_sum`; this excludes the first token,
whose forward is charged to prefill. Each arm therefore contains 127 timed
decode forwards:

| Repetition | Decode seconds | Server decode tok/s | Client completion tok/s |
|---:|---:|---:|---:|
| 1 | 3.509532 | 36.1872 | 35.6221 |
| 2 | 3.503529 | 36.2492 | 35.6882 |
| 3 | 3.506909 | 36.2142 | 35.6657 |

Median server-side decode is **36.2142 tok/s**. The maximum run spread is only
0.0620 tok/s. Strata experiment 0165's accepted register-fed baseline is
18.03 tok/s, so Strata is **2.01x slower** on decode as well as about 44x slower
on the comparable prefill page. The external target is consequently two-part:
about 900--1000 prefill tok/s and about 36 decode tok/s at TP=1.

Raw decode logs remain outside Git at `/tmp/gemma4-vllm-decode-server.log` and
`/tmp/gemma4-vllm-decode-results.jsonl`.
