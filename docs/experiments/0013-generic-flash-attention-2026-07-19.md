# Generic FlashAttention forward backend

Status: **operation gate and short full-model correctness accepted for DeepSeek
and GLM; performance and default promotion pending**.

## Contract

- Hypothesis: a shared CUDA QK/softmax/V primitive can remove host attention
  loops for DeepSeek and GLM without changing either architecture's attention,
  sparse-selection, or numerical contract.
- Primary metric: median decode tokens/s and attention nanoseconds over at
  least three interleaved repetitions; prefill is reported separately.
- Correctness gate: scalar operation oracle, CUDA fixtures for sinks, causal
  limits, indexed rows, DeepSeek's 128-plus-512 shared-KV shape, GLM's all-F32
  arithmetic, pinned full-model route/layer/logit traces, token identity, and
  `make check`.
- Memory ceiling: existing model admission ceilings and a request-bounded
  reusable CUDA workspace; no persistent score tensor.
- Rollback: keep the scalar default on any oracle, token, route, diagnostic,
  device, numerical-status, checkpoint-read, memory, or three-repetition
  throughput failure.

The implementation is ground-up C++/CUDA and adds no framework or attention
library dependency. The model-neutral descriptor owns tensor layouts, exact
row gathers, MHA/GQA/MQA head mapping, causal visible prefixes, attention
sinks, numerical policy, and a byte ceiling. Architecture adapters continue to
own projections, RoPE/YaRN, compressed caches, learned selection, and rounding
boundaries.

## Operation evidence

The CUDA-enabled suite passes on the repository's SM120 RTX 5060 Ti and two
SM86 RTX 3090 devices. Fixtures cover the tiled online-softmax path,
DeepSeek-sized 64-head/512-dimension shared KV with a 128-row window and 512
selected compressed rows, and GLM's causal all-F32 compatibility contract.

## DeepSeek full-model evidence

The accepted one-pair correctness run used the pinned
DeepSeek-V4-Flash-DSpark checkpoint, devices `0,1,2`, device MoE on both sides,
28 resident/spine/attention workers, 216 GiB host admission, 0.85 VRAM
fraction, detailed timings, greedy decoding, and two generated tokens. The
artifact is under
`results/deepseek-v4-flash-attention-correctness-exact/`.

- generated IDs `[30594, 1175]` matched;
- sequential route traces, logits, layer hashes, and operation hashes matched;
- prefill and decode checkpoint reads were zero;
- peak RSS was 145,319,044 KiB scalar and 145,321,268 KiB candidate;
- peak VRAM was unchanged at 13,240/20,294/20,294 MiB;
- candidate prefill was 4.3630 s versus 5.6693 s scalar;
- the one decode step was 0.6389 s versus 0.7457 s scalar;
- candidate prefill/decode issued 215/43 attention calls and moved
  29,642,496/6,218,496 H2D bytes plus 28,181,340/5,636,268 D2H bytes.

The apparent 1.167x one-pair decode speedup is descriptive only. It is not a
performance result under the charter's three-interleaved-repetition rule.

## GLM full-model evidence

The one-pair GLM correctness run used the pinned QuantTrio GLM-5.2 checkpoint,
devices `0,1,2`, context ceiling 256, detailed timings, sequential route
traces, greedy decoding, and two generated tokens. The artifact is under
`results/glm52-flash-attention-correctness-exact/`.

- generated IDs `[16, 13]` matched;
- all 2,325 selected expert lists matched exactly;
- routed coefficients differed by at most `1.967e-6`, within the declared
  `4e-6` all-F32 cross-device exponential contract;
- candidate prefill/decode issued 78/78 attention calls, moving
  460,072,080/322,044,216 H2D bytes and 153,354,552/5,112,120 D2H bytes;
- peak RSS was 2,277,516 KiB scalar and 2,278,232 KiB candidate.

The scalar run read 175,202,615,296 physical bytes while the candidate ran from
the warmed page cache and read only 360,448 physical bytes. Its timing is
therefore not an equal-I/O comparison and no GLM performance conclusion is
drawn. Dense GLM remains correctly reported as I/O-dependent at this admission
budget.

## Rejected numerical attempts

The first tiled-online candidate was token-identical in one short run but
diverged in BF16 layer traces and routes. Increasing online accumulation to F64
still diverged first at prefill position 1, layer 6 and changed generated
tokens. Those paths are not used by the DeepSeek adapter. The accepted adapter
selects the explicit F64-dot/F32-score/global-softmax/F32-accumulation contract
matching the established scalar oracle.

## Decision

The shared backend and both current adapters satisfy their short correctness
gate. The default remains scalar. Longer-context fixtures and a
three-repetition interleaved performance matrix with equal cache/I/O state are
required before default promotion.
