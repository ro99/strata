# Kimi-K3 runtime

Status labels below are precise on purpose. A gate that has not been measured is
marked as such, not described as passing.

## What this model is

Kimi-K3 is a 93-layer hybrid backbone with attention residuals and a latent MoE.
Everything in this section is read out of `/data/kimi-k3` — its `config.json`,
its shard headers, and `modeling_kimi_linear.py` — and pinned in
`kKimiK3ExecutionContract` (`include/strata/model_adapter.hpp`), which a mutated
config is rejected against field by field.

| | |
|---|---|
| Layers | 93: three KDA then one gated MLA, repeating, plus a final MLA at layer 92 |
| Hidden size | 7168 |
| Attention | 96 heads; MLA `q_lora_rank` 1536, `kv_lora_rank` 512, nope/rope/value head dims 128/64/128 |
| RoPE | **none** — `mla_use_nope: true`, so there is no positional rotation anywhere |
| KDA | 96 heads, head dim 128, short-conv kernel 4, decay rank 128 |
| MoE | 896 routed experts, top-16, 1 group, 2 shared experts, `routed_scaling_factor` 1.0 |
| Latent MoE | router on the raw hidden state, then 7168→3584 down, experts 3584→3072→3584, RMSNorm, 3584→7168 up |
| Dense prefix | layer 0 only (`first_k_dense_replace: 1`), intermediate 33792 |
| Attention residuals | block size 12, so layers 0, 12, …, 84 open a block: eight blocks plus the prefix |
| Activation | SiTU-GLU (`hidden_act: situ`, β 4.0, linear β 25.0) — **not** SwiGLU |
| Vocabulary | 163,840: 163,584 base pieces and 256 reserved special ids |
| Context | 1,048,576 |

Precision: **only the routed experts are quantized.** They are MXFP4
(`mxfp4-pack-quantized`, two E2M1 per byte, per-32-column E8M0 scale) — four
bits plus an eight-bit block scale, which is at the charter's floor and not
below it. Everything else, all 106.55 GiB of it, is BF16.

| | GiB | share |
|---|---|---|
| Total tensor payload | 1453.66 | 96 shards, 497,220 tensors |
| Routed experts (MXFP4) | 1347.12 | 92.7% |
| Dense spine (BF16) | 106.55 | 7.3% |

## What it costs on this machine

The governing model is `τ = max_r W_r/B_r + Σ_serial`
(`research/moe-tiered-memory-decode-optimization.md`), instantiated in
`docs/experiments/0048`. The short version:

**SATA is `argmax_r` by six to eight times.** A batch-1 decode step moves
15.6–20.7 GiB of expert misses across a ≈400 MB/s link, which is 42–56 s, while
PCIe carries 76.4 GiB at ~6.8 s and compute is negligible. `τ ≈ 0.02 tok/s`, and
that is the honest consequence of holding 1.45 TiB of weights on one SATA disk
with 245 GiB of RAM — not a defect to engineer away. Prefill is worse per page:
past roughly 256 tokens a page touches essentially every expert of every layer,
so it costs one full 1347 GiB sweep whatever the page size.

Two things follow, and they are binding:

- A mechanism that does not reduce SATA bytes cannot improve `τ`. No
  speculation, no drafting, no recompute — they add compute and PCIe load to
  shrink nothing. `docs/experiments/0025` is the worked example of that error.
- Caching cannot manufacture sparsity. 185 GiB of host arena against 1347 GiB of
  experts is the whole story, and the arena reports its hit rate rather than
  implying one.

W8A16 on the BF16 spine would halve the PCIe term and is charter-legal, but it
is worthless while SATA is `argmax`. It is not implemented.

## The NVMe constraint

Model bytes travel **SATA → host RAM → VRAM only**. Nothing derived from the
weights may be written to, staged on, or paged out to `/dev/nvme0n1` or
`/dev/sdb`; NVMe endurance is a resource the operator is protecting.

`kimi_apply_write_guard` runs before a single byte is read and *refuses*, rather
than warning, when it finds a vector open. It closes core dumps (`RLIMIT_CORE`
0), points scratch at tmpfs, disables the CUDA JIT cache, checks that the arena
can be locked, and checks that swap is not on a protected disk.

The swap path is closed by `mlockall(MCL_CURRENT | MCL_FUTURE)`, not by asking
the operator to change their system: a locked page cannot be written to swap by
definition, and `MCL_FUTURE` covers the spine and the arena, neither of which
exists at guard time. It needs `RLIMIT_MEMLOCK` to cover the resident set
(`ulimit -l unlimited`), not root. Verified on the target with a swapfile still
active on `nvme0n1`: 235.7 GiB resident, `VmLck` covering it, `VmSwap` zero.

Two things the operator should know. Locking this much creates memory pressure
that pushes *other* processes toward swap — that is not a model byte reaching
the disk, but it is a real effect. And if the lock fails, the guard falls back
to refusing while swap sits on a protected disk, naming the lock failure first.

Verification is measured, not asserted: `/sys/block/<disk>/stat` **field 7** is
cumulative sectors written (field 10 is `io_ticks` — that numbering belongs to
`/proc/diskstats`, which has three extra leading columns). The threshold is a
contemporaneous idle control, not a constant: this machine idles at 13–29 KiB/s,
so the ≈0.1 KiB/s figure an earlier handover recorded would fail every run.

## Status by gate

| Gate | What it checks | Status |
|---|---|---|
| 1 | Contract and manifest; a mutated config is rejected field by field | **green** |
| 2 | MXFP4 codec against `compressed-tensors` on real expert tensors | **green** |
| 3 | Exact operations from real tensors; chunkwise KDA ≡ the token recurrence | **green** |
| 4 | KDA, gated MLA, and the LatentMoE block against `modeling_kimi_linear.py` | **green** |
| 5 | Full-model teacher forcing, every layer | **retired as formulated** — reports, does not assert; see below |
| 6 | Greedy generation against the reference's own continuation | **green**, gap-aware; output agrees on 488 and 810 |
| 7 | Tokenizer and XTML chat rendering against the checkpoint's tokenizer | **green** |

Gate 4, measured (tolerance stated before the run: ≈5e-3 predicted from six BF16
op boundaries, gated at 2.0e-2 relative L2 and 0.999 cosine):

```
KDA prefill          relative L2 0.0061   cosine 0.999981
KDA prefill state    relative L2 0.0050   cosine 0.999988
KDA decode           relative L2 0.0055   cosine 0.999985
MLA prefill          relative L2 0.0070   cosine 0.999975
MLA decode           relative L2 0.0091   cosine 0.999959
LatentMoE block      relative L2 0.0045   cosine 0.99999
```

Gate 7, measured: 33/33 tokenizer cases encode *and* decode exactly, 4/4 XTML
conversations render and tokenize exactly. No tolerance — ids are exact or wrong.

Gates 5 and 6 first read negative, and the discriminating experiment showed the
**gates** were unsound rather than the runtime. See `docs/experiments/0049`.

Gate 5 compared per-layer hidden states against a reference computed at BF16
activation precision, while the runtime carries F32. A control run — the same
reference against itself at F32 — settles it:

| comparison | prompt layers routing differently | decode |
|---|---|---|
| runtime vs BF16 reference | 77 / 92 | 74 / 92 |
| **BF16 reference vs F32 reference** | **82 / 92** | **77 / 92** |

The reference disagrees with itself *more* than the runtime disagrees with it,
so per-layer L2 against a BF16 reference cannot separate a defect from the
reference's own rounding. The prediction that error would be flat in depth
argued from the residual stream being additive and ignored that a discrete
top-16-of-896 selection sits between the additions: it amplifies a 0.4%
activation difference instead of attenuating it. Gate 5 is therefore retired as
formulated. Its replacement — routed-set agreement against an F32-activation
reference — **reports but does not assert**, which is the honest strength of the
claim it can make.

Gate 6's decode arm was asserting greedy equality across a 0.3125 logit gap on a
14.6 scale, against 1.146 of pass-to-pass noise: it was testing the arithmetic's
tie-break, not the model. It is now gap-aware, and asserts only when the margin
is decisive.

**Output agrees everywhere**: prompt token 488 and decode token 810, identically
across the BF16 reference, the F32 reference, and the runtime. The runtime
reproduces its reference. What is *not* claimed is a bit-exact per-layer match
against a BF16 oracle, which the control above shows is not a well-posed test.

## Running it

```cpp
strata::KimiK3RuntimeConfig config;
config.maximum_context_tokens = 2048U;
config.prefill_page_tokens = 64U;
strata::KimiK3Runtime runtime;
auto ready = runtime.initialize("/path/to/kimi-k3", config);   // ~11 min
```

Initialization reads the tokenizer and `generation_config.json` first, so a
malformed vocabulary costs a second rather than five minutes, then loads the
106.55 GiB spine and sizes the expert arena from `MemAvailable` less an 8 GiB
reserve.

Three surfaces:

- `evaluate(tokens, position, logits)` — the whole backbone over a page. It
  refuses a position that is not the current length, because the KDA half is
  recurrent and cannot be rewound; a new sequence is `reset_sequence()`.
- `generate_from_tokens(prompt, n, sampling)` — pages the prompt, then samples,
  stopping on the `eos_token_id` that `generation_config.json` states (163586,
  `<|end_of_msg|>`, which is **not** the tokenizer's `[EOS]` at 163585).
- `generate_chat_stream(messages, …)` — renders XTML and decodes the reply. A
  message carrying an image is refused rather than silently flattened to text;
  see below.

`config.layer_observer` receives each layer's output as the graph runs. Nothing
on the serving path sets it; the teacher-forcing oracle does.

## Chat format

XTML, not header-delimited turns:

```
<|open|>message role="user"<|sep|>Hello<|close|>message<|sep|><|end_of_msg|>
<|open|>message role="assistant"<|sep|><|open|>response<|sep|>
```

`<|open|>`, `<|close|>`, `<|sep|>`, and `<|end_of_msg|>` are single special
tokens; tag names and attribute text go through BPE. Attribute values escape `&`
then `"`. The think channel is structural: in thinking mode every assistant
message carries `<|open|>think<|sep|><|close|>think<|sep|>` even when empty, and
a thinking conversation opens with a `thinking-effort` internal system message,
because `apply_chat_template` defaults `thinking_effort` to `max` and that text
is part of the tuned prompt.

## Tokenizer

`ModelTokenizer::load_kimi_k3(model_directory)` reads `tiktoken.model` — one
base64 piece and rank per line — and the reserved special ids from
`tokenizer_config.json`. The BPE is tiktoken's, not HuggingFace's: there is no
merge table, so encoding repeatedly joins the adjacent pair whose concatenation
has the lowest id in the vocabulary, over raw bytes rather than the GPT-2 byte
alphabet.

The pretokenizer implements Kimi's `pat_str` in the order the reference lists
its alternatives, because `fancy-regex` is leftmost-first and the order is
semantics: a Han run; then `[^\r\n\p{L}\p{N}]? [Lu Lt Lm Lo M]* [Ll Lm Lo M]+`
with an optional contraction; then the same with the two cased classes swapped;
then `\p{N}{1,3}`; then a punctuation run with an optional leading space; then
the three whitespace alternatives. The two cased alternatives are what split
`CamelCase`, and Han has its own alternative while kana do not — treating all
CJK alike splits Japanese differently from the reference.

## Not implemented

- **MoonViT-V2 vision.** The config is parsed, pinned, classified, and placed
  (0.83 GiB, device-pinned), and the 168 BF16 tensors are inventoried, but
  nothing executes them. A chat message carrying an image is an error, not a
  message with the image dropped. Tracked as
  [ro99/strata#21](https://github.com/ro99/strata/issues/21).
- **CUDA kernels.** Deliberate, and measured: SATA is `argmax_r` by 6–8× at
  batch-1 decode, so device kernels cannot move `τ` for decode. Which kernels
  matter for prefill is a question for a phase breakdown from the runtime. That
  breakdown now exists — `docs/experiments/0049` — and says the decode step is
  83% storage, so this stays deliberate.
- **Prescriptive placement.** The inventory is descriptive; the runtime does not
  yet consume `layer_device` or `device_budget_bytes`. That waits on a plan
  validated against a real load.

## Regenerating the fixtures

The oracles are Python because the reference is; the runtime stays C/C++. Each
script redirects every scratch path to tmpfs and resolves its output directory
through `refuse_forbidden_disk`, so it cannot put a byte on a protected disk.

```bash
scripts/run_kimi_k3_reference_fixture.sh    # gate 4, layer level
scripts/run_kimi_k3_backbone_fixture.sh     # gates 5 and 6, ~16 min
scripts/kimi_k3_tokenizer_fixture.py        # gate 7
```

Fixtures land in `/data/strata-results/kimi-k3-fixtures`, on the SATA disk beside
the checkpoint. `results/` in the working tree is **not** an admissible
destination: it is on `/dev/nvme0n1p2`, and reference activations are derived
from model weights.

The backbone gate is opt-in, because it loads 106.55 GiB and sweeps the routed
experts twice:

```bash
STRATA_KIMI_BACKBONE=1 ./build/strata-tests
```
