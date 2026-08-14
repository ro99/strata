# DeepSeek V4 TP2 ownership contract — corrected Stage 1

Date: 2026-08-09
Status: Stage 1 checked and Stage 3 rank-sharded loader/fixture gates completed for the matched CUDA/reference path; Stage 2 is accepted; stages 4–10 remain stopped.
Branch: `feat/dsv4-rank-local-tp2`
Scope: DeepSeek V4 only. Kimi is out of scope.

## Decision context

The hypothesis is that the current decode gap is the centralized execution
topology: every layer must execute on both TP ranks, with rank-sharded weights,
rank-local state, and real reductions at the two mathematically required layer
boundaries. The primary metric remains complete decode ms/forward at the matched
batch-one operating point. The correctness gate is the existing exact
operation/layer oracle, exact route association and rank-ordered reduction;
the memory ceiling is the accepted 0.95 per-GPU admission contract, including
the retained 151,228,416-byte full-context KV/state contract. A mismatch, an
unexplained scale slice, a fallback, or a failed memory/correctness gate rolls
back this topology branch and keeps Stage 4 onward stopped.

The installed NCCL probe establishes the accepted layer transport gate and now
has separate embedding/output association measurements. It does not authorize
runtime implementation. The current measured bottleneck is the
62.986 ms dependent callback-gap term; TP communication is the resource added
by this hypothesis, while CPU payload and aggregate GPU arithmetic are held
constant.

## Checked source and header evidence

The contract below is derived from:

- readable reference `vllm/models/deepseek_v4/attention.py`,
  `nvidia/model.py`, `compressor.py`, `vocab_parallel_embedding.py`,
  `logits_processor.py`, `routed_experts.py`, and `runner/moe_runner.py`;
- Strata `src/deepseek_runtime.cpp`, `include/strata/cuda_backend.hpp`,
  `kernels/cuda/backend.cu`, `src/deepseek_manifest.cpp`, and
  `src/deepseek_checkpoint.cpp`;
- accepted physical-page and device-integration records in experiments 0068
  and 0069; the mHC fixture record in experiment 0061;
- the retained target-format header excerpt in
  `results/dsv4-rank-local-tp2/ownership-header-evidence.txt`.

The target header records that corrected this map are:

| Location | Header evidence |
|---|---|
| Global | `embed.weight` BF16 `[129280,4096]`; `head.weight` BF16 `[129280,4096]` |
| Early layer 2 | `compressor.wkv` and `compressor.wgate` BF16 `[1024,4096]`; `indexer.compressor.wkv` and `indexer.compressor.wgate` BF16 `[256,4096]` |
| Middle layer 21 | `compressor.wkv` and `compressor.wgate` BF16 `[512,4096]` |
| Late layer 42 | `compressor.wkv` and `compressor.wgate` BF16 `[1024,4096]`; each indexer-compressor matrix BF16 `[256,4096]` |
| Router | early `ffn.gate.tid2eid` I64 `[129280,6]`; later `ffn.gate.bias` F32 `[256]` |

Experiment 0069 independently records the layer-2 plain-BF16 compressor
headers and the plain-BF16 `[256,4096]` indexer-compressor headers. These are
not FP8 matrices, and the two indexer matrices are not two `[128,4096]`
matrices.

## Notation and exact association

Matrix shapes use checkpoint order `[out,in]`. `R` means an immutable tensor is
replicated on both ranks. `C` means column-parallel: output rows are split.
`Q` means row-parallel: input columns are split and each rank produces a
partial output. “Local state replicated” below means two independent fixed
buffers with equal mathematical contents, not one centralized owner.

The target contract is hidden size 4096, 64 attention heads, head dimension
512, 8 output groups, q low-rank 1024, output low-rank 1024, `hc_mult=4`,
256 routed experts, six selected experts, and 43 decoder layers. The rank-local
attention query is 32 heads × 512. The layer input/output association is BF16;
FP32 is used for the declared TP sum where the readable reference and the
NCCL exactness arm require it, followed by one BF16 publication. The embedding
association is the reference BF16 embedding output. Router logits and mHC
mixes are FP32.

## Operation and ownership map

Every row identifies the full tensor, its ownership class, local shape, input
and output ownership, the reduction boundary, and both implementation names.

### Embedding, mHC, and norms

| Tensor/operation | Full shape and encoding | Class; rank-local shape | Input ownership → output ownership | Reduction boundary / arithmetic dtype | Strata function | Readable reference |
|---|---|---|---|---|---|---|
| `embed.weight` | BF16 `[129280,4096]` | C on vocabulary rows; `[64640,4096]` per rank | token id is replicated; each rank emits a masked local `[4096]` lookup partial | `VocabParallelEmbedding.forward` all-reduces the local output; BF16 reference association, no mHC expansion in this row | `Impl::embed`, future rank-sharded embedding loader | `VocabParallelEmbedding.__init__`, `weight_loader`, `forward` |
| Embedding output | BF16 `[4096]` | complete `[4096]` on both ranks after the embedding association | local vocab lookup partial → complete replicated vector | one association outside the 86 layer reductions | `Impl::embed` → rank-local publication | `VocabParallelEmbedding.forward` |
| mHC expansion | BF16 `[4,4096]` | R immutable rule; two rank-local state buffers | complete embedding `[4096]` → four-copy residual state | no TP reduction; `unsqueeze(-2).repeat(...,hc_mult,...)` | `device_mhc_forward_hidden` setup | `DeepseekV4Model.forward` immediately after `embed_input_ids` |
| `hc_attn_fn` | F32 `[24,16384]` (`(2+4)×4`, `4×4096`) | R; full shape on each rank | four-copy residual `[4,4096]` → mHC FP32 mix logits | no TP reduction inside mHC | `CudaDsv4MhcWeights`, `dsv4_mhc_pre_f32`/future device equivalent | `DeepseekV4DecoderLayer.hc_attn_fn`, `mhc_pre_tilelang` |
| `hc_attn_scale` | F32 `[3]` | R; `[3]` | mHC logits → pre/post/comb logits | no reduction | same mHC weight set | `DeepseekV4DecoderLayer.hc_attn_scale` |
| `hc_attn_base` | F32 `[24]` | R; `[24]` | mHC logits → pre/post/comb logits | no reduction | same mHC weight set | `DeepseekV4DecoderLayer.hc_attn_base` |
| Attention RMSNorm | BF16 `[4096]` | R; `[4096]` per rank | residual state → normalized `[4096]` layer input | norm FP32 accumulation, BF16 output; no TP reduction | `norm`, `dsv4_mhc_forward_hidden` | `attn_norm.weight`, fused norm argument to `mhc_pre_tilelang`/`mhc_fused_post_pre_tilelang` |
| Attention mHC transition state | residual BF16 `[4,4096]`; `post_mix` F32 `[4,1]`; `res_mix` F32 `[4,4]` per token | mutable, rank-local fixed buffers on both ranks | prior layer state → normalized attention input plus next-state mixes | no reduction inside transition | `device_mhc_forward_hidden`, future per-rank transition state machine | `mhc_pre_tilelang` and `mhc_fused_post_pre_tilelang` return values |
| `hc_ffn_fn` | F32 `[24,16384]` | R; full shape on each rank | attention result `[4096]` plus residual state → FFN normalized input | no reduction inside mHC | `dsv4_mhc_transition_device` / future rank-local equivalent | `DeepseekV4DecoderLayer.hc_ffn_fn`, `mhc_fused_post_pre_tilelang` |
| `hc_ffn_scale` | F32 `[3]` | R; `[3]` | FFN mHC logits → mix logits | no reduction | same mHC weight set | `DeepseekV4DecoderLayer.hc_ffn_scale` |
| `hc_ffn_base` | F32 `[24]` | R; `[24]` | FFN mHC logits → mix logits | no reduction | same mHC weight set | `DeepseekV4DecoderLayer.hc_ffn_base` |
| FFN RMSNorm | BF16 `[4096]` | R; `[4096]` per rank | post-attention branch → normalized router/FFN input | norm FP32 accumulation, BF16 output; no TP reduction | `dsv4_mhc_transition_device` | `ffn_norm.weight`, fused norm argument to `mhc_fused_post_pre_tilelang` |
| FFN mHC transition state | residual BF16 `[4,4096]`; `post_mix` F32 `[4,1]`; `res_mix` F32 `[4,4]` | mutable and rank-local; no `mhc_slot` owner | attention branch → router input and next residual state | consumes the completed attention reduction before this boundary | `dsv4_mhc_transition_device` | `DeepseekV4DecoderLayer.forward` second `mhc_fused_post_pre_tilelang` |
| Final mHC head | `hc_head_fn` F32 `[4,16384]`; scale F32 `[1]`; base F32 `[4]` | R; full shape on each rank | final four-copy state → `[4096]` | no TP reduction after the final hidden is already replicated | `enqueue_dsv4_mhc_finish_head_device` / future per-rank head | `DeepseekV4Model.hc_head_fn`, `hc_head_scale`, `hc_head_base`, `hc_head_fused_kernel_tilelang` |
| Final norm | BF16 `[4096]` | R; `[4096]` per rank | final mHC `[4096]` → normalized hidden `[4096]` | norm FP32 accumulation, BF16 output | `enqueue_dsv4_mhc_finish_head_device` | `DeepseekV4Model.norm` |

The mutable `residual`, `post_mix`, `res_mix`, hidden/branch workspaces,
transition state, status, and failure state are all duplicated as separate
rank-local fixed storage. The immutable mHC/norm views are replicated on both
ranks. No mHC operation or final dependency chain is owned by `mhc_slot` in
the intended mode.

### Attention and sparse page state

| Tensor/operation | Full shape and encoding | Class; rank-local shape | Input ownership → output ownership | Reduction boundary / arithmetic dtype | Strata function | Readable reference |
|---|---|---|---|---|---|---|
| `attn.wq_a` | FP8 E4M3 block-128 `[1024,4096]`; scale F8 E8M0 `[8,32]` | R; full source and scale on both ranks | replicated `[4096]` → replicated q-rank `[1024]` | none | current `Dsv4WeightCache::matmul`, future explicit replicated descriptor | `fused_wqa_wkv` q shard with `disable_tp=True` |
| `attn.wkv` | FP8 E4M3 block-128 `[512,4096]`; scale F8 E8M0 `[4,32]` | R; full source and scale on both ranks | replicated `[4096]` → replicated KV/score rank `[512]` | none | current compressor preparation path; future rank-local replicated descriptor | `fused_wqa_wkv` KV shard with `disable_tp=True` |
| Q/K low-rank norm and RoPE | `q_norm.weight` BF16 `[1024]`; `kv_norm.weight` BF16 `[512]`; RoPE tables FP32, rope width 64 | R; same shape per rank | replicated low-rank results → local query/KV inputs | no TP reduction | `dsv4_prepare_attention`, `prepare_attention` | `fused_q_kv_rmsnorm`, `_fused_qnorm_rope_kv_insert` |
| `attn.wq_b` | FP8 E4M3 block-128 `[32768,1024]`; scale F8 E8M0 `[256,8]` | C over output rows; `[16384,1024]` and scale `[128,8]` per rank | replicated q-rank → rank-local 32×512 query heads | none; local-head attention consumes the shard | `dsv4_prepare_attention` / future rank-local GEMM | `ColumnParallelLinear wq_b`, `n_local_heads=32` |
| Attention sinks | F32 `[64]` | C by head; `[32]` per rank | rank-local query heads → rank-local sink-aware attention | none | `physical_paged_attention` | `attn_sink` loader slices `head_rank_start:head_rank_end` |
| Main compressor `wkv`/`wgate` | Ratio 4: each plain BF16 `[1024,4096]`; ratio 128: each plain BF16 `[512,4096]` | R; exact full shape per rank; no quant scale tensor | replicated `[4096]` → replicated compressor KV/score outputs | none before rank-local page writes | `compress_state` / `dsv4_prepare_attention` | `DeepseekCompressor.fused_wkv_wgate`, `MergedColumnParallelLinear(disable_tp=True)` |
| Main compressor APE and norm | Ratio 4 APE F32 `[4,1024]`; ratio 128 APE F32 `[128,512]`; norm BF16 `[512]` | R immutable APE/norm; rank-local state destination | rank-local compressor outputs → rank-local compressed state | none | `compress_state` | `DeepseekCompressor.ape`, `.norm`, `CompressorStateCache` |
| Indexer `wq_b` | FP8 E4M3 block-128 `[8192,1024]`; scale F8 E8M0 `[64,8]` | R; full source and scale per rank | replicated q-rank → replicated 64×128 index query | none | `index_positions` / future rank-local indexer | `DeepseekV4Indexer.wq_b` (`ReplicatedLinear`) |
| Indexer `weights_proj` | BF16 `[64,4096]` | R; `[64,4096]` | replicated hidden → replicated indexer weights | none | `index_positions` | `DeepseekV4Indexer.weights_proj` (`ReplicatedLinear`) |
| Indexer compressor | Ratio-4: each plain BF16 `[256,4096]`; APE F32 `[4,256]`; norm BF16 `[128]` | R weights; rank-local indexer state | replicated hidden → rank-local page/index state | none | `index_positions`, `compress_state` | `DeepseekV4Indexer.compressor` and `DeepseekCompressor(head_dim=128)` |
| KV/page descriptors and page state | Main compressed row is the accepted exact physical-page format; rank-local page tables/descriptors and rank-local auxiliary state | R mathematical contents, distinct per-rank pages/descriptors/workspaces | rank-local compressed KV/index writes → rank-local sparse selection | no cross-rank reduction in page materialization | `physical_paged_attention`, `dsv4_paged_attention` | `DeepseekV4Attention.get_kv_cache_spec`, experiment 0068 physical-page oracle |
| Sparse selection | rank-local query `[32,512]`, index query `[64,128]`, exact page candidates/sinks/top-k metadata | rank-local execution; descriptors and selection buffers per rank | rank-local q/page state → rank-local attended `[32,512]` | none inside sparse attention | `dsv4_paged_attention` / `physical_paged_attention` | `attention_impl`, `DeepseekV4Indexer.forward`, sparse MLA backend |
| Attention output `wo_a` | FP8 E4M3 block-128 logical `[8192,4096]`, packed `[8192,4096]` U8; scale F8 E8M0 `[64,32]` | C over output rows; logical `[4096,4096]`, scale `[32,32]` per rank | rank-local 32-head attended groups → rank-local four-group `[4,1024]`/local output | none between `wo_a` and `wo_b`; source encoding must remain FP8 even if a runtime kernel converts it | `dsv4_paged_attention_to_mhc` / future rank-local `wo_a` | `ColumnParallelLinear wo_a`, `is_bmm=True`, `n_local_groups=4` |
| Attention output `wo_b` | FP8 E4M3 block-128 logical `[4096,8192]`, packed `[4096,8192]` U8; scale F8 E8M0 `[32,64]` | Q over input columns; logical `[4096,4096]`, scale `[32,32]` per rank | rank-local `wo_a` output → rank-local hidden partial `[4096]` | mandatory attention TP reduction, exact rank-ordered FP32 association then BF16 publication, before FFN mHC transition | future rank-local `dsv4_paged_attention_to_mhc` output path; current centralized output projection is not accepted | `RowParallelLinear wo_b`, `reduce_results=True` |

The exact attention association is therefore one `wo_b` reduction per layer,
not a reduction of replicated low-rank/KV projections and not a reduction
inside sparse selection.

### Router and MoE

| Tensor/operation | Full shape and encoding | Class; rank-local shape | Input ownership → output ownership | Reduction boundary / arithmetic dtype | Strata function | Readable reference |
|---|---|---|---|---|---|---|
| Router weight | BF16 `[256,4096]`, no scale tensor | R; `[256,4096]` per rank | replicated post-attention normalized `[4096]` → replicated raw logits `[256]` | router GEMM emits FP32 logits; no TP reduction | `route_moe` / future rank-local router | `DeepseekV4MoE.gate = GateLinear(..., out_dtype=torch.float32)` |
| Early hash table | I64 `[129280,6]` | R; full table per rank | replicated token id → replicated exact six expert IDs | no reduction; `8` is the I64 byte width, not a tensor dimension | `route_moe` | `DeepseekV4MoE.gate.tid2eid`, `fused_topk_bias` |
| Later correction bias | F32 `[256]` | R; `[256]` per rank | replicated raw logits → corrected scores | no reduction; exact scoring/normalization retained | `route_moe` | `DeepseekV4MoE.gate.e_score_correction_bias`, active matched header `ffn.gate.bias` |
| Top-k scoring and coefficients | six IDs and six coefficients per token; logits F32, coefficients kept at reference dtype/association | R control/state; each rank sees the same IDs and coefficients | replicated router input → replicated selected routes | exact `sqrtsoftplus`/bias/hash path, top-k 6, normalization, routed scaling; no hidden fallback | `route_moe` | `DeepseekV4MoE.forward`, `FusedMoE.select_experts`, matched `RoutedExperts` path |
| Shared expert `w1`/`w3` | Each FP8 E4M3 block-128 logical `[2048,4096]`, packed `[2048,4096]` U8; scale F8 E8M0 `[16,32]` | C over intermediate rows; logical `[1024,4096]`, scale `[8,32]` per rank | replicated router hidden → rank-local intermediate `[1024]` | no reduction until local down output | `enqueue_dsv4_host_moe_from_device_input` shared stream, future rank-local shared chain | `DeepseekV4MLP.gate_up_proj` (`MergedColumnParallelLinear`) |
| Shared expert `w2` | FP8 E4M3 block-128 logical `[4096,2048]`, packed `[4096,2048]` U8; scale `[32,16]` | Q over intermediate columns; logical `[4096,1024]`, scale `[32,8]` per rank | rank-local intermediate → rank-local hidden partial `[4096]` | joins local routed partial, then mandatory MoE TP reduction | `enqueue_dsv4_host_moe_from_device_input` shared stream, future rank-local join | `DeepseekV4MLP.down_proj` (`RowParallelLinear`) |
| Each routed expert `w1`/`w3` | FP4 E2M1 group-32; logical `[2048,4096]`; packed `[2048,2048]` U8; F8 E8M0 scale `[2048,128]` | intermediate-sharded by active matched backend; packed `[1024,2048]`, scale `[1024,128]` per rank and per expert | replicated hidden plus exact route coefficient → rank-local intermediate `[1024]` | no change to route coefficient; local down result retains expert/route association | `Dsv4TiledExpertWeights`, future rank-local CPU/GPU expert shard | active `RoutedExperts._load_w13`, `lk_moe` WNA16 configuration |
| Each routed expert `w2` | FP4 E2M1 group-32; logical `[4096,2048]`; packed `[4096,1024]` U8; scale `[4096,64]` | Q over intermediate columns; packed `[4096,512]`, scale `[4096,32]` per rank and per expert | rank-local intermediate → rank-local routed hidden partial `[4096]` | exact route coefficient applied once before down projection; no TP reduction until local join | `execute_host_routed_moe_callback`, future rank-local routed chain | active `RoutedExperts._load_w2`, `lk_moe` WNA16 configuration |
| Routed CPU destination | FP32 partial `[4096]` per rank, fixed destination and fixed status/route storage | rank-local; one destination per rank, never shared scratch | rank-local hidden/IDs/coefficients → rank-local routed partial | no cross-rank host continuation; error poisons local result and the later TP reduction | `execute_host_routed_moe_callback`, `collect_deepseek_moe` | `RoutedExperts._cpu_decode` and the selected `moe_runner.py` decode branch |
| Shared/routed overlap | shared down partial `[4096]` plus routed partial `[4096]` | rank-local auxiliary stream, join buffer, and status | same-rank shared and routed results → same-rank MoE partial `[4096]` | local exact BF16 join; then rank-ordered FP32 TP reduction and BF16 publication | `enqueue_dsv4_host_moe_from_device_input`, future rank-local join | `DeepseekV4MoE._maybe_apply_shared_experts`, `_maybe_reduce_final_output` |
| MoE output | rank-local hidden partial `[4096]`, BF16 publication | Q result on each rank; complete hidden `[4096]` on both after reduction | local joined partial → replicated next-layer hidden | mandatory second TP reduction per layer, after shared/routed join and before next mHC pre | future `tp_reduce_hidden`; current `collect_deepseek_moe` is centralized and not accepted | `FusedMoE._maybe_reduce_final_output` / next `DeepseekV4DecoderLayer.forward` boundary |

The layer therefore has exactly two hidden-width TP reductions: `wo_b` after
attention and the joined shared/routed MoE output. Across 43 layers this is the
86-reduction transport arm. Embedding and output communication are separate
from that callback-chain scope.

## Active routed-expert backend evidence

The map does not use the readable repository's `DeepseekV4MegaMoEExperts` as
evidence. That class is an expert-parallel MegaMoE backend with a different
launcher/configuration and is not the matched installed path.

The active matched path is:

1. `RoutedExperts` conditionally imports the installed `lk_moe` integration
   when `LVLLM_MOE_NUMA_ENABLED` is enabled.
2. Its loader maps `w1/w3` on dimension 0 and `w2` on dimension 1 through
   `_load_w13` and `_load_w2`, using the TP rank slice. The active matched
   configuration has `use_ep=False`, so `_get_processes_info()` returns
   `tp_size/tp_rank`, not expert-parallel ownership.
3. `_process_wna16` passes `num_processes`, `process_id`, `expert_num`,
   `hidden_size=4096`, and `intermediate_size=intermediate_size_per_partition`
   into the installed `lk_moe` configuration.
4. The measured/reference matched configuration initializes all 256 logical
   experts on each rank and sets `intermediate_size_per_partition=2048/2=1024`.
   Thus every rank owns the intermediate shard of every expert; it does not
   own only 128 experts. The readable source citation for this execution path
   is `RoutedExperts` plus the selected `_cpu_decode` branch in
   `moe_runner.py`, while experiment 0059 records the matched installed
   integration's 256-experts-per-rank configuration.

The proprietary `lk_moe` implementation is not inspected beyond its readable
Python integration and measured behavior.

## Quantized packed and scale-slice contract

The source format uses one-byte FP8 E4M3 values with one-byte F8 E8M0 scales
for 128×128 blocks, and two FP4 E2M1 values per packed byte with one F8 E8M0
scale per 32 logical input values. `src/deepseek_manifest.cpp` validates these
source shapes. The scale is a separate tensor and is never inferred from a
weight byte count.

For every row-major source tensor, let `W0` and `S0` be the manifest's absolute
safetensors payload offsets for `<base>.weight` and `<base>.scale`. The table
below gives the exact relative slice offsets and byte counts. Absolute `W0` and
`S0` are checkpoint-shard-specific values read from the manifest; no absolute
offset is guessed from tensor order.

| Tensor family | Full logical shape / encoding | Packed storage and full scales | Shard axis and exact rank-local storage | Relative slice offsets/counts; alignment |
|---|---|---|---|---|
| `attn.wq_a` | `[1024,4096]` FP8 E4M3 block-128 | weight `[1024,4096]` U8 = 4,194,304 B; scale `[8,32]` F8 E8M0 = 256 B | R; each rank gets the full 4,194,304-B weight and 256-B scale | `W_r=W0`, `S_r=S0`; replicated, not sharded; both dimensions are 128-aligned |
| `attn.wkv` | `[512,4096]` FP8 E4M3 block-128 | weight `[512,4096]` U8 = 2,097,152 B; scale `[4,32]` = 128 B | R; full on both ranks | `W_r=W0`, `S_r=S0`; 128-aligned |
| `attn.wq_b` | `[32768,1024]` FP8 E4M3 block-128 | weight = 33,554,432 B; scale `[256,8]` = 2,048 B | C rows; weight `[16384,1024]` = 16,777,216 B; scale `[128,8]` = 1,024 B | `W_r=W0+r*16,777,216`, `S_r=S0+r*1,024`; row and scale-block boundaries align at 128 |
| `indexer.wq_b` | `[8192,1024]` FP8 E4M3 block-128 | weight = 8,388,608 B; scale `[64,8]` = 512 B | R; full on both ranks | `W_r=W0`, `S_r=S0`; replicated indexer |
| `attn.wo_a` | `[8192,4096]` FP8 E4M3 block-128 | weight = 33,554,432 B; scale `[64,32]` = 2,048 B | C rows; weight `[4096,4096]` = 16,777,216 B; scale `[32,32]` = 1,024 B | `W_r=W0+r*16,777,216`, `S_r=S0+r*1,024`; source stays FP8 even if a kernel uses a BF16 conversion workspace |
| `attn.wo_b` | `[4096,8192]` FP8 E4M3 block-128 | weight = 33,554,432 B; scale `[32,64]` = 2,048 B | Q columns; logical `[4096,4096]`, weight slice 16,777,216 B; scale `[32,32]` = 1,024 B | per row `W_r(i)=W0+i*8192+r*4096`, count 4,096 B; per block-row `S_r(b)=S0+b*64+r*32`, count 32 B; each slice is 128-aligned |
| Shared `w1`, `w3` (each) | `[2048,4096]` FP8 E4M3 block-128 | weight = 8,388,608 B; scale `[16,32]` = 512 B | C rows; weight `[1024,4096]` = 4,194,304 B; scale `[8,32]` = 256 B per rank | `W_r=W0+r*4,194,304`, `S_r=S0+r*256`; intermediate row boundary and scale rows align |
| Shared `w2` | `[4096,2048]` FP8 E4M3 block-128 | weight = 8,388,608 B; scale `[32,16]` = 512 B | Q columns; weight `[4096,1024]` = 4,194,304 B; scale `[32,8]` = 256 B per rank | per row `W_r(i)=W0+i*2048+r*1024`, count 1,024 B; per block-row `S_r(b)=S0+b*16+r*8`, count 8 B |
| Routed `w1`, `w3` (each expert) | logical `[2048,4096]` FP4 E2M1 group-32 | packed `[2048,2048]` U8 = 4,194,304 B; scale `[2048,128]` F8 E8M0 = 262,144 B | C rows; packed `[1024,2048]` = 2,097,152 B; scale `[1024,128]` = 131,072 B per rank/expert | `W_r=W0+r*2,097,152`, `S_r=S0+r*131,072`; 32-column logical groups remain intact and packed columns are byte aligned |
| Routed `w2` (each expert) | logical `[4096,2048]` FP4 E2M1 group-32 | packed `[4096,1024]` U8 = 4,194,304 B; scale `[4096,64]` = 262,144 B | Q columns; packed `[4096,512]` = 2,097,152 B; scale `[4096,32]` = 131,072 B per rank/expert | per row `W_r(i)=W0+i*1024+r*512`, count 512 B; scale `S_r(i)=S0+i*64+r*32`, count 32 B; each 32-logical-value group is wholly local |

For all column-parallel rows, rank 0 and rank 1 concatenation reconstructs the
full source tensor. For row-parallel tensors, reconstruction interleaves each
rank's column slice within every row/block row. A Stage-3 loader must therefore
load the weight and scale slices together using these offsets; loading the full
tensor and discarding half is not TP ownership. The `wo_a` conversion in the
current Strata checkpoint path is an implementation detail that cannot change
the source encoding or bypass source-scale validation.

### Representative source-byte accounting

These are source-format bytes before any runtime workspace conversion:

| Scope | Full source bytes | Rank-local source bytes |
|---|---:|---:|
| One shared expert layer, all three FP8 weights plus scales | 25,167,360 B | 12,583,680 B |
| One routed expert, all three FP4 weights plus scales | 13,369,344 B | 6,684,672 B |
| 256 routed experts × 43 layers, active matched intermediate-sharded path | 147,169,738,752 B aggregate full; each rank receives half | 73,584,869,376 B/rank |
| `embed.weight` or `head.weight` | 1,059,061,760 B | 529,530,880 B |

The accepted full-context KV/state contract remains 151,228,416 B before
rank-local runtime workspaces. The endpoint and hidden-reduction probe arenas
are 412,428 B/rank and 57,356 B/rank respectively; the combined endpoint plus
layer fixed arena is 469,784 B/rank. These are application buffers, not NCCL's
opaque communicator allocation.

## Stage 3 implementation: explicit rank-shard descriptors

Stage 3 is implemented in the checkpoint-loading layer, without changing the
runtime execution path. `Dsv4RankShardDescriptor` in
`include/strata/deepseek_rank_shard.hpp` identifies the ownership class
(`Replicated`, `ContiguousRows`, or `StridedColumns`), rank/world size, shard
axis, logical and packed source shapes, source dtypes/encodings, full and
local scale shapes, absolute source offsets, local byte counts, block/group
alignment, and the exact weight and scale slice list. A slice contains both
the absolute safetensors offset and its relative payload offset/count; the
destination offset makes the rank-local payload contiguous.

`describe_dsv4_rank_shard()` resolves a quantized weight and its scale as one
validated contract. It rejects invalid rank/world, divisibility, FP8 block or
FP4 group boundaries, packed/logical shape mismatch, missing or mismatched
scales, source byte/count overflow, and non-contiguous or misaligned slices.
`load_dsv4_rank_shard()` allocates only the declared local weight and scale
vectors and fills them through `Dsv4CheckpointReader::read_slice_into()`.
Weight and scale reads are atomic at the API boundary: any failed read clears
both vectors and returns an error. It never calls the old full-tensor read for
a sharded module. Plain one-dimensional state/bias/norm tensors are accepted
only as explicitly replicated descriptors.

The descriptor is deliberately a loader contract, not a hidden runtime
fallback. The current production `load_dsv4_cuda_linear()` path and the
centralized runtime remain unchanged; rank-local attention, NCCL integration,
and full-model execution are Stage 4+ work and were not started.

### Actual target-format fixture gate

The fixture app and focused test use the read-only target checkpoint at
`models/dsv4f` (passed by absolute path from the clean worktree). They cover
68 early/middle/late tensors: `embed`/`head`; layers 2, 21, and 42 attention
ratio-4/indexer and ratio-128 families; `wq_b`, `wo_a`, `wo_b`; shared and
routed `w1/w3/w2`; router hash/bias; norms; all mHC function/scale/base
weights; and final norm/head mHC state. The actual headers, not generated
shape-reduced data, supply dtype, shape, source offset, and scale metadata.

Representative raw header/slice evidence includes:

| fixture | source header and absolute offsets | TP2 local payload per rank |
|---|---|---:|
| `embed.weight` | BF16 `[129280,4096]`, offset `96`, 1,059,061,760 B | 529,530,880 B contiguous rows |
| `head.weight` | BF16 `[129280,4096]`, offset `262564`, 1,059,061,760 B | 529,530,880 B contiguous rows |
| `layers.2.attn.wq_b` | FP8 `[32768,1024]`, scale `[256,8]`; weight offset `316283544`, scale offset `33164440` | weight 16,777,216 B + scale 1,024 B |
| `layers.21.attn.wo_a` | FP8 `[8192,4096]`, scale `[64,32]`; weight offset `217521648`, scale offset `14089968` | weight 16,777,216 B + scale 1,024 B |
| `layers.42.attn.wo_b` | FP8 `[4096,8192]`, scale `[32,64]`; weight offset `272331888`, scale offset `26959216`; strided per-row/per-block slices | weight 16,777,216 B + scale 1,024 B |
| `layers.2.ffn.experts.0.w1` | FP4 source I8 packed `[2048,2048]`, scale `[2048,128]`; weight offset `375003800`, scale offset `33166488` | packed weight 2,097,152 B + scale 131,072 B |
| `layers.42.ffn.experts.0.w2` | FP4 source I8 packed `[4096,1024]`, scale `[4096,64]`; weight offset `372995184`, scale offset `27225712` | packed weight 2,097,152 B + scale 131,072 B |

For each fixture the test loads rank 0 and rank 1, compares every loaded
weight/scale slice against the actual checkpoint in 8-MiB chunks, verifies
disjoint rank coverage and complete reconstruction (concatenation for row
shards, per-row interleaving for column shards), and checks the declared
loader call/byte counts. The negative descriptor cases cover malformed rank,
world size, row divisibility, source/scale truncation, absolute offset,
scale alignment, packed precision, and block/group shape contracts.

Raw fixture output:
`results/dsv4-rank-local-tp2/rank-shard-fixtures/rank-shard-probe.txt`.

The result is `summary=pass fixtures=68`; the two-rank fixture loader payload
is 2,701,892,824 B and exactly equals the two-rank source oracle byte count;
the checkpoint reader made 197,981 slice reads totaling 5,403,785,656 B,
including the independent source comparisons. No full sharded tensor
temporary was allocated. Focused tests report `237/270 tests passed, 33
skipped`, with both new TP2 cases passing and no failure.

### Stage 3 memory closure

The following is the manifest-wide rank-0 projection for the target format.
Quantized scales are included with their weight in every row; routed experts
are host-only in this mode and are not counted as CUDA VRAM. The `wo_a` source
remains FP8 locally and its accepted runtime conversion is counted as a
BF16-only local output, never as a full-tensor conversion.

| resource | bytes / rank or aggregate | accounting status |
|---|---:|---|
| checkpoint shard files, file-backed source | 166,886,535,336 aggregate | read-only source; not resident canonical heap |
| checkpoint tensor payload | 166,878,536,440 aggregate | manifest header total |
| canonical/mapped resident host bytes in slice loader | 0 | loader uses direct read-only slices, not a full mmap arena |
| routed-expert source bytes | 147,169,738,752 aggregate | host only; no VRAM label |
| transformed routed payload | 73,584,869,376 per rank | rank 0 NUMA 1, rank 1 NUMA 0 by matched 0059 live placement evidence |
| replicated CUDA parameters | 1,316,853,468 per rank | 939 explicit replicated modules |
| sharded CUDA source parameters excluding `wo_a` | 3,043,088,640 per rank | 217 explicit row/column modules |
| local `wo_a` FP8 weight + scale input | 721,464,320 per rank | 43 local source contracts |
| local `wo_a` BF16 converted output | 1,442,840,576 per rank | 43 × local `[4096,4096]` BF16 outputs |
| rank-local encoded loader payload | 5,081,406,428 per rank | no full sharded temporary |
| rank-local CUDA parameter projection | 5,802,782,684 per rank | replicated + sharded + BF16 `wo_a` output |
| immutable mHC views | 135,537,756 per rank | explicitly replicated; included above |
| norms | 712,704 per rank | explicitly replicated; included above |
| router/hash state | 108,834,816 per rank | explicitly replicated; included above |
| retained rank-local KV/page state | 151,228,416 per rank | accepted contract, duplicated by rank |
| mutable rank-local mHC state | 357,072 per rank | fixed residual/transition auxiliary state |
| combined NCCL/application fixed arena | 469,784 per rank | endpoint + hidden buffers; 24 B status inside |
| rank-local stream/event handles | 2 streams + 1 final event | CUDA driver bytes opaque, not silently zero |
| admission workspace reserve | 268,435,456 per rank | 256 MiB reserve |
| peak loader temporary | 529,530,880 per rank | largest actual fixture is embed/head; no full tensor |
| projected VRAM including KV/mHC/transport/workspace | 6,223,273,412 per rank; 12,446,546,824 aggregate | pass |
| accepted 0.95 per-GPU ceiling | 21,287,272,448 per rank | projected pass |
| fixture probe peak RSS | 1,261,613,056 one process | actual slice-loader process |
| two-rank fixture RSS equivalent | 2,523,226,112 | conservative source + fixture projection 149,692,964,864 B |
| declared host ceiling | 231,928,233,984 | 216 GiB; conservative projection passes |

The matched external transformed-arena run recorded approximately 77,931.7
MiB and 77,954.6 MiB total private process RSS for TP0/TP1, with the routed
payload predominantly on NUMA 1/0 respectively. Those process totals include
runtime overhead and are not relabeled as exact arena bytes; the exact
rank-local transformed payload contract is the 73,584,869,376-B figure above.

Stage 3 therefore passes its reconstruction, precision, atomic weight/scale,
no-full-temporary, and projected memory gates. It does not claim a balanced
weight lease or a device-resident runtime result because no runtime execution
was authorized in this stage.

## Embedding and output communication contracts

### Embedding

The reference instantiates `VocabParallelEmbedding`, slices vocabulary rows,
masks non-owning ranks, and calls `tensor_model_parallel_all_reduce` on the
local BF16 embedding output. For TP2 the local row interval is rank 0
`[0,64640)` and rank 1 `[64640,129280)`. The resulting complete BF16 `[4096]`
vector is then expanded separately to `[4,4096]` by mHC. This is one
communication association outside the 86 layer-local hidden reductions.

### Output head

`head.weight` is BF16 `[129280,4096]`; `ParallelLMHead` owns rank-local
vocabulary rows `[64640,4096]`. The matched CUDA platform's
`current_platform.use_all_gather()` returns `True`, so `LogitsProcessor` calls
`tensor_model_parallel_all_gather`, not a rank-0-only gather. Each rank computes
BF16 local logits `[64640]`; the all-gather publishes BF16 `[129280]` on both
ranks in rank order:

```
rank 0: token ids [0,     64640)  -> output positions [0,     64640)
rank 1: token ids [64640,129280)  -> output positions [64640,129280)
```

The application-issued communication is one NCCL BF16 all-gather with
258,560 input bytes across both ranks and 517,120 output bytes across both
ranks, plus one fixed status collective in the probe. The output dtype is
BF16: the target head is BF16 and the readable unquantized linear path has no
logit upcast before `LogitsProcessor` gathers it. Any collective or rank-local
failure poisons the full logits buffer on both ranks; no sampler may consume a
partial output. If a future matched sampler changes to rank-0 gather, that is a
separate contract change requiring a new source trace and byte gate.

## Boundary sequence authorized by this map

For each rank and each layer, the only accepted sequence is:

```
rank-local mHC pre/norm
  -> rank-local attention (32 heads, rank-local pages)
  -> rank-ordered TP reduction of wo_b
  -> rank-local mHC transition/norm
  -> rank-local router
  -> rank-local CPU routed shard + GPU shared shard overlap
  -> rank-local routed/shared join
  -> rank-ordered TP reduction of the MoE output
  -> rank-local mHC post/next-layer state
```

The two ranks have separate hidden/residual state, KV/page descriptors,
auxiliary streams, fixed workspaces, callback/status storage, failure state,
and dependent stream chains. Hidden state is replicated only after the
embedding association, after `wo_b`, after the MoE join, and wherever the
exact output head contract requires it. No shared expert, routed destination,
mHC state, or final dependency chain is centralized on `mhc_slot`.

## Stage gate

Stage 1 is a checked ownership/encoding contract and Stage 2 is accepted: the
installed NCCL FP32 86-reduction arm measured 2.845 ms net with exactness and
the corrected masked embedding ownership cases passed. Stage 3 now passes the
explicit descriptor, actual-format fixture, and projected memory gates. Stage
4 rank-local attention, Stage 5 MoE, Stage 6 mHC/runtime integration, and
Stages 7–10 remain stopped. No rank-sharded runtime execution or production
NCCL integration is included in this change.
