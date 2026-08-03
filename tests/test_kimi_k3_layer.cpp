#include "kimi_fixture.hpp"
#include "test.hpp"

#include "strata/compressed_tensors.hpp"
#include "strata/kimi_k3_checkpoint.hpp"
#include "strata/kimi_k3_expert_arena.hpp"
#include "strata/kimi_k3_layer.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

// Gate 4: one KDA layer and one gated MLA layer, real weights, against the
// checkpoint's own reference implementation for the same input.
//
// The fixture is produced by `scripts/run_kimi_k3_reference_fixture.sh`, which
// runs the real `KimiDeltaAttention` and `KimiMLAAttention` out of
// `/data/kimi-k3/modeling_kimi_linear.py` on a GPU. The comparison is therefore
// against the reference itself, not against a transcription of it.
//
// Tolerance, stated before the run rather than fitted to it: the reference
// stores activations in bfloat16 between operations while this path keeps F32,
// so every op boundary contributes about 2^-9 = 2.0e-3 of relative error and a
// layer crosses roughly six of them. Accumulating as a random walk that is
// ~5e-3 relative L2. The gate is set at 2.0e-2 relative L2 and 0.999 cosine:
// loose enough that bfloat16 rounding passes, tight enough that a transposed
// weight, a permuted head, a dropped gate, or an off-by-one in the convolution
// history — all of which are order-one errors — cannot.
//
// Measuring materially *more* than 5e-3 is a defect to investigate, not a
// datapoint to report, and measuring near zero would mean the fixture is not
// being read.

namespace {

using kimi_test::Agreement;
using kimi_test::compare;
using kimi_test::Fixture;
using kimi_test::kimi_directory;
using kimi_test::kimi_present;

constexpr float kRelativeL2Gate = 2.0e-2F;
constexpr float kCosineGate = 0.999F;

std::string fixture_path() {
    return kimi_test::fixture_path("kimi-k3-layers.fixture");
}

// The measured agreement is printed, not only gated: a number inside the gate
// but far from the predicted 5e-3 is still something to explain.
void report(const char* what, const Agreement& agreement) {
    std::cout << "  [gate 4] " << what << " relative L2 "
              << agreement.relative_l2 << ", cosine " << agreement.cosine
              << '\n';
}

// A BF16 tensor read straight out of the checkpoint, kept in its encoding
// exactly as the runtime keeps it.
struct RawTensor {
    std::vector<std::byte> bytes;
    strata::KimiBf16Matrix matrix;
};

bool read_bf16(const strata::KimiCheckpointReader& reader, const std::string& name,
               std::uint32_t rows, std::uint32_t columns, RawTensor& out) {
    const auto elements = static_cast<std::uint64_t>(rows) * columns;
    auto raw = reader.read(name, elements * 2U);
    if (!raw.ok() || raw.value.size() != elements * 2U) return false;
    out.bytes = std::move(raw.value);
    out.matrix.values = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(out.bytes.data()), elements);
    out.matrix.rows = rows;
    out.matrix.columns = columns;
    return true;
}

bool read_f32(const strata::KimiCheckpointReader& reader, const std::string& name,
              std::uint64_t elements, std::vector<float>& out) {
    auto raw = reader.read_f32(name, elements);
    if (!raw.ok() || raw.value.size() != elements) return false;
    out = std::move(raw.value);
    return true;
}

// Routed experts out of the real arena and the real coalescing reader, so the
// MoE fixture below gates the slot layout and the MXFP4 decode as well as the
// block's semantics. Mirrors `ArenaExpertSource` in the runtime; the duplication
// is deliberate — a test that reused the runtime's private class could not
// notice if the runtime and the arena disagreed on the slot layout.
class ArenaExperts final : public strata::KimiExpertSource {
public:
    ArenaExperts(const strata::KimiCheckpointReader& checkpoint,
                 strata::KimiExpertArena& arena,
                 strata::KimiExpertReader& reader)
        : checkpoint_(&checkpoint), arena_(&arena), reader_(&reader) {}

    strata::ValidationResult prepare(
        std::uint32_t layer, std::span<const std::uint32_t> experts) override {
        requested_.assign(experts.begin(), experts.end());
        std::vector<strata::KimiReadRequest> requests;
        requests.reserve(experts.size());
        for (const auto expert : experts) {
            requests.push_back(strata::KimiReadRequest{layer, expert});
        }
        return reader_->stage(*checkpoint_, *arena_, requests);
    }

    strata::ValidationResult fetch(std::uint32_t layer, std::uint32_t expert,
                                   strata::KimiExpertWeights& weights) override {
        strata::ValidationResult result;
        const auto slot = arena_->find(layer, expert);
        if (slot.empty()) {
            result.errors.emplace_back("expert is not resident after staging");
            return result;
        }
        strata::KimiExpertModules modules{};
        if (!checkpoint_->expert_modules(layer, expert, modules)) {
            result.errors.emplace_back("expert is not in the checkpoint");
            return result;
        }
        const auto layout = strata::kimi_expert_slot_layout(modules);
        const auto& c = strata::kKimiK3ExecutionContract;
        const auto* base = reinterpret_cast<const std::uint8_t*>(slot.data());
        const auto view = [&](std::uint64_t packed, std::uint64_t scale,
                              std::uint32_t rows, std::uint32_t columns) {
            strata::KimiExpertModuleView module;
            module.rows = rows;
            module.columns = columns;
            module.packed = std::span<const std::uint8_t>(
                base + packed, static_cast<std::size_t>(rows) * (columns / 2U));
            module.scales = std::span<const std::uint8_t>(
                base + scale, static_cast<std::size_t>(rows) * (columns / 32U));
            return module;
        };
        weights.gate = view(layout.gate_packed, layout.gate_scale,
                            c.expert_intermediate_size,
                            c.routed_expert_hidden_size);
        weights.up = view(layout.up_packed, layout.up_scale,
                          c.expert_intermediate_size, c.routed_expert_hidden_size);
        weights.down = view(layout.down_packed, layout.down_scale,
                            c.routed_expert_hidden_size,
                            c.expert_intermediate_size);
        return result;
    }

    [[nodiscard]] const std::vector<std::uint32_t>& requested() const noexcept {
        return requested_;
    }

private:
    const strata::KimiCheckpointReader* checkpoint_{};
    strata::KimiExpertArena* arena_{};
    strata::KimiExpertReader* reader_{};
    std::vector<std::uint32_t> requested_;
};

// The MoE block is exercised by its own fixtures; a layer probe supplies no
// experts and reports the fact rather than inventing weights.
class NoExperts : public strata::KimiExpertSource {
public:
    strata::ValidationResult prepare(std::uint32_t,
                                     std::span<const std::uint32_t>) override {
        strata::ValidationResult result;
        result.errors.emplace_back("no expert source is bound");
        return result;
    }
    strata::ValidationResult fetch(std::uint32_t, std::uint32_t,
                                   strata::KimiExpertWeights&) override {
        strata::ValidationResult result;
        result.errors.emplace_back("no expert source is bound");
        return result;
    }
};

}  // namespace

TEST_CASE("Kimi-K3 KDA layer matches the reference for prefill and decode") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    Fixture fixture;
    if (!fixture.load(fixture_path())) {
        SKIP("layer fixture is absent; run scripts/run_kimi_k3_reference_fixture.sh");
    }
    const auto& c = strata::kKimiK3ExecutionContract;
    const std::uint32_t layer = 0U;
    const auto prefix = "kda." + std::to_string(layer) + ".";

    const auto* input = fixture.find(prefix + "input");
    const auto* expected = fixture.find(prefix + "prefill_output");
    const auto* decode_input = fixture.find(prefix + "decode_input");
    const auto* decode_expected = fixture.find(prefix + "decode_output");
    REQUIRE(input != nullptr);
    REQUIRE(expected != nullptr);
    REQUIRE(decode_input != nullptr);
    REQUIRE(decode_expected != nullptr);
    const auto tokens = static_cast<std::uint32_t>(input->shape.at(0));
    REQUIRE(input->shape.at(1) == c.hidden_size);

    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& reader = *opened.value;
    const auto base =
        "language_model.model.layers." + std::to_string(layer) + ".self_attn.";
    const auto projection = c.linear_attention_heads * c.linear_head_dim;

    RawTensor q_proj, k_proj, v_proj, g_proj, o_proj, f_a, f_b, b_proj;
    REQUIRE(read_bf16(reader, base + "q_proj.weight", projection, c.hidden_size, q_proj));
    REQUIRE(read_bf16(reader, base + "k_proj.weight", projection, c.hidden_size, k_proj));
    REQUIRE(read_bf16(reader, base + "v_proj.weight", projection, c.hidden_size, v_proj));
    REQUIRE(read_bf16(reader, base + "g_proj.weight", projection, c.hidden_size, g_proj));
    REQUIRE(read_bf16(reader, base + "o_proj.weight", c.hidden_size, projection, o_proj));
    REQUIRE(read_bf16(reader, base + "f_a_proj.weight", c.linear_head_dim,
                      c.hidden_size, f_a));
    REQUIRE(read_bf16(reader, base + "f_b_proj.weight", projection,
                      c.linear_head_dim, f_b));
    REQUIRE(read_bf16(reader, base + "b_proj.weight", c.linear_attention_heads,
                      c.hidden_size, b_proj));

    std::vector<float> q_conv, k_conv, v_conv, a_log_padded, dt_bias, o_norm;
    const auto taps = static_cast<std::uint64_t>(projection) * c.short_conv_kernel;
    REQUIRE(read_f32(reader, base + "q_conv1d.weight", taps, q_conv));
    REQUIRE(read_f32(reader, base + "k_conv1d.weight", taps, k_conv));
    REQUIRE(read_f32(reader, base + "v_conv1d.weight", taps, v_conv));
    REQUIRE(read_f32(reader, base + "A_log", c.linear_head_dim, a_log_padded));
    REQUIRE(read_f32(reader, base + "dt_bias", projection, dt_bias));
    REQUIRE(read_f32(reader, base + "o_norm.weight", c.linear_head_dim, o_norm));
    // `A_log` is a per-head scalar zero-padded to head_dim. Reading it per
    // channel would apply exp(0) = 1 to a quarter of the channels, so the pad
    // is asserted before it is dropped.
    for (std::uint32_t index = c.linear_attention_heads;
         index < c.linear_head_dim; ++index) {
        REQUIRE(a_log_padded[index] == 0.0F);
    }
    std::vector<float> a_log(a_log_padded.begin(),
                             a_log_padded.begin() + c.linear_attention_heads);

    strata::KimiKdaWeights weights;
    weights.q_proj = q_proj.matrix;
    weights.k_proj = k_proj.matrix;
    weights.v_proj = v_proj.matrix;
    weights.g_proj = g_proj.matrix;
    weights.o_proj = o_proj.matrix;
    weights.f_a_proj = f_a.matrix;
    weights.f_b_proj = f_b.matrix;
    weights.b_proj = b_proj.matrix;
    weights.q_conv = q_conv;
    weights.k_conv = k_conv;
    weights.v_conv = v_conv;
    weights.a_log = a_log;
    weights.dt_bias = dt_bias;
    weights.o_norm = o_norm;

    strata::KimiStateCache cache;
    REQUIRE(cache.reset(tokens + 8U).ok());
    strata::KimiLayerScratch scratch;
    std::vector<float> output(static_cast<std::size_t>(tokens) * c.hidden_size);
    REQUIRE(strata::kimi_kda_layer(output, input->values, weights, cache, layer,
                                   tokens, scratch)
                .ok());

    const auto prefill = compare(output, expected->values);
    report("KDA prefill", prefill);
    REQUIRE(prefill.relative_l2 < kRelativeL2Gate);
    REQUIRE(prefill.cosine > kCosineGate);

    // The recurrent state itself, head by head. The decode output below already
    // depends on it, but comparing it directly says whether a failure is in the
    // recurrence or in what reads it. The reference stores `[heads, value, key]`
    // because `transpose_state_layout` is set, which is the layout this cache
    // uses too.
    const auto* reference_state = fixture.find(prefix + "prefill_state");
    REQUIRE(reference_state != nullptr);
    REQUIRE(reference_state->shape.at(0) == c.linear_attention_heads);
    REQUIRE(reference_state->shape.at(1) == c.linear_head_dim);
    REQUIRE(reference_state->shape.at(2) == c.linear_head_dim);
    const auto state_span =
        static_cast<std::size_t>(c.linear_head_dim) * c.linear_head_dim;
    std::vector<float> measured_state;
    measured_state.reserve(c.linear_attention_heads * state_span);
    for (std::uint32_t head = 0U; head < c.linear_attention_heads; ++head) {
        const auto slice = cache.recurrent_state(layer, head);
        REQUIRE(slice.size() == state_span);
        measured_state.insert(measured_state.end(), slice.begin(), slice.end());
    }
    const auto state_agreement = compare(measured_state, reference_state->values);
    report("KDA prefill state", state_agreement);
    REQUIRE(state_agreement.relative_l2 < kRelativeL2Gate);
    REQUIRE(state_agreement.cosine > kCosineGate);

    // The same state then carries one decode token. Prefill and decode share
    // one definition here, so this checks the handoff the reference performs by
    // switching from its chunk kernel to its recurrent one.
    std::vector<float> step(c.hidden_size);
    REQUIRE(strata::kimi_kda_layer(step, decode_input->values, weights, cache,
                                   layer, 1U, scratch)
                .ok());
    const auto decode = compare(step, decode_expected->values);
    report("KDA decode", decode);
    REQUIRE(decode.relative_l2 < kRelativeL2Gate);
    REQUIRE(decode.cosine > kCosineGate);
}

TEST_CASE("Kimi-K3 gated MLA layer matches the reference for prefill and decode") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    Fixture fixture;
    if (!fixture.load(fixture_path())) {
        SKIP("layer fixture is absent; run scripts/run_kimi_k3_reference_fixture.sh");
    }
    const auto& c = strata::kKimiK3ExecutionContract;
    const std::uint32_t layer = 3U;
    REQUIRE(strata::kimi_k3_full_attention_layer(layer));
    const auto prefix = "mla." + std::to_string(layer) + ".";

    const auto* input = fixture.find(prefix + "input");
    const auto* expected = fixture.find(prefix + "prefill_output");
    const auto* decode_input = fixture.find(prefix + "decode_input");
    const auto* decode_expected = fixture.find(prefix + "decode_output");
    REQUIRE(input != nullptr);
    REQUIRE(expected != nullptr);
    REQUIRE(decode_input != nullptr);
    REQUIRE(decode_expected != nullptr);
    const auto tokens = static_cast<std::uint32_t>(input->shape.at(0));

    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& reader = *opened.value;
    const auto base =
        "language_model.model.layers." + std::to_string(layer) + ".self_attn.";
    const auto query_dim = c.nope_head_dim + c.rope_head_dim;
    const auto value_span = c.attention_heads * c.value_head_dim;

    RawTensor q_a, q_b, kv_a, kv_b, o_proj, g_proj;
    REQUIRE(read_bf16(reader, base + "q_a_proj.weight", c.query_lora_rank,
                      c.hidden_size, q_a));
    REQUIRE(read_bf16(reader, base + "q_b_proj.weight",
                      c.attention_heads * query_dim, c.query_lora_rank, q_b));
    REQUIRE(read_bf16(reader, base + "kv_a_proj_with_mqa.weight",
                      c.kv_lora_rank + c.rope_head_dim, c.hidden_size, kv_a));
    REQUIRE(read_bf16(reader, base + "kv_b_proj.weight",
                      c.attention_heads * (c.nope_head_dim + c.value_head_dim),
                      c.kv_lora_rank, kv_b));
    REQUIRE(read_bf16(reader, base + "o_proj.weight", c.hidden_size, value_span,
                      o_proj));
    REQUIRE(read_bf16(reader, base + "g_proj.weight", value_span, c.hidden_size,
                      g_proj));

    std::vector<float> q_a_norm, kv_a_norm;
    REQUIRE(read_f32(reader, base + "q_a_layernorm.weight", c.query_lora_rank,
                     q_a_norm));
    REQUIRE(read_f32(reader, base + "kv_a_layernorm.weight", c.kv_lora_rank,
                     kv_a_norm));

    strata::KimiMlaWeights weights;
    weights.q_a_proj = q_a.matrix;
    weights.q_b_proj = q_b.matrix;
    weights.kv_a_proj = kv_a.matrix;
    weights.kv_b_proj = kv_b.matrix;
    weights.o_proj = o_proj.matrix;
    weights.g_proj = g_proj.matrix;
    weights.q_a_norm = q_a_norm;
    weights.kv_a_norm = kv_a_norm;

    strata::KimiStateCache cache;
    REQUIRE(cache.reset(tokens + 8U).ok());
    strata::KimiLayerScratch scratch;
    std::vector<float> output(static_cast<std::size_t>(tokens) * c.hidden_size);
    REQUIRE(strata::kimi_mla_layer(output, input->values, weights, cache, layer,
                                   0U, tokens, scratch)
                .ok());
    const auto prefill = compare(output, expected->values);
    report("MLA prefill", prefill);
    REQUIRE(prefill.relative_l2 < kRelativeL2Gate);
    REQUIRE(prefill.cosine > kCosineGate);

    REQUIRE(cache.commit(tokens - 1U).ok());
    std::vector<float> step(c.hidden_size);
    REQUIRE(strata::kimi_mla_layer(step, decode_input->values, weights, cache,
                                   layer, tokens, 1U, scratch)
                .ok());
    const auto decode = compare(step, decode_expected->values);
    report("MLA decode", decode);
    REQUIRE(decode.relative_l2 < kRelativeL2Gate);
    REQUIRE(decode.cosine > kCosineGate);
}

TEST_CASE("Kimi-K3 LatentMoE block matches the reference with real experts") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    Fixture fixture;
    if (!fixture.load(fixture_path())) {
        SKIP("layer fixture is absent; run scripts/run_kimi_k3_reference_fixture.sh");
    }
    const auto& c = strata::kKimiK3ExecutionContract;
    const std::uint32_t layer = 1U;
    REQUIRE(strata::kimi_k3_moe_layer(layer));
    const auto prefix = "moe." + std::to_string(layer) + ".";

    const auto* input = fixture.find(prefix + "input");
    const auto* expected = fixture.find(prefix + "output");
    const auto* chosen = fixture.find(prefix + "experts");
    if (input == nullptr || expected == nullptr) {
        SKIP("MoE fixture is absent; regenerate it");
    }
    REQUIRE(chosen != nullptr);
    const auto tokens = static_cast<std::uint32_t>(input->shape.at(0));

    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& reader = *opened.value;
    const auto base = "language_model.model.layers." + std::to_string(layer) +
                      ".block_sparse_moe.";
    const auto shared_inner = c.expert_intermediate_size * c.shared_experts;

    RawTensor router, latent_down, latent_up, shared_gate, shared_up, shared_down;
    REQUIRE(read_bf16(reader, base + "gate.weight", c.routed_experts,
                      c.hidden_size, router));
    REQUIRE(read_bf16(reader, base + "routed_expert_down_proj.weight",
                      c.routed_expert_hidden_size, c.hidden_size, latent_down));
    REQUIRE(read_bf16(reader, base + "routed_expert_up_proj.weight", c.hidden_size,
                      c.routed_expert_hidden_size, latent_up));
    REQUIRE(read_bf16(reader, base + "shared_experts.gate_proj.weight",
                      shared_inner, c.hidden_size, shared_gate));
    REQUIRE(read_bf16(reader, base + "shared_experts.up_proj.weight", shared_inner,
                      c.hidden_size, shared_up));
    REQUIRE(read_bf16(reader, base + "shared_experts.down_proj.weight",
                      c.hidden_size, shared_inner, shared_down));

    std::vector<float> router_bias, latent_norm;
    REQUIRE(read_f32(reader, base + "gate.e_score_correction_bias",
                     c.routed_experts, router_bias));
    REQUIRE(read_f32(reader, base + "routed_expert_norm.weight",
                     c.routed_expert_hidden_size, latent_norm));

    strata::KimiMoeWeights weights;
    weights.router = router.matrix;
    weights.router_bias = router_bias;
    weights.latent_down = latent_down.matrix;
    weights.latent_up = latent_up.matrix;
    weights.latent_norm = latent_norm;
    weights.shared_gate = shared_gate.matrix;
    weights.shared_up = shared_up.matrix;
    weights.shared_down = shared_down.matrix;

    // The real arena and the real O_DIRECT reader, so this gates the MXFP4
    // decode, the slot layout the coalescing reader writes, and the block's
    // semantics together. An arena sized for the whole selection keeps every
    // expert resident once staged.
    strata::KimiArenaConfig arena_config;
    arena_config.capacity_bytes =
        strata::KimiCheckpointReader::expert_source_bytes() *
        (static_cast<std::uint64_t>(chosen->values.size()) + 8U);
    // Locking is a production requirement, not a test one: a test that demands
    // it fails on a machine with a small RLIMIT_MEMLOCK for reasons unrelated
    // to what is under test.
    arena_config.lock_pages = false;
    strata::KimiExpertArena arena;
    REQUIRE(arena.reset(arena_config).ok());

    strata::KimiReaderConfig reader_config;
    reader_config.queue_depth = 4U;
    reader_config.direct = true;
    strata::KimiExpertReader expert_reader;
    REQUIRE(expert_reader.open(reader, reader_config).ok());

    ArenaExperts experts(reader, arena, expert_reader);
    strata::KimiLayerScratch scratch;
    std::vector<float> output(static_cast<std::size_t>(tokens) * c.hidden_size);
    const auto ran = strata::kimi_latent_moe_layer(output, input->values, weights,
                                                   experts, layer, tokens, scratch);
    if (!ran.ok()) {
        for (const auto& error : ran.errors) std::cout << "  " << error << '\n';
    }
    REQUIRE(ran.ok());

    // The routed experts this run actually read, against the ones the
    // reference's own gate selected. A routing difference would show up here as
    // a set mismatch rather than as a numerical one further downstream.
    REQUIRE(experts.requested().size() == chosen->values.size());
    for (std::size_t index = 0U; index < chosen->values.size(); ++index) {
        REQUIRE(experts.requested()[index] ==
                static_cast<std::uint32_t>(chosen->values[index]));
    }

    const auto agreement = compare(output, expected->values);
    report("LatentMoE block", agreement);
    REQUIRE(agreement.relative_l2 < kRelativeL2Gate);
    REQUIRE(agreement.cosine > kCosineGate);
}

TEST_CASE("Kimi-K3 residual stream reproduces the reference block schedule") {
    const auto& c = strata::kKimiK3ExecutionContract;
    strata::KimiResidualStream stream;
    REQUIRE(stream.reset(3U, 8U, c.attention_residual_block_size).ok());
    std::vector<float> embedding(3U * 8U, 0.5F);
    REQUIRE(stream.begin(embedding).ok());

    // Layers 0, 12, 24, ..., 84 open a block; 93 layers leave eight blocks and
    // one prefix, which is nine sources at the output site regardless of depth.
    std::uint32_t opened = 0U;
    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        if (layer % c.attention_residual_block_size == 0U) {
            REQUIRE(stream.open_block().ok());
            ++opened;
        }
    }
    REQUIRE(opened == 8U);
    REQUIRE(stream.completed_blocks() == 8U);

    // With no completed block the prefix passes through unchanged, which is the
    // reference skipping the mix at the attention site of layer 0.
    strata::KimiResidualStream fresh;
    REQUIRE(fresh.reset(1U, 4U, c.attention_residual_block_size).ok());
    const std::vector<float> seed{1.0F, -2.0F, 3.0F, 0.25F};
    REQUIRE(fresh.begin(seed).ok());
    const std::vector<float> query{0.1F, 0.2F, 0.3F, 0.4F};
    const std::vector<float> norm{1.0F, 1.0F, 1.0F, 1.0F};
    std::vector<float> mixed(4U);
    REQUIRE(fresh.mix(mixed, query, norm, c.rms_epsilon).ok());
    for (std::size_t index = 0U; index < seed.size(); ++index) {
        REQUIRE_NEAR(mixed[index], seed[index], 1.0e-6F);
    }

    // Opening a block restarts the prefix at zero, so the next `add` sets it
    // rather than accumulating onto the closed block.
    REQUIRE(fresh.open_block().ok());
    const std::vector<float> delta{0.5F, 0.5F, 0.5F, 0.5F};
    REQUIRE(fresh.add(delta).ok());
    const auto prefix = fresh.prefix();
    for (std::size_t index = 0U; index < delta.size(); ++index) {
        REQUIRE_NEAR(prefix[index], delta[index], 1.0e-6F);
    }
}

TEST_CASE("Kimi-K3 layer operands are validated against the contract") {
    const auto& c = strata::kKimiK3ExecutionContract;
    strata::KimiStateCache cache;
    REQUIRE(cache.reset(8U).ok());
    strata::KimiLayerScratch scratch;
    NoExperts experts;

    // An empty weight set must be refused rather than read out of bounds.
    strata::KimiKdaWeights kda;
    std::vector<float> input(c.hidden_size, 0.0F);
    std::vector<float> output(c.hidden_size, 0.0F);
    REQUIRE(!strata::kimi_kda_layer(output, input, kda, cache, 0U, 1U, scratch).ok());

    strata::KimiMlaWeights mla;
    REQUIRE(!strata::kimi_mla_layer(output, input, mla, cache, 3U, 0U, 1U, scratch)
                 .ok());

    strata::KimiMoeWeights moe;
    REQUIRE(!strata::kimi_latent_moe_layer(output, input, moe, experts, 1U, 1U,
                                           scratch)
                 .ok());

    // A page shape the stream was not sized for is an error, not a silent clamp.
    strata::KimiResidualStream stream;
    REQUIRE(stream.reset(1U, c.hidden_size, c.attention_residual_block_size).ok());
    strata::KimiLayerWeights layer;
    REQUIRE(!strata::kimi_decoder_layer(stream, layer, cache, experts, 0U, 0U, 4U,
                                        scratch)
                 .ok());
    REQUIRE(!strata::kimi_decoder_layer(stream, layer, cache, experts,
                                        c.layer_count, 0U, 1U, scratch)
                 .ok());
}

TEST_CASE("Kimi-K3 matmul reads BF16 rows and accumulates in F32") {
    // [2, 3] weight, two tokens. BF16 encodes these exactly, so the product is
    // exact and any transposition shows up immediately.
    const std::vector<std::uint16_t> raw{
        0x3F80U, 0x4000U, 0x4040U,   // 1, 2, 3
        0x4080U, 0x40A0U, 0x40C0U};  // 4, 5, 6
    strata::KimiBf16Matrix weight;
    weight.values = raw;
    weight.rows = 2U;
    weight.columns = 3U;
    const std::vector<float> input{1.0F, 0.0F, -1.0F, 0.5F, 0.5F, 0.5F};
    std::vector<float> output(4U);
    REQUIRE(strata::kimi_bf16_matmul(output, input, weight, 2U).ok());
    REQUIRE_NEAR(output[0], 1.0F - 3.0F, 1.0e-6F);
    REQUIRE_NEAR(output[1], 4.0F - 6.0F, 1.0e-6F);
    REQUIRE_NEAR(output[2], 0.5F * (1.0F + 2.0F + 3.0F), 1.0e-6F);
    REQUIRE_NEAR(output[3], 0.5F * (4.0F + 5.0F + 6.0F), 1.0e-6F);

    std::vector<float> wrong(3U);
    REQUIRE(!strata::kimi_bf16_matmul(wrong, input, weight, 2U).ok());
}

// The AVX2 MXFP4 path accumulates eight lanes in parallel and reduces at the
// end, so its summation order differs from the scalar path's single chain.
// That is a reordering, not a different value set.
//
// Testing it at one tolerance would confuse two different things: a decode
// defect (a wrong magnitude, a dropped sign, a misapplied group scale) and the
// conditioning of a 3584-term float sum. They separate by sweeping how far the
// per-group scales spread within a row. Reassociation error grows with the
// dynamic range of the summands; a decode defect does not care about it and
// stays put. So the narrow-spread bound is the sensitive detector, and the wide
// one only records what reassociation costs.
//
// Measured here: 5.96e-7 at one octave, rising monotonically to 1.18e-5 at
// fifteen. Fifteen octaves inside a single row is far wider than a real
// checkpoint -- the codec exists to make each group's scale track its own
// magnitude -- so it is included as a bound, not as the operating point.
TEST_CASE("kimi_mxfp4_matvec agrees with its scalar path") {
    constexpr std::uint32_t kRows = 256U;
    constexpr std::uint32_t kColumns = 3584U;  // routed_expert_hidden_size
    constexpr std::uint32_t kScaleBase = 120U;
    constexpr auto& contract = strata::kKimiK3ExecutionContract;
    REQUIRE(kColumns == contract.routed_expert_hidden_size);

    struct Arm {
        std::uint32_t spread;
        double gate;
    };
    // One octave is the decode-defect detector: with every group sharing a
    // scale there is nothing left for reassociation to lose, so the two paths
    // must agree to a handful of ulp.
    const std::array<Arm, 4U> arms{Arm{1U, 2.0e-6}, Arm{3U, 4.0e-6},
                                   Arm{7U, 1.2e-5}, Arm{15U, 2.0e-5}};

    double previous = 0.0;
    for (const auto& arm : arms) {
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(kRows) * kColumns / 2U);
        std::vector<std::uint8_t> scales(
            static_cast<std::size_t>(kRows) * kColumns / 32U);
        std::vector<float> input(kColumns);

        // A fixed linear congruential sequence, so a failure is reproducible
        // without depending on any standard library's generator.
        std::uint64_t state = 0x9E3779B97F4A7C15ULL;
        const auto next = [&state]() {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<std::uint32_t>(state >> 33U);
        };
        for (auto& byte : packed) byte = static_cast<std::uint8_t>(next() & 0xFFU);
        for (auto& byte : scales) {
            byte = static_cast<std::uint8_t>(kScaleBase + (next() % arm.spread));
        }
        for (auto& value : input) {
            value =
                static_cast<float>(static_cast<std::int32_t>(next() % 2000U) - 1000) /
                1000.0F;
        }

        strata::KimiExpertModuleView module{packed, scales, kRows, kColumns};
        REQUIRE(module.valid());

        std::vector<float> produced(kRows, 0.0F);
        const auto result = strata::kimi_mxfp4_matvec(produced, input, module);
        REQUIRE(result.ok());

        // The scalar reference is written out here rather than reached through
        // the dispatch, so the test still compares two implementations on a
        // machine where the vector path is the one that runs.
        const auto scale_stride = static_cast<std::size_t>(kColumns) / 32U;
        const auto packed_stride = static_cast<std::size_t>(kColumns) / 2U;
        double worst = 0.0;
        for (std::uint32_t row = 0U; row < kRows; ++row) {
            const auto* row_packed = packed.data() + row * packed_stride;
            const auto* row_scales = scales.data() + row * scale_stride;
            float sum = 0.0F;
            for (std::size_t group = 0U; group < scale_stride; ++group) {
                const auto scale = strata::mxfp4_scale_from_e8m0(row_scales[group]);
                const auto* bytes = row_packed + group * 16U;
                const auto* source = input.data() + group * 32U;
                float partial = 0.0F;
                for (std::size_t index = 0U; index < 16U; ++index) {
                    const auto byte = bytes[index];
                    partial +=
                        source[index * 2U] * strata::kMxfp4Values[byte & 0x0FU] +
                        source[index * 2U + 1U] * strata::kMxfp4Values[byte >> 4U];
                }
                sum += partial * scale;
            }
            const auto magnitude = std::max(std::abs(sum), 1.0F);
            worst = std::max(worst,
                             static_cast<double>(std::abs(produced[row] - sum)) /
                                 static_cast<double>(magnitude));
        }
        std::cout << "  mxfp4 scalar/vector gap at " << arm.spread
                  << " octaves: " << worst << '\n';
        REQUIRE(worst < arm.gate);
        // The shape is the evidence. If the gap stopped tracking the dynamic
        // range, the difference would not be reassociation any more.
        REQUIRE(worst >= previous);
        previous = worst;
    }
}
