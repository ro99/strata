#include "test.hpp"

#include "strata/kimi_k3_checkpoint.hpp"
#include "strata/kimi_k3_ops.hpp"
#include "strata/model_adapter.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <numeric>
#include <vector>

namespace {

// Deterministic pseudo-random values in [-scale, scale]. A fixed sequence keeps
// every comparison below reproducible without shipping a data file.
class Stream {
public:
    explicit Stream(std::uint64_t seed) : state_(seed | 1U) {}
    float next(float scale) {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const auto bits = static_cast<std::uint32_t>(state_ >> 33U);
        return (static_cast<float>(bits) / 2147483648.0F - 1.0F) * scale;
    }
    void fill(std::span<float> values, float scale) {
        for (auto& value : values) value = next(scale);
    }

private:
    std::uint64_t state_{};
};

float sigmoid_of(float value) { return 1.0F / (1.0F + std::exp(-value)); }

}  // namespace

TEST_CASE("SiTU-GLU matches its closed form and is not SwiGLU") {
    const auto& c = strata::kKimiK3ExecutionContract;
    std::vector<float> gate{0.0F, 1.0F, -1.0F, 8.0F, -8.0F, 60.0F};
    std::vector<float> up{1.0F, 2.0F, -3.0F, 100.0F, -100.0F, 0.5F};
    std::vector<float> output(gate.size());
    REQUIRE(strata::kimi_situ_glu(output, gate, up).ok());

    for (std::size_t index = 0U; index < gate.size(); ++index) {
        const auto g = gate[index];
        const auto expected =
            c.situ_gate_beta * std::tanh(g / c.situ_gate_beta) * sigmoid_of(g) *
            c.situ_linear_beta * std::tanh(up[index] / c.situ_linear_beta);
        REQUIRE_NEAR(output[index], expected, 1.0e-6F);
    }

    // Both factors of the first bracket read the gate. Feeding the up
    // projection into the sigmoid — the shape a SwiGLU port would produce —
    // gives a different answer, so the test would catch that substitution.
    const auto swapped = c.situ_gate_beta * std::tanh(gate[1] / c.situ_gate_beta) *
                         sigmoid_of(up[1]) * c.situ_linear_beta *
                         std::tanh(up[1] / c.situ_linear_beta);
    REQUIRE(std::fabs(output[1] - swapped) > 1.0e-3F);

    // Both branches are bounded, which is the point of the activation: the
    // gate factor by beta1 and the linear factor by beta2.
    REQUIRE(std::fabs(output[3]) < c.situ_gate_beta * c.situ_linear_beta + 1.0e-3F);
    REQUIRE(std::fabs(output[5]) < c.situ_gate_beta * c.situ_linear_beta + 1.0e-3F);

    std::vector<float> mismatched(gate.size() - 1U);
    REQUIRE(!strata::kimi_situ_glu(mismatched, gate, up).ok());
}

TEST_CASE("Kimi RMS norm and KDA gated output norm apply weights in order") {
    std::vector<float> input{1.0F, -2.0F, 3.0F, -4.0F};
    std::vector<float> weight{0.5F, 1.5F, -1.0F, 2.0F};
    std::vector<float> output(4U);
    REQUIRE(strata::kimi_rms_norm(output, input, weight, 0.0F).ok());
    const auto rms = std::sqrt((1.0F + 4.0F + 9.0F + 16.0F) / 4.0F);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        REQUIRE_NEAR(output[index], weight[index] * input[index] / rms, 1.0e-6F);
    }

    // The gate multiplies after the norm and its weight. Gating first would
    // change the variance the norm divides by, so the two orders differ.
    std::vector<float> gate{0.25F, -0.5F, 1.0F, 2.0F};
    std::vector<float> gated(4U);
    REQUIRE(strata::kimi_kda_output_norm(gated, input, gate, weight, 0.0F).ok());
    for (std::size_t index = 0U; index < input.size(); ++index) {
        REQUIRE_NEAR(gated[index], output[index] * sigmoid_of(gate[index]),
                     1.0e-6F);
    }
}

TEST_CASE("KDA decay is bounded by the gate lower bound") {
    const auto& c = strata::kKimiK3ExecutionContract;
    std::vector<float> logits{-1000.0F, -1.0F, 0.0F, 1.0F, 1000.0F};
    std::vector<float> bias(logits.size(), 0.0F);
    std::vector<float> logarithm(logits.size());
    REQUIRE(strata::kimi_kda_log_decay(logarithm, logits, bias, 0.0F).ok());

    for (std::size_t index = 0U; index < logits.size(); ++index) {
        const auto expected = c.kda_gate_lower_bound * sigmoid_of(logits[index]);
        REQUIRE_NEAR(logarithm[index], expected, 1.0e-5F);
        // The bound is what keeps the chunkwise causal tile finite: alpha stays
        // inside (exp(-5), 1).
        REQUIRE(logarithm[index] <= 0.0F);
        REQUIRE(logarithm[index] >= c.kda_gate_lower_bound);
    }

    // A_log scales the logit before the sigmoid, so a larger amplitude
    // saturates sooner: a negative logit decays less and a positive one decays
    // more. Reading the padded per-channel entries instead would apply
    // exp(0) = 1 here and lose that scaling entirely.
    std::vector<float> scaled(logits.size());
    REQUIRE(strata::kimi_kda_log_decay(scaled, logits, bias, 1.0F).ok());
    REQUIRE(scaled[1] > logarithm[1]);
    REQUIRE(scaled[3] < logarithm[3]);

    std::vector<float> rejected(logits.size());
    REQUIRE(!strata::kimi_kda_log_decay(rejected, logits, bias, 0.0F, 0.0F).ok());
}

TEST_CASE("the router excludes the frozen bias from the routed weights") {
    // Two experts have high scores, two have low scores but a large bias. The
    // bias must decide selection and not appear in the weights.
    std::vector<float> logits{2.0F, -2.0F, 1.0F, -1.0F};
    std::vector<float> bias{0.0F, 5.0F, 0.0F, 0.0F};
    std::vector<strata::KimiRoutedExpert> selected(2U);
    REQUIRE(strata::kimi_route_topk(selected, logits, bias).ok());

    // Expert 1 wins on bias; expert 0 wins on score.
    REQUIRE(selected[0].expert == 1U);
    REQUIRE(selected[1].expert == 0U);

    const auto score0 = sigmoid_of(2.0F);
    const auto score1 = sigmoid_of(-2.0F);
    const auto total = score0 + score1;
    REQUIRE_NEAR(selected[0].weight, score1 / total, 1.0e-6F);
    REQUIRE_NEAR(selected[1].weight, score0 / total, 1.0e-6F);
    // Renormalized over the selected set, so the weights sum to the routed
    // scale, which is 1 for this checkpoint.
    REQUIRE_NEAR(selected[0].weight + selected[1].weight,
                 strata::kKimiK3ExecutionContract.routed_scale, 1.0e-6F);

    // Ties break toward the lower index so a run reproduces.
    std::vector<float> flat(4U, 0.5F);
    std::vector<float> no_bias(4U, 0.0F);
    std::vector<strata::KimiRoutedExpert> ties(2U);
    REQUIRE(strata::kimi_route_topk(ties, flat, no_bias).ok());
    REQUIRE(ties[0].expert == 0U);
    REQUIRE(ties[1].expert == 1U);

    std::vector<strata::KimiRoutedExpert> oversized(5U);
    REQUIRE(!strata::kimi_route_topk(oversized, logits, bias).ok());
}

TEST_CASE("attention residual mixing selects over depth") {
    constexpr std::uint32_t kHidden = 8U;
    constexpr std::uint32_t kSources = 3U;
    Stream stream(11U);
    std::vector<float> sources(kHidden * kSources);
    stream.fill(sources, 1.0F);
    std::vector<float> query(kHidden);
    std::vector<float> norm(kHidden, 1.0F);
    stream.fill(query, 1.0F);
    std::vector<float> output(kHidden);
    REQUIRE(strata::kimi_attention_residual_mix(output, sources, query, norm,
                                                kHidden, kSources, 1.0e-5F).ok());

    // Recompute the reference form: scores on the normalized sources, mixture
    // over the raw ones.
    std::vector<float> scores(kSources);
    for (std::uint32_t index = 0U; index < kSources; ++index) {
        float variance = 0.0F;
        for (std::uint32_t channel = 0U; channel < kHidden; ++channel) {
            const auto value = sources[index * kHidden + channel];
            variance += value * value;
        }
        const auto scale = 1.0F / std::sqrt(variance / kHidden + 1.0e-5F);
        float sum = 0.0F;
        for (std::uint32_t channel = 0U; channel < kHidden; ++channel) {
            sum += sources[index * kHidden + channel] * scale * norm[channel] *
                   query[channel];
        }
        scores[index] = sum;
    }
    float partition = 0.0F;
    for (const auto score : scores) partition += std::exp(score);
    for (std::uint32_t channel = 0U; channel < kHidden; ++channel) {
        float expected = 0.0F;
        for (std::uint32_t index = 0U; index < kSources; ++index) {
            expected += std::exp(scores[index]) / partition *
                        sources[index * kHidden + channel];
        }
        REQUIRE_NEAR(output[channel], expected, 1.0e-5F);
    }

    // A single source is a convex combination of one vector, so it passes
    // through whatever the query weight is.
    std::vector<float> single(sources.begin(), sources.begin() + kHidden);
    std::vector<float> alone(kHidden);
    REQUIRE(strata::kimi_attention_residual_mix(alone, single, query, norm,
                                                kHidden, 1U, 1.0e-5F).ok());
    for (std::uint32_t channel = 0U; channel < kHidden; ++channel) {
        REQUIRE_NEAR(alone[channel], single[channel], 1.0e-6F);
    }
}

TEST_CASE("attention residual state keeps nine sources over 93 layers") {
    const auto& c = strata::kKimiK3ExecutionContract;
    constexpr std::uint32_t kHidden = 4U;
    strata::KimiAttentionResidualState state;
    REQUIRE(state.reset(kHidden, c.attention_residual_block_size).ok());
    std::vector<float> embedding{1.0F, 2.0F, 3.0F, 4.0F};
    REQUIRE(state.begin(embedding).ok());
    std::vector<float> query(kHidden, 0.1F);
    std::vector<float> norm(kHidden, 1.0F);
    std::vector<float> output(kHidden);
    std::vector<float> delta(kHidden, 0.25F);

    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        // Attention site, then a block boundary every `attn_res_block_size`
        // layers, exactly as the reference orders them.
        REQUIRE(state.mix(output, query, norm, c.rms_epsilon).ok());
        if (layer % c.attention_residual_block_size == 0U) {
            REQUIRE(state.open_block().ok());
        }
        REQUIRE(state.add(delta).ok());
        REQUIRE(state.mix(output, query, norm, c.rms_epsilon).ok());
        REQUIRE(state.add(delta).ok());
    }
    // Boundaries at layers 0, 12, ..., 84: eight blocks plus the live prefix.
    REQUIRE(state.completed_blocks() == 8U);
    REQUIRE(state.source_count() == 9U);
    REQUIRE(state.prefix().size() == kHidden);

    // The first block is the embedding: layer 0 opens a block from the prefix
    // before anything has been added to it.
    strata::KimiAttentionResidualState fresh;
    REQUIRE(fresh.reset(kHidden, c.attention_residual_block_size).ok());
    REQUIRE(fresh.begin(embedding).ok());
    // With nothing completed the mix passes the prefix through.
    REQUIRE(fresh.mix(output, query, norm, c.rms_epsilon).ok());
    for (std::uint32_t channel = 0U; channel < kHidden; ++channel) {
        REQUIRE_NEAR(output[channel], embedding[channel], 1.0e-6F);
    }
    REQUIRE(fresh.open_block().ok());
    // A block opened and not yet followed by an add has no prefix to mix.
    REQUIRE(!fresh.mix(output, query, norm, c.rms_epsilon).ok());
}

TEST_CASE("the short convolution is causal and carries history across steps") {
    constexpr std::uint32_t kChannels = 2U;
    constexpr std::uint32_t kKernel = 4U;
    // Channel 0 selects the newest input, channel 1 the oldest of four.
    std::vector<float> weight{0.0F, 0.0F, 0.0F, 1.0F,
                              1.0F, 0.0F, 0.0F, 0.0F};
    std::vector<float> history(kChannels * (kKernel - 1U), 0.0F);
    std::vector<float> output(kChannels);
    const float inputs[5][2] = {{1.0F, 1.0F}, {2.0F, 2.0F}, {3.0F, 3.0F},
                                {4.0F, 4.0F}, {5.0F, 5.0F}};
    std::vector<float> seen;
    for (const auto& row : inputs) {
        std::vector<float> input(row, row + kChannels);
        REQUIRE(strata::kimi_short_conv_step(output, input, weight, history,
                                             kKernel).ok());
        seen.push_back(output[0]);
        seen.push_back(output[1]);
    }
    // Channel 0 is the current input through SiLU, on every step.
    for (std::size_t step = 0U; step < 5U; ++step) {
        const auto value = static_cast<float>(step + 1U);
        REQUIRE_NEAR(seen[step * 2U], value * sigmoid_of(value), 1.0e-6F);
    }
    // Channel 1 lags by three steps and reads zero-padding before that, which
    // is what makes the convolution causal.
    REQUIRE_NEAR(seen[1], 0.0F, 1.0e-6F);
    REQUIRE_NEAR(seen[3], 0.0F, 1.0e-6F);
    REQUIRE_NEAR(seen[5], 0.0F, 1.0e-6F);
    REQUIRE_NEAR(seen[7], 1.0F * sigmoid_of(1.0F), 1.0e-6F);
    REQUIRE_NEAR(seen[9], 2.0F * sigmoid_of(2.0F), 1.0e-6F);
}

TEST_CASE("KDA chunkwise prefill equals the token recurrence") {
    // The strongest internal oracle in the bring-up: two different algorithms
    // for the same recurrence, needing no external reference. The chunk form
    // builds a dense causal tile and updates the state once; the step form
    // updates it per token.
    constexpr std::uint32_t kKeys = 16U;
    constexpr std::uint32_t kValues = 12U;
    const auto& c = strata::kKimiK3ExecutionContract;

    for (const std::uint32_t tokens : {1U, 2U, 7U, 33U, 64U}) {
        Stream stream(tokens * 7919U + 13U);
        std::vector<float> query(tokens * kKeys);
        std::vector<float> key(tokens * kKeys);
        std::vector<float> value(tokens * kValues);
        std::vector<float> logits(tokens * kKeys);
        std::vector<float> bias(kKeys);
        std::vector<float> beta(tokens);
        stream.fill(query, 1.0F);
        stream.fill(key, 1.0F);
        stream.fill(value, 1.0F);
        stream.fill(logits, 3.0F);
        stream.fill(bias, 0.5F);
        for (auto& value_of : beta) value_of = sigmoid_of(stream.next(2.0F));

        // Normalize q and k per token as the KDA path does before the kernel,
        // and scale q by the same 1/sqrt(key_dim) the kernels apply.
        const auto scale = 1.0F / std::sqrt(static_cast<float>(kKeys));
        for (std::uint32_t t = 0U; t < tokens; ++t) {
            REQUIRE(strata::kimi_l2_normalize(
                        std::span<float>(query).subspan(t * kKeys, kKeys)).ok());
            REQUIRE(strata::kimi_l2_normalize(
                        std::span<float>(key).subspan(t * kKeys, kKeys)).ok());
            for (std::uint32_t k = 0U; k < kKeys; ++k) {
                query[t * kKeys + k] *= scale;
            }
        }

        std::vector<float> logarithm(tokens * kKeys);
        for (std::uint32_t t = 0U; t < tokens; ++t) {
            REQUIRE(strata::kimi_kda_log_decay(
                        std::span<float>(logarithm).subspan(t * kKeys, kKeys),
                        std::span<const float>(logits).subspan(t * kKeys, kKeys),
                        bias, 0.3F, c.kda_gate_lower_bound).ok());
        }

        // A non-zero incoming state, so the carry across a chunk boundary is
        // exercised rather than assumed zero.
        std::vector<float> initial(kValues * kKeys);
        stream.fill(initial, 0.4F);

        std::vector<float> chunk_state(initial);
        std::vector<float> chunk_output(tokens * kValues);
        REQUIRE(strata::kimi_kda_chunk(chunk_output, chunk_state, query, key,
                                       value, logarithm, beta, tokens, kKeys,
                                       kValues).ok());

        std::vector<float> step_state(initial);
        std::vector<float> step_output(tokens * kValues);
        std::vector<float> decay(kKeys);
        for (std::uint32_t t = 0U; t < tokens; ++t) {
            for (std::uint32_t k = 0U; k < kKeys; ++k) {
                decay[k] = std::exp(logarithm[t * kKeys + k]);
            }
            REQUIRE(strata::kimi_kda_step(
                        std::span<float>(step_output).subspan(t * kValues, kValues),
                        step_state,
                        std::span<const float>(query).subspan(t * kKeys, kKeys),
                        std::span<const float>(key).subspan(t * kKeys, kKeys),
                        std::span<const float>(value).subspan(t * kValues, kValues),
                        decay, beta[t], kKeys, kValues).ok());
        }

        for (std::size_t index = 0U; index < step_output.size(); ++index) {
            REQUIRE_NEAR(chunk_output[index], step_output[index], 2.0e-4F);
        }
        for (std::size_t index = 0U; index < step_state.size(); ++index) {
            REQUIRE_NEAR(chunk_state[index], step_state[index], 2.0e-4F);
        }
    }
}

TEST_CASE("KDA chunk boundaries compose into the same state as one long chunk") {
    // Prefill pages the sequence, so the carry between pages has to be exact.
    constexpr std::uint32_t kKeys = 12U;
    constexpr std::uint32_t kValues = 10U;
    constexpr std::uint32_t kTokens = 24U;
    Stream stream(4242U);
    std::vector<float> query(kTokens * kKeys);
    std::vector<float> key(kTokens * kKeys);
    std::vector<float> value(kTokens * kValues);
    std::vector<float> logarithm(kTokens * kKeys);
    std::vector<float> beta(kTokens);
    stream.fill(query, 1.0F);
    stream.fill(key, 1.0F);
    stream.fill(value, 1.0F);
    stream.fill(logarithm, 1.0F);
    for (auto& entry : logarithm) entry = -std::fabs(entry);
    for (auto& entry : beta) entry = sigmoid_of(stream.next(2.0F));

    std::vector<float> whole_state(kValues * kKeys, 0.0F);
    std::vector<float> whole_output(kTokens * kValues);
    REQUIRE(strata::kimi_kda_chunk(whole_output, whole_state, query, key, value,
                                   logarithm, beta, kTokens, kKeys, kValues).ok());

    // Uneven pages, so the composition cannot pass by the chunk length
    // happening to divide the sequence.
    std::vector<float> paged_state(kValues * kKeys, 0.0F);
    std::vector<float> paged_output(kTokens * kValues);
    std::uint32_t begin = 0U;
    for (const std::uint32_t page : {5U, 11U, 8U}) {
        REQUIRE(strata::kimi_kda_chunk(
                    std::span<float>(paged_output)
                        .subspan(begin * kValues, page * kValues),
                    paged_state,
                    std::span<const float>(query).subspan(begin * kKeys,
                                                          page * kKeys),
                    std::span<const float>(key).subspan(begin * kKeys,
                                                        page * kKeys),
                    std::span<const float>(value).subspan(begin * kValues,
                                                          page * kValues),
                    std::span<const float>(logarithm).subspan(begin * kKeys,
                                                              page * kKeys),
                    std::span<const float>(beta).subspan(begin, page), page,
                    kKeys, kValues).ok());
        begin += page;
    }
    REQUIRE(begin == kTokens);
    for (std::size_t index = 0U; index < whole_output.size(); ++index) {
        REQUIRE_NEAR(paged_output[index], whole_output[index], 1.0e-4F);
    }
    for (std::size_t index = 0U; index < whole_state.size(); ++index) {
        REQUIRE_NEAR(paged_state[index], whole_state[index], 1.0e-4F);
    }
}

// ------------------------------------------- fixtures from real tensors

namespace {

std::string kimi_directory() {
    return (std::filesystem::path(STRATA_SOURCE_DIR) / "models/kimi-k3").string();
}

bool kimi_present() {
    return std::filesystem::exists(
        std::filesystem::path(kimi_directory()) / "model.safetensors.index.json");
}

// Row-major matvec over a BF16 tensor already decoded to F32.
std::vector<float> matvec(std::span<const float> matrix,
                          std::span<const float> input, std::size_t rows) {
    std::vector<float> output(rows);
    const auto columns = input.size();
    for (std::size_t row = 0U; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0U; column < columns; ++column) {
            sum += matrix[row * columns + column] * input[column];
        }
        output[row] = sum;
    }
    return output;
}

}  // namespace

TEST_CASE("real Kimi-K3 router selects the reference experts and weights") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& c = strata::kKimiK3ExecutionContract;

    const auto gate = opened.value->read_f32(
        "language_model.model.layers.1.block_sparse_moe.gate.weight",
        static_cast<std::uint64_t>(c.routed_experts) * c.hidden_size);
    REQUIRE(gate.ok());
    const auto bias = opened.value->read_f32(
        "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias",
        c.routed_experts);
    REQUIRE(bias.ok());

    Stream stream(2026U);
    std::vector<float> hidden(c.hidden_size);
    stream.fill(hidden, 1.0F);
    const auto logits = matvec(gate.value, hidden, c.routed_experts);

    std::vector<strata::KimiRoutedExpert> selected(c.experts_per_token);
    REQUIRE(strata::kimi_route_topk(selected, logits, bias.value).ok());

    // Oracle: the reference's own selection rule applied to the real router
    // and the real frozen bias — sigmoid scores ranked with the bias, top-16,
    // then renormalized over the unbiased scores.
    const std::uint32_t expected[16] = {485U, 592U, 452U, 335U, 747U, 439U,
                                        837U, 395U, 459U, 611U, 877U, 465U,
                                        520U, 243U, 561U, 686U};
    float total = 0.0F;
    for (std::size_t slot = 0U; slot < selected.size(); ++slot) {
        REQUIRE(selected[slot].expert == expected[slot]);
        REQUIRE(selected[slot].weight > 0.0F);
        total += selected[slot].weight;
    }
    REQUIRE_NEAR(total, c.routed_scale, 1.0e-5F);
    REQUIRE_NEAR(selected[0].weight, 0.06267262F, 1.0e-6F);
    REQUIRE_NEAR(selected[9].weight, 0.06189896F, 1.0e-6F);

    // Dropping the bias changes which experts are selected, which is what
    // makes the frozen bias load-bearing rather than decorative.
    const std::vector<float> unbiased(c.routed_experts, 0.0F);
    std::vector<strata::KimiRoutedExpert> without(c.experts_per_token);
    REQUIRE(strata::kimi_route_topk(without, logits, unbiased).ok());
    bool differs = false;
    for (std::size_t slot = 0U; slot < without.size(); ++slot) {
        if (without[slot].expert != selected[slot].expert) differs = true;
    }
    REQUIRE(differs);
}

TEST_CASE("real Kimi-K3 KDA decay matches the reference mapping") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& c = strata::kKimiK3ExecutionContract;
    const auto head_dim = c.linear_head_dim;

    const auto a_log = opened.value->read_f32(
        "language_model.model.layers.0.self_attn.A_log", head_dim);
    REQUIRE(a_log.ok());
    const auto dt_bias = opened.value->read_f32(
        "language_model.model.layers.0.self_attn.dt_bias",
        static_cast<std::uint64_t>(c.attention_heads) * head_dim);
    REQUIRE(dt_bias.ok());
    REQUIRE_NEAR(a_log.value[0], -0.524404824F, 1.0e-7F);

    Stream stream(777U);
    std::vector<float> logits(head_dim);
    stream.fill(logits, 3.0F);
    std::vector<float> logarithm(head_dim);
    REQUIRE(strata::kimi_kda_log_decay(
                logarithm, logits,
                std::span<const float>(dt_bias.value).subspan(0U, head_dim),
                a_log.value[0]).ok());

    const float expected_head[6] = {-0.03368333F, -0.09643126F, -0.04171178F,
                                    -0.21978298F, -0.06469416F, -0.08917718F};
    for (std::size_t index = 0U; index < 6U; ++index) {
        REQUIRE_NEAR(logarithm[index], expected_head[index], 1.0e-6F);
    }
    const float expected_tail[6] = {-0.34766769F, -0.22685485F, -0.10124742F,
                                    -0.21053916F, -0.13589078F, -0.31412664F};
    for (std::size_t index = 0U; index < 6U; ++index) {
        REQUIRE_NEAR(logarithm[head_dim - 6U + index], expected_tail[index],
                     1.0e-6F);
    }
}

TEST_CASE("real Kimi-K3 attention residual matches the reference mixture") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& c = strata::kKimiK3ExecutionContract;

    const auto query = opened.value->read_f32(
        "language_model.model.layers.5.self_attention_res_proj.weight",
        c.hidden_size);
    REQUIRE(query.ok());
    const auto norm = opened.value->read_f32(
        "language_model.model.layers.5.self_attention_res_norm.weight",
        c.hidden_size);
    REQUIRE(norm.ok());

    constexpr std::uint32_t kSources = 4U;
    Stream stream(31337U);
    std::vector<float> sources(static_cast<std::size_t>(kSources) * c.hidden_size);
    stream.fill(sources, 1.0F);
    std::vector<float> output(c.hidden_size);
    REQUIRE(strata::kimi_attention_residual_mix(output, sources, query.value,
                                                norm.value, c.hidden_size,
                                                kSources, c.rms_epsilon).ok());

    const float expected[6] = {-0.73717105F, -0.59122503F, -0.68096495F,
                               -0.38740554F, -0.40311846F, -0.57598102F};
    for (std::size_t index = 0U; index < 6U; ++index) {
        REQUIRE_NEAR(output[index], expected[index], 1.0e-5F);
    }
}
