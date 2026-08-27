#include "strata/models/kimi_k3/kimi_k3_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace strata {
namespace {

[[nodiscard]] float sigmoid(float value) noexcept {
    return 1.0F / (1.0F + std::exp(-value));
}

[[nodiscard]] float mean_square(std::span<const float> values) noexcept {
    float sum = 0.0F;
    for (const auto value : values) sum += value * value;
    return sum / static_cast<float>(values.size());
}

}  // namespace

ValidationResult kimi_rms_norm(std::span<float> output,
                               std::span<const float> input,
                               std::span<const float> weight, float epsilon) {
    ValidationResult result;
    if (input.empty() || output.size() != input.size() ||
        weight.size() != input.size()) {
        result.errors.emplace_back("Kimi-K3 RMS norm operands disagree in width");
        return result;
    }
    const auto scale = 1.0F / std::sqrt(mean_square(input) + epsilon);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        output[index] = weight[index] * (input[index] * scale);
    }
    return result;
}

ValidationResult kimi_situ_glu(std::span<float> output,
                               std::span<const float> gate,
                               std::span<const float> up, float gate_beta,
                               float linear_beta) {
    ValidationResult result;
    if (gate.empty() || up.size() != gate.size() || output.size() != gate.size()) {
        result.errors.emplace_back("SiTU-GLU operands disagree in width");
        return result;
    }
    if (!(gate_beta > 0.0F) || !(linear_beta > 0.0F)) {
        result.errors.emplace_back("SiTU-GLU betas must be positive");
        return result;
    }
    for (std::size_t index = 0U; index < gate.size(); ++index) {
        const auto g = gate[index];
        // Both factors read the gate projection: the tanh bounds it and the
        // sigmoid gates it. Only the second bracket reads the up projection.
        const auto bounded = gate_beta * std::tanh(g / gate_beta) * sigmoid(g);
        const auto linear = linear_beta * std::tanh(up[index] / linear_beta);
        output[index] = bounded * linear;
    }
    return result;
}

ValidationResult kimi_l2_normalize(std::span<float> values, float epsilon) {
    ValidationResult result;
    if (values.empty()) {
        result.errors.emplace_back("L2 normalization needs a non-empty vector");
        return result;
    }
    float sum = 0.0F;
    for (const auto value : values) sum += value * value;
    // The epsilon sits inside the square root, as the KDA kernels apply it.
    const auto scale = 1.0F / std::sqrt(sum + epsilon);
    for (auto& value : values) value *= scale;
    return result;
}

ValidationResult kimi_short_conv_step(std::span<float> output,
                                      std::span<const float> input,
                                      std::span<const float> weight,
                                      std::span<float> history,
                                      std::uint32_t kernel) {
    ValidationResult result;
    if (kernel < 1U) {
        result.errors.emplace_back("short convolution needs a positive kernel");
        return result;
    }
    const auto channels = input.size();
    if (channels == 0U || output.size() != channels ||
        weight.size() != channels * kernel ||
        history.size() != channels * (kernel - 1U)) {
        result.errors.emplace_back(
            "short convolution operands disagree with the channel count");
        return result;
    }
    const auto span = static_cast<std::size_t>(kernel);
    for (std::size_t channel = 0U; channel < channels; ++channel) {
        const auto* taps = weight.data() + channel * span;
        auto* past = history.data() + channel * (span - 1U);
        // Taps run oldest to newest, so the current input pairs with the last.
        float sum = taps[span - 1U] * input[channel];
        for (std::size_t offset = 0U; offset + 1U < span; ++offset) {
            sum += taps[offset] * past[offset];
        }
        for (std::size_t offset = 0U; offset + 2U < span; ++offset) {
            past[offset] = past[offset + 1U];
        }
        if (span > 1U) past[span - 2U] = input[channel];
        output[channel] = sum * sigmoid(sum);
    }
    return result;
}

ValidationResult kimi_kda_log_decay(std::span<float> logarithm,
                                    std::span<const float> logits,
                                    std::span<const float> dt_bias, float a_log,
                                    float lower_bound) {
    ValidationResult result;
    if (logits.empty() || logarithm.size() != logits.size() ||
        dt_bias.size() != logits.size()) {
        result.errors.emplace_back("KDA decay operands disagree in width");
        return result;
    }
    if (!(lower_bound < 0.0F)) {
        result.errors.emplace_back("KDA gate lower bound must be negative");
        return result;
    }
    const auto amplitude = std::exp(a_log);
    for (std::size_t index = 0U; index < logits.size(); ++index) {
        logarithm[index] =
            lower_bound * sigmoid(amplitude * (logits[index] + dt_bias[index]));
    }
    return result;
}

ValidationResult kimi_kda_step(std::span<float> output, std::span<float> state,
                               std::span<const float> query,
                               std::span<const float> key,
                               std::span<const float> value,
                               std::span<const float> decay, float beta,
                               std::uint32_t key_dim, std::uint32_t value_dim) {
    ValidationResult result;
    const auto keys = static_cast<std::size_t>(key_dim);
    const auto values = static_cast<std::size_t>(value_dim);
    if (keys == 0U || values == 0U || query.size() != keys || key.size() != keys ||
        decay.size() != keys || value.size() != values ||
        output.size() != values || state.size() != values * keys) {
        result.errors.emplace_back("KDA step operands disagree with the head shape");
        return result;
    }
    // State is [value, key]: row v holds the key-space vector that produces
    // output channel v. Decay scales the key axis, so it scales within a row.
    for (std::size_t v = 0U; v < values; ++v) {
        auto* row = state.data() + v * keys;
        float projected = 0.0F;
        for (std::size_t k = 0U; k < keys; ++k) {
            row[k] *= decay[k];
            projected += row[k] * key[k];
        }
        const auto delta = (value[v] - projected) * beta;
        float mixed = 0.0F;
        for (std::size_t k = 0U; k < keys; ++k) {
            row[k] += delta * key[k];
            mixed += row[k] * query[k];
        }
        output[v] = mixed;
    }
    return result;
}

ValidationResult kimi_kda_output_norm(std::span<float> output,
                                      std::span<const float> input,
                                      std::span<const float> gate,
                                      std::span<const float> weight,
                                      float epsilon) {
    ValidationResult result;
    if (input.empty() || output.size() != input.size() ||
        gate.size() != input.size() || weight.size() != input.size()) {
        result.errors.emplace_back("KDA output norm operands disagree in width");
        return result;
    }
    const auto scale = 1.0F / std::sqrt(mean_square(input) + epsilon);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        // Norm and its weight first, then the sigmoid gate. Gating before the
        // norm would change the variance the norm divides by.
        output[index] = weight[index] * (input[index] * scale) *
                        sigmoid(gate[index]);
    }
    return result;
}

ValidationResult kimi_route_topk(std::span<KimiRoutedExpert> selected,
                                 std::span<const float> logits,
                                 std::span<const float> selection_bias,
                                 float routed_scale) {
    ValidationResult result;
    if (logits.empty() || selection_bias.size() != logits.size() ||
        selected.empty() || selected.size() > logits.size()) {
        result.errors.emplace_back("Kimi-K3 router operands disagree in width");
        return result;
    }
    const auto experts = logits.size();
    const auto top_k = selected.size();
    std::vector<float> score(experts);
    std::vector<std::uint32_t> order(experts);
    for (std::size_t index = 0U; index < experts; ++index) {
        score[index] = sigmoid(logits[index]);
        order[index] = static_cast<std::uint32_t>(index);
    }
    // Selection ranks by score plus the frozen Quantile-Balancing bias; the
    // weights below use the unbiased score. Ties break toward the lower expert
    // index so a run reproduces.
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_k),
        order.end(), [&](std::uint32_t left, std::uint32_t right) {
            const auto ranked_left = score[left] + selection_bias[left];
            const auto ranked_right = score[right] + selection_bias[right];
            if (ranked_left != ranked_right) return ranked_left > ranked_right;
            return left < right;
        });

    float total = 0.0F;
    for (std::size_t slot = 0U; slot < top_k; ++slot) {
        total += score[order[slot]];
    }
    if (!(total > 0.0F)) {
        result.errors.emplace_back(
            "Kimi-K3 router selected experts whose scores sum to zero");
        return result;
    }
    for (std::size_t slot = 0U; slot < top_k; ++slot) {
        const auto expert = order[slot];
        selected[slot].expert = expert;
        selected[slot].weight = score[expert] / total * routed_scale;
    }
    return result;
}

ValidationResult kimi_attention_residual_mix(std::span<float> output,
                                             std::span<const float> sources,
                                             std::span<const float> query_weight,
                                             std::span<const float> norm_weight,
                                             std::uint32_t hidden_size,
                                             std::uint32_t count, float epsilon) {
    ValidationResult result;
    const auto hidden = static_cast<std::size_t>(hidden_size);
    if (hidden == 0U || count == 0U || output.size() != hidden ||
        query_weight.size() != hidden || norm_weight.size() != hidden ||
        sources.size() != hidden * count) {
        result.errors.emplace_back(
            "attention residual operands disagree with the source shape");
        return result;
    }
    std::vector<float> scores(count);
    float largest = -std::numeric_limits<float>::infinity();
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto source = sources.subspan(static_cast<std::size_t>(index) * hidden,
                                            hidden);
        const auto scale = 1.0F / std::sqrt(mean_square(source) + epsilon);
        float sum = 0.0F;
        for (std::size_t channel = 0U; channel < hidden; ++channel) {
            // The two weight vectors fold into one, which is how the reference
            // forms its score weight.
            sum += source[channel] * scale * norm_weight[channel] *
                   query_weight[channel];
        }
        scores[index] = sum;
        largest = std::max(largest, sum);
    }
    float partition = 0.0F;
    for (auto& value : scores) {
        value = std::exp(value - largest);
        partition += value;
    }
    std::fill(output.begin(), output.end(), 0.0F);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto weight = scores[index] / partition;
        const auto source = sources.subspan(static_cast<std::size_t>(index) * hidden,
                                            hidden);
        // The mixture is over the raw sources; only the scores saw the norm.
        for (std::size_t channel = 0U; channel < hidden; ++channel) {
            output[channel] += weight * source[channel];
        }
    }
    return result;
}

ValidationResult KimiAttentionResidualState::reset(std::uint32_t hidden_size,
                                                   std::uint32_t block_size) {
    ValidationResult result;
    if (hidden_size == 0U || block_size == 0U) {
        result.errors.emplace_back(
            "attention residual state needs a positive width and block size");
        return result;
    }
    hidden_size_ = hidden_size;
    block_size_ = block_size;
    completed_blocks_ = 0U;
    has_prefix_ = false;
    sources_.assign(hidden_size, 0.0F);
    return result;
}

ValidationResult KimiAttentionResidualState::begin(
    std::span<const float> embedding) {
    ValidationResult result;
    if (hidden_size_ == 0U || embedding.size() != hidden_size_) {
        result.errors.emplace_back(
            "attention residual state was not reset to this width");
        return result;
    }
    completed_blocks_ = 0U;
    has_prefix_ = true;
    sources_.assign(embedding.begin(), embedding.end());
    return result;
}

std::span<const float> KimiAttentionResidualState::prefix() const noexcept {
    if (!has_prefix_) return {};
    return std::span<const float>(sources_).subspan(
        static_cast<std::size_t>(completed_blocks_) * hidden_size_, hidden_size_);
}

ValidationResult KimiAttentionResidualState::mix(
    std::span<float> output, std::span<const float> query_weight,
    std::span<const float> norm_weight, float epsilon) const {
    ValidationResult result;
    if (!has_prefix_) {
        result.errors.emplace_back(
            "attention residual mix ran with no prefix; open_block was not "
            "followed by an add");
        return result;
    }
    if (completed_blocks_ == 0U) {
        // Only the attention site of layer 0 reaches this: there is no prior
        // block to select over, so the prefix passes through unchanged. The
        // reference expresses the same case by skipping the mix entirely.
        if (output.size() != hidden_size_) {
            result.errors.emplace_back(
                "attention residual output disagrees with the state width");
            return result;
        }
        const auto source = prefix();
        std::copy(source.begin(), source.end(), output.begin());
        return result;
    }
    return kimi_attention_residual_mix(output, sources_, query_weight,
                                       norm_weight, hidden_size_,
                                       source_count(), epsilon);
}

ValidationResult KimiAttentionResidualState::open_block() {
    ValidationResult result;
    if (!has_prefix_) {
        result.errors.emplace_back(
            "attention residual state has no prefix to close into a block");
        return result;
    }
    ++completed_blocks_;
    has_prefix_ = false;
    return result;
}

ValidationResult KimiAttentionResidualState::add(std::span<const float> delta) {
    ValidationResult result;
    if (hidden_size_ == 0U || delta.size() != hidden_size_) {
        result.errors.emplace_back(
            "attention residual delta disagrees with the state width");
        return result;
    }
    const auto base = static_cast<std::size_t>(completed_blocks_) * hidden_size_;
    if (!has_prefix_) {
        // A block was just opened: this sublayer output starts the new prefix
        // rather than adding to the one that was closed.
        sources_.resize(base + hidden_size_);
        std::copy(delta.begin(), delta.end(), sources_.begin() + base);
        has_prefix_ = true;
        return result;
    }
    for (std::size_t channel = 0U; channel < hidden_size_; ++channel) {
        sources_[base + channel] += delta[channel];
    }
    return result;
}

ValidationResult kimi_kda_chunk(std::span<float> output, std::span<float> state,
                                std::span<const float> query,
                                std::span<const float> key,
                                std::span<const float> value,
                                std::span<const float> decay_logarithm,
                                std::span<const float> beta, std::uint32_t tokens,
                                std::uint32_t key_dim, std::uint32_t value_dim) {
    ValidationResult result;
    const auto rows = static_cast<std::size_t>(tokens);
    const auto keys = static_cast<std::size_t>(key_dim);
    const auto values = static_cast<std::size_t>(value_dim);
    if (rows == 0U || keys == 0U || values == 0U || query.size() != rows * keys ||
        key.size() != rows * keys || decay_logarithm.size() != rows * keys ||
        value.size() != rows * values || output.size() != rows * values ||
        beta.size() != rows || state.size() != values * keys) {
        result.errors.emplace_back("KDA chunk operands disagree with the shape");
        return result;
    }

    // Cumulative log decay within the chunk. Every intra-chunk term below is
    // formed from a difference of two of these, so its exponent is at most
    // zero for the causal pairs it is used on.
    std::vector<float> cumulative(rows * keys);
    for (std::size_t k = 0U; k < keys; ++k) {
        float running = 0.0F;
        for (std::size_t t = 0U; t < rows; ++t) {
            running += decay_logarithm[t * keys + k];
            cumulative[t * keys + k] = running;
        }
    }

    // Pseudo-values, by forward substitution over the strictly lower-triangular
    // intra-chunk coupling. This is the UT transform written as a solve rather
    // than as an explicit inverse.
    std::vector<float> pseudo(rows * values);
    for (std::size_t t = 0U; t < rows; ++t) {
        const auto* k_t = key.data() + t * keys;
        const auto* g_t = cumulative.data() + t * keys;
        // Contribution of the incoming state: S_0^T (Gamma_t * k_t).
        for (std::size_t v = 0U; v < values; ++v) {
            const auto* row = state.data() + v * keys;
            float projected = 0.0F;
            for (std::size_t k = 0U; k < keys; ++k) {
                projected += row[k] * k_t[k] * std::exp(g_t[k]);
            }
            pseudo[t * values + v] = value[t * values + v] - projected;
        }
        for (std::size_t j = 0U; j < t; ++j) {
            const auto* k_j = key.data() + j * keys;
            const auto* g_j = cumulative.data() + j * keys;
            float coupling = 0.0F;
            for (std::size_t k = 0U; k < keys; ++k) {
                coupling += k_j[k] * k_t[k] * std::exp(g_t[k] - g_j[k]);
            }
            for (std::size_t v = 0U; v < values; ++v) {
                pseudo[t * values + v] -= coupling * pseudo[j * values + v];
            }
        }
        for (std::size_t v = 0U; v < values; ++v) {
            pseudo[t * values + v] *= beta[t];
        }
    }

    // Outputs: the decayed incoming state plus the causal intra-chunk tile.
    // The diagonal is retained, so token t reads the state after its own
    // update, matching the recurrence.
    for (std::size_t t = 0U; t < rows; ++t) {
        const auto* q_t = query.data() + t * keys;
        const auto* g_t = cumulative.data() + t * keys;
        auto* out = output.data() + t * values;
        for (std::size_t v = 0U; v < values; ++v) {
            const auto* row = state.data() + v * keys;
            float sum = 0.0F;
            for (std::size_t k = 0U; k < keys; ++k) {
                sum += row[k] * q_t[k] * std::exp(g_t[k]);
            }
            out[v] = sum;
        }
        for (std::size_t j = 0U; j <= t; ++j) {
            const auto* k_j = key.data() + j * keys;
            const auto* g_j = cumulative.data() + j * keys;
            float attention = 0.0F;
            for (std::size_t k = 0U; k < keys; ++k) {
                attention += k_j[k] * q_t[k] * std::exp(g_t[k] - g_j[k]);
            }
            for (std::size_t v = 0U; v < values; ++v) {
                out[v] += attention * pseudo[j * values + v];
            }
        }
    }

    // One state update for the whole chunk.
    const auto* g_last = cumulative.data() + (rows - 1U) * keys;
    for (std::size_t v = 0U; v < values; ++v) {
        auto* row = state.data() + v * keys;
        for (std::size_t k = 0U; k < keys; ++k) {
            row[k] *= std::exp(g_last[k]);
        }
        for (std::size_t j = 0U; j < rows; ++j) {
            const auto* k_j = key.data() + j * keys;
            const auto* g_j = cumulative.data() + j * keys;
            const auto scaled = pseudo[j * values + v];
            for (std::size_t k = 0U; k < keys; ++k) {
                row[k] += scaled * k_j[k] * std::exp(g_last[k] - g_j[k]);
            }
        }
    }
    return result;
}

}  // namespace strata
