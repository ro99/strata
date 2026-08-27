#include "strata/models/inkling/inkling_ops.hpp"

#include "strata/platform/compressed_tensors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strata {

namespace {

constexpr auto& kContract = kInklingExecutionContract;

}  // namespace

ValidationResult inkling_nvfp4_matvec_rows(
    const InklingNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end) {
    ValidationResult result;
    if (matrix.group_size == 0U || matrix.columns == 0U || matrix.rows == 0U ||
        matrix.columns % 2U != 0U ||
        matrix.packed_columns != matrix.columns / 2U ||
        matrix.scale_columns !=
            (matrix.columns + matrix.group_size - 1U) / matrix.group_size) {
        result.errors.emplace_back("NVFP4 matrix layout is not the target format");
        return result;
    }
    if (!std::isfinite(matrix.global_scale) || matrix.global_scale <= 0.0F) {
        result.errors.emplace_back("NVFP4 global scale must be finite and positive");
        return result;
    }
    if (matrix.packed.size() != matrix.rows * matrix.packed_columns ||
        matrix.scales.size() != matrix.rows * matrix.scale_columns) {
        result.errors.emplace_back("NVFP4 matrix payload size mismatch");
        return result;
    }
    if (input.size() != matrix.columns || output.size() != matrix.rows) {
        result.errors.emplace_back("NVFP4 matvec activation shape mismatch");
        return result;
    }
    if (row_begin > row_end || row_end > matrix.rows) {
        result.errors.emplace_back("NVFP4 matvec row range is out of bounds");
        return result;
    }
    const auto* packed =
        reinterpret_cast<const std::uint8_t*>(matrix.packed.data());
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(matrix.scales.data());
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const auto* packed_row = packed + row * matrix.packed_columns;
        const auto* scale_row = scales + row * matrix.scale_columns;
        float sum = 0.0F;
        for (std::uint64_t group = 0U; group < matrix.scale_columns; ++group) {
            // ModelOpt NVFP4 multiplies by the per-expert scale; see the
            // header for why the direction is pinned by measurement.
            const float scale =
                fp8_e4m3_f32(scale_row[group]) * matrix.global_scale;
            const auto begin = group * matrix.group_size;
            const auto end = std::min<std::uint64_t>(
                begin + matrix.group_size, matrix.columns);
            for (auto column = begin; column < end; ++column) {
                const auto byte = packed_row[column / 2U];
                const auto nibble = static_cast<std::uint8_t>(
                    column % 2U == 0U ? (byte & 0x0FU) : (byte >> 4U));
                sum += input[column] * fp4_e2m1_f32(nibble) * scale;
            }
        }
        output[row] = sum;
    }
    return result;
}

ValidationResult inkling_nvfp4_matvec_reference(
    const InklingNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output) {
    return inkling_nvfp4_matvec_rows(matrix, input, output, 0U, matrix.rows);
}

ValidationResult inkling_mxfp4_matvec_rows(
    const InklingMxfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end) {
    ValidationResult result;
    if (matrix.rows == 0U || matrix.columns == 0U ||
        matrix.group_size != 32U || matrix.columns % 32U != 0U ||
        matrix.packed_columns != matrix.columns / 2U ||
        matrix.scale_columns != matrix.columns / 32U ||
        matrix.packed.size() != matrix.rows * matrix.packed_columns ||
        matrix.scales.size() != matrix.rows * matrix.scale_columns) {
        result.errors.emplace_back("Inkling MXFP4 matrix layout is not group-32");
        return result;
    }
    if (input.size() != matrix.columns || output.size() != matrix.rows ||
        row_begin > row_end || row_end > matrix.rows) {
        result.errors.emplace_back("Inkling MXFP4 matvec shape mismatch");
        return result;
    }
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const auto packed = matrix.packed.subspan(
            static_cast<std::size_t>(row * matrix.packed_columns),
            static_cast<std::size_t>(matrix.packed_columns));
        const auto scales = matrix.scales.subspan(
            static_cast<std::size_t>(row * matrix.scale_columns),
            static_cast<std::size_t>(matrix.scale_columns));
        float sum = 0.0F;
        for (std::uint64_t group = 0U; group < matrix.scale_columns; ++group) {
            const float scale = mxfp4_scale_from_e8m0(
                std::to_integer<std::uint8_t>(scales[group]));
            if (!std::isfinite(scale)) {
                result.errors.emplace_back(
                    "Inkling MXFP4 matrix contains a reserved E8M0 scale");
                return result;
            }
            for (std::uint64_t offset = 0U; offset < 32U; ++offset) {
                const auto column = group * 32U + offset;
                const auto byte = std::to_integer<std::uint8_t>(
                    packed[column / 2U]);
                const auto code = static_cast<std::uint8_t>(
                    offset % 2U == 0U ? byte & 0x0FU : byte >> 4U);
                sum += input[column] * kMxfp4Values[code] * scale;
            }
        }
        output[row] = sum;
    }
    return result;
}

ValidationResult inkling_mxfp4_matvec_reference(
    const InklingMxfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output) {
    return inkling_mxfp4_matvec_rows(matrix, input, output, 0U, matrix.rows);
}

ValidationResult inkling_short_conv_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const float> history, std::span<const float> weight,
    std::uint64_t tokens, std::uint64_t channels, std::uint32_t kernel) {
    ValidationResult result;
    if (kernel == 0U || channels == 0U) {
        result.errors.emplace_back("short convolution needs a kernel and channels");
        return result;
    }
    if (input.size() != tokens * channels || output.size() != input.size()) {
        result.errors.emplace_back("short convolution stream shape mismatch");
        return result;
    }
    if (weight.size() != channels * kernel) {
        result.errors.emplace_back("short convolution weight shape mismatch");
        return result;
    }
    const auto taps = static_cast<std::uint64_t>(kernel) - 1U;
    if (history.size() % channels != 0U || history.size() > taps * channels) {
        result.errors.emplace_back("short convolution history shape mismatch");
        return result;
    }
    const auto history_rows = channels == 0U ? 0U : history.size() / channels;
    // Rows are addressed on a single timeline whose origin is the first history
    // row, so a tap that reaches back past the start of the sequence simply
    // falls off the front and contributes nothing.
    const auto origin = static_cast<std::int64_t>(history_rows);
    for (std::uint64_t token = 0U; token < tokens; ++token) {
        const auto position = origin + static_cast<std::int64_t>(token);
        for (std::uint64_t channel = 0U; channel < channels; ++channel) {
            float sum = 0.0F;
            for (std::uint32_t tap = 0U; tap < kernel; ++tap) {
                const auto row = position - static_cast<std::int64_t>(taps) +
                                 static_cast<std::int64_t>(tap);
                if (row < 0) continue;
                const float value =
                    row < origin
                        ? history[static_cast<std::uint64_t>(row) * channels +
                                  channel]
                        : input[(static_cast<std::uint64_t>(row) -
                                 history_rows) *
                                    channels +
                                channel];
                sum += weight[channel * kernel + tap] * value;
            }
            // The reference convolution carries its own residual, so the
            // module is an identity when the weights are zero.
            output[token * channels + channel] =
                input[token * channels + channel] + sum;
        }
    }
    return result;
}

float inkling_log_scaling_tau(std::uint64_t position) noexcept {
    const auto effective = static_cast<double>(position) + 1.0;
    const auto floor_value =
        static_cast<double>(kContract.log_scaling_position_floor);
    const double ratio = std::max(1.0, effective / floor_value);
    return static_cast<float>(
        1.0 + static_cast<double>(kContract.log_scaling_alpha) * std::log(ratio));
}

ValidationResult inkling_relative_logits(
    std::span<float> output, std::span<const float> relative,
    std::span<const float> projection, std::uint32_t heads,
    std::uint32_t relative_dim, std::uint32_t extent, float tau) {
    ValidationResult result;
    if (heads == 0U || relative_dim == 0U || extent == 0U) {
        result.errors.emplace_back("relative logits need heads, width and extent");
        return result;
    }
    const auto expected_output =
        static_cast<std::size_t>(heads) * static_cast<std::size_t>(extent);
    if (relative.size() != static_cast<std::size_t>(heads) * relative_dim ||
        projection.size() !=
            static_cast<std::size_t>(relative_dim) * extent ||
        output.size() != expected_output) {
        result.errors.emplace_back("relative logits shape mismatch");
        return result;
    }
    if (!std::isfinite(tau)) {
        result.errors.emplace_back("relative logits scaling must be finite");
        return result;
    }
    for (std::uint32_t head = 0U; head < heads; ++head) {
        auto* row = output.data() + static_cast<std::size_t>(head) * extent;
        std::fill(row, row + extent, 0.0F);
        for (std::uint32_t dimension = 0U; dimension < relative_dim; ++dimension) {
            const float value =
                relative[static_cast<std::size_t>(head) * relative_dim + dimension];
            const auto* projection_row =
                projection.data() + static_cast<std::size_t>(dimension) * extent;
            for (std::uint32_t bucket = 0U; bucket < extent; ++bucket) {
                row[bucket] += value * projection_row[bucket];
            }
        }
        if (tau != 1.0F) {
            for (std::uint32_t bucket = 0U; bucket < extent; ++bucket) {
                row[bucket] *= tau;
            }
        }
    }
    return result;
}

InklingRouteResult inkling_route_sigmoid_sink(
    std::span<const float> logits, std::span<const float> correction_bias,
    const RouterSpec& spec, std::uint32_t shared_experts, float global_scale) {
    InklingRouteResult result;
    const auto routed = spec.routed_experts;
    const auto top_k = spec.experts_per_token;
    if (routed == 0U || top_k == 0U || top_k > routed) {
        result.errors.emplace_back("Inkling router selection is out of range");
        return result;
    }
    if (spec.selection != RouterSelectionKind::TopK ||
        spec.scoring != RouterScoreKind::Sigmoid || !spec.selection_bias ||
        !spec.normalize_topk) {
        result.errors.emplace_back("Inkling router requires sigmoid top-k with bias");
        return result;
    }
    if (logits.size() != static_cast<std::size_t>(routed) + shared_experts) {
        result.errors.emplace_back("Inkling router logit count mismatch");
        return result;
    }
    if (correction_bias.size() != routed) {
        result.errors.emplace_back("Inkling router correction bias size mismatch");
        return result;
    }
    if (!std::isfinite(global_scale)) {
        result.errors.emplace_back("Inkling router global scale must be finite");
        return result;
    }
    for (const float value : logits) {
        if (!std::isfinite(value)) {
            result.errors.emplace_back("Inkling router logits contain a non-finite value");
            return result;
        }
    }

    // Selection runs on sigmoid(logit) + bias over the routed range. The sinks
    // are deliberately absent here: they score, but they never compete.
    std::vector<float> selection(routed);
    for (std::uint32_t expert = 0U; expert < routed; ++expert) {
        selection[expert] =
            sigmoid_f32(logits[expert]) + correction_bias[expert];
    }

    const auto active = static_cast<std::size_t>(top_k) + shared_experts;
    result.value.experts.reserve(active);
    result.value.weights.reserve(active);
    std::vector<float> raw;
    raw.reserve(active);
    for (std::uint32_t choice = 0U; choice < top_k; ++choice) {
        std::uint32_t best = 0U;
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t expert = 0U; expert < routed; ++expert) {
            // Strictly greater keeps the lower index on a tie, matching the
            // reference's argmax.
            if (selection[expert] > best_score) {
                best_score = selection[expert];
                best = expert;
            }
        }
        result.value.experts.push_back(best);
        raw.push_back(logits[best]);
        selection[best] = -std::numeric_limits<float>::infinity();
    }
    for (std::uint32_t shared = 0U; shared < shared_experts; ++shared) {
        result.value.experts.push_back(routed + shared);
        raw.push_back(logits[routed + shared]);
    }

    // Renormalization is a softmax over the log-sigmoid of the raw logits of
    // the selected experts and every sink, not over the sigmoid scores used
    // for selection, and not over the biased scores.
    std::vector<float> log_probabilities(raw.size());
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0U; index < raw.size(); ++index) {
        log_probabilities[index] = log_sigmoid_f32(raw[index]);
        maximum = std::max(maximum, log_probabilities[index]);
    }
    float total = 0.0F;
    for (std::size_t index = 0U; index < raw.size(); ++index) {
        log_probabilities[index] = std::exp(log_probabilities[index] - maximum);
        total += log_probabilities[index];
    }
    if (!(total > 0.0F) || !std::isfinite(total)) {
        result.errors.emplace_back("Inkling router renormalization did not converge");
        return result;
    }
    const float scale = spec.routed_scale * global_scale;
    for (std::size_t index = 0U; index < raw.size(); ++index) {
        result.value.weights.push_back(log_probabilities[index] / total * scale);
    }
    return result;
}

bool inkling_attention_visible(std::uint64_t query, std::uint64_t key,
                               bool local,
                               std::uint32_t sliding_window) noexcept {
    if (key > query) return false;
    if (!local) return true;
    if (sliding_window == 0U) return false;
    return query - key < sliding_window;
}

ValidationResult inkling_interleaved_swiglu_f32(std::span<float> output,
                                                std::span<const float> gate_up) {
    ValidationResult result;
    if (gate_up.size() != output.size() * 2U || output.empty()) {
        result.errors.emplace_back("interleaved SwiGLU shape mismatch");
        return result;
    }
    for (std::size_t index = 0U; index < output.size(); ++index) {
        const float gate = gate_up[index * 2U];
        const float up = gate_up[index * 2U + 1U];
        output[index] = silu_f32(gate) * up;
    }
    return result;
}

}  // namespace strata
