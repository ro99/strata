#include "strata/inkling_runtime.hpp"

#include "strata/compressed_tensors.hpp"
#include "strata/inkling_checkpoint.hpp"
#include "strata/inkling_device.hpp"
#include "strata/inkling_ops.hpp"
#include "strata/model.hpp"
#include "strata/model_adapter.hpp"
#include "strata/numerics.hpp"
#include "strata/runtime_support.hpp"
#include "strata/tokenizer.hpp"
#include "strata/trace.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <utility>
#include <random>
#include <thread>

namespace strata {
namespace {

constexpr auto& kContract = kInklingExecutionContract;
constexpr std::uint32_t kLayers = kContract.layer_count;
constexpr std::uint32_t kHidden = kContract.hidden_size;
constexpr std::uint32_t kHeadDim = kContract.head_dim;
constexpr std::uint32_t kHeads = kContract.attention_heads;
constexpr std::uint32_t kKvHeads = kContract.key_value_heads;
constexpr std::uint32_t kQueryWidth = kHeads * kHeadDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::uint32_t kRelWidth = kHeads * kContract.relative_dim;
constexpr std::uint32_t kExperts = kContract.routed_experts;
constexpr std::uint32_t kShared = kContract.shared_experts;
constexpr std::uint32_t kTopK = kContract.experts_per_token;
constexpr std::uint32_t kExpertInner = kContract.expert_intermediate_size;
constexpr std::uint32_t kExpertUp = 2U * kExpertInner;
constexpr std::uint32_t kDenseInner = kContract.dense_intermediate_size;
constexpr std::uint32_t kDenseUp = 2U * kDenseInner;
constexpr std::uint32_t kConvTaps = kContract.short_conv_kernel - 1U;

float decode_bf16(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

// The four short-convolution streams a layer owns, in reference order.
enum ConvStream : std::size_t {
    kConvK = 0U,
    kConvV,
    kConvAttn,
    kConvMlp,
    kConvStreams,
};

constexpr std::uint32_t conv_stream_width(std::size_t stream) noexcept {
    return stream == kConvK || stream == kConvV ? kKvWidth : kHidden;
}

// A BF16 matrix left in the checkpoint mapping. Decoding on use costs one
// shift per element and avoids materializing 160 GB as FP32.
struct MappedMatrix {
    InklingTensorEncoding encoding{InklingTensorEncoding::Plain};
    const std::uint16_t* data{};
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    std::uint64_t rows{};
    std::uint64_t columns{};

    [[nodiscard]] bool valid() const noexcept {
        return encoding == InklingTensorEncoding::Plain
            ? data != nullptr
            : !packed.empty() && !scales.empty();
    }
};

struct AttentionWeights {
    MappedMatrix query;
    MappedMatrix key;
    MappedMatrix value;
    MappedMatrix relative;
    MappedMatrix output;
    std::vector<float> relative_projection;
    std::vector<float> query_norm;
    std::vector<float> key_norm;
    std::vector<float> attention_norm;
    std::vector<float> mlp_norm;
    // Depthwise taps, channel-major [channels, kernel].
    std::array<std::vector<float>, kConvStreams> conv;
    bool global{};
    std::uint32_t relative_extent{};
};

// The spine projections of one layer, resident on a device. Uploading them
// once removes them from the host term permanently. Their exact footprint is
// checkpoint-format dependent and is admitted against the live VRAM budget.
struct DeviceLayer {
    std::size_t slot{};
    CudaWeight query;
    CudaWeight key;
    CudaWeight value;
    CudaWeight relative;
    CudaWeight output;
    CudaWeight dense_gate;
    CudaWeight dense_up;
    CudaWeight dense_down;
    CudaWeight gate;
    std::array<CudaWeight, 2U> shared_gate;
    std::array<CudaWeight, 2U> shared_up;
    std::array<CudaWeight, 2U> shared_down;
    CudaBuffer kv_cache;
    std::uint32_t kv_capacity_rows{};
    bool resident{};
};

struct LayerWeights {
    AttentionWeights attention;
    bool sparse{};
    MappedMatrix dense_gate_up;
    MappedMatrix dense_gate;
    MappedMatrix dense_up;
    MappedMatrix dense_down;
    float dense_global_scale{1.0F};
    MappedMatrix gate;
    std::vector<float> gate_bias;
    float gate_global_scale{1.0F};
    MappedMatrix shared_gate_up;
    MappedMatrix shared_gate;
    MappedMatrix shared_up;
    MappedMatrix shared_down;
    InklingExpertStack expert_gate;
    InklingExpertStack expert_up;
    InklingExpertStack expert_down;
    // Layer 2 only: routed experts ship plain BF16.
    const std::uint16_t* plain_gate_up{};
    const std::uint16_t* plain_down{};
};

struct MtpWeights {
    AttentionWeights attention;
    std::vector<float> embed_norm;
    std::vector<float> hidden_norm;
    MappedMatrix input_projection;
    MappedMatrix dense_gate_up;
    MappedMatrix dense_gate;
    MappedMatrix dense_up;
    MappedMatrix dense_down;
    float dense_global_scale{1.0F};
};

// Per-layer sequence state. Keys and values are stored already rounded through
// BF16, the reference cache dtype, but held as F32 so attention reads them
// directly instead of decoding the whole cache to add one row.
struct LayerState {
    std::vector<float> keys;
    std::vector<float> values;
    std::uint64_t rows{};
    std::uint64_t base_position{};
    bool device_kv_ready{};
    // Rolling conv inputs, kConvTaps rows per stream, oldest first.
    std::array<std::vector<float>, kConvStreams> conv_history;
};

void reset_conv_history(LayerState& state) {
    for (std::size_t stream = 0U; stream < kConvStreams; ++stream) {
        state.conv_history[stream].assign(
            static_cast<std::size_t>(kConvTaps) * conv_stream_width(stream),
            0.0F);
    }
}

std::uint64_t elapsed_since(
    std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
}

}  // namespace

struct InklingRuntime::Impl {
    InklingRuntimeConfig config;
    std::unique_ptr<InklingCheckpointReader> checkpoint;
    ModelTokenizer tokenizer;
    std::unique_ptr<HostWorkerPool> workers;
    std::unique_ptr<HostWorkerPool> device_workers;
    RouteTraceWriter route_trace;
    std::vector<LayerWeights> layers{kLayers};
    std::vector<MtpWeights> mtp;
    MappedMatrix embedding;
    MappedMatrix unembedding;
    std::vector<float> embedding_norm;
    std::vector<float> final_norm;
    std::vector<LayerState> state{kLayers};
    std::vector<LayerState> mtp_state;
    RouterSpec router;
    CudaBackend cuda;
    std::vector<int> devices;
    std::vector<std::size_t> device_schedule;
    std::vector<DeviceLayer> device_layers;
    std::unique_ptr<InklingExpertCache> expert_cache;
    CudaWeight device_unembed;
    std::vector<std::uint64_t> resident_spine_bytes;
    std::vector<std::uint64_t> resident_kv_bytes;
    bool cuda_enabled{};
    std::uint64_t position{};
    InklingGraphStats graph;
    std::mt19937_64 sampler;
    SamplingOptions active_sampling;
    TokenLogprob last_sample;
    bool initialized{};

    [[nodiscard]] std::size_t expert_device(std::uint32_t layer,
                                            std::uint32_t expert,
                                            std::size_t layer_slot) const {
        if (!config.enable_expert_parallel || device_schedule.size() <= 1U) {
            return layer_slot;
        }
        const auto index =
            (static_cast<std::size_t>(layer) * kExperts + expert) %
            device_schedule.size();
        return device_schedule[index];
    }

    bool write_route(std::uint64_t token_position, std::uint32_t layer,
                     const InklingRoute& route, bool prefill) {
        if (!route_trace.is_open()) return true;
        RouteEvent event;
        event.request = config.request_id;
        event.token_position = token_position;
        event.layer = layer;
        event.experts.assign(route.experts.begin(),
                             route.experts.begin() + kTopK);
        event.coefficients.assign(route.weights.begin(),
                                  route.weights.begin() + kTopK);
        event.phase = prefill ? RoutePhase::Prefill : RoutePhase::Decode;
        return route_trace.write(event).ok();
    }

    // ---- primitive kernels -------------------------------------------------

    // output[r] = sum_c weight[r][c] * input[c], FP32 accumulation over BF16
    // storage. Row blocks are independent, so this is the only place needing
    // the worker pool.
    ValidationResult matvec(const MappedMatrix& weight,
                            std::span<const float> input,
                            std::span<float> output) {
        ValidationResult result;
        if (!weight.valid() || input.size() != weight.columns ||
            output.size() != weight.rows) {
            result.errors.emplace_back("Inkling matvec shape mismatch");
            return result;
        }
        if (weight.encoding == InklingTensorEncoding::Mxfp4Group32) {
            InklingMxfp4MatrixView matrix;
            matrix.packed = weight.packed;
            matrix.scales = weight.scales;
            matrix.rows = weight.rows;
            matrix.columns = weight.columns;
            matrix.packed_columns = weight.columns / 2U;
            matrix.scale_columns = weight.columns / 32U;
            const auto blocks = std::max<std::uint64_t>(
                1U, std::min<std::uint64_t>(
                        weight.rows, workers == nullptr ? 1U : workers->size()));
            const auto per_block = (weight.rows + blocks - 1U) / blocks;
            std::vector<ValidationResult> failures(
                static_cast<std::size_t>(blocks));
            const auto body = [&](std::size_t block) {
                const auto begin = static_cast<std::uint64_t>(block) * per_block;
                const auto end = std::min(begin + per_block, weight.rows);
                failures[block] = inkling_mxfp4_matvec_rows(
                    matrix, input, output, begin, end);
            };
            if (workers == nullptr || blocks <= 1U) {
                for (std::size_t block = 0U; block < blocks; ++block) body(block);
            } else {
                result = workers->parallel_for(static_cast<std::size_t>(blocks),
                                               body);
                if (!result.ok()) return result;
            }
            for (auto& failure : failures) {
                if (!failure.ok()) return failure;
            }
            return result;
        }
        const auto rows = weight.rows;
        const auto columns = weight.columns;
        const auto* data = weight.data;
        const auto* in = input.data();
        auto* out = output.data();
        const auto blocks = std::max<std::uint64_t>(
            1U, std::min<std::uint64_t>(
                    rows, workers == nullptr ? 1U : workers->size()));
        const auto per_block = (rows + blocks - 1U) / blocks;
        const auto body = [=](std::size_t block) {
            const auto begin = static_cast<std::uint64_t>(block) * per_block;
            const auto end = std::min(begin + per_block, rows);
            for (auto row = begin; row < end; ++row) {
                const auto* weights = data + row * columns;
                float sum = 0.0F;
                for (std::uint64_t column = 0U; column < columns; ++column) {
                    sum += decode_bf16(weights[column]) * in[column];
                }
                out[row] = sum;
            }
        };
        if (workers == nullptr || blocks <= 1U) {
            for (std::uint64_t block = 0U; block < blocks; ++block) {
                body(static_cast<std::size_t>(block));
            }
            return result;
        }
        return workers->parallel_for(static_cast<std::size_t>(blocks), body);
    }

    // Runs a resident device projection, falling back to the host matvec when
    // the weight was never uploaded. Every call site passes both so a partial
    // upload degrades in speed rather than in correctness.
    ValidationResult spine_matvec(const CudaWeight& resident,
                                  const MappedMatrix& host,
                                  std::span<const float> input,
                                  std::span<float> output) {
        if (cuda_enabled && resident.valid()) {
            return cuda.matmul(resident, input, 1U, output);
        }
        return matvec(host, input, output);
    }

    static MappedMatrix matrix_slice(const MappedMatrix& stack,
                                     std::uint64_t slice,
                                     std::uint64_t rows,
                                     std::uint64_t columns) {
        MappedMatrix matrix = stack;
        matrix.rows = rows;
        matrix.columns = columns;
        if (stack.encoding == InklingTensorEncoding::Plain) {
            matrix.data = stack.data + slice * rows * columns;
        } else {
            const auto packed_bytes = rows * columns / 2U;
            const auto scale_bytes = rows * columns / 32U;
            matrix.packed = stack.packed.subspan(
                static_cast<std::size_t>(slice * packed_bytes),
                static_cast<std::size_t>(packed_bytes));
            matrix.scales = stack.scales.subspan(
                static_cast<std::size_t>(slice * scale_bytes),
                static_cast<std::size_t>(scale_bytes));
        }
        return matrix;
    }

    // The NVFP4 equivalent, over one expert slice of a stacked projection.
    ValidationResult expert_matvec(const InklingNvfp4MatrixView& matrix,
                                   std::span<const float> input,
                                   std::span<float> output) {
        const auto rows = matrix.rows;
        const auto blocks = std::max<std::uint64_t>(
            1U, std::min<std::uint64_t>(
                    rows, workers == nullptr ? 1U : workers->size()));
        const auto per_block = (rows + blocks - 1U) / blocks;
        std::vector<ValidationResult> failures(static_cast<std::size_t>(blocks));
        const auto body = [&](std::size_t block) {
            const auto begin = static_cast<std::uint64_t>(block) * per_block;
            const auto end = std::min(begin + per_block, rows);
            failures[block] =
                inkling_nvfp4_matvec_rows(matrix, input, output, begin, end);
        };
        ValidationResult result;
        if (workers == nullptr || blocks <= 1U) {
            for (std::uint64_t block = 0U; block < blocks; ++block) {
                body(static_cast<std::size_t>(block));
            }
        } else {
            result = workers->parallel_for(static_cast<std::size_t>(blocks),
                                           body);
            if (!result.ok()) return result;
        }
        for (auto& failure : failures) {
            if (!failure.ok()) return failure;
        }
        return result;
    }

    ValidationResult expert_matvec(const InklingMxfp4MatrixView& matrix,
                                   std::span<const float> input,
                                   std::span<float> output) {
        const auto blocks = std::max<std::uint64_t>(
            1U, std::min<std::uint64_t>(
                    matrix.rows, workers == nullptr ? 1U : workers->size()));
        const auto per_block = (matrix.rows + blocks - 1U) / blocks;
        std::vector<ValidationResult> failures(static_cast<std::size_t>(blocks));
        const auto body = [&](std::size_t block) {
            const auto begin = static_cast<std::uint64_t>(block) * per_block;
            const auto end = std::min(begin + per_block, matrix.rows);
            failures[block] = inkling_mxfp4_matvec_rows(
                matrix, input, output, begin, end);
        };
        ValidationResult result;
        if (workers == nullptr || blocks <= 1U) {
            for (std::size_t block = 0U; block < blocks; ++block) body(block);
        } else {
            result = workers->parallel_for(static_cast<std::size_t>(blocks), body);
            if (!result.ok()) return result;
        }
        for (auto& failure : failures) {
            if (!failure.ok()) return failure;
        }
        return result;
    }

    ValidationResult plain_expert_matvec(const std::uint16_t* base,
                                         std::uint32_t expert,
                                         std::uint64_t rows,
                                         std::uint64_t columns,
                                         std::span<const float> input,
                                         std::span<float> output) {
        ValidationResult result;
        if (base == nullptr) {
            result.errors.emplace_back("Inkling plain expert stack is not mapped");
            return result;
        }
        MappedMatrix slice;
        slice.data = base + static_cast<std::uint64_t>(expert) * rows * columns;
        slice.rows = rows;
        slice.columns = columns;
        return matvec(slice, input, output);
    }

    static void rms_norm(std::span<float> output, std::span<const float> input,
                         std::span<const float> weight) {
        double squared = 0.0;
        for (const float value : input) {
            squared += static_cast<double>(value) * value;
        }
        const auto mean =
            static_cast<float>(squared / static_cast<double>(input.size()));
        const float reciprocal = 1.0F / std::sqrt(mean + kContract.rms_epsilon);
        for (std::size_t index = 0U; index < input.size(); ++index) {
            output[index] = input[index] * reciprocal * weight[index];
        }
    }

    // Per-head RMSNorm over head_dim, which is why the attention scale is
    // 1/head_dim rather than its square root.
    static void head_norm(std::span<float> values, std::span<const float> weight,
                          std::uint32_t heads) {
        for (std::uint32_t head = 0U; head < heads; ++head) {
            auto* row = values.data() + static_cast<std::size_t>(head) * kHeadDim;
            double squared = 0.0;
            for (std::uint32_t index = 0U; index < kHeadDim; ++index) {
                squared += static_cast<double>(row[index]) * row[index];
            }
            const auto mean =
                static_cast<float>(squared / static_cast<double>(kHeadDim));
            const float reciprocal =
                1.0F / std::sqrt(mean + kContract.rms_epsilon);
            for (std::uint32_t index = 0U; index < kHeadDim; ++index) {
                row[index] = row[index] * reciprocal * weight[index];
            }
        }
    }

    // Convolves one stream for a single token and rolls its history. The
    // history holds the module's raw inputs, not its outputs, because every
    // tap reads the input stream.
    ValidationResult short_conv(LayerState& layer_state,
                                const AttentionWeights& weights,
                                std::size_t stream, std::span<float> values) {
        const auto started = std::chrono::steady_clock::now();
        const auto width = conv_stream_width(stream);
        auto& history = layer_state.conv_history[stream];
        std::vector<float> output(values.size());
        auto result = inkling_short_conv_f32(output, values, history,
                                             weights.conv[stream], 1U, width,
                                             kContract.short_conv_kernel);
        if (!result.ok()) return result;
        std::rotate(history.begin(),
                    history.begin() + static_cast<std::ptrdiff_t>(width),
                    history.end());
        std::copy(values.begin(), values.end(),
                  history.end() - static_cast<std::ptrdiff_t>(width));
        std::copy(output.begin(), output.end(), values.begin());
        graph.short_conv_nanoseconds += elapsed_since(started);
        return result;
    }

    // ---- attention ---------------------------------------------------------

    ValidationResult attention(const AttentionWeights& weights,
                               const DeviceLayer& device,
                               LayerState& layer_state,
                               std::span<const float> input,
                               std::uint64_t token_position,
                               std::span<float> output) {
        std::vector<float> query(kQueryWidth);
        std::vector<float> key(kKvWidth);
        std::vector<float> value(kKvWidth);
        std::vector<float> relative(kRelWidth);
        auto result = spine_matvec(device.query, weights.query, input, query);
        if (!result.ok()) return result;
        result = spine_matvec(device.key, weights.key, input, key);
        if (!result.ok()) return result;
        result = spine_matvec(device.value, weights.value, input, value);
        if (!result.ok()) return result;
        result = spine_matvec(device.relative, weights.relative, input, relative);
        if (!result.ok()) return result;

        // K and V are convolved after projection and before their norm.
        result = short_conv(layer_state, weights, kConvK, key);
        if (!result.ok()) return result;
        result = short_conv(layer_state, weights, kConvV, value);
        if (!result.ok()) return result;

        head_norm(query, weights.query_norm, kHeads);
        head_norm(key, weights.key_norm, kKvHeads);

        // Long-context temperature reaches the queries and the position bias,
        // and only on full-attention layers.
        const float tau =
            weights.global ? inkling_log_scaling_tau(token_position) : 1.0F;
        if (tau != 1.0F) {
            for (auto& element : query) element *= tau;
        }

        const auto extent = weights.relative_extent;
        std::vector<float> bias(static_cast<std::size_t>(kHeads) * extent);
        result = inkling_relative_logits(bias, relative,
                                         weights.relative_projection, kHeads,
                                         kContract.relative_dim, extent, tau);
        if (!result.ok()) return result;

        for (std::uint32_t index = 0U; index < kKvWidth; ++index) {
            layer_state.keys.push_back(bf16_round_f32(key[index]));
            layer_state.values.push_back(bf16_round_f32(value[index]));
        }
        ++layer_state.rows;

        const bool local = !weights.global;
        const auto window = kContract.sliding_window;
        if (local && layer_state.rows > window) {
            const auto drop = layer_state.rows - window;
            layer_state.keys.erase(
                layer_state.keys.begin(),
                layer_state.keys.begin() +
                    static_cast<std::ptrdiff_t>(drop * kKvWidth));
            layer_state.values.erase(
                layer_state.values.begin(),
                layer_state.values.begin() +
                    static_cast<std::ptrdiff_t>(drop * kKvWidth));
            layer_state.rows -= drop;
            layer_state.base_position += drop;
        }
        std::vector<float> context(kQueryWidth, 0.0F);
        const auto heads_per_kv = kHeads / kKvHeads;
        const bool use_device =
            config.enable_device_kv_attention && device.kv_cache.valid() &&
            layer_state.rows >= config.minimum_device_attention_rows;
        if (use_device) {
            if (!layer_state.device_kv_ready) {
                std::vector<std::uint16_t> encoded_keys(layer_state.keys.size());
                std::vector<std::uint16_t> encoded_values(layer_state.values.size());
                std::transform(layer_state.keys.begin(), layer_state.keys.end(),
                               encoded_keys.begin(), bf16_encode);
                std::transform(layer_state.values.begin(), layer_state.values.end(),
                               encoded_values.begin(), bf16_encode);
                result = cuda.upload_gemma4_kv(
                    device.kv_cache, encoded_keys, encoded_values,
                    static_cast<std::uint32_t>(layer_state.base_position),
                    device.kv_capacity_rows, kKvWidth);
                if (!result.ok()) return result;
                layer_state.device_kv_ready = true;
            }
            std::vector<std::uint16_t> encoded_key(kKvWidth);
            std::vector<std::uint16_t> encoded_value(kKvWidth);
            std::transform(key.begin(), key.end(), encoded_key.begin(), bf16_encode);
            std::transform(value.begin(), value.end(), encoded_value.begin(),
                           bf16_encode);
            CudaBf16KvAttentionRequest request;
            request.cache = &device.kv_cache;
            request.queries = query;
            request.next_keys = encoded_key;
            request.next_values = encoded_value;
            request.relative_bias = bias;
            request.query_heads = kHeads;
            request.key_value_heads = kKvHeads;
            request.head_dim = kHeadDim;
            request.capacity_rows = device.kv_capacity_rows;
            request.cache_start =
                static_cast<std::uint32_t>(layer_state.base_position);
            request.cached_rows = static_cast<std::uint32_t>(layer_state.rows);
            request.position = static_cast<std::uint32_t>(token_position);
            request.relative_bias_extent = extent;
            request.scale = kContract.attention_scale;
            result = cuda.bf16_kv_attention(
                devices[device.slot], request, context);
            if (!result.ok()) return result;
        } else {
            std::vector<float> scores(static_cast<std::size_t>(layer_state.rows));
            std::vector<std::uint64_t> visible;
            visible.reserve(static_cast<std::size_t>(layer_state.rows));
            for (std::uint64_t row = 0U; row < layer_state.rows; ++row) {
                const auto key_position = layer_state.base_position + row;
                if (inkling_attention_visible(token_position, key_position, local,
                                              window)) {
                    visible.push_back(row);
                }
            }
            if (visible.empty()) {
                result.errors.emplace_back("Inkling attention has no visible keys");
                return result;
            }

            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                const auto kv_head = head / heads_per_kv;
                const auto* query_row =
                    query.data() + static_cast<std::size_t>(head) * kHeadDim;
                const auto* bias_row =
                    bias.data() + static_cast<std::size_t>(head) * extent;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t index = 0U; index < visible.size(); ++index) {
                    const auto row = visible[index];
                    const auto key_position = layer_state.base_position + row;
                    const auto* key_row =
                        layer_state.keys.data() + row * kKvWidth +
                        static_cast<std::size_t>(kv_head) * kHeadDim;
                    float dot = 0.0F;
                    for (std::uint32_t element = 0U; element < kHeadDim; ++element) {
                        dot += query_row[element] * key_row[element];
                    }
                    float score = dot * kContract.attention_scale;
                    // The bias is zero outside its extent, which is what makes a
                    // global layer content-only beyond 1024 tokens.
                    const auto distance = token_position - key_position;
                    if (distance < extent) score += bias_row[distance];
                    scores[index] = score;
                    maximum = std::max(maximum, score);
                }
                float total = 0.0F;
                for (std::size_t index = 0U; index < visible.size(); ++index) {
                    scores[index] = std::exp(scores[index] - maximum);
                    total += scores[index];
                }
                auto* context_row =
                    context.data() + static_cast<std::size_t>(head) * kHeadDim;
                for (std::size_t index = 0U; index < visible.size(); ++index) {
                    const float weight = scores[index] / total;
                    const auto* value_row =
                        layer_state.values.data() + visible[index] * kKvWidth +
                        static_cast<std::size_t>(kv_head) * kHeadDim;
                    for (std::uint32_t element = 0U; element < kHeadDim; ++element) {
                        context_row[element] += weight * value_row[element];
                    }
                }
            }
        }

        result = spine_matvec(device.output, weights.output, context, output);
        if (!result.ok()) return result;
        return short_conv(layer_state, weights, kConvAttn, output);
    }

    // ---- feed-forward ------------------------------------------------------

    ValidationResult dense_mlp(const MappedMatrix& fused_gate_up,
                               const MappedMatrix& gate,
                               const MappedMatrix& up,
                               const MappedMatrix& down, float global_scale,
                               const DeviceLayer* device,
                               std::span<const float> input,
                               std::span<float> output) {
        ValidationResult result;
        const auto half = static_cast<std::uint64_t>(kDenseInner);
        std::vector<float> activated(half);
        if (cuda_enabled && device != nullptr && device->dense_gate.valid()) {
            // The device holds gate and up already de-interleaved, so the
            // strided read the host path needs disappears here.
            std::vector<float> gate_output(half);
            std::vector<float> up_output(half);
            result = cuda.matmul(device->dense_gate, input, 1U, gate_output);
            if (!result.ok()) return result;
            result = cuda.matmul(device->dense_up, input, 1U, up_output);
            if (!result.ok()) return result;
            for (std::uint64_t index = 0U; index < half; ++index) {
                activated[index] =
                    silu_f32(gate_output[index]) * up_output[index];
            }
            result = cuda.matmul(device->dense_down, activated, 1U, output);
            if (!result.ok()) return result;
            for (auto& element : output) element *= global_scale;
            return result;
        }
        if (gate.valid() && up.valid()) {
            std::vector<float> gate_output(half);
            std::vector<float> up_output(half);
            result = matvec(gate, input, gate_output);
            if (!result.ok()) return result;
            result = matvec(up, input, up_output);
            if (!result.ok()) return result;
            for (std::uint64_t index = 0U; index < half; ++index) {
                activated[index] = silu_f32(gate_output[index]) * up_output[index];
            }
        } else {
            std::vector<float> fused(fused_gate_up.rows);
            result = matvec(fused_gate_up, input, fused);
            if (!result.ok()) return result;
            result = inkling_interleaved_swiglu_f32(activated, fused);
            if (!result.ok()) return result;
        }
        result = matvec(down, activated, output);
        if (!result.ok()) return result;
        for (auto& element : output) element *= global_scale;
        return result;
    }

    ValidationResult moe(const LayerWeights& layer, const DeviceLayer& device,
                         std::uint32_t index, std::uint64_t token_position,
                         bool prefill, std::span<const float> input,
                         std::span<float> output) {
        auto started = std::chrono::steady_clock::now();
        std::vector<float> logits(layer.gate.rows);
        auto result = spine_matvec(device.gate, layer.gate, input, logits);
        if (!result.ok()) return result;
        auto route = inkling_route_sigmoid_sink(logits, layer.gate_bias, router,
                                                kShared,
                                                layer.gate_global_scale);
        if (!route.ok()) {
            result.errors = std::move(route.errors);
            return result;
        }
        if (!write_route(token_position, index, route.value, prefill)) {
            result.errors.emplace_back("cannot write Inkling route trace");
            return result;
        }
        graph.moe_router_nanoseconds += elapsed_since(started);

        if (cuda_enabled && device.resident) {
            return device_moe(layer, device, index, route.value, input, output);
        }

        std::fill(output.begin(), output.end(), 0.0F);
        std::vector<float> fused(kExpertUp);
        std::vector<float> activated(kExpertInner);
        std::vector<float> partial(kHidden);

        started = std::chrono::steady_clock::now();
        for (std::uint32_t choice = 0U; choice < kTopK; ++choice) {
            const auto expert = route.value.experts[choice];
            const float weight = route.value.weights[choice];
            result = routed_expert(layer, expert, input, fused, activated,
                                   partial);
            if (!result.ok()) return result;
            for (std::uint32_t element = 0U; element < kHidden; ++element) {
                output[element] += weight * partial[element];
            }
        }
        graph.moe_routed_nanoseconds += elapsed_since(started);

        // The sinks run on every token, weighted from the same renormalized
        // distribution as the routed experts.
        started = std::chrono::steady_clock::now();
        for (std::uint32_t shared = 0U; shared < kShared; ++shared) {
            const float weight = route.value.weights[kTopK + shared];
            const auto down = matrix_slice(layer.shared_down, shared, kHidden,
                                           kExpertInner);
            if (layer.shared_gate.valid() && layer.shared_up.valid()) {
                const auto gate = matrix_slice(layer.shared_gate, shared,
                                               kExpertInner, kHidden);
                const auto up = matrix_slice(layer.shared_up, shared,
                                             kExpertInner, kHidden);
                std::vector<float> gate_output(kExpertInner);
                std::vector<float> up_output(kExpertInner);
                result = matvec(gate, input, gate_output);
                if (!result.ok()) return result;
                result = matvec(up, input, up_output);
                if (!result.ok()) return result;
                for (std::uint32_t element = 0U; element < kExpertInner;
                     ++element) {
                    activated[element] =
                        silu_f32(gate_output[element]) * up_output[element];
                }
            } else {
                const auto gate_up = matrix_slice(
                    layer.shared_gate_up, shared, kExpertUp, kHidden);
                result = matvec(gate_up, input, fused);
                if (!result.ok()) return result;
                result = inkling_interleaved_swiglu_f32(activated, fused);
                if (!result.ok()) return result;
            }
            result = matvec(down, activated, partial);
            if (!result.ok()) return result;
            for (std::uint32_t element = 0U; element < kHidden; ++element) {
                output[element] += weight * partial[element];
            }
        }
        graph.moe_shared_nanoseconds += elapsed_since(started);
        return result;
    }

    // Two device commands per layer: one batch for the six routed experts and
    // one for both sinks. NVFP4 requires the split because a batch has one
    // encoding; MXFP4 retains it so routed and always-resident sink service
    // stay independently observable.
    ValidationResult device_moe(const LayerWeights& layer,
                                const DeviceLayer& device, std::uint32_t index,
                                const InklingRoute& route,
                                std::span<const float> input,
                                std::span<float> output) {
        ValidationResult result;
        std::fill(output.begin(), output.end(), 0.0F);

        // The NVFP4 batch requires unit coefficients: scaling before the down
        // projection is not float-equal to scaling after it, and the reference
        // scales after. Weights are applied on collection.
        const auto accumulate = [&](std::span<const float> collected,
                                    std::size_t experts, std::size_t offset) {
            for (std::size_t position = 0U; position < experts; ++position) {
                const float weight = route.weights[offset + position];
                const auto* block = collected.data() + position * kHidden;
                for (std::uint32_t element = 0U; element < kHidden; ++element) {
                    output[element] += weight * block[element];
                }
            }
        };

        const auto started = std::chrono::steady_clock::now();
        if (!config.enable_expert_parallel || devices.size() <= 1U) {
            std::vector<CudaMoeExpert> routed;
            routed.reserve(kTopK);
            std::vector<std::uint32_t> leased;
            leased.reserve(kTopK);
            const auto release_all = [&]() {
                for (const auto expert : leased) {
                    expert_cache->release(device.slot, index, expert);
                }
            };
            for (std::uint32_t choice = 0U; choice < kTopK; ++choice) {
                const auto expert = route.experts[choice];
                auto staged = expert_cache->acquire(
                    device.slot, index, expert, layer.expert_gate,
                    layer.expert_up, layer.expert_down);
                if (!staged.ok()) {
                    release_all();
                    result.errors = std::move(staged.errors);
                    return result;
                }
                leased.push_back(expert);
                routed.push_back(CudaMoeExpert{&staged.value->gate,
                                               &staged.value->up,
                                               &staged.value->down, 1.0F});
                graph.routed_expert_bytes += staged.value->device_bytes();
            }
            result = cuda.synchronize_uploads(devices[device.slot]);
            if (result.ok()) {
                result = cuda.enqueue_moe(devices[device.slot], input, 1U,
                                          routed, nullptr);
            }
            std::vector<float> collected(routed.size() * kHidden);
            if (result.ok()) {
                result = cuda.collect_moe(devices[device.slot], collected, {});
            }
            release_all();
            if (!result.ok()) return result;
            accumulate(collected, routed.size(), 0U);
            graph.moe_routed_nanoseconds += elapsed_since(started);
        } else {
        std::array<std::size_t, kTopK> owners{};
        for (std::uint32_t choice = 0U; choice < kTopK; ++choice) {
            owners[choice] = expert_device(index, route.experts[choice],
                                           device.slot);
        }
        std::vector<float> routed_collected(
            static_cast<std::size_t>(kTopK) * kHidden);
        std::vector<std::vector<std::string>> device_errors(devices.size());
        std::vector<std::uint64_t> device_bytes(devices.size());
        auto dispatched = device_workers->parallel_for(
            devices.size(), [&](std::size_t slot) {
                std::vector<std::uint32_t> choices;
                for (std::uint32_t choice = 0U; choice < kTopK; ++choice) {
                    if (owners[choice] == slot) choices.push_back(choice);
                }
                if (choices.empty()) return;

                ValidationResult status;
                std::vector<CudaMoeExpert> batch;
                std::vector<std::uint32_t> leased;
                batch.reserve(choices.size());
                leased.reserve(choices.size());
                const auto release_all = [&]() {
                    for (const auto expert : leased) {
                        expert_cache->release(slot, index, expert);
                    }
                };
                for (const auto choice : choices) {
                    const auto expert = route.experts[choice];
                    auto staged = expert_cache->acquire(
                        slot, index, expert, layer.expert_gate, layer.expert_up,
                        layer.expert_down);
                    if (!staged.ok()) {
                        release_all();
                        device_errors[slot] = std::move(staged.errors);
                        return;
                    }
                    leased.push_back(expert);
                    batch.push_back(CudaMoeExpert{&staged.value->gate,
                                                  &staged.value->up,
                                                  &staged.value->down, 1.0F});
                    device_bytes[slot] += staged.value->device_bytes();
                }
                status = cuda.synchronize_uploads(devices[slot]);
                if (status.ok()) {
                    status = cuda.enqueue_moe(devices[slot], input, 1U, batch,
                                              nullptr);
                }
                std::vector<float> local(choices.size() * kHidden);
                if (status.ok()) {
                    status = cuda.collect_moe(devices[slot], local, {});
                }
                release_all();
                if (!status.ok()) {
                    device_errors[slot] = std::move(status.errors);
                    return;
                }
                for (std::size_t choice_index = 0U;
                     choice_index < choices.size(); ++choice_index) {
                    std::copy_n(
                        local.begin() + static_cast<std::ptrdiff_t>(
                                            choice_index * kHidden),
                        kHidden,
                        routed_collected.begin() + static_cast<std::ptrdiff_t>(
                                                       choices[choice_index] *
                                                       kHidden));
                }
            });
        if (!dispatched.ok()) return dispatched;
        for (auto& errors : device_errors) {
            result.errors.insert(result.errors.end(),
                                 std::make_move_iterator(errors.begin()),
                                 std::make_move_iterator(errors.end()));
        }
        if (!result.ok()) return result;
        for (const auto bytes : device_bytes) graph.routed_expert_bytes += bytes;
        accumulate(routed_collected, kTopK, 0U);
        graph.moe_routed_nanoseconds += elapsed_since(started);
        }

        const auto shared_started = std::chrono::steady_clock::now();
        std::vector<CudaMoeExpert> sinks;
        sinks.reserve(kShared);
        for (std::uint32_t shared = 0U; shared < kShared; ++shared) {
            sinks.push_back(CudaMoeExpert{&device.shared_gate[shared],
                                          &device.shared_up[shared],
                                          &device.shared_down[shared], 1.0F});
        }
        result = cuda.enqueue_moe(devices[device.slot], input, 1U, sinks,
                                  nullptr);
        if (!result.ok()) return result;
        std::vector<float> shared_collected(sinks.size() * kHidden);
        result = cuda.collect_moe(devices[device.slot], shared_collected, {});
        if (!result.ok()) return result;
        accumulate(shared_collected, sinks.size(), kTopK);
        graph.moe_shared_nanoseconds += elapsed_since(shared_started);
        return result;
    }

    ValidationResult routed_expert(const LayerWeights& layer,
                                   std::uint32_t expert,
                                   std::span<const float> input,
                                   std::vector<float>& fused,
                                   std::vector<float>& activated,
                                   std::vector<float>& partial) {
        ValidationResult result;
        if (layer.expert_gate.encoding == InklingTensorEncoding::Plain) {
            result = plain_expert_matvec(layer.plain_gate_up, expert, kExpertUp,
                                         kHidden, input, fused);
            if (!result.ok()) return result;
            result = inkling_interleaved_swiglu_f32(activated, fused);
            if (!result.ok()) return result;
            graph.routed_expert_bytes +=
                (static_cast<std::uint64_t>(kExpertUp) * kHidden +
                 static_cast<std::uint64_t>(kHidden) * kExpertInner) *
                sizeof(std::uint16_t);
            return plain_expert_matvec(layer.plain_down, expert, kHidden,
                                       kExpertInner, activated, partial);
        }
        if (layer.expert_gate.encoding ==
            InklingTensorEncoding::Mxfp4Group32) {
            auto gate = checkpoint->mxfp4_expert_view(layer.expert_gate, expert);
            auto up = checkpoint->mxfp4_expert_view(layer.expert_up, expert);
            auto down = checkpoint->mxfp4_expert_view(layer.expert_down, expert);
            if (!gate.ok()) result.errors = std::move(gate.errors);
            if (!up.ok()) {
                result.errors.insert(result.errors.end(), up.errors.begin(),
                                     up.errors.end());
            }
            if (!down.ok()) {
                result.errors.insert(result.errors.end(), down.errors.begin(),
                                     down.errors.end());
            }
            if (!result.ok()) return result;
            auto gate_output = std::span<float>(fused).first(kExpertInner);
            auto up_output = std::span<float>(fused).subspan(kExpertInner,
                                                             kExpertInner);
            result = expert_matvec(gate.value, input, gate_output);
            if (!result.ok()) return result;
            result = expert_matvec(up.value, input, up_output);
            if (!result.ok()) return result;
            for (std::uint32_t element = 0U; element < kExpertInner; ++element) {
                activated[element] =
                    silu_f32(gate_output[element]) * up_output[element];
            }
            graph.routed_expert_bytes += layer.expert_gate.expert_bytes() +
                                         layer.expert_up.expert_bytes() +
                                         layer.expert_down.expert_bytes();
            return expert_matvec(down.value, activated, partial);
        }
        auto gate_up =
            checkpoint->nvfp4_expert_view(layer.expert_gate, expert);
        if (!gate_up.ok()) {
            result.errors = std::move(gate_up.errors);
            return result;
        }
        result = expert_matvec(gate_up.value, input, fused);
        if (!result.ok()) return result;
        result = inkling_interleaved_swiglu_f32(activated, fused);
        if (!result.ok()) return result;
        auto down = checkpoint->nvfp4_expert_view(layer.expert_down, expert);
        if (!down.ok()) {
            result.errors = std::move(down.errors);
            return result;
        }
        graph.routed_expert_bytes += layer.expert_gate.expert_bytes() +
                                     layer.expert_down.expert_bytes();
        return expert_matvec(down.value, activated, partial);
    }

    // ---- graph -------------------------------------------------------------

    ValidationResult embed(std::uint32_t token, std::span<float> hidden) {
        ValidationResult result;
        if (token >= embedding.rows) {
            result.errors.emplace_back("Inkling token id is outside the vocabulary");
            return result;
        }
        std::vector<float> raw(kHidden);
        if (embedding.encoding == InklingTensorEncoding::Plain) {
            const auto* row =
                embedding.data + static_cast<std::uint64_t>(token) * kHidden;
            for (std::uint32_t index = 0U; index < kHidden; ++index) {
                raw[index] = decode_bf16(row[index]);
            }
        } else {
            const auto row = matrix_slice(embedding, token, 1U, kHidden);
            for (std::uint32_t column = 0U; column < kHidden; ++column) {
                const auto byte = std::to_integer<std::uint8_t>(
                    row.packed[column / 2U]);
                const auto code = static_cast<std::uint8_t>(
                    column % 2U == 0U ? byte & 0x0FU : byte >> 4U);
                const auto scale = mxfp4_scale_from_e8m0(
                    std::to_integer<std::uint8_t>(row.scales[column / 32U]));
                if (!std::isfinite(scale)) {
                    result.errors.emplace_back(
                        "Inkling embedding contains a reserved E8M0 scale");
                    return result;
                }
                raw[column] = kMxfp4Values[code] * scale;
            }
        }
        rms_norm(hidden, raw, embedding_norm);
        return result;
    }

    using FeedForward = std::function<ValidationResult(std::span<const float>,
                                                       std::span<float>)>;

    ValidationResult run_block(const AttentionWeights& weights,
                               const DeviceLayer& device,
                               LayerState& layer_state, std::span<float> hidden,
                               std::uint64_t token_position,
                               const FeedForward& feed_forward) {
        std::vector<float> normalized(kHidden);
        std::vector<float> delta(kHidden);

        auto started = std::chrono::steady_clock::now();
        rms_norm(normalized, hidden, weights.attention_norm);
        auto result = attention(weights, device, layer_state, normalized,
                                token_position, delta);
        if (!result.ok()) return result;
        graph.attention_nanoseconds += elapsed_since(started);
        for (std::uint32_t index = 0U; index < kHidden; ++index) {
            hidden[index] += delta[index];
        }

        rms_norm(normalized, hidden, weights.mlp_norm);
        result = feed_forward(normalized, delta);
        if (!result.ok()) return result;
        result = short_conv(layer_state, weights, kConvMlp, delta);
        if (!result.ok()) return result;
        for (std::uint32_t index = 0U; index < kHidden; ++index) {
            hidden[index] += delta[index];
        }
        return result;
    }

    // Routed experts of one prefill page, grouped expert-major.
    //
    // The per-token path acquires a routed expert, uploads it, applies it to
    // one row and releases it -- so a page of R rows pays for the same expert
    // once per row that selected it. Inkling routes top-6 of 256 per row, so
    // at R = 64 an expert chosen by 20 rows is fetched 20 times.
    //
    // Grouping inverts that: every distinct expert of the page is acquired
    // once and applied to every row that chose it, in one batched command.
    // enqueue_moe's batching is dense over expert x row, so the rows handed to
    // it must be exactly that expert's rows -- passing the page's whole row
    // set against the union of its experts would compute |union| * R products
    // instead of R * 6, which is worse than not batching at all.
    //
    // Arithmetic is unchanged per row. Router weights are still applied on
    // collection, after the down projection, because scaling before it is not
    // float-equal to scaling after and the reference scales after.
    ValidationResult moe_page(const LayerWeights& layer,
                              const DeviceLayer& device, std::uint32_t index,
                              std::span<const InklingRoute> routes,
                              std::span<const float> inputs,
                              std::span<float> outputs) {
        ValidationResult result;
        const auto rows = static_cast<std::uint32_t>(routes.size());
        std::fill(outputs.begin(), outputs.end(), 0.0F);

        // expert -> the (row, choice) slots that selected it. The choice index
        // is carried because the reference sums a row's experts in choice
        // order, and float addition is not associative: summing them in expert
        // id order instead diverges from the token-at-a-time path in the last
        // bits, which compounds over 42 layers. Contributions are therefore
        // parked per slot and summed in choice order below.
        std::map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>>
            selections;
        for (std::uint32_t row = 0U; row < rows; ++row) {
            for (std::uint32_t choice = 0U; choice < kTopK; ++choice) {
                selections[routes[row].experts[choice]].emplace_back(row, choice);
            }
        }
        std::vector<float> contributions(
            static_cast<std::size_t>(rows) * kTopK * kHidden, 0.0F);

        std::vector<float> gathered;
        std::vector<float> collected;
        for (const auto& [expert, members] : selections) {
            CudaMoeExpert descriptor;
            auto staged = expert_cache->acquire(device.slot, index, expert,
                                                layer.expert_gate,
                                                layer.expert_up,
                                                layer.expert_down);
            if (!staged.ok()) {
                result.errors = std::move(staged.errors);
                return result;
            }
            descriptor = CudaMoeExpert{&staged.value->gate, &staged.value->up,
                                       &staged.value->down, 1.0F};
            graph.routed_expert_bytes += staged.value->device_bytes();

            const auto member_rows = static_cast<std::uint32_t>(members.size());
            gathered.resize(static_cast<std::size_t>(member_rows) * kHidden);
            for (std::uint32_t position = 0U; position < member_rows;
                 ++position) {
                const auto source =
                    static_cast<std::size_t>(members[position].first) * kHidden;
                std::copy_n(inputs.begin() +
                                static_cast<std::ptrdiff_t>(source),
                            kHidden,
                            gathered.begin() +
                                static_cast<std::ptrdiff_t>(position) * kHidden);
            }

            const std::array<CudaMoeExpert, 1U> one{descriptor};
            result = cuda.synchronize_uploads(devices[device.slot]);
            if (!result.ok()) {
                expert_cache->release(device.slot, index, expert);
                return result;
            }
            result = cuda.enqueue_moe(devices[device.slot], gathered,
                                      member_rows, one, nullptr);
            if (!result.ok()) {
                expert_cache->release(device.slot, index, expert);
                return result;
            }
            collected.assign(static_cast<std::size_t>(member_rows) * kHidden,
                             0.0F);
            result = cuda.collect_moe(devices[device.slot], collected, {});
            expert_cache->release(device.slot, index, expert);
            if (!result.ok()) return result;

            for (std::uint32_t position = 0U; position < member_rows;
                 ++position) {
                const auto [row, choice] = members[position];
                const auto* block =
                    collected.data() +
                    static_cast<std::size_t>(position) * kHidden;
                auto* slot = contributions.data() +
                             ((static_cast<std::size_t>(row) * kTopK) + choice) *
                                 kHidden;
                std::copy_n(block, kHidden, slot);
            }
        }

        // Sum each row in choice order, exactly as the per-row path does.
        for (std::uint32_t row = 0U; row < rows; ++row) {
            auto* target = outputs.data() + static_cast<std::size_t>(row) * kHidden;
            for (std::uint32_t choice = 0U; choice < kTopK; ++choice) {
                const float weight = routes[row].weights[choice];
                const auto* slot =
                    contributions.data() +
                    ((static_cast<std::size_t>(row) * kTopK) + choice) * kHidden;
                for (std::uint32_t element = 0U; element < kHidden; ++element) {
                    target[element] += weight * slot[element];
                }
            }
        }

        // Both sink experts run on every row, so this one is a genuine dense
        // batch: the whole page against both, in a single command.
        //
        // Charged to moe_shared_nanoseconds, not to the routed total. A phase
        // counter that reads zero because its stage moved under a neighbouring
        // counter is the exact trap this project has hit before, and it is
        // invisible to any logits comparison.
        const auto shared_started = std::chrono::steady_clock::now();
        std::vector<CudaMoeExpert> sinks;
        sinks.reserve(kShared);
        for (std::uint32_t shared = 0U; shared < kShared; ++shared) {
            sinks.push_back(CudaMoeExpert{&device.shared_gate[shared],
                                          &device.shared_up[shared],
                                          &device.shared_down[shared], 1.0F});
        }
        result = cuda.enqueue_moe(devices[device.slot], inputs, rows, sinks,
                                  nullptr);
        if (!result.ok()) return result;
        std::vector<float> shared_collected(
            static_cast<std::size_t>(sinks.size()) * rows * kHidden);
        result = cuda.collect_moe(devices[device.slot], shared_collected, {});
        if (!result.ok()) return result;
        graph.moe_shared_nanoseconds += elapsed_since(shared_started);
        for (std::size_t sink = 0U; sink < sinks.size(); ++sink) {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const float weight = routes[row].weights[kTopK + sink];
                const auto* block =
                    shared_collected.data() +
                    ((sink * rows) + row) * kHidden;
                auto* target =
                    outputs.data() + static_cast<std::size_t>(row) * kHidden;
                for (std::uint32_t element = 0U; element < kHidden; ++element) {
                    target[element] += weight * block[element];
                }
            }
        }
        return result;
    }

    // Prefill a page of rows. Identical arithmetic to calling forward() once
    // per token -- this is a scheduling change, not a numerical one.
    //
    // Two of the three stages in a block carry row-ordered state and stay
    // serial: attention appends this row's K/V before attending, and the four
    // short convolutions roll over the previous kConvTaps rows. The feed
    // forward between them has no cross-row state at all, which is the whole
    // reason a page can be batched here and nowhere else in this graph.
    //
    // So a page runs each layer in three passes: attention row by row in
    // order, then one batched MoE over the whole page, then the MLP
    // convolution and residual row by row in order.
    ValidationResult forward_page(std::span<const std::uint32_t> tokens,
                                  std::uint64_t base_position,
                                  std::vector<float>& hidden_rows) {
        ValidationResult result;
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        if (rows == 0U) return result;
        hidden_rows.assign(static_cast<std::size_t>(rows) * kHidden, 0.0F);

        auto started = std::chrono::steady_clock::now();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            auto slice = std::span<float>(hidden_rows)
                             .subspan(static_cast<std::size_t>(row) * kHidden,
                                      kHidden);
            std::vector<float> one(kHidden, 0.0F);
            result = embed(tokens[row], one);
            if (!result.ok()) return result;
            std::copy(one.begin(), one.end(), slice.begin());
        }
        graph.embedding_nanoseconds += elapsed_since(started);

        std::vector<float> normalized(static_cast<std::size_t>(rows) * kHidden);
        std::vector<float> deltas(static_cast<std::size_t>(rows) * kHidden);
        std::vector<InklingRoute> routes(rows);
        std::vector<float> row_delta(kHidden);

        for (std::uint32_t index = 0U; index < kLayers; ++index) {
            const auto& layer = layers[index];
            const auto& device = device_layers[index];
            auto& layer_state = state[index];

            // Attention: strictly in row order, so each row sees exactly the
            // keys and values of the rows before it.
            started = std::chrono::steady_clock::now();
            for (std::uint32_t row = 0U; row < rows; ++row) {
                auto hidden = std::span<float>(hidden_rows)
                                  .subspan(static_cast<std::size_t>(row) * kHidden,
                                           kHidden);
                auto norm = std::span<float>(normalized)
                                .subspan(static_cast<std::size_t>(row) * kHidden,
                                         kHidden);
                rms_norm(norm, hidden, layer.attention.attention_norm);
                result = attention(layer.attention, device, layer_state, norm,
                                   base_position + row, row_delta);
                if (!result.ok()) return result;
                for (std::uint32_t element = 0U; element < kHidden; ++element) {
                    hidden[element] += row_delta[element];
                }
            }
            graph.attention_nanoseconds += elapsed_since(started);

            for (std::uint32_t row = 0U; row < rows; ++row) {
                auto hidden = std::span<const float>(hidden_rows)
                                  .subspan(static_cast<std::size_t>(row) * kHidden,
                                           kHidden);
                auto norm = std::span<float>(normalized)
                                .subspan(static_cast<std::size_t>(row) * kHidden,
                                         kHidden);
                rms_norm(norm, hidden, layer.attention.mlp_norm);
            }

            // The one batched stage.
            const bool batchable =
                layer.sparse && cuda_enabled && device.resident;
            if (batchable) {
                started = std::chrono::steady_clock::now();
                for (std::uint32_t row = 0U; row < rows; ++row) {
                    auto norm = std::span<const float>(normalized)
                                    .subspan(static_cast<std::size_t>(row) * kHidden,
                                             kHidden);
                    std::vector<float> logits(layer.gate.rows);
                    result = spine_matvec(device.gate, layer.gate, norm, logits);
                    if (!result.ok()) return result;
                    auto route = inkling_route_sigmoid_sink(
                        logits, layer.gate_bias, router, kShared,
                        layer.gate_global_scale);
                    if (!route.ok()) {
                        result.errors = std::move(route.errors);
                        return result;
                    }
                    routes[row] = std::move(route.value);
                    if (!write_route(base_position + row, index, routes[row],
                                     true)) {
                        result.errors.emplace_back(
                            "cannot write Inkling route trace");
                        return result;
                    }
                }
                graph.moe_router_nanoseconds += elapsed_since(started);
                started = std::chrono::steady_clock::now();
                result = moe_page(layer, device, index, routes, normalized,
                                  deltas);
                if (!result.ok()) return result;
                graph.moe_routed_nanoseconds += elapsed_since(started);
            } else {
                for (std::uint32_t row = 0U; row < rows; ++row) {
                    auto norm = std::span<const float>(normalized)
                                    .subspan(static_cast<std::size_t>(row) * kHidden,
                                             kHidden);
                    auto out = std::span<float>(deltas)
                                   .subspan(static_cast<std::size_t>(row) * kHidden,
                                            kHidden);
                    if (layer.sparse) {
                        result = moe(layer, device, index, base_position + row,
                                     true, norm, out);
                    } else {
                        const auto dense_started =
                            std::chrono::steady_clock::now();
                        result = dense_mlp(layer.dense_gate_up,
                                           layer.dense_gate, layer.dense_up,
                                           layer.dense_down,
                                           layer.dense_global_scale, &device,
                                           norm, out);
                        graph.dense_mlp_nanoseconds +=
                            elapsed_since(dense_started);
                    }
                    if (!result.ok()) return result;
                }
            }

            // MLP convolution and residual: row order again, for conv history.
            for (std::uint32_t row = 0U; row < rows; ++row) {
                auto out = std::span<float>(deltas)
                               .subspan(static_cast<std::size_t>(row) * kHidden,
                                        kHidden);
                result = short_conv(layer_state, layer.attention, kConvMlp, out);
                if (!result.ok()) return result;
                auto hidden = std::span<float>(hidden_rows)
                                  .subspan(static_cast<std::size_t>(row) * kHidden,
                                           kHidden);
                for (std::uint32_t element = 0U; element < kHidden; ++element) {
                    hidden[element] += out[element];
                }
            }
        }
        graph.forward_tokens += rows;
        trim_sliding_caches();
        return result;
    }

    ValidationResult forward(std::uint32_t token, std::uint64_t token_position,
                             bool prefill, std::vector<float>& hidden) {
        hidden.assign(kHidden, 0.0F);
        auto started = std::chrono::steady_clock::now();
        auto result = embed(token, hidden);
        if (!result.ok()) return result;
        graph.embedding_nanoseconds += elapsed_since(started);

        for (std::uint32_t index = 0U; index < kLayers; ++index) {
            const auto& layer = layers[index];
            result = run_block(
                layer.attention, device_layers[index], state[index], hidden,
                token_position,
                [&](std::span<const float> input, std::span<float> output) {
                    if (!layer.sparse) {
                        const auto dense_started =
                            std::chrono::steady_clock::now();
                        auto status = dense_mlp(
                            layer.dense_gate_up, layer.dense_gate,
                            layer.dense_up, layer.dense_down,
                            layer.dense_global_scale, &device_layers[index],
                            input, output);
                        graph.dense_mlp_nanoseconds +=
                            elapsed_since(dense_started);
                        return status;
                    }
                    return moe(layer, device_layers[index], index,
                               token_position, prefill, input, output);
                });
            if (!result.ok()) return result;
        }
        ++graph.forward_tokens;
        trim_sliding_caches();
        return result;
    }

    // One MTP depth: the block consumes the normalized previous hidden state
    // concatenated with the normalized embedding of the token, projected back
    // to hidden width.
    ValidationResult forward_mtp(std::uint32_t depth, std::uint32_t token,
                                 std::uint64_t token_position,
                                 std::span<const float> previous_hidden,
                                 std::vector<float>& hidden) {
        ValidationResult result;
        if (depth >= mtp.size()) {
            result.errors.emplace_back("Inkling MTP depth is not loaded");
            return result;
        }
        const auto& weights = mtp[depth];
        std::vector<float> embedded(kHidden);
        result = embed(token, embedded);
        if (!result.ok()) return result;
        // The embedding side chains the backbone norm and the depth norm.
        std::vector<float> depth_embedded(kHidden);
        rms_norm(depth_embedded, embedded, weights.embed_norm);

        std::vector<float> combined(2U * kHidden);
        rms_norm(std::span<float>(combined).subspan(0U, kHidden),
                 previous_hidden, weights.hidden_norm);
        std::copy(depth_embedded.begin(), depth_embedded.end(),
                  combined.begin() + kHidden);

        hidden.assign(kHidden, 0.0F);
        result = matvec(weights.input_projection, combined, hidden);
        if (!result.ok()) return result;
        static const DeviceLayer kHostOnly;
        return run_block(
            weights.attention, kHostOnly, mtp_state[depth], hidden,
            token_position,
            [&](std::span<const float> input, std::span<float> output) {
                return dense_mlp(weights.dense_gate_up, weights.dense_gate,
                                 weights.dense_up, weights.dense_down,
                                 weights.dense_global_scale, nullptr, input,
                                 output);
            });
    }

    ValidationResult logits_from_hidden(std::span<const float> hidden,
                                        std::vector<float>& logits) {
        const auto started = std::chrono::steady_clock::now();
        std::vector<float> normalized(kHidden);
        rms_norm(normalized, hidden, final_norm);
        std::vector<float> full(unembedding.rows);
        auto result = spine_matvec(device_unembed, unembedding, normalized, full);
        if (!result.ok()) return result;
        // muP divides the logits; the padded rows are then discarded.
        const float scale = 1.0F / kContract.logits_width_multiplier;
        logits.assign(kContract.vocabulary_size, 0.0F);
        for (std::uint32_t index = 0U; index < kContract.vocabulary_size;
             ++index) {
            logits[index] = full[index] * scale;
        }
        graph.output_head_nanoseconds += elapsed_since(started);
        return result;
    }

    // Sliding layers only ever read the last `sliding_window` rows, so their
    // caches are trimmed instead of growing with the sequence.
    void trim_sliding_caches() {
        for (std::uint32_t index = 0U; index < kLayers; ++index) {
            if (inkling_global_attention_layer(index)) continue;
            auto& layer = state[index];
            if (layer.rows <= kContract.sliding_window) continue;
            const auto drop = layer.rows - kContract.sliding_window;
            layer.keys.erase(
                layer.keys.begin(),
                layer.keys.begin() + static_cast<std::ptrdiff_t>(drop * kKvWidth));
            layer.values.erase(
                layer.values.begin(),
                layer.values.begin() + static_cast<std::ptrdiff_t>(drop * kKvWidth));
            layer.rows -= drop;
            layer.base_position += drop;
        }
    }

    void reset_sequence() {
        for (auto& layer : state) {
            layer.keys.clear();
            layer.values.clear();
            layer.rows = 0U;
            layer.base_position = 0U;
            layer.device_kv_ready = false;
            reset_conv_history(layer);
        }
        for (auto& depth : mtp_state) {
            depth.keys.clear();
            depth.values.clear();
            depth.rows = 0U;
            depth.base_position = 0U;
            depth.device_kv_ready = false;
            reset_conv_history(depth);
        }
        position = 0U;
        graph = {};
    }

    // ---- loading -----------------------------------------------------------

    ParseResult<MappedMatrix> map_matrix(const std::string& name,
                                         std::uint64_t rows,
                                         std::uint64_t columns) {
        ParseResult<MappedMatrix> result;
        auto module = checkpoint->linear(name, rows, columns);
        if (!module.ok()) {
            result.errors = std::move(module.errors);
            return result;
        }
        result.value.encoding = module.value.encoding;
        if (module.value.encoding == InklingTensorEncoding::Plain) {
            auto mapped = checkpoint->view(module.value.weight->name);
            if (!mapped.ok()) {
                result.errors = std::move(mapped.errors);
                return result;
            }
            result.value.data =
                reinterpret_cast<const std::uint16_t*>(mapped.value.data());
        } else {
            auto matrix = checkpoint->mxfp4_view(module.value);
            if (!matrix.ok()) {
                result.errors = std::move(matrix.errors);
                return result;
            }
            result.value.packed = matrix.value.packed;
            result.value.scales = matrix.value.scales;
        }
        result.value.rows = rows;
        result.value.columns = columns;
        return result;
    }

    // Stacked BF16 tensors are mapped whole; the caller slices by expert.
    ParseResult<const std::uint16_t*> map_stack(const std::string& name) {
        ParseResult<const std::uint16_t*> result;
        auto mapped = checkpoint->view(name);
        if (!mapped.ok()) {
            result.errors = std::move(mapped.errors);
            return result;
        }
        result.value =
            reinterpret_cast<const std::uint16_t*>(mapped.value.data());
        return result;
    }

    ParseResult<MappedMatrix> map_stack(const InklingExpertStack& stack) {
        ParseResult<MappedMatrix> result;
        result.value.encoding = stack.encoding;
        result.value.rows = stack.rows;
        result.value.columns = stack.columns;
        if (stack.encoding == InklingTensorEncoding::Plain) {
            auto mapped = checkpoint->view(stack.weight->name);
            if (!mapped.ok()) {
                result.errors = std::move(mapped.errors);
                return result;
            }
            result.value.data =
                reinterpret_cast<const std::uint16_t*>(mapped.value.data());
            return result;
        }
        if (stack.encoding != InklingTensorEncoding::Mxfp4Group32) {
            result.errors.emplace_back(
                "only plain or MXFP4 stacks can be mapped as matrices");
            return result;
        }
        auto packed = checkpoint->view(stack.packed->name);
        auto scales = checkpoint->view(stack.scale->name);
        if (!packed.ok()) result.errors = std::move(packed.errors);
        if (!scales.ok()) {
            result.errors.insert(result.errors.end(), scales.errors.begin(),
                                 scales.errors.end());
        }
        if (!result.ok()) return result;
        result.value.packed = packed.value;
        result.value.scales = scales.value;
        return result;
    }

    ValidationResult load_vector(const std::string& name, std::uint64_t elements,
                                 std::vector<float>& output) {
        ValidationResult result;
        auto loaded = checkpoint->read_f32(name, elements);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        if (loaded.value.size() != elements) {
            result.errors.emplace_back("Inkling vector shape mismatch: " + name);
            return result;
        }
        output = std::move(loaded.value);
        return result;
    }

    ValidationResult load_attention(const std::string& prefix, bool global,
                                    std::uint32_t extent,
                                    AttentionWeights& weights) {
        ValidationResult result;
        const bool mxfp4 = checkpoint->format() ==
                           InklingCheckpointFormat::Mxfp4Group32;
        const auto attention_prefix =
            prefix + (mxfp4 ? "self_attn." : "attn.");
        weights.global = global;
        weights.relative_extent = extent;
        const auto map = [&](const std::string& name, std::uint64_t rows,
                             std::uint64_t columns, MappedMatrix& target) {
            auto mapped = map_matrix(name, rows, columns);
            if (!mapped.ok()) {
                result.errors.insert(result.errors.end(), mapped.errors.begin(),
                                     mapped.errors.end());
                return;
            }
            target = mapped.value;
        };
        map(attention_prefix + (mxfp4 ? "q_proj" : "wq_du.weight"),
            kQueryWidth, kHidden,
            weights.query);
        map(attention_prefix + (mxfp4 ? "k_proj" : "wk_dv.weight"),
            kKvWidth, kHidden, weights.key);
        map(attention_prefix + (mxfp4 ? "v_proj" : "wv_dv.weight"),
            kKvWidth, kHidden, weights.value);
        map(attention_prefix + (mxfp4 ? "r_proj" : "wr_du.weight"),
            kRelWidth, kHidden,
            weights.relative);
        map(attention_prefix + (mxfp4 ? "o_proj" : "wo_ud.weight"),
            kHidden, kQueryWidth,
            weights.output);
        if (!result.ok()) return result;

        result = load_vector(attention_prefix +
                                 (mxfp4 ? "rel_proj"
                                        : "rel_logits_proj.proj"),
                             static_cast<std::uint64_t>(kContract.relative_dim) *
                                 extent,
                             weights.relative_projection);
        if (!result.ok()) return result;
        result = load_vector(attention_prefix + "q_norm.weight", kHeadDim,
                             weights.query_norm);
        if (!result.ok()) return result;
        result = load_vector(attention_prefix + "k_norm.weight", kHeadDim,
                             weights.key_norm);
        if (!result.ok()) return result;
        result = load_vector(prefix +
                                 (mxfp4 ? "input_layernorm.weight"
                                        : "attn_norm.weight"),
                             kHidden,
                             weights.attention_norm);
        if (!result.ok()) return result;
        result = load_vector(prefix +
                                 (mxfp4 ? "post_attention_layernorm.weight"
                                        : "mlp_norm.weight"),
                             kHidden,
                             weights.mlp_norm);
        if (!result.ok()) return result;

        // Conv weights ship as [channels, 1, kernel]; the middle axis is the
        // depthwise group of one.
        const std::array<std::string, kConvStreams> names{
            attention_prefix +
                (mxfp4 ? "k_sconv.conv.weight" : "k_sconv.weight"),
            attention_prefix +
                (mxfp4 ? "v_sconv.conv.weight" : "v_sconv.weight"),
            prefix + (mxfp4 ? "attn_sconv.conv.weight"
                             : "attn_sconv.weight"),
            prefix + (mxfp4 ? "mlp_sconv.conv.weight"
                             : "mlp_sconv.weight")};
        for (std::size_t stream = 0U; stream < kConvStreams; ++stream) {
            result = load_vector(
                names[stream],
                static_cast<std::uint64_t>(conv_stream_width(stream)) *
                    kContract.short_conv_kernel,
                weights.conv[stream]);
            if (!result.ok()) return result;
        }
        return result;
    }

    ValidationResult load_layer(std::uint32_t index) {
        auto& layer = layers[index];
        const bool mxfp4 = checkpoint->format() ==
                           InklingCheckpointFormat::Mxfp4Group32;
        const auto prefix = mxfp4
            ? "language_model.model.layers." + std::to_string(index) + "."
            : inkling_layer_prefix(index);
        auto result =
            load_attention(prefix, inkling_global_attention_layer(index),
                           inkling_relative_extent(index), layer.attention);
        if (!result.ok()) return result;

        const auto mlp = prefix + "mlp.";
        layer.sparse = inkling_sparse_layer(index);
        if (!layer.sparse) {
            if (mxfp4) {
                auto gate = map_matrix(mlp + "gate_proj", kDenseInner, kHidden);
                auto up = map_matrix(mlp + "up_proj", kDenseInner, kHidden);
                if (!gate.ok()) result.errors = std::move(gate.errors);
                if (!up.ok()) {
                    result.errors.insert(result.errors.end(), up.errors.begin(),
                                         up.errors.end());
                }
                if (!result.ok()) return result;
                layer.dense_gate = gate.value;
                layer.dense_up = up.value;
            } else {
                auto gate_up =
                    map_matrix(mlp + "w13_dn.weight", kDenseUp, kHidden);
                if (!gate_up.ok()) {
                    result.errors = std::move(gate_up.errors);
                    return result;
                }
                layer.dense_gate_up = gate_up.value;
            }
            auto down = map_matrix(
                mlp + (mxfp4 ? "down_proj" : "w2_md.weight"), kHidden,
                kDenseInner);
            if (!down.ok()) {
                result.errors = std::move(down.errors);
                return result;
            }
            layer.dense_down = down.value;
            std::vector<float> scale;
            result = load_vector(mlp + "global_scale", 1U, scale);
            if (!result.ok()) return result;
            layer.dense_global_scale = scale[0];
            return result;
        }

        auto gate = map_matrix(
            mlp + (mxfp4 ? "gate_weight" : "gate.weight"),
            kExperts + kShared, kHidden);
        if (!gate.ok()) {
            result.errors = std::move(gate.errors);
            return result;
        }
        layer.gate = gate.value;
        result = load_vector(
            mlp + (mxfp4 ? "e_score_correction_bias" : "gate.bias"),
            kExperts, layer.gate_bias);
        if (!result.ok()) return result;
        std::vector<float> scale;
        result = load_vector(
            mlp + (mxfp4 ? "global_scale" : "gate.global_scale"), 1U,
            scale);
        if (!result.ok()) return result;
        layer.gate_global_scale = scale[0];

        if (mxfp4) {
            const auto load_shared = [&](const std::string& projection,
                                         std::uint64_t rows,
                                         std::uint64_t columns,
                                         MappedMatrix& target) {
                auto stack = checkpoint->expert_stack(
                    mlp + "shared_experts." + projection, index, kShared,
                    rows, columns);
                if (!stack.ok()) {
                    result.errors.insert(result.errors.end(),
                                         stack.errors.begin(), stack.errors.end());
                    return;
                }
                auto mapped = map_stack(stack.value);
                if (!mapped.ok()) {
                    result.errors.insert(result.errors.end(),
                                         mapped.errors.begin(), mapped.errors.end());
                    return;
                }
                target = mapped.value;
            };
            load_shared("gate_proj", kExpertInner, kHidden, layer.shared_gate);
            load_shared("up_proj", kExpertInner, kHidden, layer.shared_up);
            load_shared("down_proj", kHidden, kExpertInner, layer.shared_down);
            if (!result.ok()) return result;
        } else {
            auto shared_gate_up =
                map_stack(mlp + "shared_experts.shared_w13_weight");
            if (!shared_gate_up.ok()) {
                result.errors = std::move(shared_gate_up.errors);
                return result;
            }
            layer.shared_gate_up.data = shared_gate_up.value;
            layer.shared_gate_up.rows = kExpertUp;
            layer.shared_gate_up.columns = kHidden;
            auto shared_down =
                map_stack(mlp + "shared_experts.shared_w2_weight");
            if (!shared_down.ok()) {
                result.errors = std::move(shared_down.errors);
                return result;
            }
            layer.shared_down.data = shared_down.value;
            layer.shared_down.rows = kHidden;
            layer.shared_down.columns = kExpertInner;
        }

        auto stack = checkpoint->expert_stack(
            mlp + (mxfp4 ? "switch_mlp.gate_proj" : "experts.w13_weight"),
            index, kExperts, mxfp4 ? kExpertInner : kExpertUp, kHidden);
        if (!stack.ok()) {
            result.errors = std::move(stack.errors);
            return result;
        }
        layer.expert_gate = stack.value;
        if (mxfp4) {
            stack = checkpoint->expert_stack(mlp + "switch_mlp.up_proj", index,
                                             kExperts, kExpertInner, kHidden);
            if (!stack.ok()) {
                result.errors = std::move(stack.errors);
                return result;
            }
            layer.expert_up = stack.value;
        } else {
            layer.expert_up = layer.expert_gate;
        }
        stack = checkpoint->expert_stack(
            mlp + (mxfp4 ? "switch_mlp.down_proj" : "experts.w2_weight"), index,
                                         kExperts, kHidden, kExpertInner);
        if (!stack.ok()) {
            result.errors = std::move(stack.errors);
            return result;
        }
        layer.expert_down = stack.value;

        if (!mxfp4 && !inkling_quantized_expert_layer(index)) {
            auto plain = map_stack(mlp + "experts.w13_weight");
            if (!plain.ok()) {
                result.errors = std::move(plain.errors);
                return result;
            }
            layer.plain_gate_up = plain.value;
            plain = map_stack(mlp + "experts.w2_weight");
            if (!plain.ok()) {
                result.errors = std::move(plain.errors);
                return result;
            }
            layer.plain_down = plain.value;
        }
        return result;
    }

    // Proposes up to `depth` continuations of `token` from the MTP heads.
    // Advisory only: every proposal is verified by the backbone before it can
    // reach the output, so a bad draft costs time and never correctness.
    ValidationResult propose(std::uint32_t token, std::uint64_t token_position,
                             std::span<const float> hidden,
                             std::vector<std::uint32_t>& proposals);

    InklingGenerationResult generate(std::vector<std::uint32_t> prompt,
                                     std::uint32_t maximum_new_tokens,
                                     const SamplingOptions& sampling,
                                     std::span<const std::string> stop,
                                     const TokenStreamCallback& on_token);

    // Faults every routed expert tensor into page cache. Touching one byte per
    // page is enough; the kernel reads ahead, so this runs at sequential NVMe
    // speed rather than at random-read speed.
    ValidationResult warm_expert_pages() {
        ValidationResult result;
        std::uint64_t warmed = 0U;
        volatile std::uint8_t sink = 0U;
        const bool mxfp4 = checkpoint->format() ==
                           InklingCheckpointFormat::Mxfp4Group32;
        for (std::uint32_t index = 0U; index < kLayers; ++index) {
            if (!inkling_sparse_layer(index)) continue;
            const auto mlp = mxfp4
                ? "language_model.model.layers." + std::to_string(index) +
                      ".mlp."
                : inkling_layer_prefix(index) + "mlp.";
            const std::vector<std::string> suffixes = mxfp4
                ? std::vector<std::string>{
                      "switch_mlp.gate_proj.weight",
                      "switch_mlp.gate_proj.scales",
                      "switch_mlp.up_proj.weight",
                      "switch_mlp.up_proj.scales",
                      "switch_mlp.down_proj.weight",
                      "switch_mlp.down_proj.scales"}
                : std::vector<std::string>{
                      "experts.w13_weight", "experts.w13_weight.scale",
                      "experts.w2_weight", "experts.w2_weight.scale"};
            for (const auto& suffix : suffixes) {
                const auto* tensor = checkpoint->find(mlp + suffix);
                if (tensor == nullptr) continue;
                auto mapped = checkpoint->view(mlp + suffix);
                if (!mapped.ok()) continue;
                const auto* bytes =
                    reinterpret_cast<const std::uint8_t*>(mapped.value.data());
                for (std::size_t offset = 0U; offset < mapped.value.size();
                     offset += 4096U) {
                    sink = static_cast<std::uint8_t>(sink ^ bytes[offset]);
                }
                warmed += mapped.value.size();
            }
            if (config.load_progress) {
                std::fprintf(stderr, "\rwarming experts %.1f GiB",
                             static_cast<double>(warmed) /
                                 (1024.0 * 1024.0 * 1024.0));
            }
        }
        static_cast<void>(sink);
        if (config.load_progress) std::fprintf(stderr, "\n");
        return result;
    }

    // Uploads the resident spine and sizes the routed-expert cache from what
    // it leaves behind. Layers round-robin over the devices, so a 16 GiB card
    // and a 24 GiB card carry the same layer count but different cache shares.
    ValidationResult initialize_devices() {
        ValidationResult result;
        device_layers.clear();
        device_layers.resize(kLayers);
        if (!config.enable_cuda) return result;
        auto available = CudaBackend::available_devices();
        if (available.empty()) return result;
        devices.clear();
        for (const auto device : config.devices) {
            if (std::find(available.begin(), available.end(), device) !=
                available.end()) {
                devices.push_back(device);
            }
        }
        if (devices.empty()) return result;
        result = cuda.initialize(devices, true);
        if (!result.ok()) return result;
        auto plan = plan_runtime_devices(
            devices, config.vram_cache_fraction, 0U, 2ULL << 30U,
            "Inkling");
        if (!plan.ok()) {
            result.errors = std::move(plan.errors);
            return result;
        }
        device_schedule = plan.value.weighted_schedule;
        device_workers = std::make_unique<HostWorkerPool>(devices.size());
        std::vector<std::uint64_t> arena_capacities;
        if (config.use_weight_arena) {
            arena_capacities = std::move(plan.value.weight_capacities);
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                result = cuda.reserve_weight_arena(
                    devices[slot], arena_capacities[slot]);
                if (!result.ok()) return result;
            }
        }
        resident_spine_bytes.assign(devices.size(), 0U);
        resident_kv_bytes.assign(devices.size(), 0U);
        const bool configure_device_attention =
            config.enable_device_kv_attention &&
            config.maximum_context_tokens >
                config.minimum_device_attention_rows;
        if (configure_device_attention) {
            for (const auto device : devices) {
                result = cuda.validate_flash_attention_device(device);
                if (!result.ok()) return result;
            }
        }

        const auto upload_linear = [&](std::size_t slot, const std::string& name,
                                       std::uint64_t rows, std::uint64_t columns,
                                       CudaWeight& target) {
            auto module = checkpoint->linear(name, rows, columns);
            if (!module.ok()) {
                result.errors = std::move(module.errors);
                return;
            }
            auto status = load_inkling_cuda_linear(*checkpoint, module.value,
                                                   devices[slot], cuda, target);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return;
            }
            resident_spine_bytes[slot] += target.device_bytes();
        };
        const auto upload_half = [&](std::size_t slot, const std::string& name,
                                     std::uint64_t slice, std::uint64_t rows,
                                     std::uint64_t columns, bool up,
                                     CudaWeight& target) {
            auto status = load_inkling_cuda_interleaved_half(
                *checkpoint, name, slice, rows, columns, up, devices[slot], cuda,
                target);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return;
            }
            resident_spine_bytes[slot] += target.device_bytes();
        };
        const auto upload_mxfp4_expert = [&](std::size_t slot,
                                              const InklingExpertStack& stack,
                                              std::uint32_t expert,
                                              CudaWeight& target) {
            auto matrix = checkpoint->mxfp4_expert_view(stack, expert);
            if (!matrix.ok()) {
                result.errors = std::move(matrix.errors);
                return;
            }
            CudaWeightDescriptor descriptor;
            descriptor.encoding = CudaWeightEncoding::Fp4E2m1Group32;
            descriptor.dtype = SafetensorsDtype::I8;
            descriptor.rows = matrix.value.rows;
            descriptor.columns = matrix.value.columns;
            descriptor.packed_columns = matrix.value.packed_columns;
            descriptor.scale_columns = matrix.value.scale_columns;
            descriptor.group_size = 32U;
            auto status = cuda.upload(
                devices[slot], descriptor, matrix.value.packed,
                matrix.value.scales, target,
                CudaBackend::UploadCompletion::Deferred,
                CudaBackend::FragmentLayout::Prepack);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return;
            }
            resident_spine_bytes[slot] += target.device_bytes();
        };

        const bool mxfp4 = checkpoint->format() ==
                           InklingCheckpointFormat::Mxfp4Group32;

        for (std::uint32_t index = 0U; index < kLayers; ++index) {
            auto& device = device_layers[index];
            device.slot = index % devices.size();
            const auto prefix = mxfp4
                ? "language_model.model.layers." + std::to_string(index) + "."
                : inkling_layer_prefix(index);
            const auto attention =
                prefix + (mxfp4 ? "self_attn." : "attn.");
            const auto mlp = prefix + "mlp.";
            upload_linear(device.slot,
                          attention + (mxfp4 ? "q_proj" : "wq_du.weight"),
                          kQueryWidth,
                          kHidden, device.query);
            upload_linear(device.slot,
                          attention + (mxfp4 ? "k_proj" : "wk_dv.weight"),
                          kKvWidth,
                          kHidden, device.key);
            upload_linear(device.slot,
                          attention + (mxfp4 ? "v_proj" : "wv_dv.weight"),
                          kKvWidth,
                          kHidden, device.value);
            upload_linear(device.slot,
                          attention + (mxfp4 ? "r_proj" : "wr_du.weight"),
                          kRelWidth,
                          kHidden, device.relative);
            upload_linear(device.slot,
                          attention + (mxfp4 ? "o_proj" : "wo_ud.weight"),
                          kHidden,
                          kQueryWidth, device.output);
            if (!inkling_sparse_layer(index)) {
                if (mxfp4) {
                    upload_linear(device.slot, mlp + "gate_proj", kDenseInner,
                                  kHidden, device.dense_gate);
                    upload_linear(device.slot, mlp + "up_proj", kDenseInner,
                                  kHidden, device.dense_up);
                } else {
                    upload_half(device.slot, mlp + "w13_dn.weight", 0U,
                                kDenseUp, kHidden, false, device.dense_gate);
                    upload_half(device.slot, mlp + "w13_dn.weight", 0U,
                                kDenseUp, kHidden, true, device.dense_up);
                }
                upload_linear(device.slot,
                              mlp + (mxfp4 ? "down_proj" : "w2_md.weight"),
                              kHidden,
                              kDenseInner, device.dense_down);
            } else {
                upload_linear(device.slot,
                              mlp + (mxfp4 ? "gate_weight" : "gate.weight"),
                              kExperts + kShared, kHidden, device.gate);
                for (std::uint32_t shared = 0U; shared < kShared; ++shared) {
                    if (mxfp4) {
                        const auto load_stack = [&](const std::string& projection,
                                                    std::uint64_t rows,
                                                    std::uint64_t columns,
                                                    CudaWeight& target) {
                            auto stack = checkpoint->expert_stack(
                                mlp + "shared_experts." + projection, index,
                                kShared, rows, columns);
                            if (!stack.ok()) {
                                result.errors = std::move(stack.errors);
                                return;
                            }
                            upload_mxfp4_expert(device.slot, stack.value,
                                                shared, target);
                        };
                        load_stack("gate_proj", kExpertInner, kHidden,
                                   device.shared_gate[shared]);
                        load_stack("up_proj", kExpertInner, kHidden,
                                   device.shared_up[shared]);
                        load_stack("down_proj", kHidden, kExpertInner,
                                   device.shared_down[shared]);
                        continue;
                    }
                    upload_half(device.slot,
                                mlp + "shared_experts.shared_w13_weight", shared,
                                kExpertUp, kHidden, false,
                                device.shared_gate[shared]);
                    upload_half(device.slot,
                                mlp + "shared_experts.shared_w13_weight", shared,
                                kExpertUp, kHidden, true,
                                device.shared_up[shared]);
                    auto stack = checkpoint->view(
                        mlp + "shared_experts.shared_w2_weight");
                    if (!stack.ok()) {
                        result.errors = std::move(stack.errors);
                        return result;
                    }
                    const auto block = static_cast<std::size_t>(
                        kHidden * kExpertInner * sizeof(std::uint16_t));
                    CudaWeightDescriptor descriptor;
                    descriptor.encoding = CudaWeightEncoding::Plain;
                    descriptor.dtype = SafetensorsDtype::Bf16;
                    descriptor.rows = kHidden;
                    descriptor.columns = kExpertInner;
                    auto status = cuda.upload(
                        devices[device.slot], descriptor,
                        stack.value.subspan(shared * block, block), {},
                        device.shared_down[shared]);
                    if (!status.ok()) {
                        result.errors = std::move(status.errors);
                        return result;
                    }
                    resident_spine_bytes[device.slot] +=
                        device.shared_down[shared].device_bytes();
                }
            }
            if (!result.ok()) return result;
            device.resident = inkling_sparse_layer(index);
            if (configure_device_attention) {
                device.kv_capacity_rows = inkling_global_attention_layer(index)
                    ? config.maximum_context_tokens
                    : std::min(config.maximum_context_tokens,
                               kContract.sliding_window);
                const auto bytes =
                    2ULL * device.kv_capacity_rows * kKvWidth *
                    sizeof(std::uint16_t);
                result = cuda.allocate_buffer(
                    devices[device.slot], bytes, device.kv_cache);
                if (!result.ok()) return result;
                resident_kv_bytes[device.slot] += bytes;
            }
        }

        // The output head is the single largest spine tensor; it lands on the
        // device with the most headroom rather than following the layer schedule.
        std::size_t widest = 0U;
        std::uint64_t widest_free = 0U;
        for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
            auto memory = CudaBackend::device_memory(devices[slot]);
            if (!memory.ok()) continue;
            if (memory.value.free_bytes > widest_free) {
                widest_free = memory.value.free_bytes;
                widest = slot;
            }
        }
        upload_linear(widest,
                      mxfp4 ? "language_model.lm_head"
                             : "model.llm.unembed.weight",
                      kContract.padded_vocabulary_size, kHidden, device_unembed);
        if (!result.ok()) return result;

        // Whatever the spine left is expert cache. A device that cannot hold
        // several experts is worse than useless, so it is reported rather than
        // quietly configured to thrash.
        std::vector<std::uint64_t> capacities = std::move(arena_capacities);
        if (config.use_weight_arena) {
            // acquire() uploads a miss before it inserts the entry and applies
            // the LRU capacity check. Keep one worst-case plain-BF16 expert
            // inside the already admitted arena as transient upload space;
            // otherwise the bounded allocator correctly refuses the temporary
            // peak that per-weight cudaMalloc used to hide in free VRAM.
            const auto expert_upload_headroom =
                3ULL * CudaBackend::weight_storage_bytes(
                           static_cast<std::uint64_t>(kExpertInner) * kHidden *
                               sizeof(std::uint16_t),
                           0U);
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                if (resident_spine_bytes[slot] >= capacities[slot] ||
                    expert_upload_headroom >=
                        capacities[slot] - resident_spine_bytes[slot]) {
                    result.errors.emplace_back(
                        "Inkling resident spine exhausts the admitted VRAM "
                        "budget on device " + std::to_string(devices[slot]));
                    return result;
                }
                capacities[slot] -=
                    resident_spine_bytes[slot] + expert_upload_headroom;
            }
        } else {
            capacities.reserve(devices.size());
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                auto memory = CudaBackend::device_memory(devices[slot]);
                const auto free_bytes = memory.ok() ? memory.value.free_bytes : 0U;
                const auto budget = static_cast<std::uint64_t>(
                    static_cast<double>(free_bytes) *
                    config.vram_cache_fraction);
                capacities.push_back(budget);
            }
        }
        expert_cache = std::make_unique<InklingExpertCache>(
            *checkpoint, cuda, devices, capacities,
            config.direct_mapped_mxfp4_staging,
            config.defer_mapped_mxfp4_uploads);
        for (const auto device : devices) {
            result = cuda.synchronize_uploads(device);
            if (!result.ok()) return result;
        }
        cuda_enabled = true;
        return result;
    }

    ValidationResult load_mtp(std::uint32_t depth) {
        auto& weights = mtp[depth];
        const auto prefix = inkling_mtp_prefix(depth);
        auto result = load_attention(prefix + "transformer_block.",
                                     inkling_mtp_global_attention_depth(depth),
                                     inkling_mtp_relative_extent(depth),
                                     weights.attention);
        if (!result.ok()) return result;
        result = load_vector(prefix + "embed_norm.weight", kHidden,
                             weights.embed_norm);
        if (!result.ok()) return result;
        result = load_vector(prefix + "hidden_norm.weight", kHidden,
                             weights.hidden_norm);
        if (!result.ok()) return result;
        auto projection =
            map_matrix(prefix + "input_proj.weight", kHidden, 2U * kHidden);
        if (!projection.ok()) {
            result.errors = std::move(projection.errors);
            return result;
        }
        weights.input_projection = projection.value;
        const auto mlp = prefix + "transformer_block.mlp.";
        auto gate_up = map_matrix(mlp + "w13_dn.weight", kDenseUp, kHidden);
        if (!gate_up.ok()) {
            result.errors = std::move(gate_up.errors);
            return result;
        }
        weights.dense_gate_up = gate_up.value;
        auto down = map_matrix(mlp + "w2_md.weight", kHidden, kDenseInner);
        if (!down.ok()) {
            result.errors = std::move(down.errors);
            return result;
        }
        weights.dense_down = down.value;
        std::vector<float> scale;
        result = load_vector(mlp + "global_scale", 1U, scale);
        if (!result.ok()) return result;
        weights.dense_global_scale = scale[0];
        return result;
    }
};

InklingRuntime::InklingRuntime() : impl_(std::make_unique<Impl>()) {}
InklingRuntime::~InklingRuntime() = default;
InklingRuntime::InklingRuntime(InklingRuntime&&) noexcept = default;
InklingRuntime& InklingRuntime::operator=(InklingRuntime&&) noexcept = default;

ValidationResult InklingRuntime::initialize(const std::string& model_directory,
                                            const InklingRuntimeConfig& config) {
    ValidationResult result;
    if (impl_->initialized) {
        result.errors.emplace_back("Inkling runtime is already initialized");
        return result;
    }
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > kContract.maximum_context_tokens) {
        result.errors.emplace_back("Inkling context capacity is invalid");
        return result;
    }
    auto checkpoint = InklingCheckpointReader::open(model_directory);
    if (!checkpoint.ok()) {
        result.errors = std::move(checkpoint.errors);
        return result;
    }
    const bool mxfp4 = checkpoint.value->format() ==
                       InklingCheckpointFormat::Mxfp4Group32;
    const auto spec = mxfp4 ? inkling_small_mxfp4_spec()
                            : inkling_small_nvfp4_spec();
    result = mxfp4 ? validate_inkling_small_mxfp4(spec)
                   : validate_inkling_small_nvfp4(spec);
    if (!result.ok()) return result;
    auto tokenizer = ModelTokenizer::load(
        (std::filesystem::path(model_directory) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }

    impl_->config = config;
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->router = spec.router;
    impl_->sampler.seed(config.sampling_seed);
    const auto hardware = std::thread::hardware_concurrency();
    impl_->workers = std::make_unique<HostWorkerPool>(
        hardware == 0U ? 1U : static_cast<std::size_t>(hardware));
    if (!config.route_trace_path.empty()) {
        result = impl_->route_trace.open(config.route_trace_path);
        if (!result.ok()) return result;
    }

    auto& reader = *impl_->checkpoint;
    auto embedding = impl_->map_matrix(
        mxfp4 ? "language_model.model.embed_tokens"
               : "model.llm.embed.weight",
                                       kContract.padded_vocabulary_size, kHidden);
    if (!embedding.ok()) {
        result.errors = std::move(embedding.errors);
        return result;
    }
    impl_->embedding = embedding.value;
    auto unembedding = impl_->map_matrix(
        mxfp4 ? "language_model.lm_head" : "model.llm.unembed.weight",
                                         kContract.padded_vocabulary_size,
                                         kHidden);
    if (!unembedding.ok()) {
        result.errors = std::move(unembedding.errors);
        return result;
    }
    impl_->unembedding = unembedding.value;
    result = impl_->load_vector(
        mxfp4 ? "language_model.model.embed_norm.weight"
               : "model.llm.embed_norm.weight",
        kHidden,
                                impl_->embedding_norm);
    if (!result.ok()) return result;
    result = impl_->load_vector(
        mxfp4 ? "language_model.model.norm.weight" : "model.llm.norm.weight",
        kHidden,
                                impl_->final_norm);
    if (!result.ok()) return result;

    for (std::uint32_t index = 0U; index < kLayers; ++index) {
        result = impl_->load_layer(index);
        if (!result.ok()) return result;
        if (config.load_progress) {
            std::fprintf(stderr, "\rInkling layer %u/%u", index + 1U, kLayers);
        }
    }
    if (config.enable_mtp_speculation) {
        if (mxfp4) {
            result.errors.emplace_back(
                "Inkling MXFP4 checkpoint has no indexed MTP weights");
            return result;
        }
        const auto depths = config.speculation_depth == 0U
            ? kContract.mtp_layers
            : std::min(config.speculation_depth, kContract.mtp_layers);
        impl_->mtp.resize(depths);
        impl_->mtp_state.resize(depths);
        for (std::uint32_t depth = 0U; depth < depths; ++depth) {
            result = impl_->load_mtp(depth);
            if (!result.ok()) return result;
        }
    }
    if (config.load_progress) std::fprintf(stderr, "\n");
    static_cast<void>(reader);

    result = impl_->initialize_devices();
    if (!result.ok()) return result;
    if (config.warm_expert_pages && impl_->cuda_enabled) {
        result = impl_->warm_expert_pages();
        if (!result.ok()) return result;
    }

    impl_->reset_sequence();
    impl_->initialized = true;
    return result;
}

ValidationResult InklingRuntime::forward_logits(
    std::span<const std::uint32_t> tokens,
    std::vector<std::vector<float>>& logits) {
    ValidationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("Inkling runtime is not initialized");
        return result;
    }
    if (tokens.empty()) {
        result.errors.emplace_back("Inkling teacher forcing needs at least one token");
        return result;
    }
    impl_->reset_sequence();
    logits.clear();
    logits.reserve(tokens.size());
    const auto page = impl_->config.prefill_page_tokens;
    std::vector<float> hidden;
    if (page > 1U) {
        std::vector<float> rows_hidden;
        for (std::size_t begin = 0U; begin < tokens.size(); begin += page) {
            const auto count =
                std::min<std::size_t>(page, tokens.size() - begin);
            result = impl_->forward_page(tokens.subspan(begin, count), begin,
                                         rows_hidden);
            if (!result.ok()) return result;
            for (std::size_t row = 0U; row < count; ++row) {
                std::vector<float> single(
                    rows_hidden.begin() +
                        static_cast<std::ptrdiff_t>(row * kHidden),
                    rows_hidden.begin() +
                        static_cast<std::ptrdiff_t>((row + 1U) * kHidden));
                std::vector<float> out;
                result = impl_->logits_from_hidden(single, out);
                if (!result.ok()) return result;
                logits.push_back(std::move(out));
            }
        }
        impl_->position = tokens.size();
        return result;
    }
    for (std::size_t index = 0U; index < tokens.size(); ++index) {
        result = impl_->forward(tokens[index], index, true, hidden);
        if (!result.ok()) return result;
        std::vector<float> row;
        result = impl_->logits_from_hidden(hidden, row);
        if (!result.ok()) return result;
        logits.push_back(std::move(row));
    }
    impl_->position = tokens.size();
    return result;
}

InklingGenerationResult InklingRuntime::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    InklingGenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("Inkling runtime is not initialized");
        return result;
    }
    auto encoded = impl_->tokenizer.encode(prompt);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    SamplingOptions sampling;
    sampling.temperature = impl_->config.sampling_temperature;
    return impl_->generate(std::move(encoded.value), maximum_new_tokens,
                           sampling, {}, on_token);
}

InklingGenerationResult InklingRuntime::generate_chat_stream(
    std::span<const ChatMessage> messages, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    InklingGenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("Inkling runtime is not initialized");
        return result;
    }
    std::string validation_error;
    if (!validate_chat_messages(messages, validation_error)) {
        result.errors.push_back(validation_error);
        return result;
    }
    const auto rendered = render_inkling_chat_prompt(messages);
    auto encoded = impl_->tokenizer.encode(rendered);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    return impl_->generate(std::move(encoded.value), maximum_new_tokens,
                           sampling, stop, on_token);
}

ValidationResult InklingRuntime::Impl::propose(
    std::uint32_t token, std::uint64_t token_position,
    std::span<const float> hidden, std::vector<std::uint32_t>& proposals) {
    ValidationResult result;
    proposals.clear();
    if (mtp.empty()) return result;
    // Each depth continues the chain: depth i predicts the token i+1 steps
    // ahead, consuming the previous depth's hidden state and its own proposal.
    std::vector<float> chain(hidden.begin(), hidden.end());
    std::vector<float> next;
    std::vector<float> logits;
    auto current = token;
    for (std::uint32_t depth = 0U; depth < mtp.size(); ++depth) {
        result = forward_mtp(depth, current, token_position + depth, chain, next);
        if (!result.ok()) return result;
        result = logits_from_hidden(next, logits);
        if (!result.ok()) return result;
        const auto best = static_cast<std::uint32_t>(
            std::distance(logits.begin(),
                          std::max_element(logits.begin(), logits.end())));
        proposals.push_back(best);
        chain = next;
        current = best;
    }
    return result;
}

InklingGenerationResult InklingRuntime::Impl::generate(
    std::vector<std::uint32_t> prompt, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    auto& impl = *this;
    InklingGenerationResult result;
    if (prompt.empty()) {
        result.errors.emplace_back("Inkling prompt encoded to zero tokens");
        return result;
    }
    if (maximum_new_tokens == 0U) {
        result.errors.emplace_back("maximum_new_tokens must be positive");
        return result;
    }
    std::string sampling_error;
    if (!validate_sampling_options(sampling, sampling_error)) {
        result.errors.push_back(sampling_error);
        return result;
    }
    if (prompt.size() + maximum_new_tokens > impl.config.maximum_context_tokens) {
        result.errors.emplace_back(
            "Inkling prompt and generation exceed the configured context");
        return result;
    }

    impl.reset_sequence();
    impl.active_sampling = sampling;
    result.prompt_token_ids = prompt;
    result.metrics.prompt_tokens = prompt.size();
    result.metrics.prefill_tokens = prompt.size();

    std::vector<float> hidden;
    std::vector<float> logits;
    auto started = std::chrono::steady_clock::now();
    // Only the final row's hidden state feeds the first sampled token, but
    // every row still has to run: each one's K/V is what the rows after it
    // attend to.
    const auto page = impl.config.prefill_page_tokens;
    if (page > 1U) {
        std::vector<float> rows_hidden;
        for (std::size_t begin = 0U; begin < prompt.size(); begin += page) {
            const auto count =
                std::min<std::size_t>(page, prompt.size() - begin);
            auto status = impl.forward_page(
                std::span<const std::uint32_t>(prompt).subspan(begin, count),
                begin, rows_hidden);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return result;
            }
        }
        hidden.assign(rows_hidden.end() - static_cast<std::ptrdiff_t>(kHidden),
                      rows_hidden.end());
    } else {
        for (std::size_t index = 0U; index < prompt.size(); ++index) {
            auto status = impl.forward(prompt[index], index, true, hidden);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return result;
            }
        }
    }
    auto status = impl.logits_from_hidden(hidden, logits);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        return result;
    }
    result.metrics.prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    result.metrics.prefill_graph = impl.graph;
    result.metrics.prefill_cuda = impl.cuda.stats();
    if (impl.expert_cache != nullptr) {
        const auto cache = impl.expert_cache->stats();
        result.metrics.prefill_device.expert_hits = cache.hits;
        result.metrics.prefill_device.expert_misses = cache.misses;
        result.metrics.prefill_device.expert_evictions = cache.evictions;
        result.metrics.prefill_device.expert_stage_nanoseconds =
            cache.stage_nanoseconds;
        result.metrics.prefill_device.expert_staged_bytes = cache.staged_bytes;
        result.metrics.prefill_device.cache_capacity_bytes = cache.capacity_bytes;
        result.metrics.prefill_device.cache_peak_bytes = cache.peak_bytes;
        result.metrics.prefill_device.enabled = impl.cuda_enabled;
    }

    const auto end_of_text = impl.tokenizer.token_id("<|end_message|>");
    const auto end_of_stream = impl.tokenizer.token_id("<|endoftext|>");
    const auto is_stop = [&](std::uint32_t token) {
        return (end_of_text >= 0 &&
                token == static_cast<std::uint32_t>(end_of_text)) ||
               (end_of_stream >= 0 &&
                token == static_cast<std::uint32_t>(end_of_stream));
    };

    started = std::chrono::steady_clock::now();
    auto next_position = prompt.size();
    std::string text;
    // Repetition penalties read the counts of tokens produced so far, so the
    // history has to be rebuilt as generation proceeds.
    std::vector<std::uint32_t> sampled_counts(kContract.vocabulary_size, 0U);
    std::vector<std::uint32_t> sampled_tokens;
    // Speculation is measured, not acted on. Turning a proposal into a skipped
    // backbone step needs KV and conv rollback on rejection, which does not
    // exist yet; until the acceptance rate is known there is nothing to justify
    // building it, and a draft that is not consumed must never change output.
    std::vector<std::uint32_t> proposals;
    std::vector<std::uint32_t> pending_proposals;
    for (std::uint32_t produced = 0U; produced < maximum_new_tokens; ++produced) {
        const SamplingHistory history{sampled_counts, sampled_tokens};
        const auto sampled =
            sample_logits(logits, impl.active_sampling, history, impl.sampler);
        const auto token = sampled.token;
        impl.last_sample = sampled;
        result.logprobs.push_back(sampled);
        if (token < sampled_counts.size()) ++sampled_counts[token];
        sampled_tokens.push_back(token);
        if (is_stop(token)) {
            result.stopped = true;
            break;
        }
        result.generated_token_ids.push_back(token);
        auto piece = impl.tokenizer.decode_token(token);
        if (piece.ok()) {
            text += piece.value;
            if (on_token && !on_token(token, piece.value)) break;
        }
        bool hit_stop = false;
        for (const auto& sequence : stop) {
            if (!sequence.empty() && text.ends_with(sequence)) {
                text.erase(text.size() - sequence.size());
                hit_stop = true;
                break;
            }
        }
        if (hit_stop) {
            result.stopped = true;
            break;
        }
        if (produced + 1U == maximum_new_tokens) break;

        if (!pending_proposals.empty()) {
            ++result.metrics.speculation.proposed;
            if (pending_proposals.front() == token) {
                ++result.metrics.speculation.accepted;
            }
        }

        status = impl.forward(token, next_position++, false, hidden);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        if (impl.config.enable_mtp_speculation && !impl.mtp.empty()) {
            for (auto& depth : impl.mtp_state) {
                depth.keys.clear();
                depth.values.clear();
                depth.rows = 0U;
                depth.base_position = 0U;
                reset_conv_history(depth);
            }
            status = impl.propose(token, next_position, hidden, proposals);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return result;
            }
            pending_proposals = proposals;
        }
        status = impl.logits_from_hidden(hidden, logits);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
    }
    result.metrics.decode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    result.metrics.decode_tokens = result.generated_token_ids.size();
    result.metrics.graph = impl.graph;
    result.metrics.checkpoint_reads = impl.checkpoint->stats();
    result.metrics.cuda = impl.cuda.stats();
    result.metrics.device.enabled = impl.cuda_enabled;
    result.metrics.device.resident_spine_bytes = impl.resident_spine_bytes;
    result.metrics.device.resident_kv_bytes = impl.resident_kv_bytes;
    if (impl.expert_cache != nullptr) {
        const auto cache = impl.expert_cache->stats();
        result.metrics.device.expert_hits = cache.hits;
        result.metrics.device.expert_misses = cache.misses;
        result.metrics.device.expert_evictions = cache.evictions;
        result.metrics.device.expert_stage_nanoseconds = cache.stage_nanoseconds;
        result.metrics.device.expert_staged_bytes = cache.staged_bytes;
        result.metrics.device.cache_capacity_bytes = cache.capacity_bytes;
        result.metrics.device.cache_peak_bytes = cache.peak_bytes;
    }
    result.metrics.rss_bytes = process_resident_set_bytes();
    result.text = std::move(text);
    impl.position = next_position;
    return result;
}

}  // namespace strata
