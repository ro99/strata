#include "strata/gemma4_runtime.hpp"

#include "strata/attention.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/gemma4_checkpoint.hpp"
#include "strata/gemma4_image.hpp"
#include "strata/gemma4_ops.hpp"
#include "strata/model_adapter.hpp"
#include "strata/numerics.hpp"
#include "strata/runtime_support.hpp"
#include "strata/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace strata {
namespace {

constexpr auto& c = kGemma4ExecutionContract;
constexpr std::uint32_t kPrefillChunk = 128U;
constexpr std::uint64_t kMinimumAttentionWorkspace = 768ULL << 20U;

std::string layer_prefix(std::uint32_t layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

void round_bf16(std::span<float> values) {
    for (auto& value : values) value = bf16_round_f32(value);
}

float decode_bf16(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint64_t linear_bytes(const Gemma4CheckpointReader& checkpoint,
                           std::string_view base) {
    if (const auto* plain = checkpoint.find(std::string(base) + ".weight");
        plain != nullptr) return plain->bytes;
    const auto* packed = checkpoint.find(std::string(base) + ".weight_packed");
    const auto* scales = checkpoint.find(std::string(base) + ".weight_scale");
    return packed == nullptr || scales == nullptr
        ? std::numeric_limits<std::uint64_t>::max()
        : packed->bytes + scales->bytes;
}

struct Linear {
    CudaWeight weight;
    std::uint64_t rows{};
    std::uint64_t columns{};
};

struct TextLayer {
    std::size_t device{};
    Linear query;
    Linear key;
    std::optional<Linear> value;
    Linear output;
    Linear gate;
    Linear up;
    Linear down;
    std::vector<float> input_norm;
    std::vector<float> post_attention_norm;
    std::vector<float> pre_feedforward_norm;
    std::vector<float> post_feedforward_norm;
    std::vector<float> query_norm;
    std::vector<float> key_norm;
    CudaBuffer input_norm_device;
    CudaBuffer post_attention_norm_device;
    CudaBuffer pre_feedforward_norm_device;
    CudaBuffer post_feedforward_norm_device;
    CudaBuffer query_norm_device;
    CudaBuffer key_norm_device;
    float scalar{1.0F};
};

struct VisionLayer {
    Linear query;
    Linear key;
    Linear value;
    Linear output;
    Linear gate;
    Linear up;
    Linear down;
    std::vector<float> input_norm;
    std::vector<float> post_attention_norm;
    std::vector<float> pre_feedforward_norm;
    std::vector<float> post_feedforward_norm;
    std::vector<float> query_norm;
    std::vector<float> key_norm;
};

struct VisionWeights {
    std::size_t device{};
    Linear patch;
    Linear projection;
    std::vector<VisionLayer> layers;
    std::vector<float> positions;
    std::vector<float> std_bias;
    std::vector<float> std_scale;
};

struct LayerKv {
    std::vector<std::uint16_t> keys;
    std::vector<std::uint16_t> values;
    std::vector<std::uint16_t> next_keys;
    std::vector<std::uint16_t> next_values;
    CudaBuffer device;
    std::uint32_t capacity_rows{};
    std::uint32_t start{};
};

struct KvMark {
    std::uint32_t start{};
    std::size_t key_size{};
    std::size_t value_size{};
    std::vector<std::uint16_t> first_key;
    std::vector<std::uint16_t> first_value;
};

}  // namespace

struct Gemma4Runtime::Impl {
    Gemma4RuntimeConfig config;
    std::unique_ptr<Gemma4CheckpointReader> checkpoint;
    ModelTokenizer tokenizer;
    CudaBackend cuda;
    std::vector<int> devices;
    std::vector<std::size_t> schedule;
    std::vector<TextLayer> layers;
    VisionWeights vision;
    std::vector<LayerKv> kv{c.layer_count};
    Linear output_head;
    std::size_t output_device{};
    std::vector<float> final_norm;
    std::unordered_map<std::uint32_t, std::vector<float>> embedding_rows;
    std::vector<std::uint32_t> cached_token_ids;
    std::vector<std::uint32_t> sampled_counts;
    std::vector<std::uint32_t> sampled_ids;
    std::mt19937_64 sampler;
    SamplingOptions active_sampling;
    TokenLogprob last_sample;
    bool initialized{};
    bool reusable_sequence{};
    bool device_kv_ready{};

    std::size_t layer_device(std::uint32_t layer) const {
        return schedule[layer % schedule.size()];
    }

    ValidationResult linear(const Linear& weight, std::span<const float> input,
                            std::uint32_t rows, std::span<float> output) {
        auto result = cuda.matmul(weight.weight, input, rows, output);
        if (result.ok()) round_bf16(output);
        return result;
    }

    ParseResult<std::vector<float>> tensor(std::string_view name,
                                           std::uint64_t elements) {
        return checkpoint->read_f32(name, elements);
    }

    ValidationResult load_linear(std::size_t device, std::string_view name,
                                 std::uint64_t rows, std::uint64_t columns,
                                 Linear& output) {
        auto result = load_gemma4_cuda_linear(
            *checkpoint, name, rows, columns, devices[device], cuda,
            output.weight);
        if (result.ok()) {
            output.rows = rows;
            output.columns = columns;
        }
        return result;
    }

    ValidationResult upload_vector(std::size_t device,
                                   std::span<const float> values,
                                   CudaBuffer& output) {
        return cuda.upload_buffer(
            devices[device], std::as_bytes(values), output);
    }

    ValidationResult load_text_weights() {
        ValidationResult result;
        layers.resize(c.layer_count);
        const auto load_vector = [this, &result](std::string_view name,
                                                 std::uint64_t elements,
                                                 std::vector<float>& output) {
            if (!result.ok()) return;
            auto loaded = tensor(name, elements);
            if (!loaded.ok()) result.errors = std::move(loaded.errors);
            else output = std::move(loaded.value);
        };
        for (std::uint32_t layer = 0U; layer < c.layer_count && result.ok(); ++layer) {
            auto& weights = layers[layer];
            weights.device = layer_device(layer);
            const bool global = gemma4_global_attention_layer(layer);
            const auto head_dim = global ? c.global_head_dim : c.local_head_dim;
            const auto kv_heads = global ? c.global_key_value_heads
                                         : c.local_key_value_heads;
            const auto prefix = layer_prefix(layer);
            const auto attention = prefix + "self_attn.";
            const auto load = [&](std::string_view name, std::uint64_t rows,
                                  std::uint64_t columns, Linear& output) {
                if (result.ok()) result = load_linear(
                    weights.device, name, rows, columns, output);
            };
            load(attention + "q_proj", c.attention_heads * head_dim,
                 c.hidden_size, weights.query);
            load(attention + "k_proj", kv_heads * head_dim,
                 c.hidden_size, weights.key);
            if (!global) {
                weights.value.emplace();
                load(attention + "v_proj", kv_heads * head_dim,
                     c.hidden_size, *weights.value);
            }
            load(attention + "o_proj", c.hidden_size,
                 c.attention_heads * head_dim, weights.output);
            load(prefix + "mlp.gate_proj", c.intermediate_size,
                 c.hidden_size, weights.gate);
            load(prefix + "mlp.up_proj", c.intermediate_size,
                 c.hidden_size, weights.up);
            load(prefix + "mlp.down_proj", c.hidden_size,
                 c.intermediate_size, weights.down);
            load_vector(prefix + "input_layernorm.weight", c.hidden_size,
                        weights.input_norm);
            load_vector(prefix + "post_attention_layernorm.weight", c.hidden_size,
                        weights.post_attention_norm);
            load_vector(prefix + "pre_feedforward_layernorm.weight", c.hidden_size,
                        weights.pre_feedforward_norm);
            load_vector(prefix + "post_feedforward_layernorm.weight", c.hidden_size,
                        weights.post_feedforward_norm);
            load_vector(attention + "q_norm.weight", head_dim,
                        weights.query_norm);
            load_vector(attention + "k_norm.weight", head_dim,
                        weights.key_norm);
            std::vector<float> scalar;
            load_vector(prefix + "layer_scalar", 1U, scalar);
            if (result.ok()) weights.scalar = scalar.front();
            if (result.ok()) result = upload_vector(
                weights.device, weights.input_norm, weights.input_norm_device);
            if (result.ok()) result = upload_vector(
                weights.device, weights.post_attention_norm,
                weights.post_attention_norm_device);
            if (result.ok()) result = upload_vector(
                weights.device, weights.pre_feedforward_norm,
                weights.pre_feedforward_norm_device);
            if (result.ok()) result = upload_vector(
                weights.device, weights.post_feedforward_norm,
                weights.post_feedforward_norm_device);
            if (result.ok()) result = upload_vector(
                weights.device, weights.query_norm, weights.query_norm_device);
            if (result.ok()) result = upload_vector(
                weights.device, weights.key_norm, weights.key_norm_device);
            if (result.ok()) {
                auto& cache = kv[layer];
                cache.capacity_rows = global
                    ? config.maximum_context_tokens
                    : std::min(config.maximum_context_tokens,
                               c.sliding_window);
                const auto cache_elements =
                    static_cast<std::uint64_t>(cache.capacity_rows) *
                    kv_heads * head_dim * 2U;
                result = cuda.allocate_buffer(
                    devices[weights.device],
                    cache_elements * sizeof(std::uint16_t), cache.device);
                cache.next_keys.resize(
                    static_cast<std::size_t>(kv_heads) * head_dim);
                cache.next_values.resize(cache.next_keys.size());
            }
            if ((config.verbose || config.load_progress) && result.ok()) {
                std::cerr << "[load] Gemma 4 layer " << layer + 1U << '/'
                          << c.layer_count << '\n';
            }
        }
        if (!result.ok()) return result;
        output_device = layer_device(c.layer_count - 1U);
        result = load_linear(output_device,
            "model.language_model.embed_tokens", c.vocabulary_size,
            c.hidden_size, output_head);
        if (!result.ok()) return result;
        auto norm = tensor("model.language_model.norm.weight", c.hidden_size);
        if (!norm.ok()) result.errors = std::move(norm.errors);
        else final_norm = std::move(norm.value);
        return result;
    }

    ValidationResult load_vision_weights() {
        ValidationResult result;
        vision.device = 0U;
        vision.layers.resize(c.vision_layer_count);
        const auto load = [&](std::string_view name, std::uint64_t rows,
                              std::uint64_t columns, Linear& output) {
            if (result.ok()) result = load_linear(
                vision.device, name, rows, columns, output);
        };
        const auto load_vector = [this, &result](std::string_view name,
                                                 std::uint64_t elements,
                                                 std::vector<float>& output) {
            if (!result.ok()) return;
            auto loaded = tensor(name, elements);
            if (!loaded.ok()) result.errors = std::move(loaded.errors);
            else output = std::move(loaded.value);
        };
        load("model.vision_tower.patch_embedder.input_proj",
             c.vision_hidden_size, 3U * c.vision_patch_size * c.vision_patch_size,
             vision.patch);
        load("model.embed_vision.embedding_projection", c.hidden_size,
             c.vision_hidden_size, vision.projection);
        load_vector("model.vision_tower.patch_embedder.position_embedding_table",
                    2ULL * c.vision_position_embeddings * c.vision_hidden_size,
                    vision.positions);
        load_vector("model.vision_tower.std_bias", c.vision_hidden_size,
                    vision.std_bias);
        load_vector("model.vision_tower.std_scale", c.vision_hidden_size,
                    vision.std_scale);
        for (std::uint32_t layer = 0U;
             layer < c.vision_layer_count && result.ok(); ++layer) {
            auto& weights = vision.layers[layer];
            const auto prefix = "model.vision_tower.encoder.layers." +
                std::to_string(layer) + ".";
            const auto attention = prefix + "self_attn.";
            load(attention + "q_proj.linear", c.vision_hidden_size,
                 c.vision_hidden_size, weights.query);
            load(attention + "k_proj.linear", c.vision_hidden_size,
                 c.vision_hidden_size, weights.key);
            load(attention + "v_proj.linear", c.vision_hidden_size,
                 c.vision_hidden_size, weights.value);
            load(attention + "o_proj.linear", c.vision_hidden_size,
                 c.vision_hidden_size, weights.output);
            load(prefix + "mlp.gate_proj.linear", c.vision_intermediate_size,
                 c.vision_hidden_size, weights.gate);
            load(prefix + "mlp.up_proj.linear", c.vision_intermediate_size,
                 c.vision_hidden_size, weights.up);
            load(prefix + "mlp.down_proj.linear", c.vision_hidden_size,
                 c.vision_intermediate_size, weights.down);
            load_vector(prefix + "input_layernorm.weight", c.vision_hidden_size,
                        weights.input_norm);
            load_vector(prefix + "post_attention_layernorm.weight",
                        c.vision_hidden_size, weights.post_attention_norm);
            load_vector(prefix + "pre_feedforward_layernorm.weight",
                        c.vision_hidden_size, weights.pre_feedforward_norm);
            load_vector(prefix + "post_feedforward_layernorm.weight",
                        c.vision_hidden_size, weights.post_feedforward_norm);
            load_vector(attention + "q_norm.weight", c.vision_head_dim,
                        weights.query_norm);
            load_vector(attention + "k_norm.weight", c.vision_head_dim,
                        weights.key_norm);
            if ((config.verbose || config.load_progress) && result.ok()) {
                std::cerr << "[load] Gemma 4 vision layer " << layer + 1U
                          << '/' << c.vision_layer_count << '\n';
            }
        }
        return result;
    }

    ValidationResult vision_attention(
        const VisionLayer& weights, std::span<const float> input,
        const Gemma4PreparedImage& image, std::span<float> output) {
        ValidationResult result;
        const auto rows = image.positions.size() / 2U;
        std::vector<float> queries(rows * c.vision_hidden_size);
        std::vector<float> keys(queries.size());
        std::vector<float> values(queries.size());
        result = linear(weights.query, input, static_cast<std::uint32_t>(rows),
                        queries);
        if (!result.ok()) return result;
        result = linear(weights.key, input, static_cast<std::uint32_t>(rows), keys);
        if (!result.ok()) return result;
        result = linear(weights.value, input, static_cast<std::uint32_t>(rows),
                        values);
        if (!result.ok()) return result;
        std::vector<float> unscaled(c.vision_head_dim, 1.0F);
        for (std::size_t row = 0U; row < rows; ++row) {
            const auto x = image.positions[row * 2U];
            const auto y = image.positions[row * 2U + 1U];
            for (std::uint32_t head = 0U; head < c.vision_attention_heads; ++head) {
                const auto offset =
                    (row * c.vision_attention_heads + head) * c.vision_head_dim;
                auto query = std::span<float>(queries).subspan(
                    offset, c.vision_head_dim);
                auto key = std::span<float>(keys).subspan(
                    offset, c.vision_head_dim);
                auto value = std::span<float>(values).subspan(
                    offset, c.vision_head_dim);
                result = gemma4_rms_norm_bf16(
                    query, std::span<const float>(query), weights.query_norm);
                if (!result.ok()) return result;
                result = gemma4_rms_norm_bf16(
                    key, std::span<const float>(key), weights.key_norm);
                if (!result.ok()) return result;
                result = gemma4_rms_norm_bf16(
                    value, std::span<const float>(value), unscaled);
                if (!result.ok()) return result;
                result = gemma4_vision_rope_bf16(query, x, y);
                if (!result.ok()) return result;
                result = gemma4_vision_rope_bf16(key, x, y);
                if (!result.ok()) return result;
            }
        }
        const std::array<FlashAttentionSegment, 1> segments{{{keys, values, {}}}};
        FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.query_rows = static_cast<std::uint32_t>(rows);
        request.query_heads = c.vision_attention_heads;
        request.key_value_heads = c.vision_attention_heads;
        request.query_key_dim = c.vision_head_dim;
        request.value_dim = c.vision_head_dim;
        request.scale = 1.0F;
        request.numerics =
            FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum;
        request.maximum_workspace_bytes = kMinimumAttentionWorkspace;
        std::vector<float> context(queries.size());
        result = config.enable_flash_attention
            ? cuda.flash_attention(devices[vision.device], request, context)
            : flash_attention_reference_f32(request, context);
        if (!result.ok()) return result;
        round_bf16(context);
        return linear(weights.output, context, static_cast<std::uint32_t>(rows),
                      output);
    }

    ValidationResult vision_mlp(const VisionLayer& weights,
                                std::span<const float> input,
                                std::uint32_t rows,
                                std::span<float> output) {
        ValidationResult result;
        std::vector<float> gate(
            static_cast<std::size_t>(rows) * c.vision_intermediate_size);
        std::vector<float> up(gate.size());
        result = linear(weights.gate, input, rows, gate);
        if (!result.ok()) return result;
        result = linear(weights.up, input, rows, up);
        if (!result.ok()) return result;
        result = gemma4_geglu_bf16(gate, gate, up);
        if (!result.ok()) return result;
        return linear(weights.down, gate, rows, output);
    }

    ParseResult<std::vector<float>> encode_image(
        const Gemma4PreparedImage& image) {
        ParseResult<std::vector<float>> result;
        const auto rows = static_cast<std::uint32_t>(image.positions.size() / 2U);
        if (rows == 0U || image.patches.size() !=
                static_cast<std::size_t>(rows) * 3U * c.vision_patch_size *
                    c.vision_patch_size) {
            result.errors.emplace_back("Gemma 4 prepared image shape is invalid");
            return result;
        }
        std::vector<float> hidden(
            static_cast<std::size_t>(rows) * c.vision_hidden_size);
        auto status = linear(vision.patch, image.patches, rows, hidden);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto x = image.positions[static_cast<std::size_t>(row) * 2U];
            const auto y = image.positions[static_cast<std::size_t>(row) * 2U + 1U];
            if (x < 0 || y < 0 ||
                static_cast<std::uint32_t>(x) >= c.vision_position_embeddings ||
                static_cast<std::uint32_t>(y) >= c.vision_position_embeddings) {
                result.errors.emplace_back("Gemma 4 image position is out of range");
                return result;
            }
            const auto x_offset = static_cast<std::size_t>(x) * c.vision_hidden_size;
            const auto y_offset =
                (c.vision_position_embeddings + static_cast<std::size_t>(y)) *
                c.vision_hidden_size;
            const auto output_offset =
                static_cast<std::size_t>(row) * c.vision_hidden_size;
            for (std::size_t column = 0U; column < c.vision_hidden_size; ++column) {
                hidden[output_offset + column] = bf16_round_f32(
                    hidden[output_offset + column] +
                    vision.positions[x_offset + column] +
                    vision.positions[y_offset + column]);
            }
        }
        std::vector<float> normalized(hidden.size());
        std::vector<float> branch(hidden.size());
        for (const auto& weights : vision.layers) {
            status = norm_rows(normalized, hidden, weights.input_norm,
                               rows, c.vision_hidden_size);
            if (!status.ok()) break;
            status = vision_attention(weights, normalized, image, branch);
            if (!status.ok()) break;
            status = norm_rows(normalized, branch, weights.post_attention_norm,
                               rows, c.vision_hidden_size);
            if (!status.ok()) break;
            for (std::size_t index = 0U; index < hidden.size(); ++index) {
                hidden[index] = bf16_round_f32(hidden[index] + normalized[index]);
            }
            status = norm_rows(normalized, hidden, weights.pre_feedforward_norm,
                               rows, c.vision_hidden_size);
            if (!status.ok()) break;
            status = vision_mlp(weights, normalized, rows, branch);
            if (!status.ok()) break;
            status = norm_rows(normalized, branch, weights.post_feedforward_norm,
                               rows, c.vision_hidden_size);
            if (!status.ok()) break;
            for (std::size_t index = 0U; index < hidden.size(); ++index) {
                hidden[index] = bf16_round_f32(hidden[index] + normalized[index]);
            }
        }
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        const auto soft_rows = image.soft_tokens;
        if (soft_rows == 0U || rows != soft_rows *
                c.vision_pooling_kernel * c.vision_pooling_kernel) {
            result.errors.emplace_back("Gemma 4 image pooling shape is invalid");
            return result;
        }
        std::vector<float> pooled(
            static_cast<std::size_t>(soft_rows) * c.vision_hidden_size, 0.0F);
        const auto pooled_width = image.patch_width / c.vision_pooling_kernel;
        const float coefficient = 1.0F /
            static_cast<float>(c.vision_pooling_kernel * c.vision_pooling_kernel);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto x = static_cast<std::uint32_t>(
                image.positions[static_cast<std::size_t>(row) * 2U]);
            const auto y = static_cast<std::uint32_t>(
                image.positions[static_cast<std::size_t>(row) * 2U + 1U]);
            const auto destination = (x / c.vision_pooling_kernel) +
                pooled_width * (y / c.vision_pooling_kernel);
            for (std::size_t column = 0U; column < c.vision_hidden_size; ++column) {
                pooled[static_cast<std::size_t>(destination) * c.vision_hidden_size +
                       column] += hidden[static_cast<std::size_t>(row) *
                                        c.vision_hidden_size + column] * coefficient;
            }
        }
        const float scale = std::sqrt(static_cast<float>(c.vision_hidden_size));
        std::vector<float> ones(c.vision_hidden_size, 1.0F);
        for (std::uint32_t row = 0U; row < soft_rows; ++row) {
            auto values = std::span<float>(pooled).subspan(
                static_cast<std::size_t>(row) * c.vision_hidden_size,
                c.vision_hidden_size);
            for (std::size_t column = 0U; column < c.vision_hidden_size; ++column) {
                values[column] = bf16_round_f32(values[column] * scale);
                values[column] = bf16_round_f32(
                    values[column] - vision.std_bias[column]);
                values[column] = bf16_round_f32(
                    values[column] * vision.std_scale[column]);
            }
            status = gemma4_rms_norm_bf16(
                values, std::span<const float>(values), ones);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return result;
            }
        }
        result.value.resize(static_cast<std::size_t>(soft_rows) * c.hidden_size);
        status = linear(vision.projection, pooled, soft_rows, result.value);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            result.value.clear();
        }
        return result;
    }

    ValidationResult embed(std::span<const std::uint32_t> token_ids,
                           std::span<const float> replacements,
                           std::span<const std::uint8_t> replacement_mask,
                           std::span<float> output) {
        ValidationResult result;
        if (output.size() != token_ids.size() * c.hidden_size ||
            (!replacement_mask.empty() &&
             (replacement_mask.size() != token_ids.size() ||
              replacements.size() != output.size()))) {
            result.errors.emplace_back("Gemma 4 embedding output shape mismatch");
            return result;
        }
        const float scale = bf16_round_f32(std::sqrt(
            static_cast<float>(c.hidden_size)));
        for (std::size_t row = 0U; row < token_ids.size(); ++row) {
            const auto token = token_ids[row];
            if (token >= c.vocabulary_size) {
                result.errors.emplace_back("token id exceeds the Gemma 4 vocabulary");
                return result;
            }
            auto found = embedding_rows.find(token);
            if (found == embedding_rows.end()) {
                auto loaded = checkpoint->read_f32_row(
                    "model.language_model.embed_tokens.weight", token);
                if (!loaded.ok()) {
                    result.errors = std::move(loaded.errors);
                    return result;
                }
                found = embedding_rows.emplace(token, std::move(loaded.value)).first;
            }
            auto destination = output.subspan(row * c.hidden_size, c.hidden_size);
            for (std::size_t column = 0U; column < c.hidden_size; ++column) {
                destination[column] = bf16_round_f32(
                    found->second[column] * scale);
            }
            if (!replacement_mask.empty() && replacement_mask[row] != 0U) {
                std::copy(replacements.begin() +
                              static_cast<std::ptrdiff_t>(row * c.hidden_size),
                          replacements.begin() +
                              static_cast<std::ptrdiff_t>((row + 1U) * c.hidden_size),
                          destination.begin());
            }
        }
        return result;
    }

    ValidationResult norm_rows(std::span<float> output,
                               std::span<const float> input,
                               std::span<const float> weight,
                               std::uint32_t rows, std::uint32_t columns) {
        ValidationResult result;
        if (output.size() != input.size() ||
            output.size() != static_cast<std::size_t>(rows) * columns) {
            result.errors.emplace_back("Gemma 4 RMSNorm row shape mismatch");
            return result;
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = gemma4_rms_norm_bf16(
                output.subspan(static_cast<std::size_t>(row) * columns, columns),
                input.subspan(static_cast<std::size_t>(row) * columns, columns),
                weight);
            if (!result.ok()) return result;
        }
        return result;
    }

    ValidationResult attention(std::uint32_t layer,
                               std::span<const float> input,
                               std::uint32_t rows,
                               std::uint32_t position_base,
                               std::span<const std::int32_t> multimodal_groups,
                               std::span<float> output) {
        ValidationResult result;
        auto& weights = layers[layer];
        auto& cache = kv[layer];
        const bool global = gemma4_global_attention_layer(layer);
        const auto head_dim = global ? c.global_head_dim : c.local_head_dim;
        const auto kv_heads = global ? c.global_key_value_heads
                                     : c.local_key_value_heads;
        const auto query_columns = c.attention_heads * head_dim;
        const auto kv_columns = kv_heads * head_dim;
        std::vector<float> queries(static_cast<std::size_t>(rows) * query_columns);
        std::vector<float> new_keys(static_cast<std::size_t>(rows) * kv_columns);
        std::vector<float> new_values(static_cast<std::size_t>(rows) * kv_columns);
        result = linear(weights.query, input, rows, queries);
        if (!result.ok()) return result;
        result = linear(weights.key, input, rows, new_keys);
        if (!result.ok()) return result;
        if (weights.value.has_value()) {
            result = linear(*weights.value, input, rows, new_values);
            if (!result.ok()) return result;
        } else {
            new_values = new_keys;
        }
        const float theta = global ? c.global_rope_theta : c.local_rope_theta;
        const float proportion = global ? c.global_rope_proportion : 1.0F;
        std::vector<float> unscaled(head_dim, 1.0F);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            for (std::uint32_t head = 0U; head < c.attention_heads; ++head) {
                auto query = std::span<float>(queries).subspan(
                    (static_cast<std::size_t>(row) * c.attention_heads + head) *
                        head_dim,
                    head_dim);
                result = gemma4_rms_norm_bf16(
                    query, std::span<const float>(query), weights.query_norm);
                if (!result.ok()) return result;
                result = gemma4_rope_bf16(
                    query, position_base + row, theta, proportion);
                if (!result.ok()) return result;
            }
            for (std::uint32_t head = 0U; head < kv_heads; ++head) {
                auto key = std::span<float>(new_keys).subspan(
                    (static_cast<std::size_t>(row) * kv_heads + head) * head_dim,
                    head_dim);
                result = gemma4_rms_norm_bf16(
                    key, std::span<const float>(key), weights.key_norm);
                if (!result.ok()) return result;
                result = gemma4_rope_bf16(
                    key, position_base + row, theta, proportion);
                if (!result.ok()) return result;
                auto value = std::span<float>(new_values).subspan(
                    (static_cast<std::size_t>(row) * kv_heads + head) * head_dim,
                    head_dim);
                result = gemma4_rms_norm_bf16(
                    value, std::span<const float>(value), unscaled);
                if (!result.ok()) return result;
            }
        }

        const auto old_rows = cache.keys.size() / kv_columns;
        if (cache.values.size() != cache.keys.size() ||
            cache.start + old_rows != position_base) {
            result.errors.emplace_back("Gemma 4 KV cache is not contiguous");
            return result;
        }
        std::vector<float> keys((old_rows + rows) * kv_columns);
        std::vector<float> values(keys.size());
        for (std::size_t index = 0U; index < cache.keys.size(); ++index) {
            keys[index] = decode_bf16(cache.keys[index]);
            values[index] = decode_bf16(cache.values[index]);
        }
        std::copy(new_keys.begin(), new_keys.end(),
                  keys.begin() + static_cast<std::ptrdiff_t>(cache.keys.size()));
        std::copy(new_values.begin(), new_values.end(),
                  values.begin() + static_cast<std::ptrdiff_t>(cache.values.size()));

        const std::array<FlashAttentionSegment, 1> segments{{{keys, values, {}}}};
        FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.query_rows = rows;
        request.query_heads = c.attention_heads;
        request.key_value_heads = kv_heads;
        request.query_key_dim = head_dim;
        request.value_dim = head_dim;
        request.scale = 1.0F;
        request.numerics =
            FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum;
        request.maximum_workspace_bytes = std::max<std::uint64_t>(
            kMinimumAttentionWorkspace,
            static_cast<std::uint64_t>(config.maximum_context_tokens) *
                    c.global_key_value_heads * c.global_head_dim * 2U *
                    sizeof(float) +
                (512ULL << 20U));
        std::vector<std::uint32_t> causal;
        std::vector<std::uint8_t> mask;
        if (global) {
            causal.resize(rows);
            for (std::uint32_t row = 0U; row < rows; ++row) {
                causal[row] = position_base + row + 1U - cache.start;
            }
            request.causal_key_counts = causal;
        } else {
            mask.resize(static_cast<std::size_t>(rows) * (old_rows + rows));
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto query = static_cast<std::uint64_t>(position_base + row);
                for (std::size_t key_row = 0U; key_row < old_rows + rows; ++key_row) {
                    mask[static_cast<std::size_t>(row) * (old_rows + rows) + key_row] =
                        static_cast<std::uint8_t>(gemma4_text_attention_visible(
                            query, cache.start + key_row, true, c.sliding_window,
                            multimodal_groups));
                }
            }
            request.query_key_mask = mask;
        }
        std::vector<float> context(
            static_cast<std::size_t>(rows) * c.attention_heads * head_dim);
        result = config.enable_flash_attention
            ? cuda.flash_attention(devices[weights.device], request, context)
            : flash_attention_reference_f32(request, context);
        if (!result.ok()) return result;
        round_bf16(context);
        result = linear(weights.output, context, rows, output);
        if (!result.ok()) return result;

        cache.keys.reserve((old_rows + rows) * kv_columns);
        cache.values.reserve((old_rows + rows) * kv_columns);
        for (const auto value : new_keys) cache.keys.push_back(bf16_encode(value));
        for (const auto value : new_values) cache.values.push_back(bf16_encode(value));
        if (!global && old_rows + rows > c.sliding_window) {
            const auto drop_rows = old_rows + rows - c.sliding_window;
            const auto drop = drop_rows * kv_columns;
            cache.keys.erase(cache.keys.begin(),
                             cache.keys.begin() + static_cast<std::ptrdiff_t>(drop));
            cache.values.erase(cache.values.begin(),
                               cache.values.begin() + static_cast<std::ptrdiff_t>(drop));
            cache.start += static_cast<std::uint32_t>(drop_rows);
        }
        return result;
    }

    ValidationResult mlp(const TextLayer& weights,
                         std::span<const float> input,
                         std::uint32_t rows,
                         std::span<float> output) {
        ValidationResult result;
        std::vector<float> gate(static_cast<std::size_t>(rows) * c.intermediate_size);
        std::vector<float> up(gate.size());
        result = linear(weights.gate, input, rows, gate);
        if (!result.ok()) return result;
        result = linear(weights.up, input, rows, up);
        if (!result.ok()) return result;
        result = gemma4_geglu_bf16(gate, gate, up);
        if (!result.ok()) return result;
        return linear(weights.down, gate, rows, output);
    }

    ValidationResult sync_device_kv() {
        ValidationResult result;
        for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
            auto& cache = kv[layer];
            const bool global = gemma4_global_attention_layer(layer);
            const auto columns =
                (global ? c.global_key_value_heads * c.global_head_dim
                        : c.local_key_value_heads * c.local_head_dim);
            result = cuda.upload_gemma4_kv(
                cache.device, cache.keys, cache.values, cache.start,
                cache.capacity_rows, columns);
            if (!result.ok()) return result;
        }
        device_kv_ready = true;
        return result;
    }

    ValidationResult forward_decode_layers(std::span<float> hidden,
                                           std::uint32_t position) {
        ValidationResult result;
        std::uint32_t begin = 0U;
        while (begin < c.layer_count) {
            const auto device = layers[begin].device;
            auto end = begin + 1U;
            while (end < c.layer_count && layers[end].device == device) ++end;
            std::vector<CudaGemma4DecodeLayer> request;
            request.reserve(end - begin);
            for (auto layer = begin; layer < end; ++layer) {
                auto& weights = layers[layer];
                auto& cache = kv[layer];
                request.push_back({
                    &weights.query.weight,
                    &weights.key.weight,
                    weights.value.has_value() ? &weights.value->weight : nullptr,
                    &weights.output.weight,
                    &weights.gate.weight,
                    &weights.up.weight,
                    &weights.down.weight,
                    &weights.input_norm_device,
                    &weights.post_attention_norm_device,
                    &weights.pre_feedforward_norm_device,
                    &weights.post_feedforward_norm_device,
                    &weights.query_norm_device,
                    &weights.key_norm_device,
                    &cache.device,
                    cache.next_keys,
                    cache.next_values,
                    cache.capacity_rows,
                    cache.start,
                    static_cast<std::uint32_t>(
                        cache.keys.size() / cache.next_keys.size()),
                    weights.scalar,
                });
            }
            result = cuda.gemma4_decode_layers(
                devices[device], request, hidden, position, hidden);
            if (!result.ok()) return result;
            for (auto layer = begin; layer < end; ++layer) {
                auto& cache = kv[layer];
                cache.keys.insert(cache.keys.end(), cache.next_keys.begin(),
                                  cache.next_keys.end());
                cache.values.insert(cache.values.end(), cache.next_values.begin(),
                                    cache.next_values.end());
                const auto rows = cache.keys.size() / cache.next_keys.size();
                if (rows > cache.capacity_rows) {
                    const auto drop_rows = rows - cache.capacity_rows;
                    const auto drop = drop_rows * cache.next_keys.size();
                    cache.keys.erase(
                        cache.keys.begin(),
                        cache.keys.begin() + static_cast<std::ptrdiff_t>(drop));
                    cache.values.erase(
                        cache.values.begin(),
                        cache.values.begin() + static_cast<std::ptrdiff_t>(drop));
                    cache.start += static_cast<std::uint32_t>(drop_rows);
                }
            }
            begin = end;
        }
        return result;
    }

    ValidationResult forward_layers(std::span<float> hidden,
                                    std::uint32_t rows,
                                    std::uint32_t position_base,
                                    std::span<const std::int32_t> multimodal_groups) {
        if (rows == 1U && device_kv_ready && multimodal_groups.empty()) {
            return forward_decode_layers(hidden, position_base);
        }
        ValidationResult result;
        std::vector<float> normalized(hidden.size());
        std::vector<float> branch(hidden.size());
        for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
            const auto started = std::chrono::steady_clock::now();
            auto& weights = layers[layer];
            result = norm_rows(normalized, hidden, weights.input_norm,
                               rows, c.hidden_size);
            if (!result.ok()) return result;
            result = attention(layer, normalized, rows, position_base,
                               multimodal_groups, branch);
            if (!result.ok()) return result;
            result = norm_rows(normalized, branch, weights.post_attention_norm,
                               rows, c.hidden_size);
            if (!result.ok()) return result;
            for (std::size_t index = 0U; index < hidden.size(); ++index) {
                hidden[index] = bf16_round_f32(hidden[index] + normalized[index]);
            }
            result = norm_rows(normalized, hidden, weights.pre_feedforward_norm,
                               rows, c.hidden_size);
            if (!result.ok()) return result;
            result = mlp(weights, normalized, rows, branch);
            if (!result.ok()) return result;
            result = norm_rows(normalized, branch, weights.post_feedforward_norm,
                               rows, c.hidden_size);
            if (!result.ok()) return result;
            for (std::size_t index = 0U; index < hidden.size(); ++index) {
                hidden[index] = bf16_round_f32(
                    bf16_round_f32(hidden[index] + normalized[index]) *
                    weights.scalar);
            }
            if (config.verbose) {
                const auto seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << '[' << (rows > 1U ? "prefill" : "decode")
                          << "] Gemma 4 layer " << layer + 1U << '/'
                          << c.layer_count << " rows=" << rows
                          << " seconds=" << seconds << '\n';
            }
        }
        return result;
    }

    ValidationResult compute_logits(std::span<const std::uint32_t> token_ids,
                                    std::uint32_t position_base,
                                    std::span<const float> replacements,
                                    std::span<const std::uint8_t> replacement_mask,
                                    std::span<const std::int32_t> multimodal_groups,
                                    std::vector<float>& logits) {
        ValidationResult result;
        if (token_ids.empty() ||
            position_base + token_ids.size() > config.maximum_context_tokens) {
            result.errors.emplace_back(
                "Gemma 4 forward pass exceeds the configured context ceiling");
            return result;
        }
        std::vector<float> hidden(token_ids.size() * c.hidden_size);
        result = embed(token_ids, replacements, replacement_mask, hidden);
        if (!result.ok()) return result;
        result = forward_layers(hidden, static_cast<std::uint32_t>(token_ids.size()),
                                position_base, multimodal_groups);
        if (!result.ok()) return result;
        std::vector<float> normalized(c.hidden_size);
        result = gemma4_rms_norm_bf16(
            normalized, std::span<const float>(hidden).last(c.hidden_size),
            final_norm);
        if (!result.ok()) return result;
        logits.assign(c.vocabulary_size, 0.0F);
        return cuda.matmul_softcap(
            output_head.weight, normalized, c.final_logit_softcap, logits);
    }

    ParseResult<std::uint32_t> forward(std::span<const std::uint32_t> token_ids,
                                       std::uint32_t position_base,
                                       std::span<const float> replacements = {},
                                       std::span<const std::uint8_t> replacement_mask = {},
                                       std::span<const std::int32_t> multimodal_groups = {}) {
        ParseResult<std::uint32_t> result;
        std::vector<float> logits;
        std::size_t offset = 0U;
        while (offset < token_ids.size()) {
            auto count = std::min<std::size_t>(
                kPrefillChunk, token_ids.size() - offset);
            if (!multimodal_groups.empty()) {
                const auto absolute = position_base + offset;
                const auto group = multimodal_groups[absolute];
                if (group >= 0) {
                    count = 1U;
                    while (offset + count < token_ids.size() &&
                           multimodal_groups[position_base + offset + count] == group) {
                        ++count;
                    }
                } else {
                    for (std::size_t row = 1U; row < count; ++row) {
                        if (multimodal_groups[absolute + row] >= 0) {
                            count = row;
                            break;
                        }
                    }
                }
            }
            const auto replacement_offset = offset * c.hidden_size;
            auto status = compute_logits(token_ids.subspan(offset, count),
                position_base + static_cast<std::uint32_t>(offset),
                replacements.empty() ? std::span<const float>{}
                    : replacements.subspan(replacement_offset,
                                           count * c.hidden_size),
                replacement_mask.empty() ? std::span<const std::uint8_t>{}
                    : replacement_mask.subspan(offset, count),
                multimodal_groups, logits);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return result;
            }
            offset += count;
        }
        const auto cached_tokens = position_base + token_ids.size();
        FutureEntropyEvaluator lookahead;
        if (active_sampling.future_entropy_candidates != 0U) {
            lookahead = [this, cached_tokens](
                std::span<const std::uint32_t> candidates, std::uint32_t top_n,
                std::span<double> entropy) {
                return future_entropy(candidates, top_n,
                    static_cast<std::uint32_t>(cached_tokens), entropy);
            };
        }
        last_sample = sample_logits(
            logits, active_sampling, SamplingHistory{sampled_counts, sampled_ids},
            sampler, lookahead);
        if (!last_sample.ok()) {
            result.errors = last_sample.errors;
            return result;
        }
        result.value = last_sample.token;
        ++sampled_counts[result.value];
        sampled_ids.push_back(result.value);
        return result;
    }

    std::vector<KvMark> mark_kv() const {
        std::vector<KvMark> marks(kv.size());
        for (std::uint32_t layer = 0U; layer < kv.size(); ++layer) {
            const auto& cache = kv[layer];
            auto& mark = marks[layer];
            mark.start = cache.start;
            mark.key_size = cache.keys.size();
            mark.value_size = cache.values.size();
            if (!gemma4_global_attention_layer(layer) &&
                cache.keys.size() == static_cast<std::size_t>(c.sliding_window) *
                    c.local_key_value_heads * c.local_head_dim) {
                const auto row = static_cast<std::size_t>(
                    c.local_key_value_heads) * c.local_head_dim;
                mark.first_key.assign(cache.keys.begin(), cache.keys.begin() +
                    static_cast<std::ptrdiff_t>(row));
                mark.first_value.assign(cache.values.begin(), cache.values.begin() +
                    static_cast<std::ptrdiff_t>(row));
            }
        }
        return marks;
    }

    void rewind_kv(const std::vector<KvMark>& marks) {
        for (std::size_t layer = 0U; layer < kv.size(); ++layer) {
            auto& cache = kv[layer];
            const auto& mark = marks[layer];
            if (cache.start != mark.start && !mark.first_key.empty()) {
                cache.keys.insert(cache.keys.begin(), mark.first_key.begin(),
                                  mark.first_key.end());
                cache.values.insert(cache.values.begin(), mark.first_value.begin(),
                                    mark.first_value.end());
            }
            cache.keys.resize(mark.key_size);
            cache.values.resize(mark.value_size);
            cache.start = mark.start;
        }
    }

    ValidationResult future_entropy(
        std::span<const std::uint32_t> candidates, std::uint32_t top_n,
        std::uint32_t cached_tokens, std::span<double> entropy) {
        ValidationResult result;
        if (cached_tokens >= config.maximum_context_tokens ||
            candidates.size() != entropy.size()) {
            result.errors.emplace_back(
                "Gemma 4 future-entropy lookahead exceeds its cache contract");
            return result;
        }
        const auto marks = mark_kv();
        std::vector<float> logits;
        for (std::size_t index = 0U; index < candidates.size(); ++index) {
            const std::array<std::uint32_t, 1> token{candidates[index]};
            result = compute_logits(token, cached_tokens, {}, {}, {}, logits);
            rewind_kv(marks);
            if (!result.ok()) return result;
            entropy[index] = normalized_top_n_entropy(logits, top_n);
        }
        return result;
    }

    void reset_sequence() {
        reusable_sequence = false;
        device_kv_ready = false;
        cached_token_ids.clear();
        for (auto& cache : kv) {
            cache.keys.clear();
            cache.values.clear();
            cache.start = 0U;
        }
    }
};

Gemma4Runtime::Gemma4Runtime() : impl_(std::make_unique<Impl>()) {}
Gemma4Runtime::~Gemma4Runtime() = default;
Gemma4Runtime::Gemma4Runtime(Gemma4Runtime&&) noexcept = default;
Gemma4Runtime& Gemma4Runtime::operator=(Gemma4Runtime&&) noexcept = default;

ValidationResult Gemma4Runtime::initialize(
    const std::string& model_directory, const Gemma4RuntimeConfig& config) {
    ValidationResult result;
    if (impl_->initialized) {
        result.errors.emplace_back("Gemma 4 runtime is already initialized");
        return result;
    }
    result = validate_common_runtime_config(
        config.devices, config.vram_cache_fraction,
        config.sampling_temperature, "Gemma 4");
    if (!result.ok()) return result;
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > c.maximum_context_tokens) {
        result.errors.emplace_back(
            "Gemma 4 runtime context exceeds the declared model ceiling");
        return result;
    }
    impl_ = std::make_unique<Impl>();
    auto checkpoint = Gemma4CheckpointReader::open(model_directory);
    if (!checkpoint.ok()) {
        result.errors = std::move(checkpoint.errors);
        return result;
    }
    auto tokenizer = ModelTokenizer::load(
        (std::filesystem::path(model_directory) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }
    result = impl_->cuda.initialize(config.devices);
    if (!result.ok()) return result;
    if (config.enable_flash_attention) {
        for (const int device : config.devices) {
            result = impl_->cuda.validate_flash_attention_device(device);
            if (!result.ok()) return result;
        }
    }
    // A pre-solved plan already sized every layer, cache, and workspace against
    // this hardware, so it replaces the uniform per-device KV estimate below
    // with the exact per-device figure it committed to.
    const bool planned_placement = config.placement != nullptr &&
        config.placement->prescriptive &&
        config.placement->layer_device.size() == c.layer_count &&
        config.placement->device_budget_bytes.size() == config.devices.size() &&
        config.placement->request.devices == config.devices &&
        config.placement->request.maximum_context_tokens ==
            config.maximum_context_tokens;
    std::vector<std::size_t> layer_schedule(c.layer_count);
    std::vector<std::uint64_t> weight_capacities;
    if (planned_placement) {
        layer_schedule = config.placement->layer_device;
        for (std::size_t slot = 0U; slot < config.devices.size(); ++slot) {
            const auto budget = config.placement->device_budget_bytes[slot];
            const auto reserved = kMinimumAttentionWorkspace +
                placement_component_bytes(*config.placement,
                                          PlacementClass::KvCache, slot);
            if (budget <= reserved) {
                result.errors.emplace_back(
                    "Gemma 4 placement plan leaves no weight capacity on CUDA device " +
                    std::to_string(config.devices[slot]));
                return result;
            }
            weight_capacities.push_back(budget - reserved);
        }
    } else {
        const auto local_rows = std::min(
            config.maximum_context_tokens, c.sliding_window);
        const auto local_kv_bytes =
            50ULL * local_rows * c.local_key_value_heads * c.local_head_dim *
            2U * sizeof(std::uint16_t);
        const auto global_kv_bytes =
            10ULL * config.maximum_context_tokens * c.global_key_value_heads *
            c.global_head_dim * 2U * sizeof(std::uint16_t);
        const auto kv_bytes_per_device =
            (local_kv_bytes + global_kv_bytes + config.devices.size() - 1U) /
            config.devices.size();
        const auto attention_reserve =
            kMinimumAttentionWorkspace + kv_bytes_per_device;
        auto plan = plan_runtime_devices(
            config.devices, config.vram_cache_fraction, attention_reserve,
            2ULL << 30U, "Gemma 4");
        if (!plan.ok()) {
            result.errors = std::move(plan.errors);
            return result;
        }
        for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
            layer_schedule[layer] = plan.value.weighted_schedule[
                static_cast<std::size_t>(layer) *
                plan.value.weighted_schedule.size() / c.layer_count];
        }
        weight_capacities = std::move(plan.value.weight_capacities);
    }
    std::vector<std::uint64_t> planned(config.devices.size());
    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        const auto device = layer_schedule[layer];
        const bool global = gemma4_global_attention_layer(layer);
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "self_attn.";
        for (const auto& base : {
                 attention + "q_proj", attention + "k_proj",
                 attention + "o_proj", prefix + "mlp.gate_proj",
                 prefix + "mlp.up_proj", prefix + "mlp.down_proj"}) {
            const auto bytes = linear_bytes(*checkpoint.value, base);
            if (bytes == std::numeric_limits<std::uint64_t>::max() ||
                bytes > std::numeric_limits<std::uint64_t>::max() - planned[device]) {
                result.errors.emplace_back("Gemma 4 weight placement size overflows");
                return result;
            }
            planned[device] += bytes;
        }
        if (!global) planned[device] += linear_bytes(
            *checkpoint.value, attention + "v_proj");
    }
    const auto output_device = layer_schedule.back();
    planned[output_device] += linear_bytes(
        *checkpoint.value, "model.language_model.embed_tokens");
    const auto add_vision = [&](std::string_view base) {
        planned[0] += linear_bytes(*checkpoint.value, base);
    };
    add_vision("model.vision_tower.patch_embedder.input_proj");
    add_vision("model.embed_vision.embedding_projection");
    for (std::uint32_t layer = 0U; layer < c.vision_layer_count; ++layer) {
        const auto prefix = "model.vision_tower.encoder.layers." +
            std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        for (const auto& base : {
                 attention + "q_proj.linear", attention + "k_proj.linear",
                 attention + "v_proj.linear", attention + "o_proj.linear",
                 prefix + "mlp.gate_proj.linear",
                 prefix + "mlp.up_proj.linear",
                 prefix + "mlp.down_proj.linear"}) {
            add_vision(base);
        }
    }
    for (std::size_t device = 0U; device < planned.size(); ++device) {
        if (planned[device] > weight_capacities[device]) {
            result.errors.emplace_back(
                "Gemma 4 resident weights need " + format_bytes(planned[device]) +
                " on CUDA device " + std::to_string(config.devices[device]) +
                " but only " + format_bytes(weight_capacities[device]) +
                " is admitted after the KV cache and attention workspace");
            return result;
        }
    }
    impl_->config = config;
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->devices = config.devices;
    impl_->schedule = std::move(layer_schedule);
    impl_->sampler.seed(config.sampling_seed);
    try {
        for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
            const bool global = gemma4_global_attention_layer(layer);
            const auto rows = global ? config.maximum_context_tokens
                                     : std::min(config.maximum_context_tokens,
                                                c.sliding_window);
            const auto head_dim = global ? c.global_head_dim : c.local_head_dim;
            const auto heads = global ? c.global_key_value_heads
                                      : c.local_key_value_heads;
            impl_->kv[layer].keys.reserve(
                static_cast<std::size_t>(rows) * heads * head_dim);
            impl_->kv[layer].values.reserve(
                static_cast<std::size_t>(rows) * heads * head_dim);
        }
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back("cannot reserve the configured Gemma 4 BF16 KV cache");
        return result;
    } catch (const std::length_error&) {
        result.errors.emplace_back("configured Gemma 4 BF16 KV cache is too large");
        return result;
    }
    result = impl_->load_text_weights();
    if (!result.ok()) return result;
    result = impl_->load_vision_weights();
    if (!result.ok()) return result;
    impl_->initialized = true;
    return result;
}

Gemma4GenerationResult Gemma4Runtime::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    const std::array messages{ChatMessage{ChatRole::User, std::string(prompt)}};
    SamplingOptions sampling;
    sampling.temperature = impl_->config.sampling_temperature;
    sampling.seed = impl_->config.sampling_seed;
    return generate_chat_stream(messages, maximum_new_tokens, sampling, {}, on_token);
}

Gemma4GenerationResult Gemma4Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages,
    std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling,
    std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    Gemma4GenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("Gemma 4 runtime is not initialized");
        return result;
    }
    if (maximum_new_tokens == 0U) {
        result.errors.emplace_back("maximum_new_tokens must be positive");
        return result;
    }
    std::string error;
    if (!validate_sampling_options(sampling, error)) {
        result.errors.emplace_back("invalid sampling option: " + error);
        return result;
    }
    if (!validate_chat_messages(messages, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    impl_->active_sampling = sampling;
    impl_->sampled_counts.assign(c.vocabulary_size, 0U);
    impl_->sampled_ids.clear();
    impl_->sampler.seed(sampling.seed);
    std::vector<ChatMessage> active(messages.begin(), messages.end());
    ParseResult<std::vector<std::uint32_t>> encoded;
    std::vector<std::vector<float>> image_features;
    for (;;) {
        std::vector<ChatMessage> rendered = active;
        image_features.clear();
        for (auto& message : rendered) {
            if (message.parts.empty()) continue;
            message.content.clear();
            for (const auto& part : message.parts) {
                if (part.kind == ChatContentKind::Text) {
                    message.content += part.data;
                    continue;
                }
                auto prepared = prepare_gemma4_image(part.data, part.mime_type);
                if (!prepared.ok()) {
                    result.errors = std::move(prepared.errors);
                    return result;
                }
                auto features = impl_->encode_image(prepared.value);
                if (!features.ok()) {
                    result.errors = std::move(features.errors);
                    return result;
                }
                message.content += "<|image>";
                const std::string placeholder = "<|image|>";
                std::string expanded;
                expanded.reserve(static_cast<std::size_t>(
                    prepared.value.soft_tokens) * placeholder.size());
                for (std::uint32_t token = 0U;
                     token < prepared.value.soft_tokens; ++token) {
                    expanded += placeholder;
                }
                message.content += expanded;
                message.content += "<image|>";
                image_features.push_back(std::move(features.value));
            }
        }
        encoded = impl_->tokenizer.encode(render_gemma4_chat_prompt(rendered));
        if (!encoded.ok()) {
            result.errors = std::move(encoded.errors);
            return result;
        }
        if (encoded.value.size() + maximum_new_tokens <=
            impl_->config.maximum_context_tokens) break;
        if (!trim_oldest_chat_turn(active)) {
            result.errors.emplace_back(
                "prompt and requested generation exceed the context ceiling");
            return result;
        }
    }
    result.prompt_token_ids = std::move(encoded.value);
    std::vector<float> replacements;
    std::vector<std::uint8_t> replacement_mask;
    std::vector<std::int32_t> multimodal_groups;
    if (!image_features.empty()) {
        replacements.resize(result.prompt_token_ids.size() * c.hidden_size);
        replacement_mask.resize(result.prompt_token_ids.size());
        multimodal_groups.assign(result.prompt_token_ids.size(), -1);
        std::size_t image = 0U;
        std::size_t feature_row = 0U;
        for (std::size_t token = 0U; token < result.prompt_token_ids.size(); ++token) {
            if (result.prompt_token_ids[token] != 258'880U) continue;
            while (image < image_features.size() &&
                   feature_row * c.hidden_size == image_features[image].size()) {
                ++image;
                feature_row = 0U;
            }
            if (image >= image_features.size()) {
                result.errors.emplace_back(
                    "Gemma 4 image placeholders exceed encoded vision features");
                return result;
            }
            std::copy(
                image_features[image].begin() +
                    static_cast<std::ptrdiff_t>(feature_row * c.hidden_size),
                image_features[image].begin() +
                    static_cast<std::ptrdiff_t>((feature_row + 1U) * c.hidden_size),
                replacements.begin() +
                    static_cast<std::ptrdiff_t>(token * c.hidden_size));
            replacement_mask[token] = 1U;
            multimodal_groups[token] = static_cast<std::int32_t>(image);
            ++feature_row;
        }
        while (image < image_features.size() &&
               feature_row * c.hidden_size == image_features[image].size()) {
            ++image;
            feature_row = 0U;
        }
        if (image != image_features.size()) {
            result.errors.emplace_back(
                "Gemma 4 vision features exceed image placeholders");
            return result;
        }
    }
    const auto reused = impl_->config.enable_incremental_kv_continuation &&
        impl_->reusable_sequence
        ? incremental_kv_prefix_tokens(impl_->cached_token_ids,
                                       result.prompt_token_ids)
        : 0U;
    impl_->reusable_sequence = false;
    if (reused == 0U) impl_->reset_sequence();
    const auto prefill_started = std::chrono::steady_clock::now();
    const auto prefill = std::span<const std::uint32_t>(
        result.prompt_token_ids).subspan(reused);
    impl_->device_kv_ready = false;
    auto next = impl_->forward(
        prefill, static_cast<std::uint32_t>(reused),
        replacements.empty() ? std::span<const float>{}
            : std::span<const float>(replacements).subspan(
                  reused * c.hidden_size),
        replacement_mask.empty() ? std::span<const std::uint8_t>{}
            : std::span<const std::uint8_t>(replacement_mask).subspan(reused),
        multimodal_groups);
    if (next.ok()) {
        auto synced = impl_->sync_device_kv();
        if (!synced.ok()) next.errors = std::move(synced.errors);
    }
    result.metrics.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_started).count();
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    result.metrics.prefill_tokens = prefill.size();
    result.metrics.reused_prompt_tokens = reused;
    result.metrics.incremental_kv_continuation = reused != 0U;
    if (!next.ok()) {
        result.errors = std::move(next.errors);
        return result;
    }
    impl_->cached_token_ids = result.prompt_token_ids;
    constexpr std::array<std::uint32_t, 3> stop_ids{1U, 106U, 101U};
    const auto is_stop = [&stop_ids](std::uint32_t token) {
        return std::find(stop_ids.begin(), stop_ids.end(), token) != stop_ids.end();
    };
    StopSequenceBuffer output(stop);
    if (!is_stop(next.value)) {
        result.generated_token_ids.push_back(next.value);
        result.logprobs.push_back(impl_->last_sample);
        auto piece = impl_->tokenizer.decode_token(next.value);
        if (!piece.ok()) {
            result.errors = std::move(piece.errors);
            return result;
        }
        output.append(next.value, piece.value, on_token);
    }
    auto position = static_cast<std::uint32_t>(result.prompt_token_ids.size());
    const auto decode_started = std::chrono::steady_clock::now();
    while (!is_stop(next.value) && !output.stopped() && !output.cancelled() &&
           result.generated_token_ids.size() < maximum_new_tokens) {
        const std::array<std::uint32_t, 1> token{next.value};
        next = impl_->forward(token, position++);
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
        impl_->cached_token_ids.push_back(token.front());
        ++result.metrics.decode_tokens;
        if (!is_stop(next.value)) {
            result.generated_token_ids.push_back(next.value);
            result.logprobs.push_back(impl_->last_sample);
            auto piece = impl_->tokenizer.decode_token(next.value);
            if (!piece.ok()) {
                result.errors = std::move(piece.errors);
                return result;
            }
            output.append(next.value, piece.value, on_token);
        }
    }
    output.finish(on_token);
    result.text = output.text();
    result.stopped = output.stopped();
    result.metrics.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_started).count();
    result.metrics.rss_bytes = process_resident_set_bytes();
    result.metrics.device_vram_used_bytes = device_vram_used_bytes(impl_->devices);
    impl_->reusable_sequence =
        impl_->config.enable_incremental_kv_continuation && result.ok();
    return result;
}

}  // namespace strata
