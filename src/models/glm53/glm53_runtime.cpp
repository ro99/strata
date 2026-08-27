#include "strata/models/glm53/glm53_runtime.hpp"

#include "strata/engine/runtime_support.hpp"
#include "strata/models/common/tokenizer.hpp"
#include "strata/models/deepseek/deepseek_ops.hpp"
#include "strata/models/glm53/glm53_checkpoint.hpp"
#include "strata/models/kimi_k3/kimi_k3_ops.hpp"
#include "strata/platform/numerics.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <utility>

namespace strata {
namespace {

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kLayers = 45U;
constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kLinearHead = 128U;
constexpr std::uint32_t kLinearWidth = kHeads * kLinearHead;
constexpr std::uint32_t kMlaHead = 256U;
constexpr std::uint32_t kMlaWidth = kHeads * kMlaHead;
constexpr std::uint32_t kQueryRank = 1536U;
constexpr std::uint32_t kKvRank = 512U;
constexpr std::uint32_t kMhc = 4U;
constexpr std::uint32_t kVocabulary = 154880U;
constexpr std::uint32_t kExactSparseContext = 2048U;

double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

float sigmoid(float value) noexcept {
    return value >= 0.0F ? 1.0F / (1.0F + std::exp(-value))
                         : std::exp(value) / (1.0F + std::exp(value));
}

void round_bf16(std::span<float> values) noexcept {
    for (auto& value : values) value = bf16_round_f32(value);
}

void append(std::vector<std::string>& destination,
            std::vector<std::string> source) {
    for (auto& error : source) destination.push_back(std::move(error));
}

}  // namespace

struct Glm53Runtime::Impl {
    Glm53RuntimeConfig config;
    std::unique_ptr<Glm53CheckpointReader> checkpoint;
    ModelTokenizer tokenizer;
    CudaBackend cuda;
    std::vector<int> devices;
    std::array<std::vector<float>, kLayers> recurrent;
    std::array<std::array<std::vector<float>, 3U>, kLayers> convolution;
    std::array<std::vector<float>, kLayers> latents;
    bool ready{};

    [[nodiscard]] int device_for(std::uint32_t layer) const noexcept {
        return devices[layer % devices.size()];
    }

    [[nodiscard]] ValidationResult reset_sequence() {
        ValidationResult result;
        try {
            for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                if (glm53_kda_layer(layer)) {
                    recurrent[layer].assign(
                        static_cast<std::size_t>(kHeads) * kLinearHead *
                            kLinearHead,
                        0.0F);
                    for (auto& history : convolution[layer]) {
                        history.assign(static_cast<std::size_t>(kLinearWidth) * 3U,
                                       0.0F);
                    }
                } else {
                    latents[layer].assign(
                        static_cast<std::size_t>(config.maximum_context_tokens) *
                            kKvRank,
                        0.0F);
                }
            }
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "GLM-5.3 could not allocate recurrent and latent text state");
        }
        return result;
    }

    [[nodiscard]] ValidationResult linear(
        std::string_view base, std::span<const float> input,
        std::uint32_t rows, std::uint32_t columns,
        std::span<float> output, std::uint32_t layer,
        bool bf16_output = true) {
        ValidationResult result;
        if (input.size() != static_cast<std::size_t>(columns) * rows ||
            output.empty()) {
            result.errors.push_back("GLM-5.3 linear activation shape is invalid for " +
                                    std::string(base));
            return result;
        }
        const auto output_columns = output.size() / rows;
        if (output_columns * rows != output.size()) {
            result.errors.push_back("GLM-5.3 linear output shape is invalid for " +
                                    std::string(base));
            return result;
        }
        CudaWeight weight;
        result = checkpoint->load_cuda_linear(
            base, output_columns, columns, device_for(layer), cuda, weight);
        if (!result.ok()) return result;
        // PyTorch returns every published BF16/FP8 linear at the model's BF16
        // activation dtype. Keep that boundary even though the host-facing
        // CUDA API transports activations as float.
        return cuda.matmul(weight, input, rows, output, bf16_output);
    }

    [[nodiscard]] ValidationResult norm(
        std::span<float> output, std::span<const float> input,
        std::string_view weight_name) const {
        auto weight = checkpoint->read_f32(weight_name, input.size());
        if (!weight.ok()) return {std::move(weight.errors)};
        auto result = kimi_rms_norm(output, input, weight.value, 1.0e-5F);
        if (result.ok()) round_bf16(output);
        return result;
    }

    [[nodiscard]] ValidationResult mhc_pre(
        std::span<float> collapsed, Dsv4MhcMix& mix,
        std::span<const float> streams, const std::string& prefix) const {
        ValidationResult result;
        auto projection = checkpoint->read_f32(prefix + "_fn", 24U * 16384U);
        auto base = checkpoint->read_f32(prefix + "_base", 24U);
        auto scale = checkpoint->read_f32(prefix + "_scale", 3U);
        if (!projection.ok() || !base.ok() || !scale.ok()) {
            append(result.errors, std::move(projection.errors));
            append(result.errors, std::move(base.errors));
            append(result.errors, std::move(scale.errors));
            return result;
        }
        double square_sum = 0.0;
        for (const auto value : streams) square_sum += static_cast<double>(value) * value;
        const auto reciprocal = 1.0F / std::sqrt(
            static_cast<float>(square_sum /
                               static_cast<double>(streams.size())) + 1.0e-5F);
        std::vector<float> projected(24U, 0.0F);
        for (std::size_t row = 0U; row < projected.size(); ++row) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < streams.size(); ++column) {
                sum += static_cast<double>(projection.value[row * streams.size() + column]) *
                       streams[column];
            }
            projected[row] = static_cast<float>(sum) * reciprocal;
        }
        auto split = dsv4_mhc_split_sinkhorn_f32(
            projected, scale.value, base.value, kMhc, 20U, 1.0e-6F);
        if (!split.ok()) return {std::move(split.errors)};
        mix = std::move(split.value);
        round_bf16(mix.post);
        round_bf16(mix.combination);
        std::fill(collapsed.begin(), collapsed.end(), 0.0F);
        for (std::size_t stream = 0U; stream < kMhc; ++stream) {
            for (std::size_t column = 0U; column < kHidden; ++column) {
                collapsed[column] += mix.pre[stream] *
                    streams[stream * kHidden + column];
            }
        }
        round_bf16(collapsed);
        return result;
    }

    [[nodiscard]] ValidationResult attention_kda(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, const std::string& attention) {
        ValidationResult result;
        std::vector<float> query(kLinearWidth), key(kLinearWidth), value(kLinearWidth);
        result = linear(attention + "q_proj", input, 1U, kHidden, query, layer);
        if (!result.ok()) return result;
        result = linear(attention + "k_proj", input, 1U, kHidden, key, layer);
        if (!result.ok()) return result;
        result = linear(attention + "v_proj", input, 1U, kHidden, value, layer);
        if (!result.ok()) return result;
        for (std::uint32_t projection = 0U; projection < 3U; ++projection) {
            auto taps = checkpoint->read_f32(
                attention + (projection == 0U ? "q_conv1d.weight"
                              : projection == 1U ? "k_conv1d.weight"
                                                 : "v_conv1d.weight"),
                static_cast<std::uint64_t>(kLinearWidth) * 4U);
            if (!taps.ok()) return {std::move(taps.errors)};
            auto& values = projection == 0U ? query : projection == 1U ? key : value;
            auto convolved = values;
            result = kimi_short_conv_step(convolved, values, taps.value,
                                          convolution[layer][projection], 4U);
            if (!result.ok()) return result;
            values = std::move(convolved);
            round_bf16(values);
        }
        std::vector<float> low(kLinearHead), forget(kLinearWidth), beta(kHeads);
        result = linear(attention + "f_a_proj", input, 1U, kHidden, low, layer);
        if (!result.ok()) return result;
        result = linear(attention + "f_b_proj", low, 1U, kLinearHead, forget, layer);
        if (!result.ok()) return result;
        result = linear(attention + "b_proj", input, 1U, kHidden, beta, layer);
        if (!result.ok()) return result;
        for (auto& element : beta) {
            element = bf16_round_f32(sigmoid(element));
        }
        auto a_log = checkpoint->read_f32(attention + "A_log", kHeads);
        auto dt_bias = checkpoint->read_f32(attention + "dt_bias", kLinearWidth);
        auto o_norm = checkpoint->read_f32(attention + "o_norm.weight", kLinearHead);
        if (!a_log.ok() || !dt_bias.ok() || !o_norm.ok()) {
            append(result.errors, std::move(a_log.errors));
            append(result.errors, std::move(dt_bias.errors));
            append(result.errors, std::move(o_norm.errors));
            return result;
        }
        std::vector<float> gate_low(kLinearHead), gate(kLinearWidth);
        result = linear(attention + "g_a_proj", input, 1U, kHidden, gate_low, layer);
        if (!result.ok()) return result;
        result = linear(attention + "g_b_proj", gate_low, 1U, kLinearHead, gate, layer);
        if (!result.ok()) return result;
        std::vector<float> heads_out(kLinearWidth);
        const auto query_scale = 1.0F / std::sqrt(static_cast<float>(kLinearHead));
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            const auto begin = static_cast<std::size_t>(head) * kLinearHead;
            auto q = std::span<float>(query).subspan(begin, kLinearHead);
            auto k = std::span<float>(key).subspan(begin, kLinearHead);
            result = kimi_l2_normalize(q, 1.0e-6F);
            if (!result.ok()) return result;
            result = kimi_l2_normalize(k, 1.0e-6F);
            if (!result.ok()) return result;
            for (auto& element : q) element *= query_scale;
            std::vector<float> decay(kLinearHead);
            result = kimi_kda_log_decay(
                decay, std::span<const float>(forget).subspan(begin, kLinearHead),
                std::span<const float>(dt_bias.value).subspan(begin, kLinearHead),
                a_log.value[head], -5.0F);
            if (!result.ok()) return result;
            for (auto& element : decay) element = std::exp(element);
            std::vector<float> raw(kLinearHead);
            auto state = std::span<float>(recurrent[layer]).subspan(
                static_cast<std::size_t>(head) * kLinearHead * kLinearHead,
                static_cast<std::size_t>(kLinearHead) * kLinearHead);
            result = kimi_kda_step(
                raw, state, q, k,
                std::span<const float>(value).subspan(begin, kLinearHead),
                decay, beta[head], kLinearHead, kLinearHead);
            if (!result.ok()) return result;
            round_bf16(raw);
            result = kimi_kda_output_norm(
                std::span<float>(heads_out).subspan(begin, kLinearHead), raw,
                std::span<const float>(gate).subspan(begin, kLinearHead),
                o_norm.value, 1.0e-5F);
            if (!result.ok()) return result;
            round_bf16(std::span<float>(heads_out).subspan(begin, kLinearHead));
        }
        return linear(attention + "o_proj", heads_out, 1U, kLinearWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult attention_mla(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, std::uint32_t position,
        const std::string& attention) {
        ValidationResult result;
        std::vector<float> q_rank(kQueryRank), query(kMlaWidth), latent(kKvRank);
        result = linear(attention + "q_a_proj", input, 1U, kHidden, q_rank, layer);
        if (!result.ok()) return result;
        result = norm(q_rank, q_rank, attention + "q_a_layernorm.weight");
        if (!result.ok()) return result;
        result = linear(attention + "q_b_proj", q_rank, 1U, kQueryRank,
                        query, layer);
        if (!result.ok()) return result;
        result = linear(attention + "kv_a_proj_with_mqa", input, 1U, kHidden,
                        latent, layer);
        if (!result.ok()) return result;
        result = norm(latent, latent, attention + "kv_a_layernorm.weight");
        if (!result.ok()) return result;
        std::copy(latent.begin(), latent.end(),
                  latents[layer].begin() + static_cast<std::ptrdiff_t>(
                      static_cast<std::size_t>(position) * kKvRank));
        const auto history = position + 1U;
        std::vector<float> expanded(
            static_cast<std::size_t>(history) * kHeads * 2U * kMlaHead);
        result = linear(
            attention + "kv_b_proj",
            std::span<const float>(latents[layer]).first(
                static_cast<std::size_t>(history) * kKvRank),
            history, kKvRank, expanded, layer);
        if (!result.ok()) return result;
        std::vector<float> attended(kMlaWidth, 0.0F);
        const auto score_scale = 1.0F / std::sqrt(static_cast<float>(kMlaHead));
        std::vector<float> scores(history);
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            const auto* q = query.data() + static_cast<std::size_t>(head) * kMlaHead;
            float highest = -std::numeric_limits<float>::infinity();
            for (std::uint32_t token = 0U; token < history; ++token) {
                const auto* kv = expanded.data() +
                    (static_cast<std::size_t>(token) * kHeads + head) *
                        (2U * kMlaHead);
                float score = 0.0F;
                for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                    score += q[column] * kv[column];
                }
                scores[token] = score * score_scale;
                highest = std::max(highest, scores[token]);
            }
            float total = 0.0F;
            for (auto& score : scores) {
                score = std::exp(score - highest);
                total += score;
            }
            auto* destination = attended.data() +
                                static_cast<std::size_t>(head) * kMlaHead;
            for (std::uint32_t token = 0U; token < history; ++token) {
                const auto* values = expanded.data() +
                    (static_cast<std::size_t>(token) * kHeads + head) *
                        (2U * kMlaHead) + kMlaHead;
                const auto coefficient = bf16_round_f32(scores[token] / total);
                for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                    destination[column] += coefficient * values[column];
                }
            }
        }
        round_bf16(attended);
        return linear(attention + "o_proj", attended, 1U, kMlaWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult swiglu_block(
        std::span<float> output, std::span<const float> input,
        const std::string& prefix, std::uint32_t inner,
        std::uint32_t layer) {
        ValidationResult result;
        std::vector<float> gate(inner), up(inner), activated(inner);
        result = linear(prefix + "gate_proj", input, 1U, kHidden, gate, layer);
        if (!result.ok()) return result;
        result = linear(prefix + "up_proj", input, 1U, kHidden, up, layer);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < inner; ++index) {
            const auto g = std::min(gate[index], 10.0F);
            const auto u = std::clamp(up[index], -10.0F, 10.0F);
            activated[index] = g * sigmoid(g) * u;
        }
        round_bf16(activated);
        return linear(prefix + "down_proj", activated, 1U, inner,
                      output, layer);
    }

    [[nodiscard]] ValidationResult feedforward(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, const std::string& prefix) {
        if (!glm53_moe_layer(layer)) {
            return swiglu_block(output, input, prefix + "mlp.", 12288U, layer);
        }
        ValidationResult result;
        std::vector<float> logits(288U);
        // The reference router explicitly promotes both operands to F32.
        result = linear(prefix + "mlp.gate", input, 1U, kHidden, logits,
                        layer, false);
        if (!result.ok()) return result;
        auto bias = checkpoint->read_f32(
            prefix + "mlp.gate.e_score_correction_bias", 288U);
        if (!bias.ok()) return {std::move(bias.errors)};
        std::array<KimiRoutedExpert, 8U> selected{};
        result = kimi_route_topk(selected, logits, bias.value, 2.5F);
        if (!result.ok()) return result;
        std::fill(output.begin(), output.end(), 0.0F);
        std::vector<float> branch(kHidden);
        result = swiglu_block(branch, input,
                              prefix + "mlp.shared_experts.", 2048U, layer);
        if (!result.ok()) return result;
        std::copy(branch.begin(), branch.end(), output.begin());
        for (const auto& route : selected) {
            result = swiglu_block(
                branch, input,
                prefix + "mlp.experts." + std::to_string(route.expert) + ".",
                2048U, layer);
            if (!result.ok()) return result;
            for (std::size_t column = 0U; column < kHidden; ++column) {
                output[column] = bf16_round_f32(
                    output[column] + bf16_round_f32(
                        route.weight * branch[column]));
            }
        }
        return result;
    }

    [[nodiscard]] ValidationResult forward_token(
        std::uint32_t token, std::uint32_t position,
        std::span<float> logits) {
        ValidationResult result;
        auto embedding = checkpoint->read_f32_row(
            "model.language_model.embed_tokens.weight", token);
        if (!embedding.ok()) return {std::move(embedding.errors)};
        std::vector<float> streams(static_cast<std::size_t>(kMhc) * kHidden);
        for (std::uint32_t stream = 0U; stream < kMhc; ++stream) {
            std::copy(embedding.value.begin(), embedding.value.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(stream * kHidden));
        }
        std::vector<float> collapsed(kHidden), normalized(kHidden), branch(kHidden);
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto prefix = "model.language_model.layers." +
                                std::to_string(layer) + ".";
            Dsv4MhcMix mix;
            result = mhc_pre(collapsed, mix, streams, prefix + "hc_attn");
            if (!result.ok()) return result;
            result = norm(normalized, collapsed, prefix + "input_layernorm.weight");
            if (!result.ok()) return result;
            const auto attention = prefix + "self_attn.";
            result = glm53_kda_layer(layer)
                ? attention_kda(branch, normalized, layer, attention)
                : attention_mla(branch, normalized, layer, position, attention);
            if (!result.ok()) return result;
            std::vector<float> transitioned(streams.size());
            result = dsv4_mhc_post_f32(transitioned, branch, streams, mix, kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            streams = std::move(transitioned);

            result = mhc_pre(collapsed, mix, streams, prefix + "hc_ffn");
            if (!result.ok()) return result;
            result = norm(normalized, collapsed,
                          prefix + "post_attention_layernorm.weight");
            if (!result.ok()) return result;
            result = feedforward(branch, normalized, layer, prefix);
            if (!result.ok()) return result;
            transitioned.assign(streams.size(), 0.0F);
            result = dsv4_mhc_post_f32(transitioned, branch, streams, mix, kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            streams = std::move(transitioned);
            if (config.load_progress) {
                std::cerr << "\r[glm53] layer " << (layer + 1U) << '/' << kLayers
                          << std::flush;
            }
        }
        if (config.load_progress) std::cerr << '\r' << std::string(32U, ' ') << '\r';
        for (std::size_t column = 0U; column < kHidden; ++column) {
            collapsed[column] = 0.25F *
                (streams[column] + streams[kHidden + column] +
                 streams[2U * kHidden + column] + streams[3U * kHidden + column]);
        }
        round_bf16(collapsed);
        result = norm(normalized, collapsed, "model.language_model.norm.weight");
        if (!result.ok() || logits.empty()) return result;
        return linear("lm_head", normalized, 1U, kHidden, logits, kLayers - 1U);
    }
};

Glm53Runtime::Glm53Runtime() : impl_(std::make_unique<Impl>()) {}
Glm53Runtime::~Glm53Runtime() = default;
Glm53Runtime::Glm53Runtime(Glm53Runtime&&) noexcept = default;
Glm53Runtime& Glm53Runtime::operator=(Glm53Runtime&&) noexcept = default;

ValidationResult Glm53Runtime::initialize(
    const std::string& model_directory, const Glm53RuntimeConfig& config) {
    ValidationResult result;
    if (impl_->ready) {
        result.errors.emplace_back("GLM-5.3 runtime is already initialized");
        return result;
    }
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > kExactSparseContext) {
        result.errors.push_back(
            "GLM-5.3 text context must be within [1, 2048]; above 2048 the "
            "checkpoint's exact k-pool sparse indexer is required");
        return result;
    }
    impl_->config = config;
    impl_->devices = resolve_runtime_devices(config.devices);
    result = validate_common_runtime_config(
        impl_->devices, config.vram_cache_fraction,
        config.sampling_temperature, "GLM-5.3");
    if (!result.ok()) return result;
    auto tokenizer = ModelTokenizer::load(model_directory + "/tokenizer.json");
    if (!tokenizer.ok()) return {std::move(tokenizer.errors)};
    // The tokenizer has 154,820 base pieces plus 36 added special tokens.
    // The checkpoint pads its embedding and output matrices to 154,880 rows;
    // those 24 padding rows are deliberately not tokenizable.
    if (tokenizer.value.vocabulary_size() != 154856U) {
        result.errors.emplace_back(
            "GLM-5.3 tokenizer must expose 154856 usable token ids");
        return result;
    }
    auto checkpoint = Glm53CheckpointReader::open(model_directory);
    if (!checkpoint.ok()) return {std::move(checkpoint.errors)};
    result = impl_->cuda.initialize(impl_->devices);
    if (!result.ok()) return result;
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->ready = true;
    return result;
}

Glm53GenerationResult Glm53Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    Glm53GenerationResult result;
    if (!impl_->ready) {
        result.errors.emplace_back("GLM-5.3 runtime is not initialized");
        return result;
    }
    std::string error;
    if (!validate_sampling_options(sampling, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    if (!validate_chat_messages(messages, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    for (const auto& message : messages) {
        for (const auto& part : message.parts) {
            if (part.kind != ChatContentKind::Text) {
                result.errors.emplace_back(
                    "GLM-5.3 vision is not implemented; this runtime supports text-only messages");
                return result;
            }
        }
    }
    auto encoded = impl_->tokenizer.encode(
        render_glm53_chat_prompt(messages, "max", true));
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    if (encoded.value.empty() || encoded.value.size() + maximum_new_tokens >
            impl_->config.maximum_context_tokens) {
        result.errors.emplace_back(
            "GLM-5.3 prompt and requested generation exceed the admitted text context");
        return result;
    }
    auto reset = impl_->reset_sequence();
    if (!reset.ok()) {
        result.errors = std::move(reset.errors);
        return result;
    }
    result.prompt_token_ids = encoded.value;
    result.metrics.prompt_tokens = encoded.value.size();
    std::vector<float> logits(kVocabulary);
    const auto prefill_started = now_seconds();
    for (std::uint32_t position = 0U; position < encoded.value.size(); ++position) {
        auto step = impl_->forward_token(encoded.value[position], position,
                                         position + 1U == encoded.value.size()
                                             ? std::span<float>(logits)
                                             : std::span<float>{});
        if (!step.ok()) {
            result.errors = std::move(step.errors);
            return result;
        }
        ++result.metrics.prefill_tokens;
    }
    result.metrics.prefill_seconds = now_seconds() - prefill_started;
    std::mt19937_64 generator(sampling.seed);
    std::vector<std::uint32_t> counts(kVocabulary, 0U);
    std::vector<std::uint32_t> sampled;
    StopSequenceBuffer streamed(stop);
    auto position = static_cast<std::uint32_t>(encoded.value.size());
    const auto decode_started = now_seconds();
    for (std::uint32_t index = 0U; index < maximum_new_tokens; ++index) {
        auto drawn = sample_logits(logits, sampling,
                                   SamplingHistory{counts, sampled}, generator);
        if (!drawn.ok()) {
            result.errors = std::move(drawn.errors);
            return result;
        }
        if (drawn.token == 154820U || drawn.token == 154827U ||
            drawn.token == 154829U) {
            result.stopped = true;
            break;
        }
        result.generated_token_ids.push_back(drawn.token);
        result.logprobs.push_back(drawn);
        sampled.push_back(drawn.token);
        ++counts[drawn.token];
        auto piece = impl_->tokenizer.decode_token(drawn.token);
        if (!piece.ok()) {
            result.errors = std::move(piece.errors);
            return result;
        }
        streamed.append(drawn.token, piece.value, on_token);
        if (streamed.stopped() || streamed.cancelled()) break;
        if (index + 1U == maximum_new_tokens) break;
        auto step = impl_->forward_token(drawn.token, position++, logits);
        if (!step.ok()) {
            result.errors = std::move(step.errors);
            return result;
        }
        ++result.metrics.decode_tokens;
    }
    result.metrics.decode_seconds = now_seconds() - decode_started;
    streamed.finish(on_token);
    result.text = streamed.text();
    result.stopped = result.stopped || streamed.stopped();
    return result;
}

}  // namespace strata
