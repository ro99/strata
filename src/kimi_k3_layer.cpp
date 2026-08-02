#include "strata/kimi_k3_layer.hpp"

#include "strata/compressed_tensors.hpp"
#include "strata/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace strata {
namespace {

constexpr auto& kContract = kKimiK3ExecutionContract;

// The MLA LoRA layer norms take `KimiRMSNorm`'s constructor default rather than
// `rms_norm_eps`. Every other norm in the model takes `rms_norm_eps` (1e-5).
// The two differ by an order of magnitude and the reference is explicit about
// it, so they are named separately here instead of sharing one constant.
constexpr float kLoraNormEpsilon = 1.0e-6F;
// The KDA kernels put this epsilon inside the square root of the q/k L2 norm.
constexpr float kL2Epsilon = 1.0e-6F;

[[nodiscard]] float decode_bf16(std::uint16_t value) noexcept {
    const auto bits = static_cast<std::uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

[[nodiscard]] float sigmoid(float value) noexcept {
    return 1.0F / (1.0F + std::exp(-value));
}

[[nodiscard]] bool matrix_fits(const KimiBf16Matrix& matrix,
                               std::uint32_t rows, std::uint32_t columns) {
    return matrix.valid() && matrix.rows == rows && matrix.columns == columns;
}

void run_rows(std::uint32_t rows, HostWorkerPool* pool,
              const std::function<void(std::size_t)>& body) {
    if (pool != nullptr && pool->size() > 1U && rows > 1U) {
        (void)pool->parallel_for(rows, body);
        return;
    }
    for (std::uint32_t row = 0U; row < rows; ++row) body(row);
}

}  // namespace

ValidationResult kimi_bf16_matmul(std::span<float> output,
                                  std::span<const float> input,
                                  const KimiBf16Matrix& weight,
                                  std::uint32_t tokens, HostWorkerPool* pool) {
    ValidationResult result;
    const auto rows = static_cast<std::size_t>(weight.rows);
    const auto columns = static_cast<std::size_t>(weight.columns);
    const auto count = static_cast<std::size_t>(tokens);
    if (tokens == 0U || !weight.valid() || rows == 0U || columns == 0U ||
        input.size() != count * columns || output.size() != count * rows) {
        result.errors.emplace_back(
            "Kimi-K3 matmul operands disagree with the weight shape");
        return result;
    }
    // Parallel over output rows so every worker reads the same input rows and
    // writes a disjoint output column. Weight rows are contiguous, so each
    // worker streams its slice of the weight exactly once per call regardless
    // of how many tokens the page carries: that is what makes a prefill page
    // cheaper per token than a decode step, and the reason to batch at all.
    run_rows(weight.rows, pool, [&](std::size_t row) {
        const auto* values = weight.values.data() + row * columns;
        for (std::size_t token = 0U; token < count; ++token) {
            const auto* source = input.data() + token * columns;
            float sum = 0.0F;
            for (std::size_t column = 0U; column < columns; ++column) {
                sum += source[column] * decode_bf16(values[column]);
            }
            output[token * rows + row] = sum;
        }
    });
    return result;
}

bool KimiExpertModuleView::valid() const noexcept {
    if (rows == 0U || columns == 0U || columns % 32U != 0U) return false;
    const auto count = static_cast<std::size_t>(rows);
    return packed.size() == count * (columns / 2U) &&
           scales.size() == count * (columns / 32U);
}

ValidationResult kimi_mxfp4_matvec(std::span<float> output,
                                   std::span<const float> input,
                                   const KimiExpertModuleView& module,
                                   HostWorkerPool* pool) {
    ValidationResult result;
    if (!module.valid() || input.size() != module.columns ||
        output.size() != module.rows) {
        result.errors.emplace_back(
            "MXFP4 matvec operands disagree with the module shape");
        return result;
    }
    const auto columns = static_cast<std::size_t>(module.columns);
    const auto packed_stride = columns / 2U;
    const auto scale_stride = columns / 32U;
    run_rows(module.rows, pool, [&](std::size_t row) {
        const auto* packed = module.packed.data() + row * packed_stride;
        const auto* scales = module.scales.data() + row * scale_stride;
        float sum = 0.0F;
        for (std::size_t group = 0U; group < scale_stride; ++group) {
            const auto scale = mxfp4_scale_from_e8m0(scales[group]);
            const auto* bytes = packed + group * 16U;
            const auto* source = input.data() + group * 32U;
            float partial = 0.0F;
            for (std::size_t index = 0U; index < 16U; ++index) {
                const auto byte = bytes[index];
                // Low nibble first, sign in bit 3, magnitude in bits 0-2.
                const auto low = static_cast<std::uint8_t>(byte & 0x0FU);
                const auto high = static_cast<std::uint8_t>(byte >> 4U);
                const auto low_value =
                    (low & 0x08U) != 0U ? -kMxfp4Magnitudes[low & 0x07U]
                                        : kMxfp4Magnitudes[low & 0x07U];
                const auto high_value =
                    (high & 0x08U) != 0U ? -kMxfp4Magnitudes[high & 0x07U]
                                         : kMxfp4Magnitudes[high & 0x07U];
                partial += source[index * 2U] * low_value +
                           source[index * 2U + 1U] * high_value;
            }
            // One multiply per group rather than per element: the scale is
            // shared across all 32 and factors out of the partial sum.
            sum += partial * scale;
        }
        output[row] = sum;
    });
    return result;
}

// ------------------------------------------------------- residual stream

ValidationResult KimiResidualStream::reset(std::uint32_t tokens,
                                           std::uint32_t hidden_size,
                                           std::uint32_t block_size) {
    ValidationResult result;
    if (tokens == 0U || hidden_size == 0U || block_size == 0U) {
        result.errors.emplace_back(
            "attention residual stream needs positive tokens, width, and block");
        return result;
    }
    tokens_ = tokens;
    hidden_size_ = hidden_size;
    block_size_ = block_size;
    completed_blocks_ = 0U;
    blocks_.clear();
    prefix_.assign(static_cast<std::size_t>(tokens) * hidden_size, 0.0F);
    return result;
}

ValidationResult KimiResidualStream::begin(std::span<const float> embeddings) {
    ValidationResult result;
    if (embeddings.size() != prefix_.size()) {
        result.errors.emplace_back(
            "attention residual stream seed disagrees with the page shape");
        return result;
    }
    std::copy(embeddings.begin(), embeddings.end(), prefix_.begin());
    completed_blocks_ = 0U;
    blocks_.clear();
    return result;
}

ValidationResult KimiResidualStream::open_block() {
    ValidationResult result;
    if (prefix_.empty()) {
        result.errors.emplace_back("attention residual stream is not initialized");
        return result;
    }
    blocks_.insert(blocks_.end(), prefix_.begin(), prefix_.end());
    ++completed_blocks_;
    // The reference drops the prefix entirely when it opens a block; the next
    // sublayer output becomes the prefix rather than being added to it. Zeroing
    // and adding is the same value and keeps one code path.
    std::fill(prefix_.begin(), prefix_.end(), 0.0F);
    return result;
}

ValidationResult KimiResidualStream::add(std::span<const float> delta) {
    ValidationResult result;
    if (delta.size() != prefix_.size()) {
        result.errors.emplace_back(
            "attention residual delta disagrees with the page shape");
        return result;
    }
    for (std::size_t index = 0U; index < prefix_.size(); ++index) {
        prefix_[index] += delta[index];
    }
    return result;
}

ValidationResult KimiResidualStream::mix(std::span<float> output,
                                         std::span<const float> query_weight,
                                         std::span<const float> norm_weight,
                                         float epsilon,
                                         HostWorkerPool* pool) const {
    ValidationResult result;
    if (output.size() != prefix_.size() || query_weight.size() != hidden_size_ ||
        norm_weight.size() != hidden_size_) {
        result.errors.emplace_back(
            "attention residual mix operands disagree with the page shape");
        return result;
    }
    if (completed_blocks_ == 0U) {
        // Nothing to select over. The reference skips the call outright at the
        // attention site of layer 0, which is the only place this happens.
        std::copy(prefix_.begin(), prefix_.end(), output.begin());
        return result;
    }
    const auto width = static_cast<std::size_t>(hidden_size_);
    const auto sources = static_cast<std::size_t>(completed_blocks_) + 1U;
    std::vector<ValidationResult> failures(tokens_);
    run_rows(tokens_, pool, [&](std::size_t token) {
        // The mix wants the sources contiguous; blocks are stored block-major
        // so a whole block is one memcpy across the page, which is the layout
        // `open_block` writes and every layer re-reads.
        std::vector<float> gathered(sources * width);
        for (std::uint32_t block = 0U; block < completed_blocks_; ++block) {
            const auto* source =
                blocks_.data() + (static_cast<std::size_t>(block) * tokens_ + token) * width;
            std::copy(source, source + width, gathered.begin() + block * width);
        }
        const auto* running = prefix_.data() + token * width;
        std::copy(running, running + width,
                  gathered.begin() + static_cast<std::size_t>(completed_blocks_) * width);
        failures[token] = kimi_attention_residual_mix(
            output.subspan(token * width, width), gathered, query_weight,
            norm_weight, hidden_size_, static_cast<std::uint32_t>(sources),
            epsilon);
    });
    for (auto& failure : failures) {
        if (!failure.ok()) return failure;
    }
    return result;
}

// -------------------------------------------------------------- layers

ValidationResult kimi_kda_layer(std::span<float> output,
                                std::span<const float> input,
                                const KimiKdaWeights& weights,
                                KimiStateCache& cache, std::uint32_t layer,
                                std::uint32_t tokens, KimiLayerScratch& scratch,
                                HostWorkerPool* pool) {
    ValidationResult result;
    const auto hidden = static_cast<std::size_t>(kContract.hidden_size);
    const auto heads = static_cast<std::size_t>(kContract.linear_attention_heads);
    const auto head_dim = static_cast<std::size_t>(kContract.linear_head_dim);
    const auto projection = heads * head_dim;
    const auto rows = static_cast<std::size_t>(tokens);
    if (tokens == 0U || input.size() != rows * hidden ||
        output.size() != rows * hidden) {
        result.errors.emplace_back("KDA layer operands disagree with the page");
        return result;
    }
    if (!matrix_fits(weights.q_proj, static_cast<std::uint32_t>(projection),
                     kContract.hidden_size) ||
        !matrix_fits(weights.k_proj, static_cast<std::uint32_t>(projection),
                     kContract.hidden_size) ||
        !matrix_fits(weights.v_proj, static_cast<std::uint32_t>(projection),
                     kContract.hidden_size) ||
        !matrix_fits(weights.g_proj, static_cast<std::uint32_t>(projection),
                     kContract.hidden_size) ||
        !matrix_fits(weights.o_proj, kContract.hidden_size,
                     static_cast<std::uint32_t>(projection)) ||
        !matrix_fits(weights.f_a_proj, kContract.linear_head_dim,
                     kContract.hidden_size) ||
        !matrix_fits(weights.f_b_proj, static_cast<std::uint32_t>(projection),
                     kContract.linear_head_dim) ||
        !matrix_fits(weights.b_proj, kContract.linear_attention_heads,
                     kContract.hidden_size)) {
        result.errors.emplace_back("KDA layer weights disagree with the contract");
        return result;
    }
    const auto kernel = static_cast<std::size_t>(kContract.short_conv_kernel);
    if (weights.a_log.size() != heads || weights.dt_bias.size() != projection ||
        weights.o_norm.size() != head_dim ||
        weights.q_conv.size() != projection * kernel ||
        weights.k_conv.size() != projection * kernel ||
        weights.v_conv.size() != projection * kernel) {
        result.errors.emplace_back(
            "KDA layer parameter vectors disagree with the contract");
        return result;
    }

    auto& query = scratch.projection_a;
    auto& key = scratch.projection_b;
    auto& value = scratch.projection_c;
    auto& gate = scratch.projection_d;
    query.assign(rows * projection, 0.0F);
    key.assign(rows * projection, 0.0F);
    value.assign(rows * projection, 0.0F);
    gate.assign(rows * projection, 0.0F);
    result = kimi_bf16_matmul(query, input, weights.q_proj, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_bf16_matmul(key, input, weights.k_proj, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_bf16_matmul(value, input, weights.v_proj, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_bf16_matmul(gate, input, weights.g_proj, tokens, pool);
    if (!result.ok()) return result;

    // Decay logits: a rank-`head_dim` factorization of the per-channel gate.
    std::vector<float> low(rows * head_dim, 0.0F);
    std::vector<float> logits(rows * projection, 0.0F);
    result = kimi_bf16_matmul(low, input, weights.f_a_proj, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_bf16_matmul(logits, low, weights.f_b_proj, tokens, pool);
    if (!result.ok()) return result;
    // One beta per (token, head): `b_proj` is [heads, hidden], so this is a
    // [tokens, heads] matrix and each head reads its own column below.
    std::vector<float> beta(rows * heads, 0.0F);
    result = kimi_bf16_matmul(beta, input, weights.b_proj, tokens, pool);
    if (!result.ok()) return result;
    // `use_beta_sigmoid_in_kernel` in the reference; the raw projection is a
    // logit and the delta rule needs the bounded value.
    for (auto& element : beta) element = sigmoid(element);

    // The short convolutions are causal and stateful, so they run token by
    // token in order and carry their history across pages through the cache.
    for (std::uint32_t projection_index = 0U; projection_index < 3U;
         ++projection_index) {
        auto& stream = projection_index == 0U ? query
                     : projection_index == 1U ? key : value;
        const auto taps = projection_index == 0U ? weights.q_conv
                        : projection_index == 1U ? weights.k_conv
                                                 : weights.v_conv;
        auto history = cache.convolution_history(layer, projection_index);
        if (history.size() != projection * (kernel - 1U)) {
            result.errors.emplace_back(
                "KDA convolution history disagrees with the contract");
            return result;
        }
        std::vector<float> row(projection);
        for (std::size_t token = 0U; token < rows; ++token) {
            auto slice = std::span<float>(stream).subspan(token * projection,
                                                          projection);
            std::copy(slice.begin(), slice.end(), row.begin());
            result = kimi_short_conv_step(slice, row, taps, history,
                                          kContract.short_conv_kernel);
            if (!result.ok()) return result;
        }
    }

    auto& heads_out = scratch.heads;
    heads_out.assign(rows * projection, 0.0F);
    std::vector<ValidationResult> failures(heads);
    run_rows(static_cast<std::uint32_t>(heads), pool, [&](std::size_t head) {
        auto state = cache.recurrent_state(layer, static_cast<std::uint32_t>(head));
        if (state.size() != head_dim * head_dim) {
            failures[head].errors.emplace_back(
                "KDA recurrent state disagrees with the head shape");
            return;
        }
        // Head-local copies so the chunk form sees contiguous rows.
        std::vector<float> q(rows * head_dim);
        std::vector<float> k(rows * head_dim);
        std::vector<float> v(rows * head_dim);
        std::vector<float> decay(rows * head_dim);
        std::vector<float> head_beta(rows);
        const auto scale =
            1.0F / std::sqrt(static_cast<float>(head_dim));
        for (std::size_t token = 0U; token < rows; ++token) {
            const auto base = token * projection + head * head_dim;
            head_beta[token] = beta[token * heads + head];
            auto q_row = std::span<float>(q).subspan(token * head_dim, head_dim);
            auto k_row = std::span<float>(k).subspan(token * head_dim, head_dim);
            std::copy_n(query.begin() + base, head_dim, q_row.begin());
            std::copy_n(key.begin() + base, head_dim, k_row.begin());
            std::copy_n(value.begin() + base, head_dim, v.begin() + token * head_dim);
            auto normed = kimi_l2_normalize(q_row, kL2Epsilon);
            if (!normed.ok()) { failures[head] = normed; return; }
            normed = kimi_l2_normalize(k_row, kL2Epsilon);
            if (!normed.ok()) { failures[head] = normed; return; }
            // The kernel scales the query after the L2 norm, not before.
            for (auto& element : q_row) element *= scale;
            auto decayed = kimi_kda_log_decay(
                std::span<float>(decay).subspan(token * head_dim, head_dim),
                std::span<const float>(logits).subspan(base, head_dim),
                weights.dt_bias.subspan(head * head_dim, head_dim),
                weights.a_log[head], kContract.kda_gate_lower_bound);
            if (!decayed.ok()) { failures[head] = decayed; return; }
        }
        std::vector<float> raw(rows * head_dim);
        auto stepped = kimi_kda_chunk(
            raw, state, q, k, v, decay, head_beta, tokens,
            static_cast<std::uint32_t>(head_dim),
            static_cast<std::uint32_t>(head_dim));
        if (!stepped.ok()) { failures[head] = stepped; return; }
        for (std::size_t token = 0U; token < rows; ++token) {
            const auto base = token * projection + head * head_dim;
            auto normed = kimi_kda_output_norm(
                std::span<float>(heads_out).subspan(base, head_dim),
                std::span<const float>(raw).subspan(token * head_dim, head_dim),
                std::span<const float>(gate).subspan(base, head_dim),
                weights.o_norm, kContract.rms_epsilon);
            if (!normed.ok()) { failures[head] = normed; return; }
        }
    });
    for (auto& failure : failures) {
        if (!failure.ok()) return failure;
    }
    return kimi_bf16_matmul(output, heads_out, weights.o_proj, tokens, pool);
}

ValidationResult kimi_mla_layer(std::span<float> output,
                                std::span<const float> input,
                                const KimiMlaWeights& weights,
                                KimiStateCache& cache, std::uint32_t layer,
                                std::uint32_t position, std::uint32_t tokens,
                                KimiLayerScratch& scratch, HostWorkerPool* pool) {
    ValidationResult result;
    const auto hidden = static_cast<std::size_t>(kContract.hidden_size);
    const auto heads = static_cast<std::size_t>(kContract.attention_heads);
    const auto nope = static_cast<std::size_t>(kContract.nope_head_dim);
    const auto rope = static_cast<std::size_t>(kContract.rope_head_dim);
    const auto value_dim = static_cast<std::size_t>(kContract.value_head_dim);
    const auto query_dim = nope + rope;
    const auto lora = static_cast<std::size_t>(kContract.kv_lora_rank);
    const auto latent_width = lora + rope;
    const auto rows = static_cast<std::size_t>(tokens);
    if (tokens == 0U || input.size() != rows * hidden ||
        output.size() != rows * hidden) {
        result.errors.emplace_back("MLA layer operands disagree with the page");
        return result;
    }
    if (!matrix_fits(weights.q_a_proj, kContract.query_lora_rank,
                     kContract.hidden_size) ||
        !matrix_fits(weights.q_b_proj,
                     static_cast<std::uint32_t>(heads * query_dim),
                     kContract.query_lora_rank) ||
        !matrix_fits(weights.kv_a_proj,
                     static_cast<std::uint32_t>(latent_width),
                     kContract.hidden_size) ||
        !matrix_fits(weights.kv_b_proj,
                     static_cast<std::uint32_t>(heads * (nope + value_dim)),
                     kContract.kv_lora_rank) ||
        !matrix_fits(weights.o_proj, kContract.hidden_size,
                     static_cast<std::uint32_t>(heads * value_dim)) ||
        !matrix_fits(weights.g_proj,
                     static_cast<std::uint32_t>(heads * value_dim),
                     kContract.hidden_size) ||
        weights.q_a_norm.size() != kContract.query_lora_rank ||
        weights.kv_a_norm.size() != lora) {
        result.errors.emplace_back("MLA layer weights disagree with the contract");
        return result;
    }

    // Queries: low-rank, normalized between the two halves.
    std::vector<float> query_low(rows * kContract.query_lora_rank, 0.0F);
    result = kimi_bf16_matmul(query_low, input, weights.q_a_proj, tokens, pool);
    if (!result.ok()) return result;
    for (std::size_t token = 0U; token < rows; ++token) {
        auto slice = std::span<float>(query_low).subspan(
            token * kContract.query_lora_rank, kContract.query_lora_rank);
        std::vector<float> source(slice.begin(), slice.end());
        result = kimi_rms_norm(slice, source, weights.q_a_norm, kLoraNormEpsilon);
        if (!result.ok()) return result;
    }
    auto& query = scratch.projection_a;
    query.assign(rows * heads * query_dim, 0.0F);
    result = kimi_bf16_matmul(query, query_low, weights.q_b_proj, tokens, pool);
    if (!result.ok()) return result;

    // Keys and values: one compressed latent per token, shared across heads,
    // with the last `rope_head_dim` values carried unnormalized. `mla_use_nope`
    // is set, so that block is a plain slot and no rotation is applied to it.
    std::vector<float> latent(rows * latent_width, 0.0F);
    result = kimi_bf16_matmul(latent, input, weights.kv_a_proj, tokens, pool);
    if (!result.ok()) return result;
    for (std::size_t token = 0U; token < rows; ++token) {
        auto slice = std::span<float>(latent).subspan(token * latent_width, lora);
        std::vector<float> source(slice.begin(), slice.end());
        result = kimi_rms_norm(slice, source, weights.kv_a_norm, kLoraNormEpsilon);
        if (!result.ok()) return result;
        result = cache.append_latent(layer, position + static_cast<std::uint32_t>(token),
                                     std::span<const float>(latent).subspan(
                                         token * latent_width, latent_width));
        if (!result.ok()) return result;
    }

    // Rows this page can see: everything the cache has published, followed by
    // the rows just appended. The page's own rows are read from the local
    // buffer rather than from the cache, because `commit` runs once per token
    // after every MLA layer has appended and reading past it would be reading a
    // row another layer has not written yet.
    const auto committed = cache.latent_rows(layer);
    if (committed.size() != static_cast<std::size_t>(position) * latent_width) {
        result.errors.emplace_back(
            "MLA layer found the cache at a different length than the page start");
        return result;
    }
    const auto history = position + tokens;
    std::vector<float> visible(static_cast<std::size_t>(history) * latent_width);
    std::copy(committed.begin(), committed.end(), visible.begin());
    std::copy(latent.begin(), latent.end(),
              visible.begin() + static_cast<std::size_t>(position) * latent_width);

    // Expand every visible latent into per-head keys and values. This is the
    // reference's own form. It costs `history * kv_b_proj` per step, which is
    // 6.4 GMAC per layer at 512 tokens of context: about a tenth of a decode
    // step's storage-bound budget here, and the term that would have to be
    // absorbed into the query first if the context grew far past that.
    //
    // `kv_b_proj` consumes the normalized latent only, so the shared rope slot
    // is dropped before the multiply and read directly from `visible` below.
    std::vector<float> compact(static_cast<std::size_t>(history) * lora);
    for (std::uint32_t token = 0U; token < history; ++token) {
        std::copy_n(visible.data() + static_cast<std::size_t>(token) * latent_width,
                    lora, compact.begin() + static_cast<std::size_t>(token) * lora);
    }
    auto& expanded = scratch.projection_b;
    expanded.assign(static_cast<std::size_t>(history) * heads * (nope + value_dim),
                    0.0F);
    result = kimi_bf16_matmul(expanded, compact, weights.kv_b_proj, history, pool);
    if (!result.ok()) return result;

    auto& heads_out = scratch.heads;
    heads_out.assign(rows * heads * value_dim, 0.0F);
    const auto scale = 1.0F / std::sqrt(static_cast<float>(query_dim));
    run_rows(static_cast<std::uint32_t>(heads), pool, [&](std::size_t head) {
        std::vector<float> scores(history);
        for (std::size_t token = 0U; token < rows; ++token) {
            const auto* q = query.data() + (token * heads + head) * query_dim;
            const auto causal = position + static_cast<std::uint32_t>(token) + 1U;
            float highest = -std::numeric_limits<float>::infinity();
            for (std::uint32_t past = 0U; past < causal; ++past) {
                const auto* kv =
                    expanded.data() +
                    (static_cast<std::size_t>(past) * heads + head) * (nope + value_dim);
                const auto* slot = visible.data() +
                                   static_cast<std::size_t>(past) * latent_width + lora;
                float sum = 0.0F;
                for (std::size_t index = 0U; index < nope; ++index) {
                    sum += q[index] * kv[index];
                }
                // The rope-slot block is shared across heads by construction.
                for (std::size_t index = 0U; index < rope; ++index) {
                    sum += q[nope + index] * slot[index];
                }
                scores[past] = sum * scale;
                highest = std::max(highest, scores[past]);
            }
            float total = 0.0F;
            for (std::uint32_t past = 0U; past < causal; ++past) {
                scores[past] = std::exp(scores[past] - highest);
                total += scores[past];
            }
            const auto inverse = 1.0F / total;
            auto* out = heads_out.data() + (token * heads + head) * value_dim;
            for (std::uint32_t past = 0U; past < causal; ++past) {
                const auto weight = scores[past] * inverse;
                const auto* kv =
                    expanded.data() +
                    (static_cast<std::size_t>(past) * heads + head) * (nope + value_dim) +
                    nope;
                for (std::size_t index = 0U; index < value_dim; ++index) {
                    out[index] += weight * kv[index];
                }
            }
        }
    });

    // Full-rank output gate, applied before the output projection.
    auto& gate = scratch.projection_c;
    gate.assign(rows * heads * value_dim, 0.0F);
    result = kimi_bf16_matmul(gate, input, weights.g_proj, tokens, pool);
    if (!result.ok()) return result;
    for (std::size_t index = 0U; index < heads_out.size(); ++index) {
        heads_out[index] *= sigmoid(gate[index]);
    }
    return kimi_bf16_matmul(output, heads_out, weights.o_proj, tokens, pool);
}

ValidationResult kimi_dense_mlp_layer(std::span<float> output,
                                      std::span<const float> input,
                                      const KimiDenseMlpWeights& weights,
                                      std::uint32_t tokens,
                                      KimiLayerScratch& scratch,
                                      HostWorkerPool* pool) {
    ValidationResult result;
    const auto hidden = static_cast<std::size_t>(kContract.hidden_size);
    const auto inner = static_cast<std::size_t>(weights.gate.rows);
    const auto rows = static_cast<std::size_t>(tokens);
    if (tokens == 0U || input.size() != rows * hidden ||
        output.size() != rows * hidden || inner == 0U ||
        !matrix_fits(weights.gate, weights.gate.rows, kContract.hidden_size) ||
        !matrix_fits(weights.up, weights.gate.rows, kContract.hidden_size) ||
        !matrix_fits(weights.down, kContract.hidden_size, weights.gate.rows)) {
        result.errors.emplace_back("dense MLP operands disagree with the contract");
        return result;
    }
    auto& gate = scratch.expert_gate;
    auto& up = scratch.expert_up;
    auto& activated = scratch.expert_out;
    gate.assign(rows * inner, 0.0F);
    up.assign(rows * inner, 0.0F);
    activated.assign(rows * inner, 0.0F);
    result = kimi_bf16_matmul(gate, input, weights.gate, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_bf16_matmul(up, input, weights.up, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_situ_glu(activated, gate, up, kContract.situ_gate_beta,
                           kContract.situ_linear_beta);
    if (!result.ok()) return result;
    return kimi_bf16_matmul(output, activated, weights.down, tokens, pool);
}

ValidationResult kimi_latent_moe_layer(std::span<float> output,
                                       std::span<const float> input,
                                       const KimiMoeWeights& weights,
                                       KimiExpertSource& experts,
                                       std::uint32_t layer, std::uint32_t tokens,
                                       KimiLayerScratch& scratch,
                                       HostWorkerPool* pool) {
    ValidationResult result;
    const auto hidden = static_cast<std::size_t>(kContract.hidden_size);
    const auto latent_width =
        static_cast<std::size_t>(kContract.routed_expert_hidden_size);
    const auto inner = static_cast<std::size_t>(kContract.expert_intermediate_size);
    const auto shared_inner = inner * kContract.shared_experts;
    const auto rows = static_cast<std::size_t>(tokens);
    const auto top_k = static_cast<std::size_t>(kContract.experts_per_token);
    if (tokens == 0U || input.size() != rows * hidden ||
        output.size() != rows * hidden) {
        result.errors.emplace_back("LatentMoE operands disagree with the page");
        return result;
    }
    if (!matrix_fits(weights.router, kContract.routed_experts,
                     kContract.hidden_size) ||
        weights.router_bias.size() != kContract.routed_experts ||
        !matrix_fits(weights.latent_down,
                     kContract.routed_expert_hidden_size, kContract.hidden_size) ||
        !matrix_fits(weights.latent_up, kContract.hidden_size,
                     kContract.routed_expert_hidden_size) ||
        weights.latent_norm.size() != latent_width ||
        !matrix_fits(weights.shared_gate,
                     static_cast<std::uint32_t>(shared_inner), kContract.hidden_size) ||
        !matrix_fits(weights.shared_up,
                     static_cast<std::uint32_t>(shared_inner), kContract.hidden_size) ||
        !matrix_fits(weights.shared_down, kContract.hidden_size,
                     static_cast<std::uint32_t>(shared_inner))) {
        result.errors.emplace_back("LatentMoE weights disagree with the contract");
        return result;
    }

    // Routing reads the raw hidden state, before the latent projection.
    auto& logits = scratch.router_logits;
    logits.assign(rows * kContract.routed_experts, 0.0F);
    result = kimi_bf16_matmul(logits, input, weights.router, tokens, pool);
    if (!result.ok()) return result;
    auto& selection = scratch.selection;
    selection.assign(rows * top_k, KimiRoutedExpert{});
    for (std::size_t token = 0U; token < rows; ++token) {
        result = kimi_route_topk(
            std::span<KimiRoutedExpert>(selection).subspan(token * top_k, top_k),
            std::span<const float>(logits).subspan(
                token * kContract.routed_experts, kContract.routed_experts),
            weights.router_bias, kContract.routed_scale);
        if (!result.ok()) return result;
    }

    // Every expert this block will touch, deduplicated, so the source can stage
    // them at queue depth instead of one demand miss at a time. At batch one
    // this is sixteen distinct experts; across a prefill page it is closer to
    // the whole layer, which is why a page costs a full sweep of the routed set.
    std::vector<std::uint32_t> wanted;
    wanted.reserve(selection.size());
    for (const auto& entry : selection) wanted.push_back(entry.expert);
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    result = experts.prepare(layer, wanted);
    if (!result.ok()) return result;

    auto& latent = scratch.latent;
    latent.assign(rows * latent_width, 0.0F);
    result = kimi_bf16_matmul(latent, input, weights.latent_down, tokens, pool);
    if (!result.ok()) return result;

    auto& mixed = scratch.latent_mix;
    mixed.assign(rows * latent_width, 0.0F);
    auto& gate = scratch.expert_gate;
    auto& up = scratch.expert_up;
    auto& activated = scratch.expert_out;
    gate.assign(inner, 0.0F);
    up.assign(inner, 0.0F);
    activated.assign(inner, 0.0F);
    std::vector<float> expert_output(latent_width);
    // Expert-major so each expert's weights are read once per block even when
    // several tokens of a page select it.
    for (const auto expert : wanted) {
        KimiExpertWeights modules;
        result = experts.fetch(layer, expert, modules);
        if (!result.ok()) return result;
        if (modules.gate.rows != inner || modules.gate.columns != latent_width ||
            modules.up.rows != inner || modules.up.columns != latent_width ||
            modules.down.rows != latent_width || modules.down.columns != inner) {
            result.errors.emplace_back(
                "routed expert modules disagree with the latent shape");
            return result;
        }
        for (std::size_t token = 0U; token < rows; ++token) {
            float weight = 0.0F;
            for (std::size_t slot = 0U; slot < top_k; ++slot) {
                const auto& entry = selection[token * top_k + slot];
                if (entry.expert == expert) { weight = entry.weight; break; }
            }
            if (weight == 0.0F) continue;
            const auto source = std::span<const float>(latent).subspan(
                token * latent_width, latent_width);
            result = kimi_mxfp4_matvec(gate, source, modules.gate, pool);
            if (!result.ok()) return result;
            result = kimi_mxfp4_matvec(up, source, modules.up, pool);
            if (!result.ok()) return result;
            result = kimi_situ_glu(activated, gate, up, kContract.situ_gate_beta,
                                   kContract.situ_linear_beta);
            if (!result.ok()) return result;
            result = kimi_mxfp4_matvec(expert_output, activated, modules.down, pool);
            if (!result.ok()) return result;
            auto* destination = mixed.data() + token * latent_width;
            for (std::size_t index = 0U; index < latent_width; ++index) {
                destination[index] += weight * expert_output[index];
            }
        }
    }

    // The "stable" half of Stable LatentMoE: normalize the mixture in the
    // latent space before projecting it back out.
    for (std::size_t token = 0U; token < rows; ++token) {
        auto slice = std::span<float>(mixed).subspan(token * latent_width, latent_width);
        std::vector<float> source(slice.begin(), slice.end());
        result = kimi_rms_norm(slice, source, weights.latent_norm,
                               kContract.rms_epsilon);
        if (!result.ok()) return result;
    }
    result = kimi_bf16_matmul(output, mixed, weights.latent_up, tokens, pool);
    if (!result.ok()) return result;

    // Shared experts run on the raw hidden state, not the latent, and add in.
    std::vector<float> shared_gate(rows * shared_inner);
    std::vector<float> shared_up(rows * shared_inner);
    std::vector<float> shared_activated(rows * shared_inner);
    std::vector<float> shared_output(rows * hidden);
    result = kimi_bf16_matmul(shared_gate, input, weights.shared_gate, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_bf16_matmul(shared_up, input, weights.shared_up, tokens, pool);
    if (!result.ok()) return result;
    result = kimi_situ_glu(shared_activated, shared_gate, shared_up,
                           kContract.situ_gate_beta, kContract.situ_linear_beta);
    if (!result.ok()) return result;
    result = kimi_bf16_matmul(shared_output, shared_activated, weights.shared_down,
                              tokens, pool);
    if (!result.ok()) return result;
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] += shared_output[index];
    }
    return result;
}

ValidationResult kimi_decoder_layer(KimiResidualStream& stream,
                                    const KimiLayerWeights& weights,
                                    KimiStateCache& cache,
                                    KimiExpertSource& experts,
                                    std::uint32_t layer, std::uint32_t position,
                                    std::uint32_t tokens,
                                    KimiLayerScratch& scratch,
                                    HostWorkerPool* pool) {
    ValidationResult result;
    const auto hidden = static_cast<std::size_t>(kContract.hidden_size);
    const auto rows = static_cast<std::size_t>(tokens);
    if (layer >= kContract.layer_count || stream.tokens() != tokens) {
        result.errors.emplace_back("decoder layer disagrees with the page shape");
        return result;
    }
    auto& normalized = scratch.normalized;
    auto& attention = scratch.attention;
    auto& feedforward = scratch.feedforward;
    normalized.assign(rows * hidden, 0.0F);
    attention.assign(rows * hidden, 0.0F);
    feedforward.assign(rows * hidden, 0.0F);

    // The attention site selects over prior depth, then this layer may open a
    // block from the *incoming* prefix — the value before the mix, which is
    // what the reference stores.
    result = stream.mix(normalized, weights.attention_res_proj,
                        weights.attention_res_norm, kContract.rms_epsilon, pool);
    if (!result.ok()) return result;
    if (layer % kContract.attention_residual_block_size == 0U) {
        result = stream.open_block();
        if (!result.ok()) return result;
    }
    {
        std::vector<float> source(normalized.begin(), normalized.end());
        for (std::size_t token = 0U; token < rows; ++token) {
            result = kimi_rms_norm(
                std::span<float>(normalized).subspan(token * hidden, hidden),
                std::span<const float>(source).subspan(token * hidden, hidden),
                weights.input_norm, kContract.rms_epsilon);
            if (!result.ok()) return result;
        }
    }

    if (kimi_k3_kda_layer(layer)) {
        result = kimi_kda_layer(attention, normalized, weights.kda, cache, layer,
                                tokens, scratch, pool);
    } else {
        result = kimi_mla_layer(attention, normalized, weights.mla, cache, layer,
                                position, tokens, scratch, pool);
    }
    if (!result.ok()) return result;
    result = stream.add(attention);
    if (!result.ok()) return result;

    result = stream.mix(normalized, weights.mlp_res_proj, weights.mlp_res_norm,
                        kContract.rms_epsilon, pool);
    if (!result.ok()) return result;
    {
        std::vector<float> source(normalized.begin(), normalized.end());
        for (std::size_t token = 0U; token < rows; ++token) {
            result = kimi_rms_norm(
                std::span<float>(normalized).subspan(token * hidden, hidden),
                std::span<const float>(source).subspan(token * hidden, hidden),
                weights.post_attention_norm, kContract.rms_epsilon);
            if (!result.ok()) return result;
        }
    }

    if (kimi_k3_moe_layer(layer)) {
        result = kimi_latent_moe_layer(feedforward, normalized, weights.moe,
                                       experts, layer, tokens, scratch, pool);
    } else {
        result = kimi_dense_mlp_layer(feedforward, normalized, weights.dense,
                                      tokens, scratch, pool);
    }
    if (!result.ok()) return result;
    return stream.add(feedforward);
}

}  // namespace strata
