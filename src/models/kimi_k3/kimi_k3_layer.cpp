#include "strata/models/kimi_k3/kimi_k3_layer.hpp"

#include "strata/platform/compressed_tensors.hpp"
#include "strata/platform/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

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

#if defined(__x86_64__)

// One row of an MXFP4 matvec, eight elements at a time.
//
// The scalar path spends its time decoding nibbles one at a time; the decode is
// a sixteen-entry lookup, which is what an in-register permute does natively.
// `permutevar8x32_ps` selects the E2M1 magnitude by the nibble's low three
// bits, and bit 3 becomes the float's sign bit by a shift and an XOR, so no
// value is branched on and no table leaves a register.
//
// `even` and `odd` are the input already split by parity, because a byte holds
// two consecutive elements and the vector wants them in separate lanes. The
// split is done once per matvec by the caller and reused across every row,
// which is where it costs nothing: one pass over `columns` floats against
// `rows` passes over the weights.
//
// Summation order differs from the scalar path: eight lanes accumulate in
// parallel and reduce at the end. That is a different order, not a different
// value set, and a wider tree reduction is if anything better conditioned than
// a single serial chain. `kimi_mxfp4_matvec agrees with its scalar path`
// pins the agreement.
[[gnu::target("avx2,fma")]] float mxfp4_row_dot_avx2(
    const std::uint8_t* packed, const std::uint8_t* scales, const float* even,
    const float* odd, std::size_t groups) noexcept {
    const __m256 table = _mm256_loadu_ps(kMxfp4Magnitudes.data());
    const __m256i nibble_mask = _mm256_set1_epi32(0x0F);
    const __m256i magnitude_mask = _mm256_set1_epi32(0x07);
    const __m256i sign_mask = _mm256_set1_epi32(0x08);
    __m256 total = _mm256_setzero_ps();

    for (std::size_t group = 0U; group < groups; ++group) {
        const auto* bytes = packed + group * 16U;
        const auto* even_source = even + group * 16U;
        const auto* odd_source = odd + group * 16U;
        __m256 accumulator = _mm256_setzero_ps();
        for (std::size_t half = 0U; half < 2U; ++half) {
            const __m128i raw = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(bytes + half * 8U));
            const __m256i wide = _mm256_cvtepu8_epi32(raw);
            const __m256i low = _mm256_and_si256(wide, nibble_mask);
            const __m256i high =
                _mm256_and_si256(_mm256_srli_epi32(wide, 4), nibble_mask);
            const __m256 low_value = _mm256_xor_ps(
                _mm256_permutevar8x32_ps(
                    table, _mm256_and_si256(low, magnitude_mask)),
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_and_si256(low, sign_mask), 28)));
            const __m256 high_value = _mm256_xor_ps(
                _mm256_permutevar8x32_ps(
                    table, _mm256_and_si256(high, magnitude_mask)),
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_and_si256(high, sign_mask), 28)));
            accumulator = _mm256_fmadd_ps(
                low_value, _mm256_loadu_ps(even_source + half * 8U), accumulator);
            accumulator = _mm256_fmadd_ps(
                high_value, _mm256_loadu_ps(odd_source + half * 8U), accumulator);
        }
        // The group's shared scale factors out of its 32 products, so it is one
        // multiply per group rather than per element, exactly as in the scalar
        // path.
        total = _mm256_fmadd_ps(
            accumulator, _mm256_set1_ps(mxfp4_scale_from_e8m0(scales[group])),
            total);
    }

    __m128 folded = _mm_add_ps(_mm256_castps256_ps128(total),
                               _mm256_extractf128_ps(total, 1));
    folded = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    folded = _mm_add_ss(folded, _mm_shuffle_ps(folded, folded, 1));
    return _mm_cvtss_f32(folded);
}

// One row of a BF16 matvec. BF16 to F32 is the identity on the top sixteen
// bits, so the whole conversion is a widen and a shift and no table is needed.
// The scalar path went through `decode_bf16`, which is a `memcpy` per element.
[[gnu::target("avx2,fma")]] float bf16_row_dot_avx2(
    const std::uint16_t* values, const float* source,
    std::size_t columns) noexcept {
    __m256 total = _mm256_setzero_ps();
    std::size_t column = 0U;
    for (; column + 8U <= columns; column += 8U) {
        const __m128i raw =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + column));
        const __m256 weight = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
        total = _mm256_fmadd_ps(weight, _mm256_loadu_ps(source + column), total);
    }
    __m128 folded = _mm_add_ps(_mm256_castps256_ps128(total),
                               _mm256_extractf128_ps(total, 1));
    folded = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    folded = _mm_add_ss(folded, _mm_shuffle_ps(folded, folded, 1));
    float sum = _mm_cvtss_f32(folded);
    for (; column < columns; ++column) {
        sum += source[column] * decode_bf16(values[column]);
    }
    return sum;
}

[[nodiscard]] bool mxfp4_avx2_available() noexcept {
    static const bool available =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return available;
}

#endif  // __x86_64__

// One task per worker over a contiguous row range, not one task per row.
//
// A row of an expert module is about a microsecond of work. Dispatching it as
// its own task pays an indirect call through `std::function` and a share of the
// barrier for each one, and at 3072 rows across 4416 matvecs per token that
// dispatch becomes the measured cost: throughput fell going from 14 workers to
// 56 because the extra threads added barrier traffic rather than arithmetic.
// Chunking also keeps each worker on a contiguous slice of the weight matrix,
// which is what the hardware prefetcher wants.
//
// `body` is a template parameter so the inner call inlines. Taking it by
// `const std::function&` put an indirect call on the innermost loop.
template <typename Body>
void run_rows(std::uint32_t rows, HostWorkerPool* pool, Body&& body) {
    if (pool != nullptr && pool->size() > 1U && rows > 1U) {
        const auto workers = std::min<std::size_t>(pool->size(), rows);
        const auto chunk = (static_cast<std::size_t>(rows) + workers - 1U) / workers;
        (void)pool->parallel_for(workers, [&](std::size_t index) {
            const auto begin = index * chunk;
            const auto end = std::min<std::size_t>(begin + chunk, rows);
            for (std::size_t row = begin; row < end; ++row) body(row);
        });
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
#if defined(__x86_64__)
    if (mxfp4_avx2_available()) {
        const auto* weight_base = weight.values.data();
        const auto* input_base = input.data();
        auto* destination = output.data();
        run_rows(weight.rows, pool, [&](std::size_t row) {
            const auto* values = weight_base + row * columns;
            for (std::size_t token = 0U; token < count; ++token) {
                destination[token * rows + row] = bf16_row_dot_avx2(
                    values, input_base + token * columns, columns);
            }
        });
        return result;
    }
#endif

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

#if defined(__x86_64__)
    if (mxfp4_avx2_available() && columns % 32U == 0U) {
        // Split the input by parity once and let every row read it. The buffer
        // belongs to the calling thread and is only read while `run_rows` is
        // inside its synchronous parallel_for, so the workers may share it.
        thread_local std::vector<float> split;
        split.resize(columns);
        float* even = split.data();
        float* odd = split.data() + packed_stride;
        for (std::size_t index = 0U; index < packed_stride; ++index) {
            even[index] = input[index * 2U];
            odd[index] = input[index * 2U + 1U];
        }
        const auto* packed_base = module.packed.data();
        const auto* scale_base = module.scales.data();
        auto* destination = output.data();
        run_rows(module.rows, pool, [&](std::size_t row) {
            destination[row] = mxfp4_row_dot_avx2(packed_base + row * packed_stride,
                                                  scale_base + row * scale_stride,
                                                  even, odd, scale_stride);
        });
        return result;
    }
#endif

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
                // Low nibble first. The sign is bit 3, and `kMxfp4Values` has
                // it folded in, so this indexes once instead of branching on
                // it. Same values, same order, same result.
                partial += source[index * 2U] * kMxfp4Values[byte & 0x0FU] +
                           source[index * 2U + 1U] * kMxfp4Values[byte >> 4U];
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
    experts.observe(layer, wanted);
    result = experts.prepare(layer, wanted);
    if (!result.ok()) return result;

    auto& latent = scratch.latent;
    latent.assign(rows * latent_width, 0.0F);
    result = kimi_bf16_matmul(latent, input, weights.latent_down, tokens, pool);
    if (!result.ok()) return result;

    auto& mixed = scratch.latent_mix;
    mixed.assign(rows * latent_width, 0.0F);
    // Resolve every expert's views before anything runs in parallel. `fetch`
    // reaches the arena's LRU clock and its hit counters, none of which are
    // synchronised, while the views it hands back stay valid for as long as
    // `prepare` keeps the set admitted. Resolving up front is therefore what
    // makes the compute below safe to spread across workers without reaching
    // into the arena's internals at all.
    const auto slots = wanted.size();
    std::vector<KimiExpertWeights> resolved(slots);
    for (std::size_t slot = 0U; slot < slots; ++slot) {
        result = experts.fetch(layer, wanted[slot], resolved[slot]);
        if (!result.ok()) return result;
        const auto& modules = resolved[slot];
        if (modules.gate.rows != inner || modules.gate.columns != latent_width ||
            modules.up.rows != inner || modules.up.columns != latent_width ||
            modules.down.rows != latent_width || modules.down.columns != inner) {
            result.errors.emplace_back(
                "routed expert modules disagree with the latent shape");
            return result;
        }
    }

    // One parallel region per MoE block instead of one per matvec. The block
    // dispatched forty-eight of them -- three matvecs for each of sixteen
    // experts -- and each paid a barrier across every worker for about a
    // millisecond of arithmetic. On two sockets that barrier is what stopped
    // the second socket from paying for itself: 14 cores on one socket beat 28
    // across two in every arm, and spreading the pages with `numactl
    // --interleave` did not recover it. Experts are independent, so the block
    // is the natural grain -- one barrier per layer, and each worker keeps a
    // whole expert's 16.73 MiB to itself.
    //
    // Work is chunked by worker rather than dispatched per expert so the
    // accumulator and scratch can be sized by worker count. A prefill page
    // deduplicates to hundreds of experts, and a buffer per expert would run
    // to hundreds of megabytes.
    const auto width = rows * latent_width;
    const auto workers = (pool != nullptr && pool->size() > 1U && slots > 1U)
                             ? std::min<std::size_t>(pool->size(), slots)
                             : 1U;
    const auto chunk = (slots + workers - 1U) / workers;
    const auto stride = inner * 3U + latent_width;
    auto& workspace = scratch.worker_workspace;
    auto& mixture = scratch.worker_mixture;
    workspace.assign(workers * stride, 0.0F);
    mixture.assign(workers * width, 0.0F);
    std::vector<std::uint8_t> failed(workers, 0U);

    const auto run_chunk = [&](std::size_t index) {
        auto* scratch_base = workspace.data() + index * stride;
        const auto gate = std::span<float>(scratch_base, inner);
        const auto up = std::span<float>(scratch_base + inner, inner);
        const auto activated = std::span<float>(scratch_base + inner * 2U, inner);
        const auto expert_output =
            std::span<float>(scratch_base + inner * 3U, latent_width);
        auto* accumulator = mixture.data() + index * width;
        const auto begin = index * chunk;
        const auto end = std::min(begin + chunk, slots);
        for (std::size_t slot = begin; slot < end; ++slot) {
            const auto& modules = resolved[slot];
            const auto expert = wanted[slot];
            for (std::size_t token = 0U; token < rows; ++token) {
                float weight = 0.0F;
                for (std::size_t entry = 0U; entry < top_k; ++entry) {
                    const auto& candidate = selection[token * top_k + entry];
                    if (candidate.expert == expert) {
                        weight = candidate.weight;
                        break;
                    }
                }
                if (weight == 0.0F) continue;
                const auto source = std::span<const float>(latent).subspan(
                    token * latent_width, latent_width);
                // No pool: a nested parallel region would put back the barrier
                // this grain exists to remove. The matvecs run on whichever
                // worker owns the expert.
                if (!kimi_mxfp4_matvec(gate, source, modules.gate).ok() ||
                    !kimi_mxfp4_matvec(up, source, modules.up).ok() ||
                    !kimi_situ_glu(activated, gate, up, kContract.situ_gate_beta,
                                   kContract.situ_linear_beta)
                         .ok() ||
                    !kimi_mxfp4_matvec(expert_output, activated, modules.down)
                         .ok()) {
                    failed[index] = 1U;
                    return;
                }
                auto* destination = accumulator + token * latent_width;
                for (std::size_t position = 0U; position < latent_width;
                     ++position) {
                    destination[position] += weight * expert_output[position];
                }
            }
        }
    };

    if (workers > 1U) {
        (void)pool->parallel_for(workers, run_chunk);
    } else {
        run_chunk(0U);
    }

    if (std::find(failed.begin(), failed.end(), 1U) != failed.end()) {
        result.errors.emplace_back(
            "a routed expert failed to evaluate in the LatentMoE block");
        return result;
    }

    // Reduced in worker order, so the sum does not depend on the order the
    // pool happened to finish them in.
    for (std::size_t index = 0U; index < workers; ++index) {
        const auto* source = mixture.data() + index * width;
        for (std::size_t position = 0U; position < width; ++position) {
            mixed[position] += source[position];
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
    const auto mix_begin = std::chrono::steady_clock::now();
    result = stream.mix(normalized, weights.attention_res_proj,
                        weights.attention_res_norm, kContract.rms_epsilon, pool);
    scratch.residual_mix_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - mix_begin).count());
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

    const auto attention_begin = std::chrono::steady_clock::now();
    if (kimi_k3_kda_layer(layer)) {
        result = kimi_kda_layer(attention, normalized, weights.kda, cache, layer,
                                tokens, scratch, pool);
    } else {
        result = kimi_mla_layer(attention, normalized, weights.mla, cache, layer,
                                position, tokens, scratch, pool);
    }
    scratch.attention_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - attention_begin).count());
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

    const auto feedforward_begin = std::chrono::steady_clock::now();
    if (kimi_k3_moe_layer(layer)) {
        result = kimi_latent_moe_layer(feedforward, normalized, weights.moe,
                                       experts, layer, tokens, scratch, pool);
    } else {
        result = kimi_dense_mlp_layer(feedforward, normalized, weights.dense,
                                      tokens, scratch, pool);
    }
    scratch.feedforward_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - feedforward_begin).count());
    if (!result.ok()) return result;
    return stream.add(feedforward);
}

}  // namespace strata
