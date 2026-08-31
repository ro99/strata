namespace {

__device__ __forceinline__ float glm53_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float glm53_sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

// GLIBC 2.35 and Arm optimized-routines use this table and FP64 cubic for
// expf. The Arm source is MIT OR Apache-2.0 WITH LLVM-exception:
// Copyright (c) 2017-2025, Arm Limited. Keeping the operations explicitly
// rounded reproduces the host reference without a page-wide host round trip.
__device__ __constant__ std::uint64_t kGlm53KdaExpTable[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL,
    0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL,
    0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL,
    0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL,
    0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL,
    0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL};

__device__ __forceinline__ float glm53_kda_add(float left, float right) {
    return __fadd_rn(left, right);
}

__device__ __forceinline__ float glm53_kda_multiply(float left, float right) {
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float glm53_kda_expf(float value) {
    const auto value_bits = static_cast<std::uint32_t>(__float_as_uint(value));
    const auto absolute_top = (value_bits >> 20U) & 0x7FFU;
    constexpr auto boundary_top =
        (std::bit_cast<std::uint32_t>(88.0F) >> 20U) & 0x7FFU;
    if (absolute_top >= boundary_top) {
        if (value_bits == 0xFF800000U) return 0.0F;
        if (absolute_top >= 0x7F8U) return value + value;
        if (value > 0x1.62e42ep6F) return __int_as_float(0x7F800000);
        if (value < -0x1.9fe368p6F) return 0.0F;
    }
    constexpr double inverse_ln2_scaled = 0x1.71547652b82fep+0 * 32.0;
    constexpr double shift = 0x1.8p+52;
    constexpr double c0 = 0x1.c6af84b912394p-5 / (32.0 * 32.0 * 32.0);
    constexpr double c1 = 0x1.ebfce50fac4f3p-3 / (32.0 * 32.0);
    constexpr double c2 = 0x1.62e42ff0c52d6p-1 / 32.0;
    const double x = static_cast<double>(value);
    const double z = __dmul_rn(inverse_ln2_scaled, x);
    double rounded = __dadd_rn(z, shift);
    const auto integer =
        static_cast<std::uint64_t>(__double_as_longlong(rounded));
    rounded = __dadd_rn(rounded, -shift);
    const double remainder = __dadd_rn(z, -rounded);
    auto encoded = kGlm53KdaExpTable[integer & 31ULL];
    encoded += integer << 47U;
    const double scale = __longlong_as_double(
        static_cast<unsigned long long>(encoded));
    const double polynomial_high = __dadd_rn(__dmul_rn(c0, remainder), c1);
    const double square = __dmul_rn(remainder, remainder);
    double result = __dadd_rn(__dmul_rn(c2, remainder), 1.0);
    result = __dadd_rn(__dmul_rn(polynomial_high, square), result);
    result = __dmul_rn(result, scale);
    return static_cast<float>(result);
}

__device__ __forceinline__ float glm53_kda_sigmoid(float value) {
    return 1.0F / glm53_kda_add(1.0F, glm53_kda_expf(-value));
}

__global__ void glm53_kda_conv_kernel(
    float* activations, float* convolution, const float* taps,
    std::uint32_t width, std::uint32_t kernel) {
    const auto channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) return;
    const auto history_width = kernel - 1U;
    for (std::uint32_t projection = 0U; projection < 3U; ++projection) {
        auto* values = activations + static_cast<std::size_t>(projection) * width;
        auto* history = convolution +
            static_cast<std::size_t>(projection) * width * history_width +
            static_cast<std::size_t>(channel) * history_width;
        const auto* weights = taps +
            static_cast<std::size_t>(projection) * width * kernel +
            static_cast<std::size_t>(channel) * kernel;
        float sum = glm53_kda_multiply(
            weights[kernel - 1U], values[channel]);
        for (std::uint32_t offset = 0U; offset < history_width; ++offset) {
            sum = glm53_kda_add(
                sum, glm53_kda_multiply(weights[offset], history[offset]));
        }
        for (std::uint32_t offset = 0U; offset + 1U < history_width; ++offset) {
            history[offset] = history[offset + 1U];
        }
        history[history_width - 1U] = values[channel];
        values[channel] = glm53_bf16(
            glm53_kda_multiply(sum, glm53_kda_sigmoid(sum)));
    }
}

__global__ void glm53_kda_recurrence_kernel(
    float* recurrent, const float* a_log, const float* dt_bias,
    const float* norm_weight, const float* activations, float* output,
    std::uint32_t heads, std::uint32_t head_dim) {
    const auto head = blockIdx.x;
    const auto lane = threadIdx.x;
    if (head >= heads || lane >= head_dim) return;
    const auto width = heads * head_dim;
    const auto base = head * head_dim;
    const auto* query = activations;
    const auto* key = activations + width;
    const auto* value = activations + 2U * width;
    const auto* forget = activations + 3U * width;
    const auto* gate = activations + 4U * width;
    const auto* beta = activations + 5U * width;
    extern __shared__ float scratch[];
    auto* normalized_query = scratch;
    auto* normalized_key = normalized_query + head_dim;
    auto* decay = normalized_key + head_dim;
    auto* raw = decay + head_dim;
    __shared__ float query_inverse;
    __shared__ float key_inverse;
    __shared__ float output_inverse;
    if (lane == 0U) {
        float query_square = 0.0F;
        float key_square = 0.0F;
        for (std::uint32_t index = 0U; index < head_dim; ++index) {
            const auto q = query[base + index];
            const auto k = key[base + index];
            query_square = glm53_kda_add(
                query_square, glm53_kda_multiply(q, q));
            key_square = glm53_kda_add(
                key_square, glm53_kda_multiply(k, k));
        }
        query_inverse = 1.0F /
                        sqrtf(glm53_kda_add(query_square, 1.0e-6F));
        key_inverse = 1.0F /
                      sqrtf(glm53_kda_add(key_square, 1.0e-6F));
    }
    __syncthreads();
    normalized_query[lane] = glm53_kda_multiply(
        glm53_kda_multiply(query[base + lane], query_inverse),
        1.0F / sqrtf(static_cast<float>(head_dim)));
    normalized_key[lane] =
        glm53_kda_multiply(key[base + lane], key_inverse);
    const float biased_forget = glm53_kda_add(
        forget[base + lane], dt_bias[base + lane]);
    const float decay_logit = glm53_kda_multiply(
        glm53_kda_expf(a_log[head]), biased_forget);
    const float logarithm = glm53_kda_multiply(
        -5.0F, glm53_kda_sigmoid(decay_logit));
    decay[lane] = glm53_kda_expf(logarithm);
    __syncthreads();
    auto* state_row = recurrent +
        (static_cast<std::size_t>(head) * head_dim + lane) * head_dim;
    float projected = 0.0F;
    for (std::uint32_t index = 0U; index < head_dim; ++index) {
        state_row[index] =
            glm53_kda_multiply(state_row[index], decay[index]);
        projected = glm53_kda_add(
            projected,
            glm53_kda_multiply(state_row[index], normalized_key[index]));
    }
    const auto delta = glm53_kda_multiply(
        __fsub_rn(value[base + lane], projected), beta[head]);
    float mixed = 0.0F;
    for (std::uint32_t index = 0U; index < head_dim; ++index) {
        state_row[index] = glm53_kda_add(
            state_row[index],
            glm53_kda_multiply(delta, normalized_key[index]));
        mixed = glm53_kda_add(
            mixed,
            glm53_kda_multiply(state_row[index], normalized_query[index]));
    }
    raw[lane] = glm53_bf16(mixed);
    __syncthreads();
    if (lane == 0U) {
        float square = 0.0F;
        for (std::uint32_t index = 0U; index < head_dim; ++index) {
            square = glm53_kda_add(
                square, glm53_kda_multiply(raw[index], raw[index]));
        }
        output_inverse = 1.0F / sqrtf(glm53_kda_add(
            square / static_cast<float>(head_dim), 1.0e-5F));
    }
    __syncthreads();
    const float normalized = glm53_kda_multiply(
        norm_weight[lane], glm53_kda_multiply(raw[lane], output_inverse));
    output[base + lane] = glm53_bf16(glm53_kda_multiply(
        normalized, glm53_kda_sigmoid(gate[base + lane])));
}

__global__ void glm53_kda_beta_kernel(float* beta, std::uint32_t heads) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < heads) {
        beta[index] = glm53_bf16(glm53_kda_sigmoid(beta[index]));
    }
}

__global__ void glm53_mhc_gather_page_input_kernel(
    const Dsv4MhcWorkspace* slots, float* output, std::uint32_t rows,
    std::uint32_t hidden) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(rows) * hidden;
    if (index >= elements) return;
    const auto row = static_cast<std::uint32_t>(index / hidden);
    const auto column = static_cast<std::uint32_t>(index % hidden);
    output[index] = __bfloat162float(slots[row].layer_input[column]);
}

__global__ void glm53_mhc_scatter_page_branch_kernel(
    const float* input, Dsv4MhcWorkspace* slots, std::uint32_t rows,
    std::uint32_t hidden) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(rows) * hidden;
    if (index >= elements) return;
    const auto row = static_cast<std::uint32_t>(index / hidden);
    const auto column = static_cast<std::uint32_t>(index % hidden);
    slots[row].branch[column] = __float2bfloat16_rn(input[index]);
}

__global__ void glm53_kda_conv_page_row_kernel(
    float* query, float* key, float* value, float* convolution,
    const float* taps, std::uint32_t width, std::uint32_t kernel) {
    const auto channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) return;
    const auto history_width = kernel - 1U;
    float* projections[3] = {query, key, value};
    for (std::uint32_t projection = 0U; projection < 3U; ++projection) {
        auto* history = convolution +
            static_cast<std::size_t>(projection) * width * history_width +
            static_cast<std::size_t>(channel) * history_width;
        const auto* weights = taps +
            static_cast<std::size_t>(projection) * width * kernel +
            static_cast<std::size_t>(channel) * kernel;
        float sum = glm53_kda_multiply(
            weights[kernel - 1U], projections[projection][channel]);
        for (std::uint32_t offset = 0U; offset < history_width; ++offset) {
            sum = glm53_kda_add(
                sum, glm53_kda_multiply(weights[offset], history[offset]));
        }
        for (std::uint32_t offset = 0U; offset + 1U < history_width; ++offset) {
            history[offset] = history[offset + 1U];
        }
        history[history_width - 1U] = projections[projection][channel];
        projections[projection][channel] = glm53_bf16(
            glm53_kda_multiply(sum, glm53_kda_sigmoid(sum)));
    }
}

__global__ void glm53_kda_recurrence_page_row_kernel(
    float* recurrent, const float* a_log, const float* dt_bias,
    const float* norm_weight, const float* query, const float* key,
    const float* value, const float* forget, const float* beta,
    const float* gate, float* output, std::uint32_t heads,
    std::uint32_t head_dim) {
    const auto head = blockIdx.x;
    const auto lane = threadIdx.x;
    if (head >= heads || lane >= head_dim) return;
    const auto base = head * head_dim;
    extern __shared__ float scratch[];
    auto* normalized_query = scratch;
    auto* normalized_key = normalized_query + head_dim;
    auto* decay = normalized_key + head_dim;
    auto* raw = decay + head_dim;
    __shared__ float query_inverse;
    __shared__ float key_inverse;
    __shared__ float output_inverse;
    if (lane == 0U) {
        float query_square = 0.0F;
        float key_square = 0.0F;
        for (std::uint32_t index = 0U; index < head_dim; ++index) {
            query_square = glm53_kda_add(
                query_square,
                glm53_kda_multiply(query[base + index],
                                   query[base + index]));
            key_square = glm53_kda_add(
                key_square,
                glm53_kda_multiply(key[base + index], key[base + index]));
        }
        query_inverse = 1.0F /
                        sqrtf(glm53_kda_add(query_square, 1.0e-6F));
        key_inverse = 1.0F /
                      sqrtf(glm53_kda_add(key_square, 1.0e-6F));
    }
    __syncthreads();
    normalized_query[lane] = glm53_kda_multiply(
        glm53_kda_multiply(query[base + lane], query_inverse),
        1.0F / sqrtf(static_cast<float>(head_dim)));
    normalized_key[lane] =
        glm53_kda_multiply(key[base + lane], key_inverse);
    const float biased_forget = glm53_kda_add(
        forget[base + lane], dt_bias[base + lane]);
    const float decay_logit = glm53_kda_multiply(
        glm53_kda_expf(a_log[head]), biased_forget);
    const float logarithm = glm53_kda_multiply(
        -5.0F, glm53_kda_sigmoid(decay_logit));
    decay[lane] = glm53_kda_expf(logarithm);
    __syncthreads();
    auto* state_row = recurrent +
        (static_cast<std::size_t>(head) * head_dim + lane) * head_dim;
    float projected = 0.0F;
    for (std::uint32_t index = 0U; index < head_dim; ++index) {
        state_row[index] =
            glm53_kda_multiply(state_row[index], decay[index]);
        projected = glm53_kda_add(
            projected,
            glm53_kda_multiply(state_row[index], normalized_key[index]));
    }
    const auto delta = glm53_kda_multiply(
        __fsub_rn(value[base + lane], projected), beta[head]);
    float mixed = 0.0F;
    for (std::uint32_t index = 0U; index < head_dim; ++index) {
        state_row[index] = glm53_kda_add(
            state_row[index],
            glm53_kda_multiply(delta, normalized_key[index]));
        mixed = glm53_kda_add(
            mixed,
            glm53_kda_multiply(state_row[index], normalized_query[index]));
    }
    raw[lane] = glm53_bf16(mixed);
    __syncthreads();
    if (lane == 0U) {
        float square = 0.0F;
        for (std::uint32_t index = 0U; index < head_dim; ++index) {
            square = glm53_kda_add(
                square, glm53_kda_multiply(raw[index], raw[index]));
        }
        output_inverse = 1.0F / sqrtf(glm53_kda_add(
            square / static_cast<float>(head_dim), 1.0e-5F));
    }
    __syncthreads();
    const float normalized = glm53_kda_multiply(
        norm_weight[lane], glm53_kda_multiply(raw[lane], output_inverse));
    output[base + lane] = glm53_bf16(glm53_kda_multiply(
        normalized, glm53_kda_sigmoid(gate[base + lane])));
}

__global__ void glm53_moe_join_mhc_kernel(
    const float* expert_output, const float* coefficients,
    std::uint32_t routed, __nv_bfloat16* branch, std::uint32_t hidden,
    unsigned int* error_flag) {
    const auto column = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= hidden) return;
    float value = expert_output[static_cast<std::uint64_t>(routed) * hidden +
                                column];
    for (std::uint32_t expert = 0U; expert < routed; ++expert) {
        const float weighted = glm53_bf16(
            coefficients[expert] *
            expert_output[static_cast<std::uint64_t>(expert) * hidden +
                          column]);
        value = glm53_bf16(value + weighted);
    }
    if (!isfinite(value)) atomicExch(error_flag, 1U);
    branch[column] = __float2bfloat16_rn(value);
}

__global__ void glm53_swiglu_kernel(
    float* gate, const float* up, std::uint32_t elements) {
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const float g = fminf(glm53_bf16(gate[index]), 10.0F);
    const float u = fminf(fmaxf(glm53_bf16(up[index]), -10.0F), 10.0F);
    gate[index] = glm53_bf16(g * glm53_sigmoid(g) * u);
}

__global__ void glm53_rms_norm_bf16_kernel(
    float* values, const float* weights, std::uint32_t columns) {
    __shared__ float scale;
    if (threadIdx.x == 0U) {
        float sum = 0.0F;
        for (std::uint32_t column = 0U; column < columns; ++column) {
            sum = __fadd_rn(sum, __fmul_rn(values[column], values[column]));
        }
        scale = 1.0F / sqrtf(sum / static_cast<float>(columns) + 1.0e-5F);
    }
    __syncthreads();
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        values[column] = glm53_bf16(
            weights[column] * (values[column] * scale));
    }
}

// The attention scores, stopping short of the exponential.
//
// The softmax is the one place the resident path cannot follow the host: CUDA's
// `expf` and glibc's disagree in 30.4% of f32 results, and 0.000375% of those
// survive the BF16 rounding the model applies to the coefficient. At 64 heads
// times history times 11 MLA layers that is an 11.4% chance per token at
// history 46, which is exactly the rate at which record 0214 measured the
// resident chain diverging. So the device computes the scores and the host
// computes `exp`, the sum and the coefficient, on the accepted fallback's own
// code.
//
// The FP8 checkpoint still expands to F32 in the per-step workspace. Keep its
// accepted kernel unchanged while the BF16 checkpoint uses the persistent
// cache path below.
__global__ void glm53_mla_scores_f32_kernel(
    const float* query, const float* expanded, float* scores,
    std::uint32_t history, std::uint32_t heads, std::uint32_t head_dim) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= heads || threadIdx.x != 0U) return;
    const auto* q = query + static_cast<std::uint64_t>(head) * head_dim;
    // The host computes `1.0F / std::sqrt(head_dim)` once and multiplies;
    // `rsqrtf` is a ~2 ULP approximation and is not the same number.
    const float score_scale = 1.0F / sqrtf(static_cast<float>(head_dim));
    for (std::uint32_t token = 0U; token < history; ++token) {
        const auto* key = expanded +
            (static_cast<std::uint64_t>(token) * heads + head) *
                (2U * head_dim);
        // The host writes `score += q[column] * kv[column]`, which GCC
        // contracts to an FMA at -O3 on this FMA3 host.
        float score = 0.0F;
        for (std::uint32_t column = 0U; column < head_dim; ++column) {
            score = __fmaf_rn(q[column], key[column], score);
        }
        scores[static_cast<std::uint64_t>(head) * history + token] =
            score * score_scale;
    }
}

// One warp owns one independent (head, token) score. The warp coalesces each
// 32-value key load, but lane zero consumes those values in source-lane order,
// preserving the host's exact column-0..column-N FMA association. History is
// therefore parallel without turning the dot itself into a parallel reduction.
__global__ void glm53_mla_scores_bf16_kernel(
    const float* query, const __nv_bfloat16* expanded, float* scores,
    std::uint32_t history, std::uint32_t heads, std::uint32_t head_dim) {
    constexpr std::uint32_t warps_per_block = 8U;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto token = static_cast<std::uint32_t>(blockIdx.y) *
                           warps_per_block + warp;
    if (head >= heads || token >= history) return;
    const auto* q = query + static_cast<std::uint64_t>(head) * head_dim;
    const auto* key = expanded +
        (static_cast<std::uint64_t>(token) * heads + head) *
            (2U * head_dim);
    float score = 0.0F;
    for (std::uint32_t begin = 0U; begin < head_dim; begin += warpSize) {
        const auto column = begin + lane;
        const float loaded = column < head_dim
            ? __bfloat162float(key[column]) : 0.0F;
#pragma unroll
        for (std::uint32_t source = 0U; source < warpSize; ++source) {
            const float key_value = __shfl_sync(
                0xFFFF'FFFFU, loaded, static_cast<int>(source));
            if (lane == 0U && begin + source < head_dim) {
                score = __fmaf_rn(q[begin + source], key_value, score);
            }
        }
    }
    if (lane == 0U) {
        const float score_scale =
            1.0F / sqrtf(static_cast<float>(head_dim));
        scores[static_cast<std::uint64_t>(head) * history + token] =
            score * score_scale;
    }
}

// The value-weighted sum, from coefficients the host has already exponentiated,
// normalized and rounded to BF16.
__global__ void glm53_mla_weighted_kernel(
    const float* coefficients, const float* expanded, float* attended,
    std::uint32_t history, std::uint32_t heads, std::uint32_t head_dim) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= heads) return;
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += blockDim.x) {
        // Sequential over tokens, contracted, exactly as the host accumulates
        // `destination[column] += coefficient * values[column]`.
        float value = 0.0F;
        for (std::uint32_t token = 0U; token < history; ++token) {
            const auto* values = expanded +
                (static_cast<std::uint64_t>(token) * heads + head) *
                    (2U * head_dim) + head_dim;
            value += coefficients[
                static_cast<std::uint64_t>(head) * history + token] *
                values[column];
        }
        attended[static_cast<std::uint64_t>(head) * head_dim + column] =
            glm53_bf16(value);
    }
}

__global__ void glm53_mla_weighted_bf16_kernel(
    const float* coefficients, const __nv_bfloat16* expanded,
    float* attended, std::uint32_t history, std::uint32_t heads,
    std::uint32_t head_dim) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= heads) return;
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += blockDim.x) {
        float value = 0.0F;
        for (std::uint32_t token = 0U; token < history; ++token) {
            const auto* values = expanded +
                (static_cast<std::uint64_t>(token) * heads + head) *
                    (2U * head_dim) + head_dim;
            value = __fmaf_rn(
                coefficients[
                    static_cast<std::uint64_t>(head) * history + token],
                __bfloat162float(values[column]), value);
        }
        attended[static_cast<std::uint64_t>(head) * head_dim + column] =
            glm53_bf16(value);
    }
}

__global__ void glm53_mla_absorb_query_kernel(
    const float* query, const __nv_bfloat16* key_value_weights,
    float* compressed, std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t latent_dim) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(heads) * latent_dim;
    if (index >= elements) return;
    const auto head = static_cast<std::uint32_t>(index / latent_dim);
    const auto latent = static_cast<std::uint32_t>(index % latent_dim);
    float value = 0.0F;
    for (std::uint32_t column = 0U; column < head_dim; ++column) {
        const auto weight_row =
            static_cast<std::uint64_t>(head) * 2U * head_dim + column;
        value = __fadd_rn(
            value, __fmul_rn(
                query[static_cast<std::uint64_t>(head) * head_dim + column],
                __bfloat162float(
                    key_value_weights[weight_row * latent_dim + latent])));
    }
    compressed[index] = value;
}

__global__ void glm53_mla_latent_attention_kernel(
    const float* compressed_query, const float* latent_cache,
    float* weighted_latent, std::uint32_t history, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t latent_dim) {
    extern __shared__ float scores[];
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= heads) return;
    if (threadIdx.x == 0U) {
        float highest = -INFINITY;
        for (std::uint32_t token = 0U; token < history; ++token) {
            float score = 0.0F;
            for (std::uint32_t column = 0U; column < latent_dim; ++column) {
                score = __fadd_rn(
                    score,
                    __fmul_rn(
                        compressed_query[
                            static_cast<std::uint64_t>(head) * latent_dim +
                            column],
                        latent_cache[
                            static_cast<std::uint64_t>(token) * latent_dim +
                            column]));
            }
            score *= rsqrtf(static_cast<float>(head_dim));
            scores[token] = score;
            highest = fmaxf(highest, score);
        }
        float total = 0.0F;
        for (std::uint32_t token = 0U; token < history; ++token) {
            scores[token] = expf(scores[token] - highest);
            total += scores[token];
        }
        for (std::uint32_t token = 0U; token < history; ++token) {
            scores[token] = glm53_bf16(scores[token] / total);
        }
    }
    __syncthreads();
    for (std::uint32_t column = threadIdx.x; column < latent_dim;
         column += blockDim.x) {
        float value = 0.0F;
        for (std::uint32_t token = 0U; token < history; ++token) {
            value += scores[token] *
                latent_cache[static_cast<std::uint64_t>(token) * latent_dim +
                             column];
        }
        weighted_latent[static_cast<std::uint64_t>(head) * latent_dim +
                        column] = value;
    }
}

__global__ void glm53_mla_expand_value_kernel(
    const float* weighted_latent, const __nv_bfloat16* key_value_weights,
    float* attended, std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t latent_dim) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(heads) * head_dim;
    if (index >= elements) return;
    const auto head = static_cast<std::uint32_t>(index / head_dim);
    const auto column = static_cast<std::uint32_t>(index % head_dim);
    const auto weight_row = static_cast<std::uint64_t>(head) * 2U * head_dim +
                            head_dim + column;
    float value = 0.0F;
    for (std::uint32_t latent = 0U; latent < latent_dim; ++latent) {
        value = __fadd_rn(
            value,
            __fmul_rn(
                weighted_latent[
                    static_cast<std::uint64_t>(head) * latent_dim + latent],
                __bfloat162float(
                    key_value_weights[weight_row * latent_dim + latent])));
    }
    attended[index] = glm53_bf16(value);
}

enum class Glm53KernelCategory : std::uint8_t {
    Kda,
    Mla,
    Expert,
    Other,
};

template <typename DeviceState>
cudaError_t glm53_kernel_timing_begin(DeviceState& state, bool enabled,
                                      Glm53KernelCategory category) {
    if (!enabled) return cudaSuccess;
    if (state.glm53_kernel_timing_active ||
        state.glm53_kernel_timing_count >=
            state.kGlm53KernelTimingCapacity) {
        return cudaErrorInvalidValue;
    }
    const auto index = state.glm53_kernel_timing_count;
    state.glm53_kernel_category[index] =
        static_cast<std::uint8_t>(category);
    if (auto status = cudaEventRecord(state.glm53_kernel_started[index],
                                      state.stream);
        status != cudaSuccess) {
        return status;
    }
    state.glm53_kernel_timing_active = true;
    return cudaSuccess;
}

template <typename DeviceState>
cudaError_t glm53_kernel_timing_end(DeviceState& state, bool enabled) {
    if (!enabled) return cudaSuccess;
    if (!state.glm53_kernel_timing_active) return cudaErrorInvalidValue;
    const auto index = state.glm53_kernel_timing_count;
    if (auto status = cudaEventRecord(state.glm53_kernel_finished[index],
                                      state.stream);
        status != cudaSuccess) {
        state.glm53_kernel_timing_active = false;
        return status;
    }
    state.glm53_kernel_timing_active = false;
    ++state.glm53_kernel_timing_count;
    return cudaSuccess;
}

// Drain only after the stream is known complete (a natural command boundary
// or stats()). The spans deliberately enclose consecutive direct kernels, not
// transfers, so launch bookkeeping never introduces a synchronize of its own.
template <typename Impl, typename DeviceState>
cudaError_t glm53_kernel_timing_drain(Impl& impl, DeviceState& state,
                                     int device, bool wait) {
    if (!impl.detailed_timing || state.glm53_kernel_timing_count == 0U) {
        return cudaSuccess;
    }
    if (state.glm53_kernel_timing_active) return cudaErrorInvalidValue;
    if (wait) {
        if (auto status = cudaEventSynchronize(
                state.glm53_kernel_finished[
                    state.glm53_kernel_timing_count - 1U]);
            status != cudaSuccess) {
            return status;
        }
    }
    std::array<std::uint64_t, 4U> categories{};
    for (std::uint32_t index = 0U;
         index < state.glm53_kernel_timing_count; ++index) {
        float milliseconds = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &milliseconds, state.glm53_kernel_started[index],
                state.glm53_kernel_finished[index]);
            status != cudaSuccess) {
            return status;
        }
        const auto nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(milliseconds) * 1.0e6));
        categories[state.glm53_kernel_category[index]] += nanoseconds;
    }
    const auto total = categories[0U] + categories[1U] + categories[2U] +
                       categories[3U];
    {
        std::scoped_lock lock(impl.mutex);
        auto& stats = *std::find_if(
            impl.stats.devices.begin(), impl.stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.kernel_nanoseconds += total;
        stats.glm53_kda_kernel_nanoseconds += categories[0U];
        stats.glm53_mla_kernel_nanoseconds += categories[1U];
        stats.glm53_expert_kernel_nanoseconds += categories[2U];
        stats.glm53_other_kernel_nanoseconds += categories[3U];
    }
    state.glm53_kernel_timing_count = 0U;
    return cudaSuccess;
}

}  // namespace

ValidationResult CudaBackend::glm53_kda_decode(
    const CudaGlm53KdaRequest& request, std::span<float> output) {
    ValidationResult result;
    if (request.state == nullptr || !request.state->valid() ||
        request.heads == 0U || request.head_dim == 0U ||
        request.head_dim > 256U || request.convolution_kernel < 2U ||
        request.page_rows == 0U ||
        (request.page_rows != 1U && !request.mhc_source_destination)) {
        return {{"CUDA GLM-5.3 KDA command is invalid"}};
    }
    const auto width = static_cast<std::uint64_t>(request.heads) *
                       request.head_dim;
    const bool full_layer = !request.input.empty() ||
                            request.mhc_source_destination;
    const auto projected = request.output_projection != nullptr;
    const auto projected_rows = projected && request.output_projection->valid()
        ? request.output_projection->impl_->descriptor.rows
        : 0U;
    if ((!full_layer &&
         (request.query.size() != width || request.key.size() != width ||
          request.value.size() != width || request.forget.size() != width ||
          request.gate.size() != width || request.beta.size() != request.heads)) ||
        ((!request.mhc_source_destination &&
          output.size() != (projected ? projected_rows : width)) ||
         (request.mhc_source_destination && !output.empty()))) {
        return {{"CUDA GLM-5.3 KDA operands have incompatible shapes"}};
    }
    const auto recurrent_floats = width * request.head_dim;
    const auto convolution_floats = 3ULL * width *
                                    (request.convolution_kernel - 1U);
    const auto tap_floats = 3ULL * width * request.convolution_kernel;
    const auto required_state_floats = recurrent_floats + convolution_floats +
        tap_floats + request.heads + width + request.head_dim;
    if (required_state_floats >
        std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        request.state->device_bytes() < required_state_floats * sizeof(float)) {
        return {{"CUDA GLM-5.3 KDA persistent state has an invalid extent"}};
    }
    const int device = request.state->device();
    const auto projection_encoding = projected
        ? request.output_projection->impl_->descriptor.encoding
        : CudaWeightEncoding::Plain;
    const auto plain_bf16_projection = projected &&
        projection_encoding == CudaWeightEncoding::Plain &&
        request.output_projection->impl_->descriptor.dtype ==
            SafetensorsDtype::Bf16;
    const auto fp8_projection = projected &&
        projection_encoding == CudaWeightEncoding::Fp8E4m3Block128F32 &&
        request.output_projection->impl_->fragment_prepacked;
    if (projected &&
        (request.output_projection->device() != device ||
         request.output_projection->impl_->descriptor.columns != width ||
         (!plain_bf16_projection && !fp8_projection))) {
        return {{"CUDA GLM-5.3 KDA output projection is not resident in the "
                 "required register-fed layout"}};
    }
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"CUDA GLM-5.3 KDA state targets an uninitialized device"}};
    }
    auto& device_state = found->second;
    if (device_state.moe_in_flight) {
        return {{"CUDA GLM-5.3 KDA cannot overlap an in-flight MoE command"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for GLM-5.3 KDA");
    }
    if (full_layer) {
        const auto hidden = projected_rows;
        const auto workspace_floats = 2ULL * hidden + 6ULL * width +
                                      2ULL * request.head_dim + request.heads;
        if (!projected ||
            (!request.mhc_source_destination &&
             request.input.size() != hidden) ||
            (request.mhc_source_destination && !request.input.empty()) ||
            (request.page_rows == 1U &&
             required_state_floats + workspace_floats >
                 request.state->device_bytes() / sizeof(float))) {
            return {{"CUDA GLM-5.3 fused KDA layer workspace is invalid"}};
        }
        const std::array<const CudaWeight*, 9U> projections{
            request.query_projection, request.key_projection,
            request.value_projection, request.forget_a_projection,
            request.beta_projection, request.gate_a_projection,
            request.forget_b_projection, request.gate_b_projection,
            request.output_projection};
        if (std::any_of(projections.begin(), projections.end(),
                        [device](const CudaWeight* weight) {
                            if (weight == nullptr || !weight->valid() ||
                                weight->device() != device) {
                                return true;
                            }
                            const auto& descriptor = weight->impl_->descriptor;
                            const auto plain_bf16 =
                                descriptor.encoding == CudaWeightEncoding::Plain &&
                                descriptor.dtype == SafetensorsDtype::Bf16;
                            const auto regfed_fp8 =
                                descriptor.encoding ==
                                    CudaWeightEncoding::Fp8E4m3Block128F32 &&
                                weight->impl_->fragment_prepacked;
                            return !plain_bf16 && !regfed_fp8;
                        })) {
            return {{"CUDA GLM-5.3 fused KDA layer weights are not resident "
                     "BF16 or register-fed FP8 tensors"}};
        }
        const auto shape = [](const CudaWeight* weight, std::uint64_t rows,
                              std::uint64_t columns) {
            return weight->impl_->descriptor.rows == rows &&
                   weight->impl_->descriptor.columns == columns;
        };
        if (!shape(request.query_projection, width, hidden) ||
            !shape(request.key_projection, width, hidden) ||
            !shape(request.value_projection, width, hidden) ||
            !shape(request.forget_a_projection, request.head_dim, hidden) ||
            !shape(request.beta_projection, request.heads, hidden) ||
            !shape(request.gate_a_projection, request.head_dim, hidden) ||
            !shape(request.forget_b_projection, width, request.head_dim) ||
            !shape(request.gate_b_projection, width, request.head_dim) ||
            !shape(request.output_projection, hidden, width)) {
            return {{"CUDA GLM-5.3 fused KDA layer weight shapes are invalid"}};
        }
        if (request.mhc_source_destination &&
            (!device_state.dsv4_mhc_supported ||
             device_state.dsv4_mhc_stage != 1U ||
             device_state.dsv4_mhc_workspace == nullptr ||
             device_state.dsv4_mhc_branch_ready ||
             device_state.dsv4_mhc_failed)) {
            return {{"CUDA GLM-5.3 fused KDA mHC command order is invalid"}};
        }
        if (request.page_rows > 1U) {
            const auto rows = request.page_rows;
            if (device_state.dsv4_mhc_slot_arena == nullptr ||
                device_state.dsv4_mhc_slot_capacity < rows ||
                device_state.dsv4_mhc_saved_slots.size() < rows ||
                device_state.dsv4_mhc_active_slot != rows - 1U) {
                return {{"CUDA GLM-5.3 KDA page mHC slots are invalid"}};
            }
            for (std::uint32_t row = 0U; row + 1U < rows; ++row) {
                const auto& slot = device_state.dsv4_mhc_saved_slots[row];
                if (slot.stage != 1U || slot.branch_ready) {
                    return {{"CUDA GLM-5.3 KDA page mHC order is invalid"}};
                }
            }
            const auto page_workspace_floats =
                static_cast<std::uint64_t>(rows) * workspace_floats;
            if (page_workspace_floats >
                device_state.output_bytes / sizeof(float)) {
                return {{"CUDA GLM-5.3 KDA page exceeds its admitted workspace"}};
            }
            auto* page = device_state.output;
            auto* device_input = page;
            auto* query = device_input + static_cast<std::uint64_t>(rows) * hidden;
            auto* key = query + static_cast<std::uint64_t>(rows) * width;
            auto* value = key + static_cast<std::uint64_t>(rows) * width;
            auto* forget = value + static_cast<std::uint64_t>(rows) * width;
            auto* gate = forget + static_cast<std::uint64_t>(rows) * width;
            auto* beta = gate + static_cast<std::uint64_t>(rows) * width;
            auto* forget_low = beta +
                static_cast<std::uint64_t>(rows) * request.heads;
            auto* gate_low = forget_low +
                static_cast<std::uint64_t>(rows) * request.head_dim;
            auto* heads_output = gate_low +
                static_cast<std::uint64_t>(rows) * request.head_dim;
            auto* final_output = heads_output +
                static_cast<std::uint64_t>(rows) * width;
            constexpr unsigned int threads = 256U;
            const auto input_elements =
                static_cast<std::uint64_t>(rows) * hidden;
            if (auto status = glm53_kernel_timing_begin(
                    device_state, impl_->detailed_timing,
                    Glm53KernelCategory::Kda);
                status != cudaSuccess) {
                return cuda_error(status,
                                  "start GLM-5.3 KDA page kernel timing");
            }
            glm53_mhc_gather_page_input_kernel<<<
                static_cast<unsigned int>((input_elements + threads - 1U) /
                                          threads),
                threads, 0U, device_state.stream>>>(
                device_state.dsv4_mhc_slot_arena, device_input, rows,
                static_cast<std::uint32_t>(hidden));
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(status, "gather GLM-5.3 KDA page input");
            }
            const auto project_page = [&](const CudaWeight* weight,
                                          const float* source,
                                          float* destination,
                                          std::uint64_t output_rows,
                                          const char* operation)
                -> ValidationResult {
                const auto& descriptor = weight->impl_->descriptor;
                if (descriptor.encoding == CudaWeightEncoding::Plain) {
                    constexpr unsigned int warps = threads / 32U;
                    const dim3 grid(
                        static_cast<unsigned int>(
                            (output_rows + warps - 1U) / warps),
                        static_cast<unsigned int>(
                            (rows + kBf16MatvecRowTile - 1U) /
                            kBf16MatvecRowTile),
                        1U);
                    bf16_matvec_rows_kernel<kBf16MatvecRowTile><<<
                        grid, threads, 0U, device_state.stream>>>(
                        destination, source,
                        static_cast<const __nv_bfloat16*>(
                            weight->impl_->weights),
                        rows, descriptor.columns, output_rows);
                } else if (auto status = launch_regfed_fp8_f32_rows(
                               device_state.moe_regfed, descriptor,
                               weight->impl_->weights,
                               weight->impl_->scales,
                               weight->impl_->fragment_prepacked, source,
                               destination, rows, device_state.stream);
                           status != cudaSuccess) {
                    return cuda_error(status, operation);
                }
                const auto elements =
                    static_cast<std::uint64_t>(rows) * output_rows;
                round_bf16_rows_kernel<<<
                    static_cast<unsigned int>((elements + threads - 1U) /
                                              threads),
                    threads, 0U, device_state.stream>>>(destination, elements);
                if (auto status = cudaGetLastError(); status != cudaSuccess) {
                    return cuda_error(status, operation);
                }
                return {};
            };
            for (const auto& command : std::array{
                     std::tuple{request.query_projection, device_input, query,
                                width, "project GLM-5.3 KDA page query"},
                     std::tuple{request.key_projection, device_input, key,
                                width, "project GLM-5.3 KDA page key"},
                     std::tuple{request.value_projection, device_input, value,
                                width, "project GLM-5.3 KDA page value"},
                     std::tuple{request.forget_a_projection, device_input,
                                forget_low,
                                static_cast<std::uint64_t>(request.head_dim),
                                "project GLM-5.3 KDA page forget A"},
                     std::tuple{request.beta_projection, device_input, beta,
                                static_cast<std::uint64_t>(request.heads),
                                "project GLM-5.3 KDA page beta"},
                     std::tuple{request.gate_a_projection, device_input,
                                gate_low,
                                static_cast<std::uint64_t>(request.head_dim),
                                "project GLM-5.3 KDA page gate A"}}) {
                auto projected_result = std::apply(project_page, command);
                if (!projected_result.ok()) return projected_result;
            }
            auto projected_result = project_page(
                request.forget_b_projection, forget_low, forget, width,
                "project GLM-5.3 KDA page forget B");
            if (!projected_result.ok()) return projected_result;
            projected_result = project_page(
                request.gate_b_projection, gate_low, gate, width,
                "project GLM-5.3 KDA page gate B");
            if (!projected_result.ok()) return projected_result;
            const auto beta_elements =
                static_cast<std::uint64_t>(rows) * request.heads;
            glm53_kda_beta_kernel<<<
                static_cast<unsigned int>((beta_elements + threads - 1U) /
                                          threads),
                threads, 0U, device_state.stream>>>(
                beta, static_cast<std::uint32_t>(beta_elements));
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(status, "activate GLM-5.3 KDA page beta");
            }
            auto* packed = static_cast<float*>(request.state->impl_->data);
            auto* recurrent = packed;
            auto* convolution = recurrent + recurrent_floats;
            const auto* taps = convolution + convolution_floats;
            const auto* a_log = taps + tap_floats;
            const auto* dt_bias = a_log + request.heads;
            const auto* norm_weight = dt_bias + width;
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto wide_offset = static_cast<std::uint64_t>(row) * width;
                glm53_kda_conv_page_row_kernel<<<
                    static_cast<unsigned int>((width + threads - 1U) / threads),
                    threads, 0U, device_state.stream>>>(
                    query + wide_offset, key + wide_offset,
                    value + wide_offset, convolution, taps,
                    static_cast<std::uint32_t>(width),
                    request.convolution_kernel);
                glm53_kda_recurrence_page_row_kernel<<<
                    request.heads, request.head_dim,
                    static_cast<std::size_t>(4U * request.head_dim *
                                             sizeof(float)),
                    device_state.stream>>>(
                    recurrent, a_log, dt_bias, norm_weight,
                    query + wide_offset, key + wide_offset,
                    value + wide_offset, forget + wide_offset,
                    beta + static_cast<std::uint64_t>(row) * request.heads,
                    gate + wide_offset, heads_output + wide_offset,
                    request.heads, request.head_dim);
            }
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(status, "launch GLM-5.3 KDA page recurrence");
            }
            projected_result = project_page(
                request.output_projection, heads_output, final_output, hidden,
                "project GLM-5.3 KDA page output");
            if (!projected_result.ok()) return projected_result;
            glm53_mhc_scatter_page_branch_kernel<<<
                static_cast<unsigned int>((input_elements + threads - 1U) /
                                          threads),
                threads, 0U, device_state.stream>>>(
                final_output, device_state.dsv4_mhc_slot_arena, rows,
                static_cast<std::uint32_t>(hidden));
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(status, "publish GLM-5.3 KDA page branch");
            }
            if (auto status = glm53_kernel_timing_end(
                    device_state, impl_->detailed_timing);
                status != cudaSuccess) {
                return cuda_error(status,
                                  "finish GLM-5.3 KDA page kernel timing");
            }
            for (std::uint32_t row = 0U; row + 1U < rows; ++row) {
                device_state.dsv4_mhc_saved_slots[row].branch_ready = true;
            }
            device_state.dsv4_mhc_branch_ready = true;
            return {};
        }
        const auto input_bytes = hidden * sizeof(float);
        const auto output_bytes = hidden * sizeof(float);
        const auto grow_pinned = [](std::byte*& pointer,
                                    std::uint64_t& capacity,
                                    std::uint64_t required) -> cudaError_t {
            if (required <= capacity) return cudaSuccess;
            void* replacement = nullptr;
            if (auto status = cudaMallocHost(&replacement, required);
                status != cudaSuccess) return status;
            if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
            pointer = static_cast<std::byte*>(replacement);
            capacity = required;
            return cudaSuccess;
        };
        if (auto status = grow_pinned(
                device_state.matmul_host_input,
                device_state.matmul_host_input_bytes, input_bytes);
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate fused GLM-5.3 KDA input staging");
        }
        if (auto status = grow_pinned(
                device_state.matmul_host_output,
                device_state.matmul_host_output_bytes, output_bytes);
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate fused GLM-5.3 KDA output staging");
        }
        if (!request.mhc_source_destination) {
            std::memcpy(device_state.matmul_host_input, request.input.data(),
                        input_bytes);
        }
        auto* packed = static_cast<float*>(request.state->impl_->data);
        auto* workspace = packed + required_state_floats;
        auto* device_input = workspace;
        auto* activations = device_input + hidden;
        auto* query = activations;
        auto* key = query + width;
        auto* value = key + width;
        auto* forget = value + width;
        auto* gate = forget + width;
        auto* beta = gate + width;
        auto* forget_low = beta + request.heads;
        auto* gate_low = forget_low + request.head_dim;
        auto* heads_output = gate_low + request.head_dim;
        auto* final_output = heads_output + width;
        if (request.mhc_source_destination) {
            if (auto status = glm53_kernel_timing_begin(
                    device_state, impl_->detailed_timing,
                    Glm53KernelCategory::Kda);
                status != cudaSuccess) {
                return cuda_error(status,
                                  "start resident GLM-5.3 KDA kernel timing");
            }
            constexpr std::uint32_t threads = 256U;
            constexpr std::uint32_t blocks =
                (kDsv4MhcHidden + threads - 1U) / threads;
            dsv4_bf16_to_fp32<<<blocks, threads, 0U, device_state.stream>>>(
                device_state.dsv4_mhc_workspace->layer_input, device_input,
                kDsv4MhcHidden);
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(status,
                                  "convert resident GLM-5.3 KDA input");
            }
        } else if (auto status = cudaMemcpyAsync(
                       device_input, device_state.matmul_host_input,
                       input_bytes, cudaMemcpyHostToDevice,
                       device_state.stream);
                   status != cudaSuccess) {
            return cuda_error(status,
                              "upload fused GLM-5.3 KDA layer input");
        }
        if (!request.mhc_source_destination) {
            if (auto status = glm53_kernel_timing_begin(
                    device_state, impl_->detailed_timing,
                    Glm53KernelCategory::Kda);
                status != cudaSuccess) {
                return cuda_error(status,
                                  "start fused GLM-5.3 KDA kernel timing");
            }
        }
        const auto project = [&](const CudaWeight* weight,
                                 float* source, float* destination,
                                 std::uint64_t rows,
                                 const char* operation) -> ValidationResult {
            auto& projection = *weight->impl_;
            const auto plain_bf16 =
                projection.descriptor.encoding == CudaWeightEncoding::Plain &&
                projection.descriptor.dtype == SafetensorsDtype::Bf16;
            if (plain_bf16) {
                constexpr unsigned int threads = 256U;
                constexpr unsigned int warps_per_block = threads / 32U;
                const auto blocks = static_cast<unsigned int>(
                    (rows + warps_per_block - 1U) / warps_per_block);
                bf16_matvec_kernel<<<blocks, threads, 0U,
                    device_state.stream>>>(
                    destination, source,
                    static_cast<const __nv_bfloat16*>(projection.weights),
                    projection.descriptor.columns, rows);
                if (auto status = cudaGetLastError(); status != cudaSuccess) {
                    return cuda_error(status, operation);
                }
            } else {
                if (auto status = launch_regfed_fp8_f32_rows(
                        device_state.moe_regfed, projection.descriptor,
                        projection.weights, projection.scales,
                        projection.fragment_prepacked, source, destination,
                        1U, device_state.stream);
                    status != cudaSuccess) {
                    return cuda_error(status, operation);
                }
            }
            round_bf16_rows_kernel<<<
                static_cast<unsigned int>((rows + 255U) / 256U), 256U, 0U,
                device_state.stream>>>(destination, rows);
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(status, operation);
            }
            return {};
        };
        for (const auto& command : std::array{
                 std::tuple{request.query_projection, device_input, query,
                            width, "project fused GLM-5.3 KDA query"},
                 std::tuple{request.key_projection, device_input, key, width,
                            "project fused GLM-5.3 KDA key"},
                 std::tuple{request.value_projection, device_input, value,
                            width, "project fused GLM-5.3 KDA value"},
                 std::tuple{request.forget_a_projection, device_input,
                            forget_low,
                            static_cast<std::uint64_t>(request.head_dim),
                            "project fused GLM-5.3 KDA forget A"},
                 std::tuple{request.beta_projection, device_input, beta,
                            static_cast<std::uint64_t>(request.heads),
                            "project fused GLM-5.3 KDA beta"},
                 std::tuple{request.gate_a_projection, device_input, gate_low,
                            static_cast<std::uint64_t>(request.head_dim),
                            "project fused GLM-5.3 KDA gate A"}}) {
            auto projected_result = std::apply(project, command);
            if (!projected_result.ok()) return projected_result;
        }
        auto projected_result = project(
            request.forget_b_projection, forget_low, forget, width,
            "project fused GLM-5.3 KDA forget B");
        if (!projected_result.ok()) return projected_result;
        projected_result = project(
            request.gate_b_projection, gate_low, gate, width,
            "project fused GLM-5.3 KDA gate B");
        if (!projected_result.ok()) return projected_result;
        glm53_kda_beta_kernel<<<1U, 128U, 0U, device_state.stream>>>(
            beta, request.heads);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status, "activate fused GLM-5.3 KDA beta");
        }
        auto* recurrent = packed;
        auto* convolution = recurrent + recurrent_floats;
        const auto* taps = convolution + convolution_floats;
        const auto* a_log = taps + tap_floats;
        const auto* dt_bias = a_log + request.heads;
        const auto* norm_weight = dt_bias + width;
        constexpr std::uint32_t threads = 256U;
        glm53_kda_conv_kernel<<<
            static_cast<unsigned int>((width + threads - 1U) / threads),
            threads, 0U, device_state.stream>>>(
            activations, convolution, taps,
            static_cast<std::uint32_t>(width), request.convolution_kernel);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status,
                              "launch fused GLM-5.3 KDA convolution");
        }
        glm53_kda_recurrence_kernel<<<
            request.heads, request.head_dim,
            static_cast<std::size_t>(4U * request.head_dim * sizeof(float)),
            device_state.stream>>>(
            recurrent, a_log, dt_bias, norm_weight, activations,
            heads_output, request.heads, request.head_dim);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status,
                              "launch fused GLM-5.3 KDA recurrence");
        }
        projected_result = project(
            request.output_projection, heads_output, final_output, hidden,
            "project fused GLM-5.3 KDA output");
        if (!projected_result.ok()) return projected_result;
        if (request.mhc_source_destination) {
            constexpr std::uint32_t threads = 256U;
            constexpr std::uint32_t blocks =
                (kDsv4MhcHidden + threads - 1U) / threads;
            dsv4_fp32_to_bf16<<<blocks, threads, 0U, device_state.stream>>>(
                final_output, device_state.dsv4_mhc_workspace->branch,
                kDsv4MhcHidden);
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(status,
                                  "publish resident GLM-5.3 KDA branch");
            }
            if (auto status = glm53_kernel_timing_end(
                    device_state, impl_->detailed_timing);
                status != cudaSuccess) {
                return cuda_error(status,
                                  "finish resident GLM-5.3 KDA kernel timing");
            }
            device_state.dsv4_mhc_branch_ready = true;
            return {};
        }
        if (auto status = glm53_kernel_timing_end(
                device_state, impl_->detailed_timing);
            status != cudaSuccess) {
            return cuda_error(status,
                              "finish fused GLM-5.3 KDA kernel timing");
        }
        if (auto status = cudaMemcpyAsync(
                device_state.matmul_host_output, final_output, output_bytes,
                cudaMemcpyDeviceToHost, device_state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "download fused GLM-5.3 KDA layer output");
        }
        const auto wait_started = std::chrono::steady_clock::now();
        if (auto status = cudaStreamSynchronize(device_state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "synchronize fused GLM-5.3 KDA layer");
        }
        if (auto status = glm53_kernel_timing_drain(
                *impl_, device_state, device, false);
            status != cudaSuccess) {
            return cuda_error(status, "measure fused GLM-5.3 KDA kernels");
        }
        std::memcpy(output.data(), device_state.matmul_host_output,
                    output_bytes);
        const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
        {
            std::scoped_lock lock(impl_->mutex);
            auto& stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [device](const auto& value) { return value.device == device; });
            stats.activation_h2d_bytes += input_bytes;
            stats.activation_d2h_bytes += output_bytes;
            record_synchronization(stats, SynchronizationSubsystem::Other,
                                   1U, wait_nanoseconds);
        }
        return {};
    }
    const auto input_floats = 5ULL * width + request.heads;
    const auto input_bytes = input_floats * sizeof(float);
    const auto output_bytes = (projected ? projected_rows : width) *
                              sizeof(float);
    if (input_bytes > device_state.input_bytes) {
        if (device_state.input != nullptr) {
            static_cast<void>(cudaFree(device_state.input));
        }
        if (auto status = cudaMalloc(&device_state.input, input_bytes);
            status != cudaSuccess) {
            device_state.input = nullptr;
            device_state.input_bytes = 0U;
            return cuda_error(status, "allocate GLM-5.3 KDA input workspace");
        }
        device_state.input_bytes = input_bytes;
    }
    if (output_bytes > device_state.output_bytes) {
        if (device_state.output != nullptr) {
            static_cast<void>(cudaFree(device_state.output));
        }
        if (auto status = cudaMalloc(&device_state.output, output_bytes);
            status != cudaSuccess) {
            device_state.output = nullptr;
            device_state.output_bytes = 0U;
            return cuda_error(status, "allocate GLM-5.3 KDA output workspace");
        }
        device_state.output_bytes = output_bytes;
    }
    const auto grow_pinned = [](std::byte*& pointer, std::uint64_t& capacity,
                                std::uint64_t required) -> cudaError_t {
        if (required <= capacity) return cudaSuccess;
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(&replacement, required);
            status != cudaSuccess) return status;
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = required;
        return cudaSuccess;
    };
    if (auto status = grow_pinned(device_state.matmul_host_input,
                                  device_state.matmul_host_input_bytes,
                                  input_bytes);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 KDA input staging");
    }
    if (auto status = grow_pinned(device_state.matmul_host_output,
                                  device_state.matmul_host_output_bytes,
                                  output_bytes);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 KDA output staging");
    }
    auto* staged = reinterpret_cast<float*>(device_state.matmul_host_input);
    std::uint64_t cursor = 0U;
    for (const auto values : {request.query, request.key, request.value,
                              request.forget, request.gate}) {
        std::copy(values.begin(), values.end(), staged + cursor);
        cursor += width;
    }
    std::copy(request.beta.begin(), request.beta.end(), staged + cursor);
    if (auto status = cudaMemcpyAsync(
            device_state.input, staged, input_bytes, cudaMemcpyHostToDevice,
            device_state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload GLM-5.3 KDA activations");
    }
    if (auto status = glm53_kernel_timing_begin(
            device_state, impl_->detailed_timing,
            Glm53KernelCategory::Kda);
        status != cudaSuccess) {
        return cuda_error(status, "start GLM-5.3 KDA kernel timing");
    }
    auto* packed = static_cast<float*>(request.state->impl_->data);
    auto* recurrent = packed;
    auto* convolution = recurrent + recurrent_floats;
    const auto* taps = convolution + convolution_floats;
    const auto* a_log = taps + tap_floats;
    const auto* dt_bias = a_log + request.heads;
    const auto* norm_weight = dt_bias + width;
    constexpr std::uint32_t threads = 256U;
    glm53_kda_conv_kernel<<<
        static_cast<unsigned int>((width + threads - 1U) / threads), threads,
        0U, device_state.stream>>>(
        device_state.input, convolution, taps, static_cast<std::uint32_t>(width),
        request.convolution_kernel);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch GLM-5.3 fused short convolution");
    }
    glm53_kda_recurrence_kernel<<<
        request.heads, request.head_dim,
        static_cast<std::size_t>(4U * request.head_dim * sizeof(float)),
        device_state.stream>>>(
        recurrent, a_log, dt_bias, norm_weight, device_state.input,
        device_state.output, request.heads, request.head_dim);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch GLM-5.3 fused KDA recurrence");
    }
    const float* device_result = device_state.output;
    if (projected) {
        auto& projection = *request.output_projection->impl_;
        if (plain_bf16_projection) {
            constexpr unsigned int threads = 256U;
            constexpr unsigned int warps_per_block = threads / 32U;
            const auto blocks = static_cast<unsigned int>(
                (projected_rows + warps_per_block - 1U) / warps_per_block);
            bf16_matvec_kernel<<<blocks, threads, 0U, device_state.stream>>>(
                device_state.input, device_state.output,
                static_cast<const __nv_bfloat16*>(projection.weights),
                width, projected_rows);
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(
                    status, "launch GLM-5.3 fused BF16 KDA output projection");
            }
        } else {
            if (auto status = launch_regfed_fp8_f32_rows(
                    device_state.moe_regfed, projection.descriptor,
                    projection.weights, projection.scales,
                    projection.fragment_prepacked, device_state.output,
                    device_state.input, 1U, device_state.stream);
                status != cudaSuccess) {
                return cuda_error(
                    status, "launch GLM-5.3 fused FP8 KDA output projection");
            }
        }
        round_bf16_rows_kernel<<<
            static_cast<unsigned int>((projected_rows + 255U) / 256U), 256U,
            0U, device_state.stream>>>(device_state.input, projected_rows);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status,
                              "round GLM-5.3 KDA projected activation");
        }
        device_result = device_state.input;
    }
    if (auto status = glm53_kernel_timing_end(
            device_state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish GLM-5.3 KDA kernel timing");
    }
    if (auto status = cudaMemcpyAsync(
            device_state.matmul_host_output, device_result, output_bytes,
            cudaMemcpyDeviceToHost, device_state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 KDA activation");
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(device_state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize GLM-5.3 KDA recurrence");
    }
    if (auto status = glm53_kernel_timing_drain(
            *impl_, device_state, device, false);
        status != cudaSuccess) {
        return cuda_error(status, "measure GLM-5.3 KDA kernels");
    }
    std::memcpy(output.data(), device_state.matmul_host_output, output_bytes);
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.activation_h2d_bytes += input_bytes;
        stats.activation_d2h_bytes += output_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Other, 1U,
                               wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::glm53_mhc_router(
    int device, const CudaWeight& router, std::span<float> logits) {
    if (!router.valid() || router.device() != device || logits.size() != 288U) {
        return {{"CUDA GLM-5.3 resident router command is invalid"}};
    }
    const auto& descriptor = router.impl_->descriptor;
    if (descriptor.encoding != CudaWeightEncoding::Plain ||
        descriptor.dtype != SafetensorsDtype::Bf16 ||
        descriptor.rows != logits.size() ||
        descriptor.columns != kDsv4MhcHidden) {
        return {{"CUDA GLM-5.3 resident router requires a 288x4096 BF16 "
                 "projection"}};
    }
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"CUDA GLM-5.3 resident router targets an uninitialized "
                 "device"}};
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.dsv4_mhc_branch_ready ||
        state.dsv4_mhc_failed || state.moe_in_flight) {
        return {{"CUDA GLM-5.3 resident router violates mHC command order"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for GLM-5.3 resident router");
    }
    const auto bytes = logits.size_bytes();
    if (bytes > state.matmul_host_output_bytes) {
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(&replacement, bytes);
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate GLM-5.3 resident router staging");
        }
        if (state.matmul_host_output != nullptr) {
            static_cast<void>(cudaFreeHost(state.matmul_host_output));
        }
        state.matmul_host_output = static_cast<std::byte*>(replacement);
        state.matmul_host_output_bytes = bytes;
    }
    constexpr unsigned int threads = 256U;
    constexpr unsigned int warps_per_block = threads / 32U;
    const auto blocks = static_cast<unsigned int>(
        (descriptor.rows + warps_per_block - 1U) / warps_per_block);
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Other);
        status != cudaSuccess) {
        return cuda_error(status, "start GLM-5.3 router kernel timing");
    }
    bf16_input_matvec_kernel<<<blocks, threads, 0U, state.stream>>>(
        state.dsv4_mhc_workspace->glm53_router_logits,
        state.dsv4_mhc_workspace->layer_input,
        static_cast<const __nv_bfloat16*>(router.impl_->weights),
        descriptor.columns, descriptor.rows);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch GLM-5.3 resident router");
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish GLM-5.3 router kernel timing");
    }
    if (auto status = cudaMemcpyAsync(
            state.matmul_host_output,
            state.dsv4_mhc_workspace->glm53_router_logits, bytes,
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 resident router logits");
    }
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status,
                          "synchronize GLM-5.3 resident router logits");
    }
    if (auto status = glm53_kernel_timing_drain(
            *impl_, state, device, false);
        status != cudaSuccess) {
        return cuda_error(status, "measure GLM-5.3 router kernel");
    }
    std::memcpy(logits.data(), state.matmul_host_output, bytes);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.activation_d2h_bytes += bytes;
    }
    return {};
}

ValidationResult CudaBackend::glm53_mhc_swiglu(
    int device, const CudaWeight& gate, const CudaWeight& up,
    const CudaWeight& down, std::uint32_t intermediate) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || intermediate == 0U) {
        return {{"CUDA GLM-5.3 resident SwiGLU command is invalid"}};
    }
    auto& state = found->second;
    const auto valid = [device](const CudaWeight& weight,
                                std::uint64_t rows,
                                std::uint64_t columns) {
        if (!weight.valid() || weight.device() != device ||
            weight.impl_->descriptor.rows != rows ||
            weight.impl_->descriptor.columns != columns) return false;
        const auto& descriptor = weight.impl_->descriptor;
        return (descriptor.encoding == CudaWeightEncoding::Plain &&
                descriptor.dtype == SafetensorsDtype::Bf16) ||
               (descriptor.encoding ==
                    CudaWeightEncoding::Fp8E4m3Block128F32 &&
                weight.impl_->fragment_prepacked);
    };
    if (!valid(gate, intermediate, kDsv4MhcHidden) ||
        !valid(up, intermediate, kDsv4MhcHidden) ||
        !valid(down, kDsv4MhcHidden, intermediate) ||
        !state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.dsv4_mhc_branch_ready ||
        state.dsv4_mhc_failed || state.moe_in_flight) {
        return {{"CUDA GLM-5.3 resident SwiGLU weights or command order are "
                 "invalid"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for GLM-5.3 resident SwiGLU");
    }
    const auto hidden_bytes =
        static_cast<std::uint64_t>(kDsv4MhcHidden) * sizeof(float);
    const auto activation_bytes =
        static_cast<std::uint64_t>(intermediate) * 2U * sizeof(float);
    const auto ensure = [&](float*& pointer, std::uint64_t& capacity,
                            std::uint64_t required) -> cudaError_t {
        if (capacity >= required) return cudaSuccess;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (auto status = cudaMalloc(&pointer, required);
            status != cudaSuccess) return status;
        capacity = required;
        return cudaSuccess;
    };
    if (auto status = ensure(state.moe_hidden, state.moe_hidden_bytes,
                             hidden_bytes); status != cudaSuccess) {
        return cuda_error(status, "allocate resident SwiGLU hidden buffer");
    }
    if (auto status = ensure(state.moe_activations,
                             state.moe_activation_bytes, activation_bytes);
        status != cudaSuccess) {
        return cuda_error(status,
                          "allocate resident SwiGLU activation buffer");
    }
    constexpr unsigned int threads = 256U;
    constexpr unsigned int warps = threads / 32U;
    constexpr unsigned int hidden_blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Other);
        status != cudaSuccess) {
        return cuda_error(status,
                          "start resident GLM-5.3 SwiGLU kernel timing");
    }
    dsv4_bf16_to_fp32<<<hidden_blocks, threads, 0U, state.stream>>>(
        state.dsv4_mhc_workspace->layer_input, state.moe_hidden,
        kDsv4MhcHidden);
    auto* gate_output = state.moe_activations;
    auto* up_output = gate_output + intermediate;
    const auto project = [&](const CudaWeight& weight, float* source,
                             float* destination,
                             std::uint64_t rows) -> cudaError_t {
        const auto& descriptor = weight.impl_->descriptor;
        if (descriptor.encoding == CudaWeightEncoding::Plain) {
            const auto blocks = static_cast<unsigned int>(
                (rows + warps - 1U) / warps);
            bf16_matvec_kernel<<<blocks, threads, 0U, state.stream>>>(
                destination, source,
                static_cast<const __nv_bfloat16*>(weight.impl_->weights),
                descriptor.columns, rows);
            return cudaGetLastError();
        }
        return launch_regfed_fp8_f32_rows(
            state.moe_regfed, descriptor, weight.impl_->weights,
            weight.impl_->scales, weight.impl_->fragment_prepacked,
            source, destination, 1U, state.stream);
    };
    if (auto status = project(gate, state.moe_hidden, gate_output,
                              intermediate); status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 SwiGLU gate");
    }
    if (auto status = project(up, state.moe_hidden, up_output, intermediate);
        status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 SwiGLU up");
    }
    round_bf16_rows_kernel<<<
        static_cast<unsigned int>((intermediate + threads - 1U) / threads),
        threads, 0U, state.stream>>>(gate_output, intermediate);
    round_bf16_rows_kernel<<<
        static_cast<unsigned int>((intermediate + threads - 1U) / threads),
        threads, 0U, state.stream>>>(up_output, intermediate);
    glm53_swiglu_kernel<<<
        static_cast<unsigned int>((intermediate + threads - 1U) / threads),
        threads, 0U, state.stream>>>(gate_output, up_output, intermediate);
    if (auto status = project(down, gate_output, state.moe_hidden,
                              kDsv4MhcHidden); status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 SwiGLU down");
    }
    round_bf16_rows_kernel<<<hidden_blocks, threads, 0U, state.stream>>>(
        state.moe_hidden, kDsv4MhcHidden);
    dsv4_fp32_to_bf16<<<hidden_blocks, threads, 0U, state.stream>>>(
        state.moe_hidden, state.dsv4_mhc_workspace->branch,
        kDsv4MhcHidden);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch resident GLM-5.3 SwiGLU");
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status,
                          "finish resident GLM-5.3 SwiGLU kernel timing");
    }
    state.dsv4_mhc_branch_ready = true;
    return {};
}

ValidationResult CudaBackend::glm53_mla_prepare_history(
    const CudaGlm53MlaRequest& request, std::uint32_t history) {
    if (request.state == nullptr || !request.state->valid() ||
        request.key_value_b == nullptr || !request.key_value_b->valid() ||
        request.maximum_context == 0U || history > request.maximum_context ||
        request.heads == 0U || request.head_dim == 0U ||
        request.query_rank == 0U || request.key_value_rank == 0U) {
        return {{"CUDA GLM-5.3 MLA history preparation is invalid"}};
    }
    const auto device = request.state->device();
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() ||
        request.key_value_b->device() != device) {
        return {{"CUDA GLM-5.3 MLA history targets an invalid device"}};
    }
    const auto width = static_cast<std::uint64_t>(request.heads) *
                       request.head_dim;
    const auto expanded_width = 2ULL * width;
    const auto cache_floats =
        static_cast<std::uint64_t>(request.maximum_context) *
        request.key_value_rank;
    const auto state_floats = cache_floats + request.query_rank +
                              request.key_value_rank;
    const auto required_bytes = state_floats * sizeof(float) +
        static_cast<std::uint64_t>(request.maximum_context) *
            expanded_width * sizeof(__nv_bfloat16);
    const auto& descriptor = request.key_value_b->impl_->descriptor;
    if (descriptor.encoding != CudaWeightEncoding::Plain ||
        descriptor.dtype != SafetensorsDtype::Bf16 ||
        descriptor.rows != expanded_width ||
        descriptor.columns != request.key_value_rank ||
        request.state->device_bytes() < required_bytes) {
        return {{"CUDA GLM-5.3 MLA history cache shape is invalid"}};
    }
    auto& state = found->second;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for MLA history preparation");
    }
    auto& buffer = *request.state->impl_;
    buffer.glm53_mla_expanded_rows = 0U;
    buffer.glm53_mla_maximum_context = request.maximum_context;
    buffer.glm53_mla_expanded_width = expanded_width;
    if (history == 0U) {
        if (auto status = cudaStreamSynchronize(state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "synchronize empty MLA history");
        }
        return {};
    }

    auto* packed = static_cast<float*>(buffer.data);
    auto* expanded = reinterpret_cast<__nv_bfloat16*>(
        packed + state_floats);
    constexpr unsigned int threads = 256U;
    constexpr unsigned int warps = threads / 32U;
    const dim3 grid(
        static_cast<unsigned int>((expanded_width + warps - 1U) / warps),
        static_cast<unsigned int>(
            (history + kBf16MatvecRowTile - 1U) /
            kBf16MatvecRowTile),
        1U);
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Mla);
        status != cudaSuccess) {
        return cuda_error(status, "start MLA history preparation timing");
    }
    bf16_matvec_rows_to_bf16_kernel<kBf16MatvecRowTile><<<
        grid, threads, 0U, state.stream>>>(
        expanded, packed,
        static_cast<const __nv_bfloat16*>(
            request.key_value_b->impl_->weights),
        history, request.key_value_rank, expanded_width);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "project GLM-5.3 MLA history");
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish MLA history preparation timing");
    }
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize MLA history preparation");
    }
    if (auto status = glm53_kernel_timing_drain(
            *impl_, state, device, false);
        status != cudaSuccess) {
        return cuda_error(status, "measure MLA history preparation");
    }
    buffer.glm53_mla_expanded_rows = history;
    return {};
}

ValidationResult CudaBackend::glm53_mla_decode_to_mhc(
    const CudaGlm53MlaRequest& request, std::span<float> scores) {
    if (request.state == nullptr || !request.state->valid() ||
        request.position >= request.maximum_context || request.heads == 0U ||
        request.head_dim == 0U || request.query_rank == 0U ||
        request.key_value_rank == 0U) {
        return {{"CUDA GLM-5.3 resident MLA command is invalid"}};
    }
    const auto device = request.state->device();
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"CUDA GLM-5.3 resident MLA targets an uninitialized device"}};
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.dsv4_mhc_branch_ready ||
        state.dsv4_mhc_failed || state.moe_in_flight) {
        return {{"CUDA GLM-5.3 resident MLA violates mHC command order"}};
    }
    const auto width = static_cast<std::uint64_t>(request.heads) *
                       request.head_dim;
    const auto expanded_width = 2ULL * width;
    const auto cache_floats =
        static_cast<std::uint64_t>(request.maximum_context) *
        request.key_value_rank;
    const auto state_floats = cache_floats + request.query_rank +
                              request.key_value_rank;
    if (request.state->device_bytes() < state_floats * sizeof(float)) {
        return {{"CUDA GLM-5.3 resident MLA state extent is invalid"}};
    }
    const auto valid = [device](const CudaWeight* weight,
                                std::uint64_t rows,
                                std::uint64_t columns) {
        if (weight == nullptr || !weight->valid() ||
            weight->device() != device ||
            weight->impl_->descriptor.rows != rows ||
            weight->impl_->descriptor.columns != columns) return false;
        const auto& descriptor = weight->impl_->descriptor;
        return (descriptor.encoding == CudaWeightEncoding::Plain &&
                descriptor.dtype == SafetensorsDtype::Bf16) ||
               (descriptor.encoding ==
                    CudaWeightEncoding::Fp8E4m3Block128F32 &&
                weight->impl_->fragment_prepacked);
    };
    if (!valid(request.query_a, request.query_rank, kDsv4MhcHidden) ||
        !valid(request.key_value_a, request.key_value_rank,
               kDsv4MhcHidden) ||
        !valid(request.query_b, width, request.query_rank) ||
        !valid(request.key_value_b, expanded_width,
               request.key_value_rank) ||
        !valid(request.output, kDsv4MhcHidden, width)) {
        return {{"CUDA GLM-5.3 resident MLA projection shapes are invalid"}};
    }
    const auto& kv_descriptor = request.key_value_b->impl_->descriptor;
    const auto history = static_cast<std::uint64_t>(request.position) + 1U;
    // Keep the accepted host fallback's projection/attention association.
    // Absorbing the KV-B projection into the query is algebraically equivalent,
    // but not bit-equivalent after the model's BF16 projection boundaries.
    // The resident path therefore expands the latent history exactly as the
    // fallback does before applying attention.
    //
    // BF16 KV-B is expanded once into an exact BF16 cache appended to this
    // layer's sequence state. FP8 KV-B retains the accepted F32 workspace path
    // because its register-fed projection currently publishes F32 rows.
    const auto reserved_history = std::max<std::uint64_t>(
        history, request.maximum_context);
    const auto persistent_cache_bytes = state_floats * sizeof(float) +
        static_cast<std::uint64_t>(request.maximum_context) *
            expanded_width * sizeof(__nv_bfloat16);
    const bool persistent_bf16 =
        kv_descriptor.encoding == CudaWeightEncoding::Plain &&
        kv_descriptor.dtype == SafetensorsDtype::Bf16 &&
        request.state->device_bytes() >= persistent_cache_bytes;
    if (persistent_bf16 &&
        (request.state->impl_->glm53_mla_maximum_context !=
             request.maximum_context ||
         request.state->impl_->glm53_mla_expanded_width != expanded_width)) {
        return {{"CUDA GLM-5.3 resident MLA cache was not prepared"}};
    }
    // The trailing `heads * reserved_history` holds the scores on the way out
    // and the normalized coefficients on the way back in.
    const auto workspace_floats = kDsv4MhcHidden + request.query_rank + width +
        (persistent_bf16 ? 0U : reserved_history * expanded_width) +
        width + kDsv4MhcHidden +
        static_cast<std::uint64_t>(request.heads) * reserved_history;
    const auto workspace_bytes = workspace_floats * sizeof(float);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for GLM-5.3 resident MLA");
    }
    if (workspace_bytes > state.glm53_mla_workspace_bytes) {
        if (state.glm53_mla_workspace != nullptr) {
            static_cast<void>(cudaFree(state.glm53_mla_workspace));
        }
        state.glm53_mla_workspace = nullptr;
        state.glm53_mla_workspace_bytes = 0U;
        if (auto status = cudaMalloc(&state.glm53_mla_workspace,
                                     workspace_bytes);
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate GLM-5.3 resident MLA workspace");
        }
        state.glm53_mla_workspace_bytes = workspace_bytes;
    }
    auto* packed = static_cast<float*>(request.state->impl_->data);
    auto* q_norm = packed + cache_floats;
    auto* kv_norm = q_norm + request.query_rank;
    auto* workspace = reinterpret_cast<float*>(state.glm53_mla_workspace);
    auto* input = workspace;
    auto* q_rank = input + kDsv4MhcHidden;
    auto* query = q_rank + request.query_rank;
    auto* expanded = query + width;
    auto* attended = expanded +
        (persistent_bf16 ? 0U : reserved_history * expanded_width);
    auto* output = attended + width;
    auto* coefficients = output + kDsv4MhcHidden;
    auto* expanded_bf16 = reinterpret_cast<__nv_bfloat16*>(
        packed + state_floats);
    auto* latent = packed + static_cast<std::uint64_t>(request.position) *
                                request.key_value_rank;
    constexpr unsigned int threads = 256U;
    constexpr unsigned int warps = threads / 32U;
    constexpr unsigned int hidden_blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Mla);
        status != cudaSuccess) {
        return cuda_error(status, "start resident GLM-5.3 MLA kernel timing");
    }
    dsv4_bf16_to_fp32<<<hidden_blocks, threads, 0U, state.stream>>>(
        state.dsv4_mhc_workspace->layer_input, input, kDsv4MhcHidden);
    const auto project_one = [&](const CudaWeight* weight, float* source,
                                 float* destination,
                                 std::uint64_t rows) -> cudaError_t {
        const auto& descriptor = weight->impl_->descriptor;
        if (descriptor.encoding == CudaWeightEncoding::Plain) {
            const auto blocks = static_cast<unsigned int>(
                (rows + warps - 1U) / warps);
            bf16_matvec_kernel<<<blocks, threads, 0U, state.stream>>>(
                destination, source,
                static_cast<const __nv_bfloat16*>(weight->impl_->weights),
                descriptor.columns, rows);
        } else if (auto status = launch_regfed_fp8_f32_rows(
                       state.moe_regfed, descriptor, weight->impl_->weights,
                       weight->impl_->scales,
                       weight->impl_->fragment_prepacked, source, destination,
                       1U, state.stream); status != cudaSuccess) {
            return status;
        }
        round_bf16_rows_kernel<<<
            static_cast<unsigned int>((rows + threads - 1U) / threads),
            threads, 0U, state.stream>>>(destination, rows);
        return cudaGetLastError();
    };
    if (auto status = project_one(request.query_a, input, q_rank,
                                  request.query_rank); status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 MLA query A");
    }
    if (auto status = project_one(request.key_value_a, input, latent,
                                  request.key_value_rank);
        status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 MLA KV A");
    }
    glm53_rms_norm_bf16_kernel<<<1U, threads, 0U, state.stream>>>(
        q_rank, q_norm, request.query_rank);
    glm53_rms_norm_bf16_kernel<<<1U, threads, 0U, state.stream>>>(
        latent, kv_norm, request.key_value_rank);
    if (auto status = project_one(request.query_b, q_rank, query, width);
        status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 MLA query B");
    }
    if (persistent_bf16) {
        auto& expanded_rows = request.state->impl_->glm53_mla_expanded_rows;
        if (expanded_rows > history) {
            return {{"CUDA GLM-5.3 MLA expansion cache is ahead of history"}};
        }
        const auto missing = static_cast<std::uint32_t>(
            history - expanded_rows);
        if (missing != 0U) {
            const dim3 kv_grid(
                static_cast<unsigned int>(
                    (expanded_width + warps - 1U) / warps),
                static_cast<unsigned int>(
                    (missing + kBf16MatvecRowTile - 1U) /
                    kBf16MatvecRowTile),
                1U);
            auto* expansion_destination = expanded_bf16 +
                static_cast<std::uint64_t>(expanded_rows) * expanded_width;
            const auto* expansion_source = packed +
                static_cast<std::uint64_t>(expanded_rows) *
                    request.key_value_rank;
            const auto* expansion_weights =
                static_cast<const __nv_bfloat16*>(
                    request.key_value_b->impl_->weights);
            if (missing == 1U) {
                // The generic tile intentionally fills every accumulator in a
                // partial tile. Decode appends exactly one row, so Tile=1
                // avoids computing that row sixteen times.
                bf16_matvec_rows_to_bf16_kernel<1U><<<
                    kv_grid, threads, 0U, state.stream>>>(
                    expansion_destination, expansion_source,
                    expansion_weights, missing, request.key_value_rank,
                    expanded_width);
            } else {
                bf16_matvec_rows_to_bf16_kernel<kBf16MatvecRowTile><<<
                    kv_grid, threads, 0U, state.stream>>>(
                    expansion_destination, expansion_source,
                    expansion_weights, missing, request.key_value_rank,
                    expanded_width);
            }
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(
                    status, "append GLM-5.3 MLA expansion cache");
            }
            expanded_rows = static_cast<std::uint32_t>(history);
        }
    } else if (kv_descriptor.encoding == CudaWeightEncoding::Plain) {
        const dim3 kv_grid(
            static_cast<unsigned int>((expanded_width + warps - 1U) / warps),
            static_cast<unsigned int>(
                (history + kBf16MatvecRowTile - 1U) /
                kBf16MatvecRowTile),
            1U);
        bf16_matvec_rows_kernel<kBf16MatvecRowTile><<<
            kv_grid, threads, 0U, state.stream>>>(
            expanded, packed,
            static_cast<const __nv_bfloat16*>(
                request.key_value_b->impl_->weights),
            static_cast<std::uint32_t>(history), request.key_value_rank,
            expanded_width);
    } else if (auto status = launch_regfed_fp8_f32_rows(
                   state.moe_regfed, kv_descriptor,
                   request.key_value_b->impl_->weights,
                   request.key_value_b->impl_->scales,
                   request.key_value_b->impl_->fragment_prepacked, packed,
                   expanded, static_cast<std::uint32_t>(history),
                   state.stream); status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 MLA KV B");
    }
    if (!persistent_bf16) {
        round_bf16_rows_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>(
                    (history * expanded_width + threads - 1U) / threads,
                    65535U)),
            threads, 0U, state.stream>>>(expanded,
                                         history * expanded_width);
    }
    // Stop at the scores and hand them back. The caller applies `exp`, the
    // normalization and the BF16 coefficient rounding on the host, then calls
    // `glm53_mla_decode_finish` to complete the layer.
    if (scores.size() != request.heads * history) {
        return {{"CUDA GLM-5.3 resident MLA score span has an invalid shape"}};
    }
    if (persistent_bf16) {
        constexpr unsigned int score_warps_per_block = threads / 32U;
        const dim3 score_grid(
            request.heads,
            static_cast<unsigned int>(
                (history + score_warps_per_block - 1U) /
                score_warps_per_block),
            1U);
        glm53_mla_scores_bf16_kernel<<<
            score_grid, threads, 0U, state.stream>>>(
            query, expanded_bf16, coefficients,
            static_cast<std::uint32_t>(history), request.heads,
            request.head_dim);
    } else {
        glm53_mla_scores_f32_kernel<<<
            request.heads, threads, 0U, state.stream>>>(
            query, expanded, coefficients,
            static_cast<std::uint32_t>(history), request.heads,
            request.head_dim);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch resident GLM-5.3 MLA scores");
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish resident GLM-5.3 MLA kernel timing");
    }
    if (auto status = cudaMemcpyAsync(
            scores.data(), coefficients, scores.size_bytes(),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download resident GLM-5.3 MLA scores");
    }
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "complete resident GLM-5.3 MLA scores");
    }
    if (auto status = glm53_kernel_timing_drain(
            *impl_, state, device, false);
        status != cudaSuccess) {
        return cuda_error(status, "measure resident GLM-5.3 MLA kernels");
    }
    state.glm53_mla_scores_pending = true;
    return {};
}

ValidationResult CudaBackend::glm53_mla_decode_finish(
    const CudaGlm53MlaRequest& request,
    std::span<const float> normalized_coefficients) {
    const auto device = request.state == nullptr ? -1 : request.state->device();
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"CUDA GLM-5.3 resident MLA targets an uninitialized device"}};
    }
    auto& state = found->second;
    if (!state.glm53_mla_scores_pending) {
        return {{"CUDA GLM-5.3 resident MLA finish has no pending scores"}};
    }
    const auto history = static_cast<std::uint64_t>(request.position) + 1U;
    if (normalized_coefficients.size() != request.heads * history) {
        return {{"CUDA GLM-5.3 resident MLA coefficient span is invalid"}};
    }
    const auto width = static_cast<std::uint64_t>(request.heads) *
                       request.head_dim;
    const auto expanded_width = 2ULL * width;
    if (request.key_value_b == nullptr || !request.key_value_b->valid()) {
        return {{"CUDA GLM-5.3 resident MLA finish is missing KV-B"}};
    }
    const auto cache_floats =
        static_cast<std::uint64_t>(request.maximum_context) *
        request.key_value_rank;
    const auto state_floats = cache_floats + request.query_rank +
                              request.key_value_rank;
    const auto persistent_cache_bytes = state_floats * sizeof(float) +
        static_cast<std::uint64_t>(request.maximum_context) *
            expanded_width * sizeof(__nv_bfloat16);
    const auto& kv_descriptor = request.key_value_b->impl_->descriptor;
    const bool persistent_bf16 =
        kv_descriptor.encoding == CudaWeightEncoding::Plain &&
        kv_descriptor.dtype == SafetensorsDtype::Bf16 &&
        request.state->device_bytes() >= persistent_cache_bytes;
    if (persistent_bf16 &&
        request.state->impl_->glm53_mla_expanded_rows < history) {
        return {{"CUDA GLM-5.3 MLA expansion cache is incomplete"}};
    }
    const auto reserved_history = std::max<std::uint64_t>(
        history, request.maximum_context);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for GLM-5.3 resident MLA");
    }
    auto* workspace = reinterpret_cast<float*>(state.glm53_mla_workspace);
    auto* query = workspace + kDsv4MhcHidden + request.query_rank;
    auto* expanded = query + width;
    auto* attended = expanded +
        (persistent_bf16 ? 0U : reserved_history * expanded_width);
    auto* output = attended + width;
    auto* coefficients = output + kDsv4MhcHidden;
    auto* packed = static_cast<float*>(request.state->impl_->data);
    auto* expanded_bf16 = reinterpret_cast<__nv_bfloat16*>(
        packed + state_floats);
    constexpr unsigned int threads = 256U;
    const auto hidden_blocks =
        static_cast<unsigned int>((kDsv4MhcHidden + threads - 1U) / threads);
    if (auto status = cudaMemcpyAsync(
            coefficients, normalized_coefficients.data(),
            normalized_coefficients.size_bytes(), cudaMemcpyHostToDevice,
            state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload resident GLM-5.3 MLA coefficients");
    }
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Mla);
        status != cudaSuccess) {
        return cuda_error(status, "start resident GLM-5.3 MLA finish timing");
    }
    if (persistent_bf16) {
        glm53_mla_weighted_bf16_kernel<<<
            request.heads, threads, 0U, state.stream>>>(
            coefficients, expanded_bf16, attended,
            static_cast<std::uint32_t>(history), request.heads,
            request.head_dim);
    } else {
        glm53_mla_weighted_kernel<<<
            request.heads, threads, 0U, state.stream>>>(
            coefficients, expanded, attended,
            static_cast<std::uint32_t>(history), request.heads,
            request.head_dim);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch resident GLM-5.3 MLA weighted sum");
    }
    const auto project_one = [&](const CudaWeight* weight, const float* source,
                                 float* destination,
                                 std::uint64_t rows) -> cudaError_t {
        const auto& descriptor = weight->impl_->descriptor;
        if (descriptor.encoding == CudaWeightEncoding::Plain) {
            bf16_matvec_kernel<<<
                static_cast<unsigned int>((rows + 7U) / 8U), threads, 0U,
                state.stream>>>(
                destination, source,
                static_cast<const __nv_bfloat16*>(weight->impl_->weights),
                descriptor.columns, rows);
        } else if (auto status = launch_regfed_fp8_f32_rows(
                       state.moe_regfed, descriptor, weight->impl_->weights,
                       weight->impl_->scales,
                       weight->impl_->fragment_prepacked, source, destination,
                       1U, state.stream); status != cudaSuccess) {
            return status;
        }
        round_bf16_rows_kernel<<<
            static_cast<unsigned int>((rows + threads - 1U) / threads),
            threads, 0U, state.stream>>>(destination, rows);
        return cudaGetLastError();
    };
    if (auto status = project_one(request.output, attended, output,
                                  kDsv4MhcHidden);
        status != cudaSuccess) {
        return cuda_error(status, "project resident GLM-5.3 MLA output");
    }
    dsv4_fp32_to_bf16<<<hidden_blocks, threads, 0U, state.stream>>>(
        output, state.dsv4_mhc_workspace->branch, kDsv4MhcHidden);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch resident GLM-5.3 MLA");
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish resident GLM-5.3 MLA finish timing");
    }
    state.glm53_mla_scores_pending = false;
    state.dsv4_mhc_branch_ready = true;
    return {};
}

namespace {

// The three GLM-5.3 expert matrices are fixed by the architecture:
// gate and up are `intermediate x hidden`, down is `hidden x intermediate`,
// and E4M3 block scales are one per 128x128 tile. BF16 matrices have no scale.
[[nodiscard]] ValidationResult glm53_expert_shape(
    const CudaGlm53Expert& expert) {
    if (expert.hidden == 0U || expert.intermediate == 0U ||
        expert.hidden % 128U != 0U || expert.intermediate % 128U != 0U) {
        return {{"CUDA GLM-5.3 expert has an invalid shape"}};
    }
    const CudaBuffer* weights[3] = {
        expert.gate_weights, expert.up_weights, expert.down_weights};
    for (const auto* buffer : weights) {
        if (buffer == nullptr || !buffer->valid()) {
            return {{"CUDA GLM-5.3 expert is missing a matrix"}};
        }
    }
    const std::uint64_t projection =
        static_cast<std::uint64_t>(expert.intermediate) * expert.hidden;
    if (expert.encoding == CudaGlm53ExpertEncoding::Bf16) {
        if (expert.gate_scales != nullptr || expert.up_scales != nullptr ||
            expert.down_scales != nullptr ||
            expert.gate_weights->device_bytes() != 2U * projection ||
            expert.up_weights->device_bytes() != 2U * projection ||
            expert.down_weights->device_bytes() != 2U * projection) {
            return {{"CUDA GLM-5.3 BF16 expert matrix is mis-sized"}};
        }
        return {};
    }
    const CudaBuffer* scales[3] = {
        expert.gate_scales, expert.up_scales, expert.down_scales};
    for (const auto* buffer : scales) {
        if (buffer == nullptr || !buffer->valid()) {
            return {{"CUDA GLM-5.3 FP8 expert is missing block scales"}};
        }
    }
    const std::uint64_t gate_up_scales =
        static_cast<std::uint64_t>(expert.intermediate / 128U) *
        (expert.hidden / 128U) * sizeof(float);
    const std::uint64_t down_scales =
        static_cast<std::uint64_t>(expert.hidden / 128U) *
        (expert.intermediate / 128U) * sizeof(float);
    if (expert.gate_weights->device_bytes() != projection ||
        expert.up_weights->device_bytes() != projection ||
        expert.down_weights->device_bytes() != projection ||
        expert.gate_scales->device_bytes() != gate_up_scales ||
        expert.up_scales->device_bytes() != gate_up_scales ||
        expert.down_scales->device_bytes() != down_scales) {
        return {{"CUDA GLM-5.3 expert matrix is mis-sized"}};
    }
    return {};
}

// Eight routed experts plus the shared one is the widest batch GLM-5.3 can
// present at a single decode row, so the device scratch is sized for it once.
constexpr std::size_t kGlm53MaxDeviceExperts = 9U;

}  // namespace

ValidationResult CudaBackend::enqueue_glm53_expert_gate_up(
    int device, std::span<const CudaGlm53Expert> experts,
    std::span<const float> input) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 expert batch targets an uninitialized CUDA device"}};
    }
    auto& state = found->second;
    if (state.glm53_shared_gate_up_in_flight ||
        state.glm53_shared_down_in_flight) {
        return {{"a GLM-5.3 expert batch is already in flight"}};
    }
    if (experts.empty() || experts.size() > kGlm53MaxDeviceExperts) {
        return {{"GLM-5.3 expert batch has an invalid width"}};
    }
    const auto hidden = experts.front().hidden;
    const auto intermediate = experts.front().intermediate;
    for (const auto& expert : experts) {
        if (auto shape = glm53_expert_shape(expert); !shape.ok()) {
            return shape;
        }
        if (expert.hidden != hidden || expert.intermediate != intermediate) {
            return {{"GLM-5.3 expert batch mixes shapes"}};
        }
        if (expert.encoding != experts.front().encoding) {
            return {{"GLM-5.3 expert batch mixes encodings"}};
        }
        const CudaBuffer* buffers[6] = {
            expert.gate_weights, expert.up_weights, expert.down_weights,
            expert.gate_scales, expert.up_scales, expert.down_scales};
        for (const auto* buffer : buffers) {
            if (buffer == nullptr) continue;
            if (buffer->device() != device) {
                return {{"GLM-5.3 expert matrix is on another device"}};
            }
        }
    }
    if (input.size() != hidden) {
        return {{"GLM-5.3 expert batch input has an invalid shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for the expert batch");
    }
    if (state.glm53_shared_input == nullptr) {
        // Sized for the widest batch the architecture can present -- the eight
        // routed experts plus the shared one -- so a batch that grows between
        // layers never reallocates inside a timed step.
        const auto allocate = [](float*& pointer, std::uint64_t floats) {
            void* memory = nullptr;
            const auto status = cudaMalloc(&memory, floats * sizeof(float));
            if (status == cudaSuccess) pointer = static_cast<float*>(memory);
            return status;
        };
        const std::uint64_t batch_intermediate =
            static_cast<std::uint64_t>(kGlm53MaxDeviceExperts) * intermediate;
        const std::uint64_t batch_hidden =
            static_cast<std::uint64_t>(kGlm53MaxDeviceExperts) * hidden;
        if (auto status = allocate(state.glm53_shared_input, hidden);
            status != cudaSuccess) {
            return cuda_error(status, "allocate GLM-5.3 expert batch input");
        }
        if (auto status = allocate(state.glm53_shared_gate, batch_intermediate);
            status != cudaSuccess) {
            return cuda_error(status, "allocate GLM-5.3 expert batch gate");
        }
        if (auto status = allocate(state.glm53_shared_up, batch_intermediate);
            status != cudaSuccess) {
            return cuda_error(status, "allocate GLM-5.3 expert batch up");
        }
        if (auto status =
                allocate(state.glm53_shared_activation, batch_intermediate);
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate GLM-5.3 expert batch activation");
        }
        if (auto status = allocate(state.glm53_shared_output, batch_hidden);
            status != cudaSuccess) {
            return cuda_error(status, "allocate GLM-5.3 expert batch output");
        }
        const std::uint64_t staging_floats =
            std::max(batch_hidden, std::uint64_t{2U} * batch_intermediate);
        void* staging = nullptr;
        if (auto status =
                cudaMallocHost(&staging, staging_floats * sizeof(float));
            status != cudaSuccess) {
            return cuda_error(status, "allocate GLM-5.3 expert batch staging");
        }
        state.glm53_shared_staging = static_cast<float*>(staging);
        state.glm53_shared_staging_floats =
            static_cast<std::uint32_t>(staging_floats);
        state.glm53_shared_hidden = hidden;
        state.glm53_shared_intermediate = intermediate;
        state.glm53_shared_encoding = experts.front().encoding;
    }
    if (state.glm53_shared_hidden != hidden ||
        state.glm53_shared_intermediate != intermediate ||
        state.glm53_shared_encoding != experts.front().encoding) {
        return {{"GLM-5.3 expert batch shape changed after admission"}};
    }
    std::copy(input.begin(), input.end(), state.glm53_shared_staging);
    if (auto status = cudaMemcpyAsync(
            state.glm53_shared_input, state.glm53_shared_staging,
            hidden * sizeof(float), cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload GLM-5.3 expert batch input");
    }
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Expert);
        status != cudaSuccess) {
        return cuda_error(status, "start GLM-5.3 expert gate/up timing");
    }
    constexpr std::uint32_t threads = 256U;
    const std::uint32_t blocks = (intermediate * 8U + threads - 1U) / threads;
    const auto scale_columns = hidden / 128U;
    for (std::size_t index = 0U; index < experts.size(); ++index) {
        const auto& expert = experts[index];
        const auto offset = index * intermediate;
        if (expert.encoding == CudaGlm53ExpertEncoding::Bf16) {
            glm53_shared_expert_bf16_dot_kernel<<<
                blocks, threads, 0U, state.stream>>>(
                state.glm53_shared_gate + offset,
                static_cast<const unsigned short*>(
                    expert.gate_weights->impl_->data),
                state.glm53_shared_input, intermediate, hidden);
            glm53_shared_expert_bf16_dot_kernel<<<
                blocks, threads, 0U, state.stream>>>(
                state.glm53_shared_up + offset,
                static_cast<const unsigned short*>(
                    expert.up_weights->impl_->data),
                state.glm53_shared_input, intermediate, hidden);
        } else {
            glm53_shared_expert_dot_kernel<<<blocks, threads, 0U, state.stream>>>(
                state.glm53_shared_gate + offset,
                static_cast<const unsigned char*>(
                    expert.gate_weights->impl_->data),
                static_cast<const float*>(expert.gate_scales->impl_->data),
                state.glm53_shared_input, intermediate, hidden, scale_columns);
            glm53_shared_expert_dot_kernel<<<blocks, threads, 0U, state.stream>>>(
                state.glm53_shared_up + offset,
                static_cast<const unsigned char*>(
                    expert.up_weights->impl_->data),
                static_cast<const float*>(expert.up_scales->impl_->data),
                state.glm53_shared_input, intermediate, hidden, scale_columns);
        }
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch GLM-5.3 expert batch gate and up");
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish GLM-5.3 expert gate/up timing");
    }
    const auto batch_floats = experts.size() * intermediate;
    if (auto status = cudaMemcpyAsync(
            state.glm53_shared_staging, state.glm53_shared_gate,
            batch_floats * sizeof(float), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 expert batch gate");
    }
    if (auto status = cudaMemcpyAsync(
            state.glm53_shared_staging + batch_floats, state.glm53_shared_up,
            batch_floats * sizeof(float), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 expert batch up");
    }
    state.glm53_shared_gate_up_in_flight = true;
    state.glm53_shared_batch = static_cast<std::uint32_t>(experts.size());
    return {};
}

ValidationResult CudaBackend::collect_glm53_expert_gate_up(
    int device, std::span<float> gate, std::span<float> up) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 expert batch targets an uninitialized CUDA device"}};
    }
    auto& state = found->second;
    if (!state.glm53_shared_gate_up_in_flight) {
        return {{"no GLM-5.3 expert batch gate and up command is in flight"}};
    }
    const std::size_t expected =
        static_cast<std::size_t>(state.glm53_shared_batch) *
        state.glm53_shared_intermediate;
    if (gate.size() != expected || up.size() != expected) {
        return {{"GLM-5.3 expert batch gate and up output has an invalid shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for the expert batch");
    }
    const auto status = cudaStreamSynchronize(state.stream);
    state.glm53_shared_gate_up_in_flight = false;
    if (status != cudaSuccess) {
        return cuda_error(status, "complete GLM-5.3 expert batch gate and up");
    }
    if (auto timing_status = glm53_kernel_timing_drain(
            *impl_, state, device, false);
        timing_status != cudaSuccess) {
        return cuda_error(timing_status, "measure GLM-5.3 expert gate/up");
    }
    std::copy_n(state.glm53_shared_staging, expected, gate.begin());
    std::copy_n(state.glm53_shared_staging + expected, expected, up.begin());
    return {};
}

ValidationResult CudaBackend::enqueue_glm53_expert_down(
    int device, std::span<const CudaGlm53Expert> experts,
    std::span<const float> activations) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 expert batch targets an uninitialized CUDA device"}};
    }
    auto& state = found->second;
    if (state.glm53_shared_gate_up_in_flight ||
        state.glm53_shared_down_in_flight) {
        return {{"a GLM-5.3 expert batch is already in flight"}};
    }
    if (state.glm53_shared_input == nullptr) {
        return {{"GLM-5.3 expert batch down ran before its gate and up"}};
    }
    if (experts.empty() || experts.size() != state.glm53_shared_batch) {
        return {{"GLM-5.3 expert batch down does not match its gate and up"}};
    }
    const auto hidden = state.glm53_shared_hidden;
    const auto intermediate = state.glm53_shared_intermediate;
    for (const auto& expert : experts) {
        if (auto shape = glm53_expert_shape(expert); !shape.ok()) {
            return shape;
        }
        if (expert.hidden != hidden || expert.intermediate != intermediate) {
            return {{"GLM-5.3 expert batch mixes shapes"}};
        }
        if (expert.encoding != state.glm53_shared_encoding) {
            return {{"GLM-5.3 expert batch down changes encoding"}};
        }
        const CudaBuffer* buffers[6] = {
            expert.gate_weights, expert.up_weights, expert.down_weights,
            expert.gate_scales, expert.up_scales, expert.down_scales};
        for (const auto* buffer : buffers) {
            if (buffer == nullptr) continue;
            if (buffer->device() != device) {
                return {{"GLM-5.3 expert matrix is on another device"}};
            }
        }
    }
    if (activations.size() != experts.size() * intermediate) {
        return {{"GLM-5.3 expert batch activation has an invalid shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for the expert batch");
    }
    std::copy(activations.begin(), activations.end(),
              state.glm53_shared_staging);
    if (auto status = cudaMemcpyAsync(
            state.glm53_shared_activation, state.glm53_shared_staging,
            activations.size() * sizeof(float),
            cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload GLM-5.3 expert batch activation");
    }
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Expert);
        status != cudaSuccess) {
        return cuda_error(status, "start GLM-5.3 expert down timing");
    }
    constexpr std::uint32_t threads = 256U;
    const std::uint32_t blocks = (hidden * 8U + threads - 1U) / threads;
    for (std::size_t index = 0U; index < experts.size(); ++index) {
        const auto& expert = experts[index];
        if (expert.encoding == CudaGlm53ExpertEncoding::Bf16) {
            glm53_shared_expert_bf16_dot_kernel<<<
                blocks, threads, 0U, state.stream>>>(
                state.glm53_shared_output + index * hidden,
                static_cast<const unsigned short*>(
                    expert.down_weights->impl_->data),
                state.glm53_shared_activation + index * intermediate, hidden,
                intermediate);
        } else {
            glm53_shared_expert_dot_kernel<<<blocks, threads, 0U, state.stream>>>(
                state.glm53_shared_output + index * hidden,
                static_cast<const unsigned char*>(
                    expert.down_weights->impl_->data),
                static_cast<const float*>(expert.down_scales->impl_->data),
                state.glm53_shared_activation + index * intermediate, hidden,
                intermediate, intermediate / 128U);
        }
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch GLM-5.3 expert batch down");
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish GLM-5.3 expert down timing");
    }
    if (auto status = cudaMemcpyAsync(
            state.glm53_shared_staging, state.glm53_shared_output,
            experts.size() * hidden * sizeof(float), cudaMemcpyDeviceToHost,
            state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 expert batch down");
    }
    state.glm53_shared_down_in_flight = true;
    return {};
}

ValidationResult CudaBackend::collect_glm53_expert_down(
    int device, std::span<float> output) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 expert batch targets an uninitialized CUDA device"}};
    }
    auto& state = found->second;
    if (!state.glm53_shared_down_in_flight) {
        return {{"no GLM-5.3 expert batch down command is in flight"}};
    }
    const std::size_t expected =
        static_cast<std::size_t>(state.glm53_shared_batch) *
        state.glm53_shared_hidden;
    if (output.size() != expected) {
        return {{"GLM-5.3 expert batch output has an invalid shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for the expert batch");
    }
    const auto status = cudaStreamSynchronize(state.stream);
    state.glm53_shared_down_in_flight = false;
    if (status != cudaSuccess) {
        return cuda_error(status, "complete GLM-5.3 expert batch down");
    }
    if (auto timing_status = glm53_kernel_timing_drain(
            *impl_, state, device, false);
        timing_status != cudaSuccess) {
        return cuda_error(timing_status, "measure GLM-5.3 expert down");
    }
    std::copy_n(state.glm53_shared_staging, expected, output.begin());
    return {};
}
