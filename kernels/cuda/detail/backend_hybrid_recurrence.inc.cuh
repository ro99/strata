namespace {

__device__ __forceinline__ float glm53_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float glm53_sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
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
        float sum = weights[kernel - 1U] * values[channel];
        for (std::uint32_t offset = 0U; offset < history_width; ++offset) {
            sum += weights[offset] * history[offset];
        }
        for (std::uint32_t offset = 0U; offset + 1U < history_width; ++offset) {
            history[offset] = history[offset + 1U];
        }
        history[history_width - 1U] = values[channel];
        values[channel] = glm53_bf16(sum * glm53_sigmoid(sum));
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
            query_square += q * q;
            key_square += k * k;
        }
        query_inverse = rsqrtf(query_square + 1.0e-6F) /
                        sqrtf(static_cast<float>(head_dim));
        key_inverse = rsqrtf(key_square + 1.0e-6F);
    }
    __syncthreads();
    normalized_query[lane] = query[base + lane] * query_inverse;
    normalized_key[lane] = key[base + lane] * key_inverse;
    decay[lane] = expf(-5.0F * glm53_sigmoid(
        expf(a_log[head]) * (forget[base + lane] + dt_bias[base + lane])));
    __syncthreads();
    auto* state_row = recurrent +
        (static_cast<std::size_t>(head) * head_dim + lane) * head_dim;
    float projected = 0.0F;
    for (std::uint32_t index = 0U; index < head_dim; ++index) {
        state_row[index] *= decay[index];
        projected += state_row[index] * normalized_key[index];
    }
    const auto delta =
        (value[base + lane] - projected) * beta[head];
    float mixed = 0.0F;
    for (std::uint32_t index = 0U; index < head_dim; ++index) {
        state_row[index] += delta * normalized_key[index];
        mixed += state_row[index] * normalized_query[index];
    }
    raw[lane] = glm53_bf16(mixed);
    __syncthreads();
    if (lane == 0U) {
        float square = 0.0F;
        for (std::uint32_t index = 0U; index < head_dim; ++index) {
            square += raw[index] * raw[index];
        }
        output_inverse = rsqrtf(
            square / static_cast<float>(head_dim) + 1.0e-5F);
    }
    __syncthreads();
    output[base + lane] = glm53_bf16(
        norm_weight[lane] * raw[lane] * output_inverse *
        glm53_sigmoid(gate[base + lane]));
}

__global__ void glm53_kda_beta_kernel(float* beta, std::uint32_t heads) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < heads) beta[index] = glm53_bf16(glm53_sigmoid(beta[index]));
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
// One block per head with a single active thread, because the host accumulates
// each score sequentially over the columns and the association has to match.
__global__ void glm53_mla_scores_kernel(
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

}  // namespace

ValidationResult CudaBackend::glm53_kda_decode(
    const CudaGlm53KdaRequest& request, std::span<float> output) {
    ValidationResult result;
    if (request.state == nullptr || !request.state->valid() ||
        request.heads == 0U || request.head_dim == 0U ||
        request.head_dim > 256U || request.convolution_kernel < 2U) {
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
            required_state_floats + workspace_floats >
                request.state->device_bytes() / sizeof(float)) {
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
            device_state.dsv4_mhc_branch_ready = true;
            return {};
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
    bf16_input_matvec_kernel<<<blocks, threads, 0U, state.stream>>>(
        state.dsv4_mhc_workspace->glm53_router_logits,
        state.dsv4_mhc_workspace->layer_input,
        static_cast<const __nv_bfloat16*>(router.impl_->weights),
        descriptor.columns, descriptor.rows);
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
    state.dsv4_mhc_branch_ready = true;
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
    const auto history = static_cast<std::uint64_t>(request.position) + 1U;
    // Keep the accepted host fallback's projection/attention association.
    // Absorbing the KV-B projection into the query is algebraically equivalent,
    // but not bit-equivalent after the model's BF16 projection boundaries.
    // The resident path therefore expands the latent history exactly as the
    // fallback does before applying attention.
    //
    // Size the workspace from the admitted context rather than the current
    // position. The expanded history grows by `expanded_width` floats every
    // token, so sizing it from `history` would cudaFree and cudaMalloc inside a
    // timed decode step -- which M4's gate forbids outright, and which would
    // also make a step's cost depend on when the buffer last happened to grow.
    // Reserving once costs `maximum_context * expanded_width` floats: 268 MB at
    // 2,048 tokens on this model, whose 64 heads and 256 head dim make
    // `expanded_width` 32,768 because it holds key and value together. That
    // sits inside the per-device workspace reserve the planner already
    // withholds.
    //
    // The offsets below use the reserved extent too, so a pointer into the
    // workspace means the same thing at every position.
    const auto reserved_history = std::max<std::uint64_t>(
        history, request.maximum_context);
    // The trailing `heads * reserved_history` holds the scores on the way out
    // and the normalized coefficients on the way back in.
    const auto workspace_floats = kDsv4MhcHidden + request.query_rank + width +
        reserved_history * expanded_width + width + kDsv4MhcHidden +
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
    auto* attended = expanded + reserved_history * expanded_width;
    auto* output = attended + width;
    auto* coefficients = output + kDsv4MhcHidden;
    auto* latent = packed + static_cast<std::uint64_t>(request.position) *
                                request.key_value_rank;
    constexpr unsigned int threads = 256U;
    constexpr unsigned int warps = threads / 32U;
    constexpr unsigned int hidden_blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
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
    const auto& kv_descriptor = request.key_value_b->impl_->descriptor;
    if (kv_descriptor.encoding == CudaWeightEncoding::Plain) {
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
    round_bf16_rows_kernel<<<
        static_cast<unsigned int>(
            std::min<std::uint64_t>(
                (history * expanded_width + threads - 1U) / threads,
                65535U)),
        threads, 0U, state.stream>>>(expanded, history * expanded_width);
    // Stop at the scores and hand them back. The caller applies `exp`, the
    // normalization and the BF16 coefficient rounding on the host, then calls
    // `glm53_mla_decode_finish` to complete the layer.
    if (scores.size() != request.heads * history) {
        return {{"CUDA GLM-5.3 resident MLA score span has an invalid shape"}};
    }
    glm53_mla_scores_kernel<<<request.heads, threads, 0U, state.stream>>>(
        query, expanded, coefficients,
        static_cast<std::uint32_t>(history), request.heads, request.head_dim);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch resident GLM-5.3 MLA scores");
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
    const auto reserved_history = std::max<std::uint64_t>(
        history, request.maximum_context);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for GLM-5.3 resident MLA");
    }
    auto* workspace = reinterpret_cast<float*>(state.glm53_mla_workspace);
    auto* query = workspace + kDsv4MhcHidden + request.query_rank;
    auto* expanded = query + width;
    auto* attended = expanded + reserved_history * expanded_width;
    auto* output = attended + width;
    auto* coefficients = output + kDsv4MhcHidden;
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
    glm53_mla_weighted_kernel<<<request.heads, threads, 0U, state.stream>>>(
        coefficients, expanded, attended,
        static_cast<std::uint32_t>(history), request.heads, request.head_dim);
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
    std::copy_n(state.glm53_shared_staging, expected, output.begin());
    return {};
}
