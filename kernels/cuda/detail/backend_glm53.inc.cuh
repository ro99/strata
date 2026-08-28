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
    const auto projected = request.output_projection != nullptr;
    const auto projected_rows = projected && request.output_projection->valid()
        ? request.output_projection->impl_->descriptor.rows
        : 0U;
    if (request.query.size() != width || request.key.size() != width ||
        request.value.size() != width || request.forget.size() != width ||
        request.gate.size() != width || request.beta.size() != request.heads ||
        output.size() != (projected ? projected_rows : width)) {
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
        request.state->device_bytes() != required_state_floats * sizeof(float)) {
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
            if (auto status = launch_regfed_fp8_matvec(
                    device_state.moe_regfed, projection.descriptor,
                    projection.weights, projection.scales,
                    projection.fragment_prepacked, device_state.output,
                    device_state.input, device_state.stream);
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
