// The host FP4 expert claims bit-identity with the device kernel, not a
// tolerance. These tests hold it to that: same bytes in, same bit pattern out.

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_host_expert.hpp"
#include "test.hpp"

namespace {

constexpr std::uint64_t kHidden = 512U;
constexpr std::uint64_t kIntermediate = 256U;
constexpr float kSwigluLimit = 10.0F;

struct HostFp4 {
    std::vector<std::byte> packed;
    std::vector<std::byte> scales;
};

// Byte-for-byte the generator tests/test_cuda_backend.cpp uploads, so the host
// and the device are reading the same weights rather than similar ones.
HostFp4 make_fp4(std::uint64_t rows, std::uint64_t columns, std::uint8_t seed) {
    const auto packed_columns = (columns + 1U) / 2U;
    const auto scale_columns = (columns + 31U) / 32U;
    HostFp4 result;
    result.packed.resize(static_cast<std::size_t>(rows * packed_columns));
    for (std::uint64_t row = 0U; row < rows; ++row) {
        for (std::uint64_t packed = 0U; packed < packed_columns; ++packed) {
            const auto low = static_cast<std::uint8_t>(
                (seed + row * 3U + packed * 5U) & 0x0FU);
            const auto high = static_cast<std::uint8_t>(
                (seed + row * 7U + packed * 11U + 1U) & 0x0FU);
            result.packed[static_cast<std::size_t>(row * packed_columns + packed)] =
                static_cast<std::byte>(low | static_cast<std::uint8_t>(high << 4U));
        }
    }
    result.scales.resize(static_cast<std::size_t>(rows * scale_columns));
    for (std::size_t index = 0U; index < result.scales.size(); ++index) {
        result.scales[index] = static_cast<std::byte>(
            0x78U + static_cast<std::uint8_t>((index + seed) % 3U));
    }
    return result;
}

strata::CudaWeight upload(strata::CudaBackend& backend, int device,
                          std::uint64_t rows, std::uint64_t columns,
                          const HostFp4& source) {
    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::Fp4E2m1Group32;
    descriptor.dtype = strata::SafetensorsDtype::I8;
    descriptor.rows = rows;
    descriptor.columns = columns;
    descriptor.packed_columns = (columns + 1U) / 2U;
    descriptor.scale_columns = (columns + 31U) / 32U;
    descriptor.group_size = 32U;
    strata::CudaWeight result;
    REQUIRE(backend.upload(device, descriptor, source.packed, source.scales,
                           result)
                .ok());
    return result;
}

// Deliberately NOT multiples of a power of two. An earlier version of this
// helper used 0.125F steps, every one of which is exactly representable in
// E4M3, so it could not detect that the host path was skipping the input
// quantizer the device applies. Test data must not be exact in the format
// under test.
std::vector<float> make_hidden_dim(std::uint64_t dimension) {
    std::vector<float> hidden(static_cast<std::size_t>(dimension));
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        const auto step = static_cast<float>(index % 37U) * 0.0271828F;
        hidden[index] = (index % 2U == 0U ? 1.0F : -1.0F) * (0.1013F + step);
    }
    return hidden;
}

std::vector<float> make_hidden() { return make_hidden_dim(kHidden); }

strata::Dsv4HostExpertWeights view(const HostFp4& w1, const HostFp4& w3,
                                   const HostFp4& w2) {
    strata::Dsv4HostExpertWeights weights;
    weights.w1_packed = w1.packed;
    weights.w1_scales = w1.scales;
    weights.w3_packed = w3.packed;
    weights.w3_scales = w3.scales;
    weights.w2_packed = w2.packed;
    weights.w2_scales = w2.scales;
    return weights;
}

}  // namespace

TEST_CASE("DeepSeek host FP4 expert AVX2 path is bit-identical to its scalar path") {
    const auto w1 = make_fp4(kIntermediate, kHidden, 1U);
    const auto w3 = make_fp4(kIntermediate, kHidden, 6U);
    const auto w2 = make_fp4(kHidden, kIntermediate, 11U);
    const auto hidden = make_hidden();
    const auto weights = view(w1, w3, w2);

    std::vector<float> scalar_output(static_cast<std::size_t>(kHidden));
    std::vector<float> vector_output(static_cast<std::size_t>(kHidden));
    std::vector<float> scratch(static_cast<std::size_t>(kIntermediate));

    REQUIRE(strata::dsv4_host_expert_fp4(scalar_output, hidden, weights, scratch,
                                         kHidden, kIntermediate, 0.75F,
                                         kSwigluLimit, false)
                .ok());
    REQUIRE(strata::dsv4_host_expert_fp4(vector_output, hidden, weights, scratch,
                                         kHidden, kIntermediate, 0.75F,
                                         kSwigluLimit, true)
                .ok());
    for (std::size_t index = 0U; index < scalar_output.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(scalar_output[index]) ==
                std::bit_cast<std::uint32_t>(vector_output[index]));
    }
}

TEST_CASE("DeepSeek host FP4 expert rejects mismatched extents") {
    const auto w1 = make_fp4(kIntermediate, kHidden, 1U);
    const auto w3 = make_fp4(kIntermediate, kHidden, 6U);
    const auto w2 = make_fp4(kHidden, kIntermediate, 11U);
    const auto hidden = make_hidden();
    auto weights = view(w1, w3, w2);
    weights.w2_scales = weights.w2_scales.first(weights.w2_scales.size() - 1U);

    std::vector<float> output(static_cast<std::size_t>(kHidden));
    std::vector<float> scratch(static_cast<std::size_t>(kIntermediate));
    REQUIRE(!strata::dsv4_host_expert_fp4(output, hidden, weights, scratch,
                                          kHidden, kIntermediate, 0.75F,
                                          kSwigluLimit, true)
                 .ok());
}

// The production shape. The 512/256 case above exercises the phase structure
// but not the loop trip counts the real expert uses: 4096 columns is 16 offset
// steps against 2, and 2048 is 8. A kernel can be exact at one and wrong at the
// other, and decode only needs one flipped ULP to diverge a greedy argmax.
TEST_CASE("DeepSeek host FP4 expert matches the device at the production shape") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    constexpr std::uint64_t hidden = 4096U;
    constexpr std::uint64_t intermediate = 2048U;
    const int device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    const auto w1 = make_fp4(intermediate, hidden, 1U);
    const auto w3 = make_fp4(intermediate, hidden, 6U);
    const auto w2 = make_fp4(hidden, intermediate, 11U);
    auto device_w1 = upload(backend, device, intermediate, hidden, w1);
    auto device_w3 = upload(backend, device, intermediate, hidden, w3);
    auto device_w2 = upload(backend, device, hidden, intermediate, w2);

    const auto hidden_state = make_hidden_dim(hidden);
    constexpr float coefficient = 0.75F;
    const std::array<strata::CudaDeepSeekMoeExpert, 1> routed{{
        {&device_w1, &device_w3, &device_w2, coefficient},
    }};
    REQUIRE(backend
                .enqueue_deepseek_moe(device, hidden_state, routed, nullptr,
                                      kSwigluLimit)
                .ok());
    std::vector<float> device_output(static_cast<std::size_t>(hidden));
    std::vector<float> shared_output;
    REQUIRE(backend.collect_deepseek_moe(device, device_output, shared_output)
                .ok());

    const auto weights = view(w1, w3, w2);
    std::vector<float> host_output(static_cast<std::size_t>(hidden));
    std::vector<float> scratch(static_cast<std::size_t>(intermediate));
    REQUIRE(strata::dsv4_host_expert_fp4(host_output, hidden_state, weights,
                                         scratch, hidden, intermediate,
                                         coefficient, kSwigluLimit, true)
                .ok());
    for (std::size_t index = 0U; index < host_output.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(host_output[index]) ==
                std::bit_cast<std::uint32_t>(device_output[index]));
    }
}

TEST_CASE("DeepSeek host FP4 expert reproduces the device kernel bit for bit") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    const int device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    const auto w1 = make_fp4(kIntermediate, kHidden, 1U);
    const auto w3 = make_fp4(kIntermediate, kHidden, 6U);
    const auto w2 = make_fp4(kHidden, kIntermediate, 11U);
    auto device_w1 = upload(backend, device, kIntermediate, kHidden, w1);
    auto device_w3 = upload(backend, device, kIntermediate, kHidden, w3);
    auto device_w2 = upload(backend, device, kHidden, kIntermediate, w2);

    const auto hidden = make_hidden();
    constexpr float coefficient = 0.75F;
    const std::array<strata::CudaDeepSeekMoeExpert, 1> routed{{
        {&device_w1, &device_w3, &device_w2, coefficient},
    }};
    REQUIRE(backend
                .enqueue_deepseek_moe(device, hidden, routed, nullptr,
                                      kSwigluLimit)
                .ok());
    std::vector<float> device_output(static_cast<std::size_t>(kHidden));
    std::vector<float> shared_output;
    REQUIRE(backend.collect_deepseek_moe(device, device_output, shared_output)
                .ok());

    const auto weights = view(w1, w3, w2);
    std::vector<float> host_output(static_cast<std::size_t>(kHidden));
    std::vector<float> scratch(static_cast<std::size_t>(kIntermediate));
    REQUIRE(strata::dsv4_host_expert_fp4(host_output, hidden, weights, scratch,
                                         kHidden, kIntermediate, coefficient,
                                         kSwigluLimit, true)
                .ok());
    for (std::size_t index = 0U; index < host_output.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(host_output[index]) ==
                std::bit_cast<std::uint32_t>(device_output[index]));
    }
}
