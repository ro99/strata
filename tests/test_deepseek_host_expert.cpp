// The host FP4 expert claims bit-identity with the device kernel, not a
// tolerance. These tests hold it to that: same bytes in, same bit pattern out.

#include <array>
#include <bit>
#include <cmath>
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

float fp4(std::byte packed, bool high) {
    constexpr std::array<float, 16> values{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
    const auto byte = std::to_integer<std::uint8_t>(packed);
    return values[high ? byte >> 4U : byte & 0x0FU];
}

float e8m0(std::byte encoded) {
    const auto byte = std::to_integer<std::uint8_t>(encoded);
    return std::bit_cast<float>(static_cast<std::uint32_t>(byte) << 23U);
}

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

TEST_CASE("DeepSeek tiled expert transform preserves every matrix value") {
    constexpr std::uint64_t hidden = 64U;
    constexpr std::uint64_t intermediate = 64U;
    const auto w1 = make_fp4(intermediate, hidden, 1U);
    const auto w3 = make_fp4(intermediate, hidden, 6U);
    const auto w2 = make_fp4(hidden, intermediate, 11U);
    const auto canonical = view(w1, w3, w2);
    const auto shard_bytes = strata::dsv4_tiled_expert_shard_bytes(
        hidden, intermediate);
    REQUIRE(shard_bytes > 0U);
    std::vector<std::byte> storage(static_cast<std::size_t>(2U * shard_bytes));
    for (std::uint64_t shard = 0U; shard < 2U; ++shard) {
        REQUIRE(strata::dsv4_transform_tiled_expert_shard(
                    std::span<std::byte>(storage).subspan(
                        static_cast<std::size_t>(shard * shard_bytes),
                        static_cast<std::size_t>(shard_bytes)),
                    canonical, hidden, intermediate, shard)
                    .ok());
    }
    auto tiled = strata::dsv4_tiled_expert_weights(
        std::span<const std::byte>(storage).first(shard_bytes), hidden,
        intermediate);
    REQUIRE(tiled.ok());
    const auto input = make_hidden_dim(hidden);
    std::array<float, 16U> actual{};
    strata::dsv4_tiled_expert_matvec16(
        actual, input, tiled.value.w13_packed, tiled.value.w13_scales,
        intermediate, 0U);
    for (std::size_t row = 0U; row < actual.size(); ++row) {
        float expected = 0.0F;
        for (std::size_t column = 0U; column < input.size(); ++column) {
            expected = std::fma(
                input[column],
                fp4(w1.packed[row * (hidden / 2U) + column / 2U],
                    column % 2U != 0U) *
                    e8m0(w1.scales[row * (hidden / 32U) + column / 32U]),
                expected);
        }
        REQUIRE(std::bit_cast<std::uint32_t>(actual[row]) ==
                std::bit_cast<std::uint32_t>(expected));
    }
}

TEST_CASE("DeepSeek tiled expert row batch holds at the production shape") {
    // The 64-wide case below cannot see a reduction that only differs once
    // the column loop is long enough to be blocked, and the runtime drives
    // this primitive at 4,096 input columns for gate/up and 1,024 for down.
    const auto check = [](std::uint64_t hidden, std::uint64_t intermediate,
                          std::size_t rows, std::size_t members) {
        const auto w1 = make_fp4(intermediate, hidden, 1U);
        const auto w3 = make_fp4(intermediate, hidden, 6U);
        const auto w2 = make_fp4(hidden, intermediate, 11U);
        const auto canonical = view(w1, w3, w2);
        const auto shard_bytes = strata::dsv4_tiled_expert_shard_bytes(
            hidden, intermediate);
        std::vector<std::byte> storage(static_cast<std::size_t>(shard_bytes));
        REQUIRE(strata::dsv4_transform_tiled_expert_shard(
                    storage, canonical, hidden, intermediate, 0U).ok());
        auto tiled = strata::dsv4_tiled_expert_weights(
            storage, hidden, intermediate);
        REQUIRE(tiled.ok());

        std::vector<float> input(rows * static_cast<std::size_t>(hidden));
        std::uint32_t seed = 0x2f6e21U;
        for (auto& value : input) {
            seed = seed * 1'664'525U + 1'013'904'223U;
            value = static_cast<float>(static_cast<std::int32_t>(
                        (seed >> 9U) % 4'096U) - 2'048) / 8'192.0F;
        }
        std::vector<std::uint32_t> selected(members);
        for (std::size_t index = 0U; index < members; ++index) {
            selected[index] = static_cast<std::uint32_t>((index * 5U + 1U) % rows);
        }
        // The runtime writes 16 outputs at a time into a 32-wide scratch, so
        // the strided half-block layout is part of what has to match.
        constexpr std::uint64_t block = 32U;
        std::vector<float> expected(members * block);
        std::vector<float> actual(members * block);
        for (std::uint64_t offset = 0U; offset + block <= intermediate;
             offset += block) {
            for (std::size_t member = 0U; member < members; ++member) {
                for (std::uint64_t half = 0U; half < 2U; ++half) {
                    strata::dsv4_tiled_expert_matvec16(
                        std::span<float>(expected)
                            .subspan(member * block + half * 16U, 16U)
                            .first<16U>(),
                        std::span<const float>(input).subspan(
                            static_cast<std::size_t>(selected[member]) * hidden,
                            static_cast<std::size_t>(hidden)),
                        tiled.value.w13_packed, tiled.value.w13_scales,
                        2U * intermediate, offset + half * 16U);
                }
            }
            for (std::uint64_t half = 0U; half < 2U; ++half) {
                strata::dsv4_tiled_expert_matvec16_rows(
                    std::span<float>(actual).subspan(half * 16U), block, input,
                    hidden, selected, tiled.value.w13_packed,
                    tiled.value.w13_scales, 2U * intermediate,
                    offset + half * 16U);
            }
            for (std::size_t index = 0U; index < expected.size(); ++index) {
                REQUIRE(std::bit_cast<std::uint32_t>(actual[index]) ==
                        std::bit_cast<std::uint32_t>(expected[index]));
            }
        }
    };
    check(4096U, 1024U, 7U, 5U);
    check(1024U, 4096U, 7U, 4U);
}

TEST_CASE("DeepSeek tiled expert row batch is bit-identical to scalar rows") {
    constexpr std::uint64_t hidden = 64U;
    constexpr std::uint64_t intermediate = 64U;
    const auto w1 = make_fp4(intermediate, hidden, 1U);
    const auto w3 = make_fp4(intermediate, hidden, 6U);
    const auto w2 = make_fp4(hidden, intermediate, 11U);
    const auto canonical = view(w1, w3, w2);
    const auto shard_bytes = strata::dsv4_tiled_expert_shard_bytes(
        hidden, intermediate);
    std::vector<std::byte> storage(static_cast<std::size_t>(shard_bytes));
    REQUIRE(strata::dsv4_transform_tiled_expert_shard(
                storage, canonical, hidden, intermediate, 0U)
                .ok());
    auto tiled = strata::dsv4_tiled_expert_weights(
        storage, hidden, intermediate);
    REQUIRE(tiled.ok());

    constexpr std::size_t rows = 7U;
    std::vector<float> input(rows * hidden);
    for (std::size_t row = 0U; row < rows; ++row) {
        auto generated = make_hidden_dim(hidden);
        for (std::size_t column = 0U; column < hidden; ++column) {
            input[row * hidden + column] =
                generated[column] + static_cast<float>(row) * 0.03125F;
        }
    }
    const std::array<std::uint32_t, 5U> selected{6U, 1U, 4U, 0U, 3U};
    std::vector<float> expected(selected.size() * intermediate);
    std::vector<float> actual(selected.size() * intermediate);
    for (std::size_t row = 0U; row < selected.size(); ++row) {
        for (std::uint64_t offset = 0U; offset < intermediate; offset += 16U) {
            strata::dsv4_tiled_expert_matvec16(
                std::span<float>(expected)
                    .subspan(row * intermediate + offset, 16U)
                    .first<16U>(),
                std::span<const float>(input).subspan(
                    static_cast<std::size_t>(selected[row]) * hidden, hidden),
                tiled.value.w13_packed, tiled.value.w13_scales,
                intermediate, offset);
        }
    }
    for (std::uint64_t offset = 0U; offset < intermediate; offset += 16U) {
        strata::dsv4_tiled_expert_matvec16_rows(
            std::span<float>(actual).subspan(offset), intermediate, input,
            hidden, selected, tiled.value.w13_packed,
            tiled.value.w13_scales, intermediate, offset);
    }
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(actual[index]) ==
                std::bit_cast<std::uint32_t>(expected[index]));
    }
}

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

TEST_CASE("DeepSeek transformed expert shards match the scalar oracle") {
    // Prefill reads the routed experts in the transform's layout, which puts
    // 32 output rows of a block in 32 consecutive bytes. A warp owns the block
    // and each lane sums its own row over the whole reduction, so the terms
    // are the canonical ones in a different order. That is a reassociation,
    // not a different computation, and the oracle is what says so.
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    constexpr std::uint64_t hidden = 4096U;
    constexpr std::uint64_t intermediate = 2048U;
    constexpr std::uint64_t shards = strata::kCudaDsv4TiledShards;
    constexpr std::uint32_t rows = 3U;
    const int device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    const auto w1 = make_fp4(intermediate, hidden, 1U);
    const auto w3 = make_fp4(intermediate, hidden, 6U);
    const auto w2 = make_fp4(hidden, intermediate, 11U);
    const auto weights = view(w1, w3, w2);

    const auto shard_bytes = strata::dsv4_tiled_expert_shard_bytes(
        hidden, intermediate, shards);
    REQUIRE(shard_bytes != 0U);
    std::array<strata::CudaWeight, shards> shard_weights;
    for (std::uint64_t shard = 0U; shard < shards; ++shard) {
        std::vector<std::byte> storage(static_cast<std::size_t>(shard_bytes));
        REQUIRE(strata::dsv4_transform_tiled_expert_shard(
                    storage, weights, hidden, intermediate, shard, shards).ok());
        strata::CudaWeightDescriptor descriptor;
        descriptor.encoding = strata::CudaWeightEncoding::Fp4E2m1Tiled32;
        descriptor.dtype = strata::SafetensorsDtype::I8;
        descriptor.rows = hidden;
        descriptor.columns = intermediate / shards;
        descriptor.group_size = 32U;
        REQUIRE(backend.upload(device, descriptor, storage, {},
                               shard_weights[shard]).ok());
    }

    std::vector<float> hidden_state(rows * static_cast<std::size_t>(hidden));
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto generated = make_hidden_dim(hidden);
        for (std::uint64_t column = 0U; column < hidden; ++column) {
            hidden_state[row * hidden + column] =
                generated[static_cast<std::size_t>(column)] +
                static_cast<float>(row) * 0.0173F;
        }
    }
    const std::array<std::uint32_t, rows> page_rows{2U, 0U, 1U};
    const std::array<float, rows> coefficients{0.75F, 1.25F, 0.5F};

    strata::CudaDeepSeekMoeRowGroup group;
    for (std::size_t shard = 0U; shard < shards; ++shard) {
        group.tiled_shards[shard] = &shard_weights[shard];
    }
    group.rows = page_rows;
    group.coefficients = coefficients;
    const std::array<strata::CudaDeepSeekMoeRowGroup, 1> page{group};
    REQUIRE(backend.enqueue_deepseek_moe_rows(
                device, hidden_state, rows, page, nullptr, {}, kSwigluLimit)
                .ok());
    std::vector<float> routed(rows * static_cast<std::size_t>(hidden));
    REQUIRE(backend.collect_deepseek_moe_rows(device, routed, {}).ok());

    std::vector<float> oracle(static_cast<std::size_t>(hidden));
    std::vector<float> scratch(static_cast<std::size_t>(intermediate));
    for (std::uint32_t slot = 0U; slot < rows; ++slot) {
        const auto input = std::span<const float>(hidden_state)
            .subspan(static_cast<std::size_t>(page_rows[slot]) * hidden, hidden);
        REQUIRE(strata::dsv4_host_expert_fp4(
                    oracle, input, weights, scratch, hidden, intermediate,
                    coefficients[slot], kSwigluLimit, true).ok());
        float magnitude = 0.0F;
        for (const float value : oracle) {
            magnitude = std::max(magnitude, std::fabs(value));
        }
        REQUIRE(magnitude > 0.0F);
        // One BF16 mantissa step of the row's largest oracle magnitude. Both
        // the oracle and the kernel round their result to BF16, so this is the
        // granularity the reassociation could possibly show up at.
        const float tolerance = magnitude / 256.0F;
        for (std::uint64_t column = 0U; column < hidden; ++column) {
            const auto index =
                static_cast<std::size_t>(slot) * hidden + column;
            REQUIRE(std::fabs(routed[index] -
                              oracle[static_cast<std::size_t>(column)]) <=
                    tolerance);
        }
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
