#include "test.hpp"

#include "strata/deepseek_kv_cache.hpp"
#include "strata/deepseek_ops.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <vector>

namespace {

std::vector<float> row(std::size_t width, float value) {
    return std::vector<float>(width, value);
}

void require_bit_equal(std::span<const float> actual,
                       std::span<const float> expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(actual[index]) ==
                std::bit_cast<std::uint32_t>(expected[index]));
    }
}

}  // namespace

TEST_CASE("DeepSeek compact KV codecs are exact or reject the write") {
    constexpr auto head_dim = strata::kDeepSeekV4ExecutionContract.head_dim;
    constexpr auto rope_dim =
        strata::kDeepSeekV4ExecutionContract.rope_head_dim;
    std::vector<float> kv(head_dim);
    for (std::size_t group = 0U; group < (head_dim - rope_dim) / 64U; ++group) {
        const float scale = std::ldexp(1.0F, static_cast<int>(group) - 3);
        for (std::size_t column = 0U; column < 64U; ++column) {
            const auto code = column == 63U ? std::uint8_t{0x7EU}
                : static_cast<std::uint8_t>(column + 1U);
            kv[group * 64U + column] =
                strata::dsv4_fp8_e4m3_f32(code) * scale;
        }
    }
    for (std::size_t column = 0U; column < rope_dim; ++column) {
        kv[head_dim - rope_dim + column] = std::ldexp(
            static_cast<float>(static_cast<int>(column % 7U) - 3), -4);
    }
    const auto kv_format = strata::dsv4_kv_format(
        strata::Dsv4KvBlockKind::Sliding);
    std::vector<std::byte> encoded_kv(static_cast<std::size_t>(
        strata::dsv4_kv_row_bytes(strata::Dsv4KvBlockKind::Sliding,
                                  kv_format)));
    REQUIRE(strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format, kv, encoded_kv).ok());
    std::vector<float> decoded_kv(head_dim);
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format,
        encoded_kv, decoded_kv).ok());
    require_bit_equal(decoded_kv, kv);

    auto lossy = kv;
    lossy[0] = 1.1F;
    REQUIRE(!strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format,
        lossy, encoded_kv).ok());
    encoded_kv[head_dim - rope_dim] = std::byte{0xFFU};
    REQUIRE(!strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format,
        encoded_kv, decoded_kv).ok());

    constexpr auto index_dim =
        strata::kDeepSeekV4ExecutionContract.index_head_dim;
    std::vector<float> index(index_dim);
    for (std::size_t group = 0U; group < index_dim / 32U; ++group) {
        const float scale = std::ldexp(1.0F, static_cast<int>(group) - 2);
        for (std::size_t column = 0U; column < 32U; ++column) {
            const auto code = column == 31U ? std::uint8_t{7U}
                : static_cast<std::uint8_t>(column % 8U);
            index[group * 32U + column] =
                strata::dsv4_fp4_e2m1_f32(code) * scale;
        }
    }
    const auto index_format = strata::dsv4_kv_format(
        strata::Dsv4KvBlockKind::LearnedIndex);
    std::vector<std::byte> encoded_index(static_cast<std::size_t>(
        strata::dsv4_kv_row_bytes(strata::Dsv4KvBlockKind::LearnedIndex,
                                  index_format)));
    REQUIRE(strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_format,
        index, encoded_index).ok());
    std::vector<float> decoded_index(index_dim);
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_format,
        encoded_index, decoded_index).ok());
    require_bit_equal(decoded_index, index);
    encoded_index[index_dim / 2U] = std::byte{0xFFU};
    REQUIRE(!strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_format,
        encoded_index, decoded_index).ok());

    std::vector<std::byte> oracle(kv.size() * sizeof(float));
    REQUIRE(strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, strata::Dsv4KvFormat::F32,
        lossy, oracle).ok());
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, strata::Dsv4KvFormat::F32,
        oracle, decoded_kv).ok());
    require_bit_equal(decoded_kv, lossy);
}

TEST_CASE("DeepSeek KV cache keeps typed block tables and masks stale rows") {
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 2U;
    config.sliding_window_rows = 4U;
    config.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache cache(config);
    REQUIRE(cache.validate().ok());
    const auto created = cache.create_sequence();
    REQUIRE(created.ok());
    REQUIRE(cache.stats().host_used_bytes == 0U);
    REQUIRE(cache.stats().used_blocks == 0U);

    const auto kv = row(strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    for (std::uint64_t position = 0U; position < 5U; ++position) {
        REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, kv).ok());
    }
    REQUIRE(!cache.row(created.value, strata::Dsv4KvBlockKind::Sliding,
                       0U, 0U).ok());
    REQUIRE(cache.row(created.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 1U).ok());

    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Csa,
                         1U, 4U, 0U, kv).ok());
    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Hca,
                         2U, 128U, 0U, kv).ok());
    const auto index = row(
        strata::kDeepSeekV4ExecutionContract.index_head_dim, 2.0F);
    REQUIRE(cache.append(created.value,
                         strata::Dsv4KvBlockKind::LearnedIndex,
                         1U, 4U, 0U, index).ok());

    const auto csa = cache.block_table(
        created.value, strata::Dsv4KvBlockKind::Csa, 1U);
    const auto hca = cache.block_table(
        created.value, strata::Dsv4KvBlockKind::Hca, 2U);
    const auto learned = cache.block_table(
        created.value, strata::Dsv4KvBlockKind::LearnedIndex, 1U);
    REQUIRE(csa.ok() && csa.value.size() == 1U);
    REQUIRE(hca.ok() && hca.value.size() == 1U);
    REQUIRE(learned.ok() && learned.value.size() == 1U);
    REQUIRE(csa.value[0].compression_ratio == 4U);
    REQUIRE(hca.value[0].compression_ratio == 128U);
    REQUIRE(learned.value[0].row_width ==
            strata::kDeepSeekV4ExecutionContract.index_head_dim);
    REQUIRE(csa.value[0].format ==
            strata::Dsv4KvFormat::Fp8E4m3Group64Bf16Rope);
    REQUIRE(hca.value[0].format ==
            strata::Dsv4KvFormat::Fp8E4m3Group64Bf16Rope);
    REQUIRE(learned.value[0].format ==
            strata::Dsv4KvFormat::Fp4E2m1Group32);
    REQUIRE(learned.value[0].format_version == strata::kDsv4KvFormatVersion);
    REQUIRE(learned.value[0].physical_bytes > learned.value[0].payload_bytes);
    const auto compact = cache.learned_index_segments(
        created.value, 1U, 1U);
    REQUIRE(compact.ok() && compact.value.size() == 1U);
    REQUIRE(compact.value[0].rows == 1U);
    REQUIRE(compact.value[0].device_buffer == nullptr);
    std::vector<float> compact_decoded(index.size());
    const auto compact_row_bytes = strata::dsv4_kv_row_bytes(
        strata::Dsv4KvBlockKind::LearnedIndex,
        strata::Dsv4KvFormat::Fp4E2m1Group32);
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex,
        strata::Dsv4KvFormat::Fp4E2m1Group32,
        compact.value[0].host_bytes.subspan(
            compact.value[0].byte_offset, compact_row_bytes),
        compact_decoded).ok());
    require_bit_equal(compact_decoded, index);

    REQUIRE(cache.truncate_sequence(created.value, 4U).ok());
    REQUIRE(!cache.row(created.value, strata::Dsv4KvBlockKind::Hca,
                       2U, 0U).ok());
    REQUIRE(cache.reset_sequence(created.value).ok());
    REQUIRE(cache.stats().used_blocks == 0U);
    REQUIRE(cache.release_sequence(created.value).ok());
    REQUIRE(!cache.release_sequence(created.value).ok());
}

TEST_CASE("DeepSeek KV forks copy partial blocks only on write") {
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 4U;
    config.sliding_window_rows = 8U;
    config.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache cache(config);
    const auto original = cache.create_sequence();
    REQUIRE(original.ok());
    const auto first = row(strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    REQUIRE(cache.append(original.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 0U, first).ok());
    const auto fork = cache.fork_sequence(original.value);
    REQUIRE(fork.ok());
    const auto shared = cache.block_table(
        original.value, strata::Dsv4KvBlockKind::Sliding, 0U);
    REQUIRE(shared.ok() && shared.value.size() == 1U);
    REQUIRE(shared.value[0].owner_sequence == original.value);
    REQUIRE(shared.value[0].refcount == 2U);

    const auto original_second = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 2.0F);
    const auto fork_second = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 3.0F);
    REQUIRE(cache.append(original.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 1U, original_second).ok());
    REQUIRE(cache.append(fork.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 1U, fork_second).ok());
    const auto from_original = cache.row(
        original.value, strata::Dsv4KvBlockKind::Sliding, 0U, 1U);
    const auto from_fork = cache.row(
        fork.value, strata::Dsv4KvBlockKind::Sliding, 0U, 1U);
    REQUIRE(from_original.ok() && from_original.value.front() == 2.0F);
    REQUIRE(from_fork.ok() && from_fork.value.front() == 3.0F);
    REQUIRE(cache.stats().copy_on_write_blocks == 1U);

    REQUIRE(cache.release_sequence(original.value).ok());
    REQUIRE(cache.stats().used_blocks == 1U);
    REQUIRE(cache.release_sequence(fork.value).ok());
    REQUIRE(cache.stats().used_blocks == 0U);
}

TEST_CASE("DeepSeek KV truncation cannot expose stale shared rows") {
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 4U;
    config.sliding_window_rows = 8U;
    config.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache cache(config);
    const auto original = cache.create_sequence();
    REQUIRE(original.ok());
    for (std::uint64_t position = 0U; position < 4U; ++position) {
        const auto values = row(
            strata::kDeepSeekV4ExecutionContract.head_dim,
            static_cast<float>(position));
        REQUIRE(cache.append(original.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, values).ok());
    }
    const auto fork = cache.fork_sequence(original.value);
    REQUIRE(fork.ok());
    REQUIRE(cache.truncate_sequence(fork.value, 2U).ok());
    REQUIRE(!cache.row(fork.value, strata::Dsv4KvBlockKind::Sliding,
                       0U, 2U).ok());
    const auto replacement = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 9.0F);
    REQUIRE(cache.append(fork.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 2U, replacement).ok());
    REQUIRE(cache.row(fork.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 2U).value.front() == 9.0F);
    REQUIRE(cache.row(original.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 2U).value.front() == 2.0F);
}

TEST_CASE("DeepSeek KV host allocation fails at its own ceiling") {
    const auto block_bytes = strata::dsv4_kv_block_bytes(
        strata::Dsv4KvBlockKind::Sliding,
        strata::dsv4_kv_format(strata::Dsv4KvBlockKind::Sliding), 2U);
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 2U;
    config.sliding_window_rows = 4U;
    config.host_capacity_bytes = block_bytes;
    strata::Dsv4KvCache cache(config);
    const auto created = cache.create_sequence();
    REQUIRE(created.ok());
    const auto values = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 0U, values).ok());
    REQUIRE(!cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                          1U, 1U, 0U, values).ok());
    REQUIRE(cache.stats().host_used_bytes == block_bytes);
    REQUIRE(cache.stats().host_write_bytes == strata::dsv4_kv_row_bytes(
        strata::Dsv4KvBlockKind::Sliding,
        strata::dsv4_kv_format(strata::Dsv4KvBlockKind::Sliding)));
    REQUIRE(cache.stats().used_blocks == 1U);
}

TEST_CASE("DeepSeek KV device eviction protects in-flight blocks") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const std::array<int, 1> selected{devices.front()};
    REQUIRE(backend.initialize(selected, false).ok());

    const auto block_bytes = strata::dsv4_kv_block_bytes(
        strata::Dsv4KvBlockKind::Sliding,
        strata::dsv4_kv_format(strata::Dsv4KvBlockKind::Sliding), 2U);
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 2U;
    config.sliding_window_rows = 4U;
    config.host_capacity_bytes = block_bytes * 2U;
    config.devices = {devices.front()};
    config.device_capacity_bytes = {block_bytes};
    strata::Dsv4KvCache cache(config, &backend);
    const auto created = cache.create_sequence();
    REQUIRE(created.ok());
    const auto values = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    for (std::uint64_t position = 0U; position < 3U; ++position) {
        REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, values).ok());
    }

    auto first = cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 0U, 0U);
    REQUIRE(first.ok() && first.value.valid());
    REQUIRE(!cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 2U, 0U).ok());
    first.value = {};
    auto second = cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 2U, 0U);
    REQUIRE(second.ok() && second.value.valid());
    second.value = {};
    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 3U, values).ok());
    auto refreshed = cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 2U, 0U);
    REQUIRE(refreshed.ok() && refreshed.value.valid());
    REQUIRE(cache.stats().promotions == 3U);
    REQUIRE(cache.stats().evictions == 1U);
    REQUIRE(backend.stats().activation_h2d_bytes == block_bytes * 3U);
    REQUIRE(backend.stats().synchronization_calls == 3U);
}
