#include "test.hpp"

#include "strata/deepseek_kv_cache.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace {

std::vector<float> row(std::size_t width, float value) {
    return std::vector<float>(width, value);
}

}  // namespace

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
    constexpr std::uint64_t block_bytes =
        2ULL * strata::kDeepSeekV4ExecutionContract.head_dim * sizeof(float);
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
    REQUIRE(cache.stats().host_write_bytes == values.size() * sizeof(float));
    REQUIRE(cache.stats().used_blocks == 1U);
}

TEST_CASE("DeepSeek KV device eviction protects in-flight blocks") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const std::array<int, 1> selected{devices.front()};
    REQUIRE(backend.initialize(selected, false).ok());

    constexpr std::uint64_t block_bytes =
        2ULL * strata::kDeepSeekV4ExecutionContract.head_dim * sizeof(float);
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
    REQUIRE(cache.stats().promotions == 2U);
    REQUIRE(cache.stats().evictions == 1U);
    REQUIRE(backend.stats().activation_h2d_bytes == block_bytes * 2U);
    REQUIRE(backend.stats().synchronization_calls == 2U);
}
