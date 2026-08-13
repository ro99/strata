#include "test.hpp"

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/dsv4_rank_local_kv.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace {

constexpr std::uint32_t kHeadDim =
    strata::kDeepSeekV4ExecutionContract.head_dim;
constexpr std::uint32_t kIndexHeadDim =
    strata::kDeepSeekV4ExecutionContract.index_head_dim;
constexpr std::uint32_t kLayer = 2U;
constexpr std::uint32_t kRatio = 4U;

constexpr std::array<std::size_t, strata::kDsv4RankLocalWorld> kSlots{0U, 1U};

// Physical reservation needs a CUDA-backed cache with two device slots: the
// whole point of the transaction is that one logical row lands on both ranks'
// devices.
struct RankLocalFixture {
    std::optional<strata::CudaBackend> backend;
    std::optional<strata::Dsv4KvCache> cache;
    strata::Dsv4SequenceHandle sequence{};
    bool available{};

    RankLocalFixture() {
        const auto devices = strata::CudaBackend::available_devices();
        if (!strata::CudaBackend::compiled() || devices.size() < 2U) return;
        backend.emplace();
        const std::array<int, 2> selected{devices[0], devices[1]};
        if (!backend->initialize(selected, false).ok()) return;
        strata::Dsv4KvCacheConfig config;
        config.block_rows = strata::kDsv4PhysicalKvBlockRows;
        config.sliding_window_rows =
            strata::kDeepSeekV4ExecutionContract.sliding_window;
        config.host_capacity_bytes = 256ULL << 20U;
        config.physical_layout = true;
        config.devices.assign(selected.begin(), selected.end());
        config.device_capacity_bytes.assign(2U, 64ULL << 20U);
        cache.emplace(config, &*backend);
        auto created = cache->create_sequence();
        if (!created.ok()) return;
        sequence = created.value;
        available = true;
    }
};

std::vector<float> row(std::size_t width, float value) {
    return std::vector<float>(width, value);
}

// Reservation is contiguous per sequence, so a test that wants position N has
// to have published every position below it first.
[[nodiscard]] bool publish_token(strata::Dsv4KvCache& cache,
                                 strata::Dsv4SequenceHandle sequence,
                                 std::uint32_t position, float value) {
    strata::Dsv4RankLocalKvTransaction transaction(
        cache, sequence, kSlots, position);
    if (!transaction.reserve_layer(
            kLayer, position, kRatio, strata::Dsv4KvBlockKind::Csa).ok()) {
        return false;
    }
    const auto bytes = static_cast<std::size_t>(
        transaction.patch_bytes(kLayer));
    std::vector<std::byte> first(bytes);
    std::vector<std::byte> second(bytes);
    const std::array<std::span<std::byte>, 2> patches{first, second};
    if (!transaction.commit_layer(
            kLayer, row(kHeadDim, value), row(kHeadDim, value),
            patches).ok()) {
        return false;
    }
    return transaction.commit().ok();
}

}  // namespace

TEST_CASE("rank-local selection agreement reports the first divergence") {
    const std::array<std::uint32_t, 4> base{7U, 11U, 19U, 23U};
    const std::array<std::uint32_t, 4> same{7U, 11U, 19U, 23U};
    auto agreement = strata::dsv4_rank_local_selection_agreement(base, same);
    REQUIRE(agreement.agree);
    REQUIRE(agreement.first_mismatch == base.size());

    // Both ranks score replicated state independently, so a divergence is a
    // replica fault. It has to be located, not merely detected, because the
    // token is aborted rather than reconciled.
    const std::array<std::uint32_t, 4> diverged{7U, 11U, 20U, 23U};
    agreement = strata::dsv4_rank_local_selection_agreement(base, diverged);
    REQUIRE(!agreement.agree);
    REQUIRE(agreement.first_mismatch == 2U);
    REQUIRE(agreement.left == 19U);
    REQUIRE(agreement.right == 20U);

    // A short selection is a divergence too, not a prefix match.
    const std::array<std::uint32_t, 2> truncated{7U, 11U};
    agreement = strata::dsv4_rank_local_selection_agreement(base, truncated);
    REQUIRE(!agreement.agree);
    REQUIRE(agreement.left == 4U);
    REQUIRE(agreement.right == 2U);

    const std::span<const std::uint32_t> empty;
    agreement = strata::dsv4_rank_local_selection_agreement(empty, empty);
    REQUIRE(agreement.agree);
}

TEST_CASE("rank-local KV reserves a logical row once and replicates it") {
    RankLocalFixture fixture;
    if (!fixture.available) return;

    // A logical row belongs to the sequence, not to a device. Reserving it per
    // rank would advance the table's end row twice and reject the second call
    // as non-contiguous, so reservation happens once and every other rank
    // leases the same block.
    strata::Dsv4RankLocalKvTransaction transaction(
        *fixture.cache, fixture.sequence, kSlots, 0U);
    REQUIRE(transaction.reserve_layer(
        kLayer, 0U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());
    REQUIRE(transaction.reserved_layers() == 1U);

    // Both ranks patch the same offsets and extents on their own device.
    std::vector<strata::CudaDsv4AttentionPageWrite> first;
    std::vector<strata::CudaDsv4AttentionPageWrite> second;
    REQUIRE(transaction.page_writes(kLayer, 0U, first).ok());
    REQUIRE(transaction.page_writes(kLayer, 1U, second).ok());
    REQUIRE(first.size() == second.size());
    REQUIRE(!first.empty());
    std::uint64_t total = 0U;
    for (std::size_t index = 0U; index < first.size(); ++index) {
        REQUIRE(first[index].data_offset == second[index].data_offset);
        REQUIRE(first[index].scale_offset == second[index].scale_offset);
        REQUIRE(first[index].data_bytes == second[index].data_bytes);
        REQUIRE(first[index].scale_bytes == second[index].scale_bytes);
        // Different devices, so necessarily different buffers.
        REQUIRE(first[index].buffer != second[index].buffer);
        total += first[index].data_bytes + first[index].scale_bytes;
    }
    REQUIRE(total == transaction.patch_bytes(kLayer));

    // An unreserved layer or an out-of-range rank has no page writes to give.
    REQUIRE(!transaction.page_writes(kLayer + 1U, 0U, first).ok());
    REQUIRE(!transaction.page_writes(kLayer, 2U, first).ok());
    REQUIRE(transaction.patch_bytes(kLayer + 1U) == 0U);
}

TEST_CASE("rank-local KV publishes one encode into every rank's staging") {
    RankLocalFixture fixture;
    if (!fixture.available) return;

    // Positions 0..2 first, so position 3 lands on the ratio-4 compressor
    // boundary and reserves a compressed and a learned-index row too.
    for (std::uint32_t position = 0U; position < 3U; ++position) {
        REQUIRE(publish_token(*fixture.cache, fixture.sequence, position,
                              static_cast<float>(position + 1U)));
    }

    constexpr std::uint32_t position = 3U;
    strata::Dsv4RankLocalKvTransaction transaction(
        *fixture.cache, fixture.sequence, kSlots, position);
    REQUIRE(transaction.reserve_layer(
        kLayer, position, kRatio, strata::Dsv4KvBlockKind::Csa, kRatio).ok());

    // Sliding, compressed and learned-index rows: three page writes.
    std::vector<strata::CudaDsv4AttentionPageWrite> writes;
    REQUIRE(transaction.page_writes(kLayer, 0U, writes).ok());
    REQUIRE(writes.size() == 3U);

    // Reservation advances block-table metadata and makes the row
    // addressable, but writes nothing: the page still holds its zeroed bytes.
    // What commit publishes is content, and what abort withdraws is the row.
    {
        const auto reserved = fixture.cache->row(
            fixture.sequence, strata::Dsv4KvBlockKind::Sliding, kLayer,
            position);
        REQUIRE(reserved.ok());
        REQUIRE(std::all_of(reserved.value.begin(), reserved.value.end(),
                            [](float value) { return value == 0.0F; }));
    }

    const auto bytes = static_cast<std::size_t>(
        transaction.patch_bytes(kLayer));
    std::vector<std::byte> first(bytes);
    std::vector<std::byte> second(bytes);
    const std::array<std::span<std::byte>, 2> patches{first, second};
    REQUIRE(transaction.commit_layer(
        kLayer, row(kHeadDim, 0.5F), row(kHeadDim, 0.25F), patches,
        row(kIndexHeadDim, 0.125F)).ok());

    // The row is encoded once and copied, so the two ranks cannot disagree.
    REQUIRE(first == second);

    REQUIRE(transaction.commit().ok());
    REQUIRE(transaction.committed());

    // The canonical host page now carries the committed row, and the pinned
    // staging carries exactly the bytes each rank's queued patch will apply.
    const auto stored = fixture.cache->row(
        fixture.sequence, strata::Dsv4KvBlockKind::Sliding, kLayer, position);
    REQUIRE(stored.ok());
    REQUIRE(std::all_of(stored.value.begin(), stored.value.end(),
                        [](float value) { return value == 0.5F; }));
}

TEST_CASE("rank-local KV refuses a double commit, a re-reserve and a bad span") {
    RankLocalFixture fixture;
    if (!fixture.available) return;

    strata::Dsv4RankLocalKvTransaction transaction(
        *fixture.cache, fixture.sequence, kSlots, 0U);
    REQUIRE(transaction.reserve_layer(
        kLayer, 0U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());
    // Reserving the same layer twice within one token would give the position
    // two rows.
    REQUIRE(!transaction.reserve_layer(
        kLayer, 0U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());

    const auto bytes = static_cast<std::size_t>(
        transaction.patch_bytes(kLayer));

    // Patch staging is fixed per command; a short or long span is a defect in
    // the caller's accounting and must not be silently absorbed. The check
    // runs before any encode, so a rejected commit leaves the layer
    // uncommitted rather than half written.
    std::vector<std::byte> sized(bytes);
    std::vector<std::byte> shortened(bytes - 1U);
    const std::array<std::span<std::byte>, 2> mismatched{sized, shortened};
    REQUIRE(!transaction.commit_layer(
        kLayer, row(kHeadDim, 1.0F), {}, mismatched).ok());
    std::vector<std::byte> lengthened(bytes + 1U);
    const std::array<std::span<std::byte>, 2> overlong{sized, lengthened};
    REQUIRE(!transaction.commit_layer(
        kLayer, row(kHeadDim, 1.0F), {}, overlong).ok());

    std::vector<std::byte> first(bytes);
    std::vector<std::byte> second(bytes);
    const std::array<std::span<std::byte>, 2> patches{first, second};
    REQUIRE(transaction.commit_layer(
        kLayer, row(kHeadDim, 1.0F), {}, patches).ok());
    REQUIRE(!transaction.commit_layer(
        kLayer, row(kHeadDim, 1.0F), {}, patches).ok());
    REQUIRE(transaction.commit().ok());
    // A committed transaction is closed to everything.
    REQUIRE(!transaction.commit().ok());
    REQUIRE(!transaction.reserve_layer(
        kLayer, 1U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());
}

TEST_CASE("rank-local KV rolls a partial token back on abort") {
    RankLocalFixture fixture;
    if (!fixture.available) return;

    REQUIRE(publish_token(*fixture.cache, fixture.sequence, 0U, 2.0F));
    const auto published = fixture.cache->block_table(
        fixture.sequence, strata::Dsv4KvBlockKind::Sliding, kLayer);
    REQUIRE(published.ok());

    // A token that reserves but never commits, then leaves scope, must not
    // survive as a visible row. The destructor aborts, so an early return
    // anywhere in a 43-layer chain cannot publish a partial position.
    {
        strata::Dsv4RankLocalKvTransaction transaction(
            *fixture.cache, fixture.sequence, kSlots, 1U);
        REQUIRE(transaction.reserve_layer(
            kLayer, 1U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());
    }
    REQUIRE(!fixture.cache->row(fixture.sequence,
                                strata::Dsv4KvBlockKind::Sliding, kLayer,
                                1U).ok());

    // The accepted token is untouched by the rollback.
    const auto after = fixture.cache->block_table(
        fixture.sequence, strata::Dsv4KvBlockKind::Sliding, kLayer);
    REQUIRE(after.ok());
    REQUIRE(after.value.size() == published.value.size());
    REQUIRE(fixture.cache->row(fixture.sequence,
                               strata::Dsv4KvBlockKind::Sliding, kLayer,
                               0U).ok());

    // An explicit abort is idempotent and still refuses to publish.
    {
        strata::Dsv4RankLocalKvTransaction transaction(
            *fixture.cache, fixture.sequence, kSlots, 1U);
        REQUIRE(transaction.reserve_layer(
            kLayer, 1U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());
        transaction.abort();
        transaction.abort();
        REQUIRE(!transaction.commit().ok());
        REQUIRE(!transaction.committed());
        REQUIRE(!transaction.reserve_layer(
            kLayer, 1U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());
    }

    // The position the aborted tokens occupied still accepts a real one.
    REQUIRE(publish_token(*fixture.cache, fixture.sequence, 1U, 3.0F));
}

TEST_CASE("rank-local KV replication requires two distinct device slots") {
    RankLocalFixture fixture;
    if (!fixture.available) return;

    // Both ranks on one slot would give one device two copies and the other
    // none, which is not replication.
    const std::array<std::size_t, 2> collided{1U, 1U};
    strata::Dsv4RankLocalKvTransaction transaction(
        *fixture.cache, fixture.sequence, collided, 0U);
    REQUIRE(!transaction.reserve_layer(
        kLayer, 0U, kRatio, strata::Dsv4KvBlockKind::Csa).ok());
    REQUIRE(transaction.reserved_layers() == 0U);
}
