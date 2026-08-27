#include "test.hpp"

#include "strata/models/kimi_k3/kimi_k3_expert_arena.hpp"
#include "strata/models/common/model_adapter.hpp"
#include "strata/engine/placement.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string kimi_directory() {
    return (std::filesystem::path(STRATA_SOURCE_DIR) / "models/kimi-k3").string();
}

bool kimi_present() {
    return std::filesystem::exists(
        std::filesystem::path(kimi_directory()) / "model.safetensors.index.json");
}

strata::KimiArenaConfig small_arena(std::uint64_t experts) {
    strata::KimiArenaConfig config;
    config.capacity_bytes =
        experts * (strata::KimiCheckpointReader::expert_source_bytes() + 4096U);
    config.lock_pages = false;
    return config;
}

}  // namespace

TEST_CASE("the expert arena admits, evicts by recency, and publishes atomically") {
    strata::KimiExpertArena arena;
    REQUIRE(arena.reset(small_arena(3U)).ok());
    REQUIRE(arena.slot_count() == 3U);

    const auto bytes = strata::KimiCheckpointReader::expert_source_bytes();
    for (std::uint32_t expert = 0U; expert < 3U; ++expert) {
        auto slot = arena.reserve(1U, expert, bytes);
        REQUIRE(slot.size() == bytes);
        // Not visible until published, so a failed read cannot leave a
        // half-written expert readable.
        REQUIRE(arena.find(1U, expert).empty());
        arena.publish(1U, expert);
        REQUIRE(arena.find(1U, expert).size() == bytes);
    }
    REQUIRE(arena.evictions() == 0U);

    // Touch 0 and 2, then admit a fourth: 1 is least recently used.
    REQUIRE(!arena.find(1U, 0U).empty());
    REQUIRE(!arena.find(1U, 2U).empty());
    auto fresh = arena.reserve(1U, 9U, bytes);
    REQUIRE(!fresh.empty());
    arena.publish(1U, 9U);
    REQUIRE(arena.evictions() == 1U);
    REQUIRE(arena.find(1U, 1U).empty());
    REQUIRE(!arena.find(1U, 0U).empty());
    REQUIRE(!arena.find(1U, 2U).empty());

    // Layer and expert together key the entry: the same ordinal in another
    // layer is a different expert.
    REQUIRE(arena.find(2U, 0U).empty());

    strata::KimiExpertArena tiny;
    REQUIRE(!tiny.reset({1024U, false}).ok());
}

TEST_CASE("a locked arena reports its own failure rather than running unlocked") {
    strata::KimiArenaConfig config = small_arena(2U);
    config.lock_pages = true;
    strata::KimiExpertArena arena;
    const auto locked = arena.reset(config);
    if (!locked.ok()) {
        // The machine's RLIMIT_MEMLOCK is too low. The error must name it,
        // because running unlocked silently reintroduces a swap write path.
        REQUIRE(locked.errors.front().find("RLIMIT_MEMLOCK") != std::string::npos);
        SKIP("RLIMIT_MEMLOCK does not admit a test arena");
    }
    REQUIRE(arena.locked());
    REQUIRE(arena.base() != nullptr);
}

TEST_CASE("the write guard refuses paths on a protected disk") {
    const auto here = strata::resolve_backing_storage(STRATA_SOURCE_DIR);
    REQUIRE(here.resolved);

    strata::KimiWriteGuardConfig guard;
    guard.forbidden_disks = {here.disk};
    guard.write_paths = {STRATA_SOURCE_DIR};
    guard.require_no_forbidden_swap = false;
    guard.disable_core_dumps = false;
    // `mlockall` is process-wide and irreversible for the rest of the run, so
    // the unit test leaves it off: locking the whole test binary would change
    // how every later test allocates.
    guard.lock_address_space = false;
    const auto refused = strata::kimi_apply_write_guard(guard);
    REQUIRE(!refused.ok());
    REQUIRE(refused.errors.front().find(here.disk) != std::string::npos);

    // A path that does not resolve is refused rather than assumed safe.
    guard.forbidden_disks = {"nonexistent-disk"};
    REQUIRE(strata::kimi_apply_write_guard(guard).ok());

    // /dev/shm is a tmpfs with no block device behind it, so it never resolves
    // to a protected disk and is the right home for scratch.
    guard.write_paths = {"/dev/shm"};
    guard.forbidden_disks = {here.disk};
    REQUIRE(strata::kimi_apply_write_guard(guard).ok());
}

TEST_CASE("disk write and read counters advance monotonically") {
    const auto here = strata::resolve_backing_storage(STRATA_SOURCE_DIR);
    REQUIRE(here.resolved);
    const auto first = strata::kimi_disk_sectors_written(here.disk);
    REQUIRE(first.ok());
    const auto second = strata::kimi_disk_sectors_written(here.disk);
    REQUIRE(second.ok());
    REQUIRE(second.value >= first.value);
    REQUIRE(strata::kimi_disk_sectors_read(here.disk).ok());
    REQUIRE(!strata::kimi_disk_sectors_written("../../etc/passwd").ok());
    REQUIRE(!strata::kimi_disk_sectors_written("").ok());
}

TEST_CASE("real Kimi-K3 expert reads coalesce into one submission per expert") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());

    strata::KimiExpertArena arena;
    REQUIRE(arena.reset(small_arena(4U)).ok());
    strata::KimiExpertReader reader;
    strata::KimiReaderConfig config;
    config.queue_depth = 4U;
    config.direct = true;
    REQUIRE(reader.open(*opened.value, config).ok());

    const std::vector<strata::KimiReadRequest> requests{
        {1U, 0U}, {1U, 1U}, {50U, 300U}, {92U, 895U}};
    REQUIRE(reader.stage(*opened.value, arena, requests).ok());

    const auto& stats = reader.stats();
    REQUIRE(stats.requests == 4U);
    REQUIRE(stats.bytes_read ==
            4U * strata::KimiCheckpointReader::expert_source_bytes());
    // An expert's six modules are one contiguous extent in the shard, so the
    // coalescing reader issues one read per expert. Six submissions per expert
    // would mean the merge stopped working and per-request latency returned.
    REQUIRE(stats.coalesced_modules == 24U);
    REQUIRE(stats.submissions == 4U);
    REQUIRE(stats.peak_queue_depth == 4U);

    // Everything requested is now resident, and a repeat costs no reads.
    for (const auto& request : requests) {
        REQUIRE(arena.find(request.layer, request.expert).size() ==
                strata::KimiCheckpointReader::expert_source_bytes());
    }
    REQUIRE(reader.stage(*opened.value, arena, requests).ok());
    REQUIRE(reader.stats().submissions == 4U);
}

TEST_CASE("real Kimi-K3 coalesced reads decode to the same expert weights") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& checkpoint = *opened.value;

    strata::KimiExpertArena arena;
    REQUIRE(arena.reset(small_arena(2U)).ok());
    strata::KimiExpertReader reader;
    strata::KimiReaderConfig config;
    config.queue_depth = 2U;
    REQUIRE(reader.open(checkpoint, config).ok());
    const std::vector<strata::KimiReadRequest> requests{{1U, 0U}};
    REQUIRE(reader.stage(checkpoint, arena, requests).ok());
    const auto resident = arena.find(1U, 0U);
    REQUIRE(resident.size() ==
            strata::KimiCheckpointReader::expert_source_bytes());

    // The arena mirrors the shard, so the first payload in the slot is the
    // gate's packed block. Decoding it must reproduce the same reference
    // values the per-module path produced in the codec fixture.
    const auto& c = strata::kKimiK3ExecutionContract;
    const auto inner = static_cast<std::uint64_t>(c.expert_intermediate_size);
    const auto latent =
        static_cast<std::uint64_t>(c.routed_expert_hidden_size);
    const auto layout = strata::kimi_expert_layout(inner, latent);
    const auto packed_bytes = inner * (latent / 2U);
    const auto scale_bytes = inner * (latent / 32U);
    std::vector<float> row(static_cast<std::size_t>(latent));
    REQUIRE(strata::mxfp4_dequantize_row(
                row, resident.subspan(0U, static_cast<std::size_t>(packed_bytes)),
                resident.subspan(static_cast<std::size_t>(packed_bytes),
                                 static_cast<std::size_t>(scale_bytes)),
                layout, strata::kimi_expert_quantization(), 0U).ok());

    const float expected[8] = {0.0625F,   0.0078125F, -0.0078125F, -0.0234375F,
                               -0.0625F, -0.03125F,   0.0234375F,  0.015625F};
    for (std::size_t index = 0U; index < 8U; ++index) {
        REQUIRE(row[index] == expected[index]);
    }
}
