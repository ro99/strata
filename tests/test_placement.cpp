#include "test.hpp"

#include "strata/engine/placement.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>

#include <unistd.h>

namespace {

constexpr std::uint64_t kGigabyte = 1ULL << 30U;

strata::PlacementHardware make_hardware(
    const std::vector<std::uint64_t>& device_gigabytes,
    std::uint64_t host_gigabytes = 64U) {
    strata::PlacementHardware hardware;
    for (std::size_t slot = 0U; slot < device_gigabytes.size(); ++slot) {
        strata::PlacementDevice device;
        device.id = static_cast<int>(slot);
        device.name = "cuda:" + std::to_string(slot);
        device.total_bytes = device_gigabytes[slot] * kGigabyte;
        device.free_bytes = device.total_bytes;
        hardware.devices.push_back(std::move(device));
    }
    hardware.host_total_bytes = host_gigabytes * kGigabyte;
    hardware.host_available_bytes = host_gigabytes * kGigabyte;
    return hardware;
}

strata::PlacementRequest make_request(std::size_t devices,
                                      std::uint32_t context = 4096U) {
    strata::PlacementRequest request;
    request.model = strata::PlacementModel::Gemma4;
    request.model_directory = "/models/test";
    request.devices.resize(devices);
    std::iota(request.devices.begin(), request.devices.end(), 0);
    request.vram_cache_fraction = 0.9;
    request.maximum_context_tokens = context;
    return request;
}

// A dense model of `layers` equal layers, each `layer_gigabytes` in VRAM and
// read once per decode step.
strata::PlacementInventory make_dense_inventory(std::uint32_t layers,
                                                std::uint64_t layer_gigabytes) {
    strata::PlacementInventory inventory;
    inventory.model = strata::PlacementModel::Gemma4;
    inventory.model_name = "dense-test";
    inventory.layer_count = layers;
    inventory.maximum_context_tokens = 4096U;
    inventory.per_device_workspace_bytes = kGigabyte / 2U;
    inventory.minimum_device_budget_bytes = kGigabyte;
    inventory.prescriptive = true;
    for (std::uint32_t layer = 0U; layer < layers; ++layer) {
        strata::PlacementItem item;
        item.component = strata::PlacementClass::FeedForward;
        item.layer = static_cast<std::int32_t>(layer);
        item.device_bytes = layer_gigabytes * kGigabyte;
        item.source_bytes = item.device_bytes;
        item.decode_read_bytes = item.device_bytes;
        inventory.items.push_back(item);
    }
    return inventory;
}

std::uint64_t block_bytes(const strata::PlacementPlan& plan,
                          const strata::PlacementInventory& inventory,
                          std::size_t slot) {
    std::uint64_t bytes = 0U;
    for (const auto& item : inventory.items) {
        if (item.layer < 0) continue;
        if (plan.layer_device[static_cast<std::size_t>(item.layer)] != slot) continue;
        bytes += item.device_bytes;
    }
    return bytes;
}

}  // namespace

TEST_CASE("placement keeps a dense model contiguous and inside every budget") {
    const auto inventory = make_dense_inventory(12U, 1U);
    const auto hardware = make_hardware({8U, 8U});
    const auto result = solve_placement(inventory, hardware, make_request(2U));
    REQUIRE(result.ok());
    const auto& plan = result.value;
    REQUIRE(plan.fits);
    REQUIRE(!plan.io_dependent);
    REQUIRE(plan.layer_device.size() == 12U);
    REQUIRE(std::is_sorted(plan.layer_device.begin(), plan.layer_device.end()));
    REQUIRE(plan.cross_device_activation_hops == 1U);
    for (std::size_t slot = 0U; slot < 2U; ++slot) {
        REQUIRE(plan.device_resident_bytes[slot] <= plan.device_budget_bytes[slot]);
    }
    REQUIRE(plan.decode_device_read_bytes == 12U * kGigabyte);
    REQUIRE(plan.decode_host_to_device_bytes == 0U);
    REQUIRE(plan.decode_storage_read_bytes == 0U);
}

TEST_CASE("placement gives the larger GPU the larger share of layers") {
    const auto inventory = make_dense_inventory(20U, 1U);
    const auto hardware = make_hardware({8U, 24U});
    const auto result = solve_placement(inventory, hardware, make_request(2U));
    REQUIRE(result.ok());
    const auto& plan = result.value;
    REQUIRE(plan.fits);
    const auto first = block_bytes(plan, inventory, 0U);
    const auto second = block_bytes(plan, inventory, 1U);
    REQUIRE(first + second == 20U * kGigabyte);
    REQUIRE(second > first);
    // Proportional to capacity, not to device count: an even split would put
    // ten layers on the 8 GB card, which does not fit after its workspace.
    REQUIRE(first <= plan.device_budget_bytes[0U]);
}

TEST_CASE("placement refuses a dense model that outgrows aggregate VRAM") {
    const auto inventory = make_dense_inventory(40U, 1U);
    const auto hardware = make_hardware({8U, 8U});
    const auto result = solve_placement(inventory, hardware, make_request(2U));
    REQUIRE(!result.ok());
    // Caching cannot manufacture sparsity in a dense model: the planner reports
    // the shortfall instead of silently spilling a densely read class.
    REQUIRE(result.errors.front().find("admitted VRAM budget") != std::string::npos);
}

TEST_CASE("sparse experts spill to host and then to NVMe") {
    auto inventory = make_dense_inventory(4U, 1U);
    strata::PlacementItem experts;
    experts.component = strata::PlacementClass::RoutedExpert;
    experts.device_bytes = 40U * kGigabyte;
    experts.host_bytes = 40U * kGigabyte;
    experts.source_bytes = experts.device_bytes;
    experts.decode_read_bytes = 4U * kGigabyte;
    experts.spillable = true;
    inventory.items.push_back(experts);

    const auto roomy = solve_placement(inventory, make_hardware({16U, 16U}, 64U),
                                       make_request(2U));
    REQUIRE(roomy.ok());
    REQUIRE(!roomy.value.io_dependent);
    REQUIRE(roomy.value.host_resident_bytes > 0U);
    REQUIRE(roomy.value.storage_resident_bytes == 0U);

    const auto cramped = solve_placement(inventory, make_hardware({16U, 16U}, 8U),
                                         make_request(2U));
    REQUIRE(cramped.ok());
    REQUIRE(cramped.value.io_dependent);
    REQUIRE(cramped.value.storage_resident_bytes > 0U);
    REQUIRE(!cramped.value.fits);
    REQUIRE(cramped.value.decode_storage_read_bytes > 0U);
}

TEST_CASE("per-step read volume is conserved across tiers") {
    auto inventory = make_dense_inventory(4U, 1U);
    strata::PlacementItem experts;
    experts.component = strata::PlacementClass::RoutedExpert;
    experts.device_bytes = 40U * kGigabyte;
    experts.host_bytes = 40U * kGigabyte;
    experts.decode_read_bytes = 4U * kGigabyte;
    experts.spillable = true;
    inventory.items.push_back(experts);
    const auto result = solve_placement(inventory, make_hardware({16U, 16U}, 8U),
                                        make_request(2U));
    REQUIRE(result.ok());
    const auto& plan = result.value;
    REQUIRE(plan.decode_device_read_bytes + plan.decode_host_to_device_bytes +
                plan.decode_storage_read_bytes ==
            8U * kGigabyte);
}

TEST_CASE("a descriptive plan reproduces the weighted round robin") {
    auto inventory = make_dense_inventory(6U, 1U);
    inventory.contiguous_layer_blocks = false;
    inventory.prescriptive = false;
    const auto result = solve_placement(inventory, make_hardware({16U, 16U}),
                                        make_request(2U));
    REQUIRE(result.ok());
    const auto& plan = result.value;
    REQUIRE(plan.weighted_schedule.size() == 4U);
    for (std::size_t layer = 0U; layer < plan.layer_device.size(); ++layer) {
        REQUIRE(plan.layer_device[layer] ==
                plan.weighted_schedule[layer % plan.weighted_schedule.size()]);
    }
    REQUIRE(!plan.prescriptive);
}

TEST_CASE("a fixed-slot item stays on the device its runtime requires") {
    auto inventory = make_dense_inventory(8U, 1U);
    strata::PlacementItem vision;
    vision.component = strata::PlacementClass::Vision;
    vision.fixed_device_slot = 0;
    vision.device_bytes = 2U * kGigabyte;
    vision.source_bytes = vision.device_bytes;
    inventory.items.push_back(vision);
    const auto result = solve_placement(inventory, make_hardware({8U, 8U}),
                                        make_request(2U));
    REQUIRE(result.ok());
    const auto& plan = result.value;
    REQUIRE(strata::placement_component_bytes(plan, strata::PlacementClass::Vision,
                                              0U) == 2U * kGigabyte);
    REQUIRE(strata::placement_component_bytes(plan, strata::PlacementClass::Vision,
                                              1U) == 0U);
    // The vision tower is not read while decoding text.
    REQUIRE(plan.decode_device_read_bytes == 8U * kGigabyte);
}

TEST_CASE("a plan survives a JSON round trip") {
    const auto inventory = make_dense_inventory(12U, 1U);
    auto result = solve_placement(inventory, make_hardware({8U, 24U}),
                                  make_request(2U, 8192U));
    REQUIRE(result.ok());
    result.value.model_identity = "0123456789abcdef";
    result.value.maximum_context_tokens_that_fit = 16384U;
    const auto encoded = strata::encode_placement_plan(result.value);
    const auto decoded = strata::decode_placement_plan(encoded);
    REQUIRE(decoded.ok());
    const auto& before = result.value;
    const auto& after = decoded.value;
    REQUIRE(after.version == before.version);
    REQUIRE(after.model_identity == before.model_identity);
    REQUIRE(after.request.devices == before.request.devices);
    REQUIRE(after.request.maximum_context_tokens ==
            before.request.maximum_context_tokens);
    REQUIRE(after.layer_device == before.layer_device);
    REQUIRE(after.device_resident_bytes == before.device_resident_bytes);
    REQUIRE(after.components.size() == before.components.size());
    REQUIRE(after.decode_device_read_bytes == before.decode_device_read_bytes);
    REQUIRE(after.maximum_context_tokens_that_fit == 16384U);
    REQUIRE(after.fits == before.fits);
}

TEST_CASE("a plan file is keyed by request and checkpoint identity") {
    const auto first = make_request(2U, 4096U);
    auto second = first;
    second.maximum_context_tokens = 8192U;
    auto third = first;
    third.devices = {0, 1, 2};
    REQUIRE(strata::placement_plan_filename(first, "aaaa") !=
            strata::placement_plan_filename(second, "aaaa"));
    REQUIRE(strata::placement_plan_filename(first, "aaaa") !=
            strata::placement_plan_filename(third, "aaaa"));
    REQUIRE(strata::placement_plan_filename(first, "aaaa") !=
            strata::placement_plan_filename(first, "bbbb"));
    REQUIRE(strata::placement_plan_filename(first, "aaaa") ==
            strata::placement_plan_filename(first, "aaaa"));
    // A directory component must never escape the cache directory.
    strata::PlacementRequest hostile(first);
    hostile.model_directory = "/models/../../etc";
    const auto name = strata::placement_plan_filename(hostile, "../../etc/passwd");
    REQUIRE(name.find('/') == std::string::npos);
}

TEST_CASE("the plan cache stores, reloads, and rejects a stale plan") {
    const auto directory = std::filesystem::temp_directory_path() /
        ("strata-plan-test-" + std::to_string(::getpid()));
    std::filesystem::remove_all(directory);
    const auto hardware = make_hardware({8U, 24U});
    const auto request = make_request(2U);
    auto planned = solve_placement(make_dense_inventory(12U, 1U), hardware, request);
    REQUIRE(planned.ok());
    planned.value.model_identity = "cafebabecafebabe";
    REQUIRE(strata::store_placement_plan(directory.string(), planned.value).ok());

    const auto hit = strata::load_placement_plan(directory.string(), request,
                                                 hardware, "cafebabecafebabe");
    REQUIRE(hit.ok());
    REQUIRE(hit.value.version == strata::kPlacementPlanVersion);
    REQUIRE(hit.value.layer_device == planned.value.layer_device);

    // A checkpoint that changed under the plan is a miss, not a silent reuse.
    const auto changed = strata::load_placement_plan(directory.string(), request,
                                                     hardware, "0000000000000000");
    REQUIRE(changed.ok());
    REQUIRE(changed.value.version == 0U);

    // So is different hardware.
    const auto other_hardware = make_hardware({8U, 16U});
    const auto swapped = strata::load_placement_plan(
        directory.string(), request, other_hardware, "cafebabecafebabe");
    REQUIRE(swapped.ok());
    REQUIRE(swapped.value.version == 0U);
    std::filesystem::remove_all(directory);
}

TEST_CASE("verification fails when free VRAM no longer covers the plan") {
    const auto hardware = make_hardware({8U, 8U});
    const auto planned = solve_placement(make_dense_inventory(12U, 1U), hardware,
                                         make_request(2U));
    REQUIRE(planned.ok());
    REQUIRE(strata::verify_placement_plan(planned.value, hardware).ok());

    auto squeezed = hardware;
    squeezed.devices[0U].free_bytes = kGigabyte;
    const auto verified = strata::verify_placement_plan(planned.value, squeezed);
    REQUIRE(!verified.ok());
    REQUIRE(verified.errors.front().find("is free") != std::string::npos);

    auto swapped = hardware;
    swapped.devices[1U].total_bytes = 48U * kGigabyte;
    REQUIRE(!strata::verify_placement_plan(planned.value, swapped).ok());
}

TEST_CASE("byte formatting is stable across magnitudes") {
    REQUIRE(strata::format_bytes(0U) == "0 B");
    REQUIRE(strata::format_bytes(1024U) == "1.00 KiB");
    REQUIRE(strata::format_bytes(kGigabyte * 35U / 10U) == "3.50 GiB");
}

// ------------------------------------------------------- storage tier (v2)

namespace {

// One densely read class that may not spill past the host tier, and one sparse
// class that may reach storage. This is Kimi-K3's shape: a BF16 spine larger
// than VRAM alongside a routed-expert set larger than host memory.
strata::PlacementInventory make_tiered_inventory(
    std::uint64_t dense_gigabytes, std::uint64_t sparse_gigabytes) {
    strata::PlacementInventory inventory;
    inventory.model = strata::PlacementModel::KimiK3;
    inventory.model_name = "tiered-test";
    inventory.layer_count = 4U;
    inventory.maximum_context_tokens = 2048U;
    inventory.per_device_workspace_bytes = kGigabyte / 2U;
    inventory.minimum_device_budget_bytes = kGigabyte;
    inventory.host_reserve_bytes = 0U;
    inventory.contiguous_layer_blocks = false;

    strata::PlacementItem dense;
    dense.component = strata::PlacementClass::Attention;
    dense.device_bytes = dense_gigabytes * kGigabyte;
    dense.host_bytes = dense.device_bytes;
    dense.source_bytes = dense.device_bytes;
    dense.decode_read_bytes = dense.device_bytes;
    dense.spillable = true;
    dense.deepest_tier = strata::PlacementTier::Host;
    inventory.items.push_back(dense);

    strata::PlacementItem sparse;
    sparse.component = strata::PlacementClass::RoutedExpert;
    sparse.device_bytes = sparse_gigabytes * kGigabyte;
    sparse.host_bytes = sparse.device_bytes;
    sparse.source_bytes = sparse.device_bytes;
    sparse.decode_read_bytes = kGigabyte;
    sparse.spillable = true;
    sparse.device_cache_only = true;
    sparse.deepest_tier = strata::PlacementTier::Storage;
    inventory.items.push_back(sparse);
    return inventory;
}

}  // namespace

TEST_CASE("a dense class spills to host but never to storage") {
    // 20 GiB of dense weights against 13.4 GiB of admitted VRAM and a routed
    // set larger than host memory: a third of the dense class must stream from
    // host every step, which is a fit, not a defect.
    const auto hardware = make_hardware({8U, 8U}, 64U);
    auto request = make_request(2U);
    request.model = strata::PlacementModel::KimiK3;
    const auto planned = solve_placement(make_tiered_inventory(20U, 400U),
                                         hardware, request);
    REQUIRE(planned.ok());
    const auto& plan = planned.value;
    REQUIRE(plan.io_dependent);
    REQUIRE(plan.storage_resident_bytes != 0U);

    // The dense class claims VRAM ahead of the sparse one: under a max over
    // resources, VRAM spent on a class read every step beats VRAM spent on a
    // class read once per routed hit.
    const auto attention = strata::placement_component_bytes(
        plan, strata::PlacementClass::Attention, 0U);
    REQUIRE(attention != 0U);
    REQUIRE(strata::placement_component_bytes(
                plan, strata::PlacementClass::RoutedExpert, 0U) == 0U);
    // Only the sparse class reaches storage.
    for (const auto& component : plan.components) {
        if (component.component == strata::PlacementClass::Attention) {
            REQUIRE(component.tier != strata::PlacementTier::Storage);
        }
    }
}

TEST_CASE("a dense class larger than device plus host is an error, not a spill") {
    // 200 GiB of densely read weights against 16 GiB of VRAM and 64 GiB of
    // host: the charter's I/O-dependent case, which must be reported.
    const auto hardware = make_hardware({8U, 8U}, 64U);
    auto request = make_request(2U);
    request.model = strata::PlacementModel::KimiK3;
    const auto planned = solve_placement(make_tiered_inventory(200U, 40U),
                                         hardware, request);
    REQUIRE(!planned.ok());
    REQUIRE(planned.errors.front().find("may not spill past the host tier") !=
            std::string::npos);
    REQUIRE(planned.errors.front().find("I/O dependent") != std::string::npos);
}

TEST_CASE("the storage tier is admitted wherever the checkpoint lives") {
    auto hardware = make_hardware({8U, 8U}, 64U);
    hardware.storage.path = "/models/test";
    hardware.storage.device = "nvme0n1p2";
    hardware.storage.disk = "nvme0n1";
    hardware.storage.nvme = true;
    hardware.storage.resolved = true;

    auto request = make_request(2U);
    request.model = strata::PlacementModel::KimiK3;
    const auto inventory = make_tiered_inventory(12U, 400U);

    // Which block device backs the checkpoint sets `B_storage`, so it changes
    // the cost the plan reports and not whether the plan is legal. Refusing the
    // fast device to spare its endurance only forces the run onto the slow one,
    // which is the regression this contract exists to prevent.
    const auto nvme = solve_placement(inventory, hardware, request);
    REQUIRE(nvme.ok());
    REQUIRE(nvme.value.io_dependent);
    REQUIRE(nvme.value.storage_resident_bytes != 0U);

    auto sata = hardware;
    sata.storage.device = "sda1";
    sata.storage.disk = "sda";
    sata.storage.nvme = false;
    const auto spinning = solve_placement(inventory, sata, request);
    REQUIRE(spinning.ok());
    REQUIRE(spinning.value.io_dependent);
    REQUIRE(spinning.value.storage_resident_bytes ==
            nvme.value.storage_resident_bytes);

    // An unresolved backing device no longer decides admission either: the plan
    // still reports the tier, and the guard that protects a disk from writes is
    // the one that needs the device resolved.
    auto unknown = hardware;
    unknown.storage.resolved = false;
    unknown.storage.nvme = false;
    REQUIRE(solve_placement(inventory, unknown, request).ok());
}

TEST_CASE("an explicitly streamed component is accounted as storage resident") {
    auto inventory = make_dense_inventory(1U, 1U);
    inventory.prescriptive = false;
    auto& item = inventory.items.front();
    item.preferred_tier = strata::PlacementTier::Storage;
    item.host_bytes = item.source_bytes;
    item.device_bytes = 0U;
    const auto planned = solve_placement(
        inventory, make_hardware({8U}), make_request(1U));
    REQUIRE(planned.ok());
    REQUIRE(planned.value.host_resident_bytes == 0U);
    REQUIRE(planned.value.storage_resident_bytes == kGigabyte);
    REQUIRE(planned.value.decode_storage_read_bytes == kGigabyte);
    REQUIRE(planned.value.io_dependent);
}

TEST_CASE("the backing block device of a real path resolves") {
    const auto storage = strata::resolve_backing_storage(STRATA_SOURCE_DIR);
    REQUIRE(storage.resolved);
    REQUIRE(!storage.device.empty());
    REQUIRE(!storage.disk.empty());
    // A partition resolves to its whole disk, and the disk name is a prefix of
    // the partition name on every naming scheme this runs on.
    REQUIRE(storage.device.rfind(storage.disk, 0U) == 0U);
    REQUIRE(storage.nvme == (storage.disk.rfind("nvme", 0U) == 0U));

    const auto missing = strata::resolve_backing_storage("/nonexistent/path");
    REQUIRE(!missing.resolved);
}

TEST_CASE("the plan cache round-trips the storage tier") {
    auto hardware = make_hardware({8U, 8U}, 64U);
    hardware.storage.path = "/models/test";
    hardware.storage.device = "sda1";
    hardware.storage.disk = "sda";
    hardware.storage.resolved = true;
    auto request = make_request(2U);
    request.model = strata::PlacementModel::KimiK3;
    auto planned = solve_placement(make_tiered_inventory(12U, 400U), hardware,
                                   request);
    REQUIRE(planned.ok());

    const auto decoded = strata::decode_placement_plan(
        strata::encode_placement_plan(planned.value));
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value.version == strata::kPlacementPlanVersion);
    REQUIRE(decoded.value.request.model == strata::PlacementModel::KimiK3);
    REQUIRE(decoded.value.hardware.storage.disk == "sda");
    REQUIRE(decoded.value.hardware.storage.resolved);
    REQUIRE(!decoded.value.hardware.storage.nvme);
    REQUIRE(decoded.value.storage_resident_bytes ==
            planned.value.storage_resident_bytes);
    REQUIRE(decoded.value.decode_storage_read_bytes ==
            planned.value.decode_storage_read_bytes);
}

TEST_CASE("a plan made for a different peer topology is refused") {
    // Capacity mismatches fail loudly on their own. A peer mismatch would not:
    // the run would just take a slower path and report it as the fast one.
    auto hardware = make_hardware({8U, 8U}, 64U);
    hardware.high_speed_peer = {0U, 1U, 1U, 0U};
    auto planned = solve_placement(make_tiered_inventory(12U, 400U), hardware,
                                   make_request(2U));
    REQUIRE(planned.ok());
    REQUIRE(strata::verify_placement_plan(planned.value, hardware).ok());

    auto without_link = hardware;
    without_link.high_speed_peer = {0U, 0U, 0U, 0U};
    const auto refused =
        strata::verify_placement_plan(planned.value, without_link);
    REQUIRE(!refused.ok());
}

TEST_CASE("identical checkpoint and topology produce an identical plan") {
    // M1's acceptance gate requires that the same checkpoint and topology
    // produce deterministic placement and memory accounting. Comparing the
    // encoded plans byte for byte tests the accounting too, not only the
    // layer assignment: any field that drifted between two solves of the same
    // inputs would change the encoding.
    const auto hardware = make_hardware({8U, 8U}, 64U);
    const auto request = make_request(2U);
    const auto inventory = make_tiered_inventory(12U, 400U);

    const auto first = solve_placement(inventory, hardware, request);
    const auto second = solve_placement(inventory, hardware, request);
    REQUIRE(first.ok());
    REQUIRE(second.ok());
    REQUIRE(strata::encode_placement_plan(first.value) ==
            strata::encode_placement_plan(second.value));

    // The same must hold across an encode/decode cycle, because a cached plan
    // is reused in place of a fresh solve and the two must be interchangeable.
    const auto decoded = strata::decode_placement_plan(
        strata::encode_placement_plan(first.value));
    REQUIRE(decoded.ok());
    REQUIRE(strata::encode_placement_plan(decoded.value) ==
            strata::encode_placement_plan(first.value));
}

TEST_CASE("the plan cache round-trips device locality and the peer matrix") {
    // v3 added both. They are recorded rather than only probed because M3
    // admits or refuses rank-local execution on exactly this, and a plan that
    // does not carry the topology it was built for cannot be verified against
    // the machine that later reuses it.
    auto hardware = make_hardware({8U, 8U}, 64U);
    hardware.devices[0].numa_node = 1;
    // -1 is the driver declining to say, and must survive a round trip through
    // an unsigned-only JSON reader rather than decoding as node 0.
    hardware.devices[1].numa_node = -1;
    hardware.high_speed_peer = {0U, 1U, 1U, 0U};
    auto planned = solve_placement(make_tiered_inventory(12U, 400U), hardware,
                                   make_request(2U));
    REQUIRE(planned.ok());

    const auto decoded = strata::decode_placement_plan(
        strata::encode_placement_plan(planned.value));
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value.hardware.devices.size() == 2U);
    REQUIRE(decoded.value.hardware.devices[0].numa_node == 1);
    REQUIRE(decoded.value.hardware.devices[1].numa_node == -1);
    REQUIRE(decoded.value.hardware.high_speed_peer.size() == 4U);
    REQUIRE(decoded.value.hardware.peer_is_high_speed(0U, 1U));
    REQUIRE(decoded.value.hardware.peer_is_high_speed(1U, 0U));
    // The diagonal stays clear: a device is not its own peer, and treating it
    // as one would let a single-GPU topology look peer-linked.
    REQUIRE(!decoded.value.hardware.peer_is_high_speed(0U, 0U));
}

TEST_CASE("an absent peer matrix reports no high-speed link rather than crashing") {
    strata::PlacementHardware hardware = make_hardware({8U, 8U}, 64U);
    hardware.high_speed_peer.clear();
    REQUIRE(!hardware.peer_is_high_speed(0U, 1U));
    // Out-of-range slots are a caller error, not a reason to read past the end.
    REQUIRE(!hardware.peer_is_high_speed(0U, 99U));
}

TEST_CASE("a v1 plan is discarded rather than reinterpreted") {
    // v1 spelled the overflow tier `nvme` and named its fields for it. Reading
    // those bytes as v2 storage figures would report them as coming from a
    // device they were never measured on.
    const auto v1 = R"({"version": 1, "model_type": "glm", "model_name": "x",
        "model_identity": "abc", "model_directory": "/m",
        "nvme_streamed_bytes": 123, "decode_nvme_read_bytes": 456})";
    const auto decoded = strata::decode_placement_plan(v1);
    REQUIRE(!decoded.ok());
    REQUIRE(decoded.errors.front().find("schema version") != std::string::npos);
}
