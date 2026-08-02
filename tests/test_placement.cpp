#include "test.hpp"

#include "strata/placement.hpp"

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
    REQUIRE(plan.decode_nvme_read_bytes == 0U);
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
    REQUIRE(roomy.value.nvme_streamed_bytes == 0U);

    const auto cramped = solve_placement(inventory, make_hardware({16U, 16U}, 8U),
                                         make_request(2U));
    REQUIRE(cramped.ok());
    REQUIRE(cramped.value.io_dependent);
    REQUIRE(cramped.value.nvme_streamed_bytes > 0U);
    REQUIRE(!cramped.value.fits);
    REQUIRE(cramped.value.decode_nvme_read_bytes > 0U);
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
                plan.decode_nvme_read_bytes ==
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
