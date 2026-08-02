#include "strata/placement.hpp"

#include "strata/cuda_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <system_error>
#include <unordered_set>

namespace strata {
namespace {

constexpr std::size_t kClassCount =
    static_cast<std::size_t>(PlacementClass::Count);

[[nodiscard]] bool add(std::uint64_t& target, std::uint64_t value) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - target) return false;
    target += value;
    return true;
}

// Report order, not declaration order: weights first, then the state and
// scratch that sit alongside them.
constexpr std::array<PlacementClass, kClassCount> kReportOrder{
    PlacementClass::Attention,    PlacementClass::FeedForward,
    PlacementClass::RoutedExpert, PlacementClass::SharedExpert,
    PlacementClass::Router,       PlacementClass::Norm,
    PlacementClass::Embedding,    PlacementClass::OutputHead,
    PlacementClass::Vision,       PlacementClass::KvCache,
    PlacementClass::Workspace};

[[nodiscard]] std::uint64_t read_meminfo_kilobytes(std::string_view key) {
    std::ifstream input("/proc/meminfo");
    std::string name;
    std::uint64_t value = 0U;
    std::string unit;
    while (input >> name >> value >> unit) {
        if (!name.empty() && name.back() == ':' &&
            std::string_view(name).substr(0U, name.size() - 1U) == key) {
            return value;
        }
    }
    return 0U;
}

// Fill each slot with as many consecutive layers as it can hold. For a fixed
// slot order this maximizes the layers consumed by every prefix, so it decides
// feasibility exactly: if it cannot place them all, no contiguous partition can.
[[nodiscard]] bool maximal_fill(std::span<const std::uint64_t> layer_bytes,
                                std::span<const std::uint64_t> headroom,
                                std::size_t first_layer, std::size_t first_slot,
                                std::vector<std::size_t>& cut) {
    std::size_t layer = first_layer;
    for (std::size_t slot = first_slot; slot < headroom.size(); ++slot) {
        std::uint64_t block = 0U;
        while (layer < layer_bytes.size() &&
               layer_bytes[layer] <= headroom[slot] - block) {
            block += layer_bytes[layer];
            ++layer;
        }
        cut[slot] = layer;
    }
    return layer == layer_bytes.size();
}

// Contiguous, byte-aware layer assignment. Cut points start at the maximal fill
// that proves the set fits, then move forward while that brings each block
// closer to its capacity-proportional share and the suffix still packs. Blocks
// stay contiguous, so a decode step crosses devices once per boundary instead
// of once per layer.
[[nodiscard]] bool assign_contiguous_blocks(
    std::span<const std::uint64_t> layer_bytes,
    std::span<const std::uint64_t> headroom,
    std::vector<std::size_t>& assignment) {
    const auto layers = layer_bytes.size();
    const auto slots = headroom.size();
    assignment.assign(layers, 0U);
    if (slots == 0U) return false;
    std::vector<std::size_t> cut(slots, layers);
    if (!maximal_fill(layer_bytes, headroom, 0U, 0U, cut)) return false;

    std::uint64_t total_headroom = 0U;
    std::uint64_t total_layer_bytes = 0U;
    for (const auto bytes : headroom) {
        if (!add(total_headroom, bytes)) return false;
    }
    for (const auto bytes : layer_bytes) {
        if (!add(total_layer_bytes, bytes)) return false;
    }
    if (total_headroom != 0U) {
        for (std::size_t slot = 0U; slot + 1U < slots; ++slot) {
            const auto begin = slot == 0U ? std::size_t{0U} : cut[slot - 1U];
            const auto target = static_cast<std::uint64_t>(
                (static_cast<long double>(total_layer_bytes) *
                 static_cast<long double>(headroom[slot])) /
                static_cast<long double>(total_headroom));
            std::uint64_t block = 0U;
            for (std::size_t index = begin; index < cut[slot]; ++index) {
                block += layer_bytes[index];
            }
            std::vector<std::size_t> probe(cut);
            while (cut[slot] > begin && block > target) {
                const auto candidate = cut[slot] - 1U;
                const auto reduced = block - layer_bytes[candidate];
                const auto before = block - target;
                const auto after = reduced >= target ? reduced - target
                                                     : target - reduced;
                if (after >= before) break;
                if (!maximal_fill(layer_bytes, headroom, candidate, slot + 1U,
                                  probe)) {
                    break;
                }
                cut[slot] = candidate;
                for (std::size_t later = slot + 1U; later < slots; ++later) {
                    cut[later] = probe[later];
                }
                block = reduced;
            }
        }
    }

    std::size_t begin = 0U;
    for (std::size_t slot = 0U; slot < slots; ++slot) {
        std::uint64_t block = 0U;
        for (std::size_t index = begin; index < cut[slot]; ++index) {
            assignment[index] = slot;
            block += layer_bytes[index];
        }
        if (block > headroom[slot]) return false;
        begin = cut[slot];
    }
    return begin == layers;
}

// Reproduces the VRAM-proportional round-robin the GLM and DeepSeek runtimes
// already use, so a descriptive plan reports the placement those runtimes
// actually perform rather than one the planner would have preferred.
[[nodiscard]] bool weighted_round_robin(const PlacementHardware& hardware,
                                        std::vector<std::size_t>& schedule) {
    schedule.clear();
    std::uint64_t smallest = std::numeric_limits<std::uint64_t>::max();
    for (const auto& device : hardware.devices) {
        smallest = std::min(smallest, device.total_bytes);
    }
    if (smallest == 0U) return false;
    for (std::size_t slot = 0U; slot < hardware.devices.size(); ++slot) {
        const auto total = hardware.devices[slot].total_bytes;
        if (total > (std::numeric_limits<std::uint64_t>::max() - smallest / 2U) / 2U) {
            return false;
        }
        const auto shares = std::max<std::uint64_t>(
            1U, (total * 2U + smallest / 2U) / smallest);
        for (std::uint64_t count = 0U; count < shares; ++count) {
            schedule.push_back(slot);
        }
    }
    return !schedule.empty();
}

struct ClassAccumulator {
    std::uint64_t bytes{};
    std::uint64_t decode_read_bytes{};
    std::vector<std::uint64_t> device_bytes;
    // Bytes per tier, so a class is labelled by where the bulk of it lives
    // rather than by whichever fragment happened to be accounted first.
    std::array<std::uint64_t, 3U> tier_bytes{};
    bool present{};

    [[nodiscard]] PlacementTier dominant_tier() const noexcept {
        const auto largest = std::max_element(tier_bytes.begin(), tier_bytes.end());
        return static_cast<PlacementTier>(largest - tier_bytes.begin());
    }
    [[nodiscard]] bool mixed() const noexcept {
        return std::count_if(tier_bytes.begin(), tier_bytes.end(),
                             [](std::uint64_t value) { return value != 0U; }) > 1;
    }
};

}  // namespace

std::string_view to_string(PlacementModel model) noexcept {
    switch (model) {
        case PlacementModel::Glm52: return "glm";
        case PlacementModel::DeepSeekV4: return "deepseek";
        case PlacementModel::Gemma4: return "gemma4";
        case PlacementModel::KimiK3: return "kimi-k3";
    }
    return "unknown";
}

std::string_view to_string(PlacementTier tier) noexcept {
    switch (tier) {
        case PlacementTier::Device: return "device";
        case PlacementTier::Host: return "host";
        case PlacementTier::Storage: return "storage";
    }
    return "unknown";
}

std::string_view to_string(PlacementClass component) noexcept {
    switch (component) {
        case PlacementClass::Attention: return "attention";
        case PlacementClass::FeedForward: return "feed-forward";
        case PlacementClass::RoutedExpert: return "routed-expert";
        case PlacementClass::SharedExpert: return "shared-expert";
        case PlacementClass::Router: return "router";
        case PlacementClass::Norm: return "norm";
        case PlacementClass::Embedding: return "embedding";
        case PlacementClass::OutputHead: return "output-head";
        case PlacementClass::Vision: return "vision";
        case PlacementClass::KvCache: return "kv-cache";
        case PlacementClass::Workspace: return "workspace";
        case PlacementClass::Count: break;
    }
    return "unknown";
}

bool parse_placement_model(std::string_view text, PlacementModel& model) noexcept {
    if (text == "glm") { model = PlacementModel::Glm52; return true; }
    if (text == "deepseek") { model = PlacementModel::DeepSeekV4; return true; }
    if (text == "gemma4") { model = PlacementModel::Gemma4; return true; }
    if (text == "kimi-k3") { model = PlacementModel::KimiK3; return true; }
    return false;
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr std::array<const char*, 5> units{"B", "KiB", "MiB", "GiB", "TiB"};
    auto value = static_cast<double>(bytes);
    std::size_t unit = 0U;
    while (value >= 1024.0 && unit + 1U < units.size()) {
        value /= 1024.0;
        ++unit;
    }
    std::array<char, 40> text{};
    std::snprintf(text.data(), text.size(), unit == 0U ? "%.0f %s" : "%.2f %s",
                  value, units[unit]);
    return text.data();
}

std::uint64_t placement_component_bytes(const PlacementPlan& plan,
                                        PlacementClass component,
                                        std::size_t slot) noexcept {
    for (const auto& entry : plan.components) {
        if (entry.component != component) continue;
        return slot < entry.device_bytes.size() ? entry.device_bytes[slot] : 0U;
    }
    return 0U;
}

std::uint64_t PlacementPlan::total_device_bytes() const noexcept {
    std::uint64_t total = 0U;
    for (const auto bytes : device_resident_bytes) {
        if (!add(total, bytes)) return std::numeric_limits<std::uint64_t>::max();
    }
    return total;
}

PlacementStorage resolve_backing_storage(const std::string& path) {
    PlacementStorage storage;
    storage.path = path;
    struct stat status {};
    if (path.empty() || ::stat(path.c_str(), &status) != 0) return storage;
    // Major zero is an anonymous device: tmpfs, ramfs, and friends have no
    // block device behind them. That is a resolved answer, not an unknown one,
    // and it is the safest target a run can pick for scratch.
    if (major(status.st_dev) == 0U) {
        storage.memory_backed = true;
        storage.resolved = true;
        return storage;
    }
    // st_dev names the block device the file lives on; /sys/dev/block maps that
    // major:minor back to a kernel device name.
    std::array<char, 64> node{};
    std::snprintf(node.data(), node.size(), "/sys/dev/block/%u:%u",
                  major(status.st_dev), minor(status.st_dev));
    std::error_code code;
    const auto resolved = std::filesystem::canonical(node.data(), code);
    if (code) return storage;
    storage.device = resolved.filename().string();

    // A partition's sysfs node sits under its whole disk, so the parent
    // directory names the disk whenever the leaf is a partition.
    storage.disk = storage.device;
    if (std::filesystem::exists(resolved / "partition", code) && !code) {
        storage.disk = resolved.parent_path().filename().string();
    }
    storage.nvme = storage.disk.rfind("nvme", 0U) == 0U;
    std::ifstream rotational(
        (std::filesystem::path("/sys/block") / storage.disk / "queue/rotational")
            .string());
    int spinning = 0;
    if (rotational >> spinning) storage.rotational = spinning != 0;
    storage.resolved = true;
    return storage;
}

ParseResult<PlacementHardware> probe_placement_hardware(
    std::span<const int> devices, const std::string& model_directory) {
    ParseResult<PlacementHardware> result;
    if (devices.empty()) {
        result.errors.emplace_back(
            "placement probe requires at least one CUDA device");
        return result;
    }
    std::unordered_set<int> unique;
    for (const int device : devices) {
        if (device < 0 || !unique.insert(device).second) {
            result.errors.emplace_back(
                "placement devices must be unique non-negative ids");
            return result;
        }
        auto memory = CudaBackend::device_memory(device);
        if (!memory.ok()) {
            result.errors = std::move(memory.errors);
            return result;
        }
        PlacementDevice entry;
        entry.id = device;
        entry.name = "cuda:" + std::to_string(device);
        entry.total_bytes = memory.value.total_bytes;
        entry.free_bytes = memory.value.free_bytes;
        result.value.devices.push_back(std::move(entry));
    }
    constexpr std::uint64_t kilobyte = 1024U;
    result.value.host_total_bytes = read_meminfo_kilobytes("MemTotal") * kilobyte;
    result.value.host_available_bytes =
        read_meminfo_kilobytes("MemAvailable") * kilobyte;
    result.value.storage = resolve_backing_storage(model_directory);
    return result;
}

PlacementPlanResult solve_placement(const PlacementInventory& inventory,
                                    const PlacementHardware& hardware,
                                    const PlacementRequest& request) {
    PlacementPlanResult result;
    auto& plan = result.value;
    const auto slots = hardware.devices.size();
    if (slots == 0U || request.devices.size() != slots) {
        result.errors.emplace_back(
            "placement request and hardware probe describe different devices");
        return result;
    }
    if (!std::isfinite(request.vram_cache_fraction) ||
        request.vram_cache_fraction <= 0.0 ||
        request.vram_cache_fraction > 0.95) {
        result.errors.emplace_back("VRAM cache fraction must be in (0, 0.95]");
        return result;
    }
    plan.version = kPlacementPlanVersion;
    plan.request = request;
    plan.hardware = hardware;
    plan.model_name = inventory.model_name;
    plan.prescriptive = inventory.prescriptive;
    plan.device_budget_bytes.assign(slots, 0U);
    plan.device_resident_bytes.assign(slots, 0U);
    plan.device_expert_cache_bytes.assign(slots, 0U);

    std::vector<std::uint64_t> capacity(slots, 0U);
    for (std::size_t slot = 0U; slot < slots; ++slot) {
        const auto budget = static_cast<std::uint64_t>(
            static_cast<double>(hardware.devices[slot].free_bytes) *
            request.vram_cache_fraction);
        plan.device_budget_bytes[slot] = budget;
        if (budget < inventory.minimum_device_budget_bytes ||
            budget <= inventory.per_device_workspace_bytes) {
            result.errors.emplace_back(
                std::string(inventory.model_name) + " CUDA device " +
                std::to_string(request.devices[slot]) +
                " does not meet the admitted VRAM budget: " +
                format_bytes(budget) + " available against " +
                format_bytes(std::max(inventory.minimum_device_budget_bytes,
                                      inventory.per_device_workspace_bytes)) +
                " required");
            return result;
        }
        capacity[slot] = budget - inventory.per_device_workspace_bytes;
        plan.device_resident_bytes[slot] = inventory.per_device_workspace_bytes;
    }

    // Fixed-slot items are pinned by the runtime's own graph, not by the
    // planner: subtract them before the layer blocks are sized.
    std::vector<std::uint64_t> headroom(capacity);
    for (const auto& item : inventory.items) {
        if (item.preferred_tier != PlacementTier::Device || item.spillable) continue;
        if (item.fixed_device_slot < 0) continue;
        const auto slot = static_cast<std::size_t>(item.fixed_device_slot);
        if (slot >= slots) {
            result.errors.emplace_back(
                "placement item targets a device slot outside the request");
            return result;
        }
        if (item.device_bytes > headroom[slot]) {
            result.errors.emplace_back(
                std::string(to_string(item.component)) +
                " weights do not fit the admitted budget of CUDA device " +
                std::to_string(request.devices[slot]));
            return result;
        }
        headroom[slot] -= item.device_bytes;
    }

    const auto layers = inventory.layer_count;
    std::vector<std::uint64_t> layer_bytes(layers, 0U);
    std::uint64_t unbound_device_bytes = 0U;
    for (const auto& item : inventory.items) {
        if (item.preferred_tier != PlacementTier::Device || item.spillable) continue;
        if (item.fixed_device_slot >= 0) continue;
        if (item.layer < 0 || static_cast<std::uint32_t>(item.layer) >= layers) {
            if (!add(unbound_device_bytes, item.device_bytes)) {
                result.errors.emplace_back("placement byte total overflows");
                return result;
            }
            continue;
        }
        if (!add(layer_bytes[static_cast<std::size_t>(item.layer)],
                 item.device_bytes)) {
            result.errors.emplace_back("placement layer byte total overflows");
            return result;
        }
    }

    plan.layer_device.assign(layers, 0U);
    if (inventory.contiguous_layer_blocks) {
        if (!assign_contiguous_blocks(layer_bytes, headroom, plan.layer_device)) {
            std::uint64_t required = unbound_device_bytes;
            for (const auto bytes : layer_bytes) required += bytes;
            std::uint64_t available = 0U;
            for (const auto bytes : headroom) available += bytes;
            result.errors.emplace_back(
                inventory.model_name + " resident layers need " +
                format_bytes(required) + " but the admitted VRAM budget leaves " +
                format_bytes(available) +
                " after workspace and fixed placements");
            return result;
        }
        plan.weighted_schedule = plan.layer_device;
    } else {
        if (!weighted_round_robin(hardware, plan.weighted_schedule)) {
            result.errors.emplace_back(
                "CUDA device schedule weight is degenerate or overflows");
            return result;
        }
        for (std::uint32_t layer = 0U; layer < layers; ++layer) {
            plan.layer_device[layer] =
                plan.weighted_schedule[layer % plan.weighted_schedule.size()];
        }
    }

    std::array<ClassAccumulator, kClassCount> totals;
    for (auto& entry : totals) entry.device_bytes.assign(slots, 0U);
    const auto account = [&](PlacementClass component, PlacementTier tier,
                             std::uint64_t bytes, std::uint64_t reads,
                             std::size_t slot, bool has_slot) {
        auto& entry = totals[static_cast<std::size_t>(component)];
        entry.present = true;
        entry.bytes += bytes;
        entry.decode_read_bytes += reads;
        entry.tier_bytes[static_cast<std::size_t>(tier)] += bytes;
        if (has_slot) entry.device_bytes[slot] += bytes;
    };

    // Resident, non-spillable items first: they are read on every step, so
    // spilling one to reclaim VRAM for a sparsely read class is negative under
    // a max over resources.
    std::vector<std::uint64_t> committed(slots, 0U);
    for (const auto& item : inventory.items) {
        if (item.spillable) continue;
        if (item.preferred_tier != PlacementTier::Device) {
            if (!add(plan.host_resident_bytes, item.host_bytes)) {
                result.errors.emplace_back("placement host byte total overflows");
                return result;
            }
            account(item.component, item.preferred_tier, item.host_bytes,
                    item.decode_read_bytes, 0U, false);
            if (item.preferred_tier == PlacementTier::Host) {
                plan.decode_host_to_device_bytes += item.decode_read_bytes;
            } else {
                plan.decode_storage_read_bytes += item.decode_read_bytes;
            }
            continue;
        }
        std::size_t slot = 0U;
        if (item.fixed_device_slot >= 0) {
            slot = static_cast<std::size_t>(item.fixed_device_slot);
        } else if (item.layer >= 0 &&
                   static_cast<std::uint32_t>(item.layer) < layers) {
            slot = plan.layer_device[static_cast<std::size_t>(item.layer)];
        } else {
            slot = static_cast<std::size_t>(
                std::min_element(committed.begin(), committed.end()) -
                committed.begin());
        }
        committed[slot] += item.device_bytes;
        account(item.component, PlacementTier::Device, item.device_bytes,
                item.decode_read_bytes, slot, true);
        plan.decode_device_read_bytes += item.decode_read_bytes;
    }
    for (std::size_t slot = 0U; slot < slots; ++slot) {
        if (committed[slot] > capacity[slot]) {
            result.errors.emplace_back(
                inventory.model_name + " resident set needs " +
                format_bytes(committed[slot]) + " on CUDA device " +
                std::to_string(request.devices[slot]) + " but only " +
                format_bytes(capacity[slot]) + " is admitted");
            return result;
        }
        plan.device_resident_bytes[slot] += committed[slot];
        plan.device_expert_cache_bytes[slot] = capacity[slot] - committed[slot];
    }
    if (inventory.per_device_workspace_bytes != 0U) {
        for (std::size_t slot = 0U; slot < slots; ++slot) {
            account(PlacementClass::Workspace, PlacementTier::Device,
                    inventory.per_device_workspace_bytes, 0U, slot, true);
        }
    }

    // Spillable classes take what device capacity is left, then host memory.
    // Anything beyond both is read from the checkpoint on every step that needs
    // it. Shallower `deepest_tier` first: a class that may not reach storage has
    // the harder constraint, and on the models that mix the two it is also the
    // densely read one, so giving it VRAM ahead of a sparse class is strictly
    // better under a max over resources.
    std::vector<const PlacementItem*> spill_order;
    for (const auto& item : inventory.items) {
        if (item.spillable) spill_order.push_back(&item);
    }
    std::stable_sort(spill_order.begin(), spill_order.end(),
                     [](const PlacementItem* left, const PlacementItem* right) {
                         return static_cast<std::uint8_t>(left->deepest_tier) <
                                static_cast<std::uint8_t>(right->deepest_tier);
                     });
    std::uint64_t expert_capacity = 0U;
    for (const auto bytes : plan.device_expert_cache_bytes) {
        if (!add(expert_capacity, bytes)) {
            result.errors.emplace_back("placement expert capacity overflows");
            return result;
        }
    }
    std::uint64_t host_capacity = inventory.host_capacity_bytes == 0U
        ? hardware.host_available_bytes
        : std::min(hardware.host_available_bytes, inventory.host_capacity_bytes);
    const bool has_spill = std::any_of(
        inventory.items.begin(), inventory.items.end(),
        [](const PlacementItem& item) { return item.spillable; });
    const auto host_withheld =
        inventory.host_workspace_bytes + inventory.host_reserve_bytes;
    if (host_withheld >= host_capacity) {
        host_capacity = 0U;
    } else {
        host_capacity -= host_withheld;
    }
    if (plan.host_resident_bytes >= host_capacity) {
        host_capacity = 0U;
    } else {
        host_capacity -= plan.host_resident_bytes;
    }
    for (const auto* entry_item : spill_order) {
        const auto& item = *entry_item;
        const auto total = item.device_bytes == 0U ? 1U : item.device_bytes;
        const auto device_share = std::min(item.device_bytes, expert_capacity);
        expert_capacity -= device_share;
        // A cache over host memory does not shrink the host copy; a move does.
        const auto outside = item.device_cache_only
            ? (item.host_bytes == 0U ? item.device_bytes : item.host_bytes)
            : item.device_bytes - device_share;
        const auto host_taken = std::min(outside, host_capacity);
        host_capacity -= host_taken;
        const auto nvme_share = outside - host_taken;
        if (nvme_share != 0U && item.deepest_tier != PlacementTier::Storage) {
            // A densely read class that will not fit device plus host is the
            // charter's I/O-dependent case. Report it; do not absorb it by
            // silently streaming it from the checkpoint on every step.
            result.errors.emplace_back(
                inventory.model_name + " " + std::string(to_string(item.component)) +
                " needs " + format_bytes(outside) +
                " outside VRAM but the host tier admits only " +
                format_bytes(host_taken) + ", leaving " +
                format_bytes(nvme_share) +
                " with nowhere to go: this class is read on every decode step "
                "and may not spill past the host tier, so the model is I/O "
                "dependent at this operating point");
            return result;
        }
        // Per-step reads split by resident fraction. Routing decides the real
        // hit rate; this is a uniform-routing bound, flagged in the notes.
        const auto reads = item.decode_read_bytes;
        const auto device_reads = static_cast<std::uint64_t>(
            (static_cast<long double>(reads) *
             static_cast<long double>(device_share)) /
            static_cast<long double>(total));
        const auto streamed = reads - device_reads;
        const auto host_reads = nvme_share == 0U ? streamed
            : static_cast<std::uint64_t>(
                  (static_cast<long double>(streamed) *
                   static_cast<long double>(host_taken)) /
                  static_cast<long double>(outside == 0U ? 1U : outside));
        const auto nvme_reads = streamed - host_reads;
        plan.decode_device_read_bytes += device_reads;
        plan.decode_host_to_device_bytes += host_reads;
        plan.decode_storage_read_bytes += nvme_reads;
        if (!add(plan.host_resident_bytes, host_taken) ||
            !add(plan.storage_resident_bytes, nvme_share)) {
            result.errors.emplace_back("placement spill byte total overflows");
            return result;
        }
        auto& entry = totals[static_cast<std::size_t>(item.component)];
        entry.present = true;
        entry.decode_read_bytes += reads;
        if (item.device_cache_only) {
            // The device share is a copy, so it sizes the VRAM columns without
            // inflating the class total, which stays the canonical set.
            entry.bytes += outside;
            entry.tier_bytes[static_cast<std::size_t>(PlacementTier::Host)] +=
                host_taken;
            entry.tier_bytes[static_cast<std::size_t>(PlacementTier::Storage)] +=
                nvme_share;
        } else {
            entry.bytes += item.device_bytes;
            entry.tier_bytes[static_cast<std::size_t>(PlacementTier::Device)] +=
                device_share;
            entry.tier_bytes[static_cast<std::size_t>(PlacementTier::Host)] +=
                host_taken;
            entry.tier_bytes[static_cast<std::size_t>(PlacementTier::Storage)] +=
                nvme_share;
        }
        // Spread the admitted cache over devices in proportion to the capacity
        // each one has left; the runtime's cache fills the same way. Demand
        // loading fills it, so those bytes are committed VRAM and count toward
        // what each device holds.
        std::uint64_t cache_total = 0U;
        for (const auto bytes : plan.device_expert_cache_bytes) cache_total += bytes;
        if (cache_total != 0U && device_share != 0U) {
            for (std::size_t slot = 0U; slot < slots; ++slot) {
                const auto share = static_cast<std::uint64_t>(
                    (static_cast<long double>(device_share) *
                     static_cast<long double>(plan.device_expert_cache_bytes[slot])) /
                    static_cast<long double>(cache_total));
                entry.device_bytes[slot] += share;
                plan.device_resident_bytes[slot] += share;
            }
        }
    }

    for (const auto component : kReportOrder) {
        const auto& entry = totals[static_cast<std::size_t>(component)];
        if (!entry.present || entry.bytes == 0U) continue;
        PlacementComponentTotals totals_entry;
        totals_entry.component = component;
        totals_entry.tier = entry.dominant_tier();
        totals_entry.bytes = entry.bytes;
        totals_entry.decode_read_bytes = entry.decode_read_bytes;
        totals_entry.device_bytes = entry.device_bytes;
        plan.components.push_back(std::move(totals_entry));
        if (entry.mixed()) {
            std::string breakdown;
            for (std::size_t tier = 0U; tier < entry.tier_bytes.size(); ++tier) {
                if (entry.tier_bytes[tier] == 0U) continue;
                if (!breakdown.empty()) breakdown += ", ";
                breakdown += format_bytes(entry.tier_bytes[tier]);
                breakdown += ' ';
                breakdown += to_string(static_cast<PlacementTier>(tier));
            }
            plan.notes.push_back(std::string(to_string(component)) +
                                 " spans tiers: " + breakdown +
                                 " (the row is labelled by the larger share)");
        }
    }

    plan.cross_device_activation_hops = 0U;
    for (std::size_t layer = 1U; layer < plan.layer_device.size(); ++layer) {
        if (plan.layer_device[layer] != plan.layer_device[layer - 1U]) {
            ++plan.cross_device_activation_hops;
        }
    }
    plan.io_dependent = plan.storage_resident_bytes != 0U;
    plan.fits = !plan.io_dependent;
    // NVMe endurance is protected by refusal, not by preference: a plan that
    // would source model bytes from an NVMe-backed path is an error even when
    // it otherwise fits.
    if (request.forbid_nvme_residency && plan.storage_resident_bytes != 0U) {
        if (!hardware.storage.resolved) {
            result.errors.emplace_back(
                "cannot confirm the checkpoint's backing block device, and this "
                "run forbids NVMe residency; resolve " +
                request.model_directory + " or clear the restriction");
            return result;
        }
        if (hardware.storage.nvme) {
            result.errors.emplace_back(
                "this run forbids NVMe residency but " + request.model_directory +
                " is backed by " + hardware.storage.disk + ", so the " +
                format_bytes(plan.storage_resident_bytes) +
                " storage tier would read model bytes from NVMe");
            return result;
        }
    }
    if (has_spill && inventory.host_reserve_bytes != 0U) {
        plan.notes.emplace_back(
            "host tier withholds " + format_bytes(inventory.host_reserve_bytes) +
            " for activations, worker stacks, and page cache");
    }
    if (plan.io_dependent) {
        std::string where = "the checkpoint";
        if (hardware.storage.resolved) {
            where = hardware.storage.disk + " (" +
                    (hardware.storage.nvme ? "nvme"
                                           : hardware.storage.rotational
                                                 ? "rotational"
                                                 : "non-rotational, non-nvme") +
                    ')';
        }
        plan.notes.emplace_back(
            "steady-state decode reads " +
            format_bytes(plan.decode_storage_read_bytes) + " per step from " +
            where + ": this configuration is I/O dependent");
    }
    if (plan.decode_host_to_device_bytes != 0U) {
        plan.notes.emplace_back(
            "steady-state decode moves " +
            format_bytes(plan.decode_host_to_device_bytes) +
            " host-to-device per step; the split assumes uniform routing, so "
            "measure the hit rate on a decode route trace before trusting it");
    }
    for (std::size_t slot = 0U; slot < slots; ++slot) {
        const auto assigned = std::count(plan.layer_device.begin(),
                                         plan.layer_device.end(), slot);
        if (assigned == 0) {
            plan.notes.push_back("CUDA device " +
                                 std::to_string(request.devices[slot]) +
                                 " holds no layers at this operating point");
        }
    }
    if (!inventory.prescriptive) {
        plan.notes.emplace_back(
            "descriptive plan: it reports and admits the placement this runtime "
            "already performs and does not change it");
    }
    return result;
}

ValidationResult verify_placement_plan(const PlacementPlan& plan,
                                       const PlacementHardware& hardware) {
    ValidationResult result;
    if (plan.version != kPlacementPlanVersion) {
        result.errors.emplace_back("placement plan schema version is not current");
        return result;
    }
    if (hardware.devices.size() != plan.device_resident_bytes.size()) {
        result.errors.emplace_back(
            "placement plan device count does not match the current hardware");
        return result;
    }
    for (std::size_t slot = 0U; slot < hardware.devices.size(); ++slot) {
        if (hardware.devices[slot].id != plan.hardware.devices[slot].id ||
            hardware.devices[slot].total_bytes !=
                plan.hardware.devices[slot].total_bytes) {
            result.errors.emplace_back(
                "placement plan was made for different CUDA hardware");
            return result;
        }
        const auto required = plan.device_resident_bytes[slot];
        if (required > hardware.devices[slot].free_bytes) {
            result.errors.emplace_back(
                "placement plan needs " + format_bytes(required) +
                " on CUDA device " + std::to_string(hardware.devices[slot].id) +
                " but only " + format_bytes(hardware.devices[slot].free_bytes) +
                " is free; free VRAM or re-run with --replan");
        }
    }
    if (plan.host_resident_bytes > hardware.host_available_bytes) {
        result.errors.emplace_back(
            "placement plan needs " + format_bytes(plan.host_resident_bytes) +
            " of host memory but only " +
            format_bytes(hardware.host_available_bytes) + " is available");
    }
    return result;
}

std::string render_placement_report(const PlacementPlan& plan) {
    std::ostringstream text;
    const auto slots = plan.hardware.devices.size();
    text << "placement plan  " << to_string(plan.request.model) << "  "
         << plan.model_name << '\n'
         << "  checkpoint    " << plan.request.model_directory << '\n'
         << "  context       " << plan.request.maximum_context_tokens
         << " tokens\n"
         << "  devices       ";
    for (std::size_t slot = 0U; slot < slots; ++slot) {
        if (slot != 0U) text << ", ";
        text << plan.hardware.devices[slot].name << " ("
             << format_bytes(plan.hardware.devices[slot].free_bytes) << " free of "
             << format_bytes(plan.hardware.devices[slot].total_bytes) << ')';
    }
    text << "\n  host          " << format_bytes(plan.hardware.host_available_bytes)
         << " available of " << format_bytes(plan.hardware.host_total_bytes)
         << '\n';
    text << "  storage       ";
    if (plan.hardware.storage.memory_backed) {
        text << "memory backed (no block device)";
    } else if (plan.hardware.storage.resolved) {
        text << plan.hardware.storage.device << " on "
             << plan.hardware.storage.disk << " ("
             << (plan.hardware.storage.nvme ? "nvme" : "non-nvme") << ", "
             << (plan.hardware.storage.rotational ? "rotational"
                                                  : "non-rotational")
             << ')';
    } else {
        text << "unresolved";
    }
    text << "\n\n";

    constexpr int kNameWidth = 16;
    constexpr int kCellWidth = 12;
    text << std::left << std::setw(kNameWidth) << "component" << std::right
         << std::setw(kCellWidth) << "size" << std::setw(9) << "tier";
    for (const auto& device : plan.hardware.devices) {
        text << std::setw(kCellWidth) << device.name;
    }
    text << std::setw(kCellWidth) << "per step" << '\n';
    text << std::string(static_cast<std::size_t>(
             kNameWidth + kCellWidth + 9 + kCellWidth) +
             slots * static_cast<std::size_t>(kCellWidth), '-') << '\n';
    for (const auto& component : plan.components) {
        text << std::left << std::setw(kNameWidth) << to_string(component.component)
             << std::right << std::setw(kCellWidth) << format_bytes(component.bytes)
             << std::setw(9) << to_string(component.tier);
        for (std::size_t slot = 0U; slot < slots; ++slot) {
            const auto bytes = slot < component.device_bytes.size()
                ? component.device_bytes[slot] : 0U;
            text << std::setw(kCellWidth)
                 << (bytes == 0U ? std::string("-") : format_bytes(bytes));
        }
        text << std::setw(kCellWidth)
             << (component.decode_read_bytes == 0U
                     ? std::string("-")
                     : format_bytes(component.decode_read_bytes))
             << '\n';
    }
    text << std::string(static_cast<std::size_t>(
             kNameWidth + kCellWidth + 9 + kCellWidth) +
             slots * static_cast<std::size_t>(kCellWidth), '-') << '\n';
    text << std::left << std::setw(kNameWidth) << "device total" << std::right
         << std::setw(kCellWidth) << format_bytes(plan.total_device_bytes())
         << std::setw(9) << "";
    for (const auto bytes : plan.device_resident_bytes) {
        text << std::setw(kCellWidth) << format_bytes(bytes);
    }
    text << '\n';
    text << std::left << std::setw(kNameWidth) << "admitted budget" << std::right
         << std::setw(kCellWidth) << "" << std::setw(9) << "";
    for (const auto bytes : plan.device_budget_bytes) {
        text << std::setw(kCellWidth) << format_bytes(bytes);
    }
    text << "\n\n";

    text << "  layer blocks  ";
    if (plan.layer_device.empty()) {
        text << "none";
    } else {
        std::size_t begin = 0U;
        for (std::size_t layer = 1U; layer <= plan.layer_device.size(); ++layer) {
            const bool boundary = layer == plan.layer_device.size() ||
                plan.layer_device[layer] != plan.layer_device[begin];
            if (!boundary) continue;
            if (begin != 0U) text << ", ";
            text << plan.hardware.devices[plan.layer_device[begin]].name << '='
                 << begin << ".." << layer - 1U;
            begin = layer;
        }
    }
    text << "\n  hops          " << plan.cross_device_activation_hops
         << " cross-device activation transfers per decode step\n";
    text << "  host resident " << format_bytes(plan.host_resident_bytes) << '\n'
         << "  storage resid " << format_bytes(plan.storage_resident_bytes) << '\n';
    text << "  decode reads  " << format_bytes(plan.decode_device_read_bytes)
         << " device, " << format_bytes(plan.decode_host_to_device_bytes)
         << " host-to-device, " << format_bytes(plan.decode_storage_read_bytes)
         << " storage  (bytes per step, not a duration)\n";
    if (plan.maximum_context_tokens_that_fit != 0U) {
        text << "  max context   " << plan.maximum_context_tokens_that_fit
             << " tokens at this placement\n";
    }
    text << "  verdict       "
         << (plan.io_dependent
                 ? "I/O dependent: steady-state decode reads the checkpoint"
                 : plan.decode_host_to_device_bytes != 0U
                     ? "fits with a host tier: decode streams weights over PCIe"
                     : "fits: every weight, cache, and workspace is device resident")
         << '\n';
    for (const auto& note : plan.notes) {
        text << "  note          " << note << '\n';
    }
    return text.str();
}

}  // namespace strata
