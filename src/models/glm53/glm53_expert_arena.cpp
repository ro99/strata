#include "strata/models/glm53/glm53_expert_arena.hpp"

#include "strata/models/glm53/glm53_checkpoint.hpp"
#include "strata/platform/hardware_profile.hpp"
#include "strata/platform/numa_topology.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace strata {
namespace {

constexpr std::uint64_t kTensorAlignment = 64U;
constexpr std::uint64_t kFallbackHugePageBytes = 2ULL << 20U;
constexpr std::uint64_t kNumaStripeBytes = 64ULL << 20U;
constexpr std::size_t kMaximumReadWorkers = 8U;
constexpr double kMinimumHugeBacking = 0.95;

[[nodiscard]] bool arena_tensor(const Glm53ManifestTensor& tensor,
                                bool include_mtp_experts) noexcept {
    // Layer 45 is classified wholesale as Mtp by the manifest, including its
    // routed/shared MoE weights. Keep those host-executed payloads in the
    // arena too so enabling exact MTP cannot reach around the replacement
    // representation and refault a duplicate mapped expert set.
    const bool mtp_expert = include_mtp_experts &&
        tensor.role == Glm53TensorRole::Mtp &&
        (tensor.name.find(".mlp.experts.") != std::string::npos ||
         tensor.name.find(".mlp.shared_experts.") != std::string::npos);
    const bool expert = tensor.role == Glm53TensorRole::RoutedExpert ||
                        tensor.role == Glm53TensorRole::SharedExpert ||
                        mtp_expert;
    return expert &&
           (tensor.component == Glm53TensorComponent::Weight ||
            tensor.component == Glm53TensorComponent::Scale);
}

[[nodiscard]] bool align_up(std::uint64_t value, std::uint64_t alignment,
                            std::uint64_t& output) noexcept {
    if (alignment == 0U) return false;
    const auto remainder = value % alignment;
    const auto increment = remainder == 0U ? 0U : alignment - remainder;
    if (value > std::numeric_limits<std::uint64_t>::max() - increment) {
        return false;
    }
    output = value + increment;
    return true;
}

[[nodiscard]] std::uint64_t huge_page_bytes() noexcept {
    std::ifstream thp_size(
        "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size");
    std::uint64_t transparent_bytes = 0U;
    if (thp_size >> transparent_bytes && transparent_bytes != 0U) {
        return transparent_bytes;
    }
    const auto reported = host_hardware_profile().huge_page_bytes;
    return reported == 0U ? kFallbackHugePageBytes : reported;
}

[[nodiscard]] std::uint64_t layout_bytes(
    const Glm53IndexManifest& manifest, bool include_mapping_tail,
    bool include_mtp_experts) noexcept {
    std::uint64_t cursor = 0U;
    for (const auto& tensor : manifest.tensors) {
        if (!arena_tensor(tensor, include_mtp_experts)) continue;
        std::uint64_t tensor_bytes = 0U;
        if (!align_up(tensor.source_bytes, kTensorAlignment, tensor_bytes) ||
            tensor_bytes >
                std::numeric_limits<std::uint64_t>::max() - cursor) {
            return 0U;
        }
        cursor += tensor_bytes;
    }
    if (cursor == 0U) return 0U;
    if (include_mapping_tail && !align_up(cursor, huge_page_bytes(), cursor)) {
        return 0U;
    }
    return cursor;
}

[[nodiscard]] bool parse_hex(std::string_view value,
                             std::uintptr_t& output) noexcept {
    return std::from_chars(value.data(), value.data() + value.size(), output,
                           16).ec == std::errc{};
}

[[nodiscard]] bool parse_decimal(std::string_view value,
                                 std::uint64_t& output) noexcept {
    while (!value.empty() && value.front() == ' ') value.remove_prefix(1U);
    const auto parsed = std::from_chars(value.data(),
                                        value.data() + value.size(), output);
    return parsed.ec == std::errc{};
}

// Sum only VMAs overlapping this arena. /proc/meminfo's AnonHugePages is a
// host-wide figure and can make an arena look backed when another process owns
// every huge page.
[[nodiscard]] std::uint64_t anonymous_huge_bytes(
    const std::byte* base, std::uint64_t bytes) noexcept {
    std::ifstream input("/proc/self/smaps");
    if (!input || base == nullptr || bytes == 0U) return 0U;
    const auto arena_begin = reinterpret_cast<std::uintptr_t>(base);
    const auto arena_end = arena_begin + bytes;
    bool overlaps = false;
    std::uint64_t total_kib = 0U;
    std::string line;
    while (std::getline(input, line)) {
        const auto dash = line.find('-');
        const auto space = line.find(' ');
        if (dash != std::string::npos && space != std::string::npos &&
            dash < space) {
            std::uintptr_t begin = 0U;
            std::uintptr_t end = 0U;
            overlaps = parse_hex(std::string_view(line).substr(0U, dash), begin) &&
                       parse_hex(std::string_view(line).substr(
                                     dash + 1U, space - dash - 1U), end) &&
                       end > arena_begin && begin < arena_end;
            continue;
        }
        constexpr std::string_view key = "AnonHugePages:";
        if (overlaps && line.compare(0U, key.size(), key) == 0) {
            std::uint64_t kib = 0U;
            if (parse_decimal(std::string_view(line).substr(key.size()), kib)) {
                total_kib += kib;
            }
        }
    }
    return total_kib * 1024U;
}

[[nodiscard]] std::vector<std::uint64_t> numa_resident_bytes(
    const std::byte* base, std::uint64_t bytes, int nodes) noexcept {
    if (base == nullptr || bytes == 0U || nodes <= 0) return {};
    std::ifstream input("/proc/self/numa_maps");
    if (!input) return {};
    const auto arena_begin = reinterpret_cast<std::uintptr_t>(base);
    const auto arena_end = arena_begin + bytes;
    std::vector<std::uint64_t> result(static_cast<std::size_t>(nodes));
    bool matched = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string address;
        if (!(fields >> address)) continue;
        std::uintptr_t begin = 0U;
        if (!parse_hex(address, begin) || begin < arena_begin ||
            begin >= arena_end) {
            continue;
        }
        matched = true;
        std::vector<std::uint64_t> line_pages(
            static_cast<std::size_t>(nodes));
        const auto system_page = sysconf(_SC_PAGESIZE);
        std::uint64_t page_bytes = system_page > 0
            ? static_cast<std::uint64_t>(system_page) : 4096U;
        std::string token;
        while (fields >> token) {
            constexpr std::string_view page_key = "kernelpagesize_kB=";
            if (token.starts_with(page_key)) {
                std::uint64_t kib = 0U;
                if (parse_decimal(std::string_view(token).substr(
                                      page_key.size()), kib) && kib != 0U) {
                    page_bytes = kib * 1024U;
                }
                continue;
            }
            if (token.size() < 4U || token.front() != 'N') continue;
            const auto equals = token.find('=');
            if (equals == std::string::npos) continue;
            int node = -1;
            std::uint64_t pages = 0U;
            const auto node_text = std::string_view(token).substr(
                1U, equals - 1U);
            if (std::from_chars(node_text.data(),
                                node_text.data() + node_text.size(), node).ec !=
                    std::errc{} ||
                node < 0 || node >= nodes ||
                !parse_decimal(std::string_view(token).substr(equals + 1U),
                               pages)) {
                continue;
            }
            line_pages[static_cast<std::size_t>(node)] = pages;
        }
        for (std::size_t node = 0U; node < line_pages.size(); ++node) {
            result[node] += line_pages[node] * page_bytes;
        }
    }
    return matched ? result : std::vector<std::uint64_t>{};
}

}  // namespace

Glm53ExpertArena::~Glm53ExpertArena() { release(); }

void Glm53ExpertArena::release() noexcept {
    if (base_ != nullptr && mapped_bytes_ != 0U) {
        static_cast<void>(
            munmap(base_, static_cast<std::size_t>(mapped_bytes_)));
    }
    base_ = nullptr;
    mapped_bytes_ = 0U;
    extents_.clear();
    stats_ = {};
    complete_ = false;
}

std::uint64_t Glm53ExpertArena::required_bytes(
    const Glm53IndexManifest& manifest, bool include_mtp_experts) noexcept {
    return layout_bytes(manifest, true, include_mtp_experts);
}

ValidationResult Glm53ExpertArena::stage(
    const Glm53CheckpointReader& checkpoint,
    std::uint64_t host_memory_ceiling_bytes, bool include_mtp_experts) {
    ValidationResult result;
    if (complete_ || base_ != nullptr) {
        result.errors.emplace_back("GLM-5.3 expert arena is already staged");
        return result;
    }

    std::vector<const Glm53ManifestTensor*> tensors;
    tensors.reserve(checkpoint.manifest().tensors.size());
    for (const auto& tensor : checkpoint.manifest().tensors) {
        if (arena_tensor(tensor, include_mtp_experts)) {
            tensors.push_back(&tensor);
        }
    }
    std::sort(tensors.begin(), tensors.end(), [](const auto* left,
                                                 const auto* right) {
        if (left->shard != right->shard) return left->shard < right->shard;
        if (left->source_offset != right->source_offset) {
            return left->source_offset < right->source_offset;
        }
        return left->name < right->name;
    });

    const auto logical_bytes = layout_bytes(
        checkpoint.manifest(), false, include_mtp_experts);
    const auto mapping_bytes = required_bytes(
        checkpoint.manifest(), include_mtp_experts);
    if (logical_bytes == 0U || mapping_bytes == 0U || tensors.empty()) {
        result.errors.emplace_back(
            "GLM-5.3 checkpoint has no admissible expert arena layout");
        return result;
    }
    if (host_memory_ceiling_bytes == 0U ||
        mapping_bytes > host_memory_ceiling_bytes) {
        result.errors.emplace_back(
            "GLM-5.3 expert arena exceeds the discovered host-memory ceiling");
        return result;
    }
    const auto& hardware = host_hardware_profile();
    if (!hardware.transparent_huge_pages_enabled) {
        result.errors.emplace_back(
            "GLM-5.3 expert arena requires transparent huge pages");
        return result;
    }
    const auto started = std::chrono::steady_clock::now();

    const auto alignment = huge_page_bytes();
    if (mapping_bytes > std::numeric_limits<std::uint64_t>::max() - alignment) {
        result.errors.emplace_back("GLM-5.3 expert arena allocation overflows");
        return result;
    }
    const auto reservation_bytes = mapping_bytes + alignment;
    if (reservation_bytes > std::numeric_limits<std::size_t>::max()) {
        result.errors.emplace_back(
            "GLM-5.3 expert arena exceeds the process address range");
        return result;
    }
    void* reservation = mmap(nullptr, static_cast<std::size_t>(reservation_bytes),
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (reservation == MAP_FAILED) {
        result.errors.emplace_back(
            "cannot allocate GLM-5.3 expert arena: " +
            std::string(std::strerror(errno)));
        return result;
    }
    const auto reservation_begin =
        reinterpret_cast<std::uintptr_t>(reservation);
    const auto aligned_begin =
        (reservation_begin + alignment - 1U) / alignment * alignment;
    const auto prefix = aligned_begin - reservation_begin;
    const auto suffix = reservation_bytes - prefix - mapping_bytes;
    if (prefix != 0U) {
        static_cast<void>(munmap(reservation, static_cast<std::size_t>(prefix)));
    }
    if (suffix != 0U) {
        static_cast<void>(munmap(reinterpret_cast<void*>(aligned_begin +
                                     mapping_bytes),
                                 static_cast<std::size_t>(suffix)));
    }
    base_ = reinterpret_cast<std::byte*>(aligned_begin);
    mapped_bytes_ = mapping_bytes;

#if defined(MADV_HUGEPAGE)
    stats_.hugepage_accepted =
        madvise(base_, static_cast<std::size_t>(mapped_bytes_),
                MADV_HUGEPAGE) == 0;
#endif
    if (!stats_.hugepage_accepted) {
        const auto reason = std::strerror(errno);
        release();
        result.errors.emplace_back(
            "cannot request huge pages for GLM-5.3 expert arena: " +
            std::string(reason));
        return result;
    }

    stats_.numa_balanced = true;
    if (hardware.numa.multi_node()) {
        for (std::uint64_t offset = 0U; offset < mapped_bytes_;
             offset += kNumaStripeBytes) {
            const auto stripe_bytes = std::min(
                kNumaStripeBytes, mapped_bytes_ - offset);
            const auto node = static_cast<int>(
                (offset / kNumaStripeBytes) %
                static_cast<std::uint64_t>(hardware.numa.nodes));
            if (!numa_bind_range(base_ + offset, stripe_bytes, node)) {
                stats_.numa_balanced = false;
                break;
            }
        }
    }
    if (!stats_.numa_balanced) {
        release();
        result.errors.emplace_back(
            "cannot stripe GLM-5.3 expert arena across NUMA nodes");
        return result;
    }

    // Allocate THPs before pread fills arbitrary tensor offsets. Directly
    // first-touching the destination from parallel pread calls left 42% of
    // this 182 GiB arena in 4 KiB pages. Late MADV_COLLAPSE cannot repair it
    // without transiently allocating replacement pages, because the resident
    // arena itself consumes nearly all immediately free RAM. One first-touch
    // worker per NUMA node instead constructs the final pages in place; the
    // subsequent exact reads overwrite rather than replace them.
    const auto prefault_workers = std::max(hardware.numa.nodes, 1);
    std::vector<std::thread> prefault;
    prefault.reserve(static_cast<std::size_t>(prefault_workers));
    for (int worker = 0; worker < prefault_workers; ++worker) {
        prefault.emplace_back([&, worker] {
            const auto stride = kNumaStripeBytes *
                static_cast<std::uint64_t>(prefault_workers);
            for (std::uint64_t offset = kNumaStripeBytes *
                     static_cast<std::uint64_t>(worker);
                 offset < mapped_bytes_; offset += stride) {
                const auto stripe_bytes =
                    std::min(kNumaStripeBytes, mapped_bytes_ - offset);
                std::memset(base_ + offset, 0,
                            static_cast<std::size_t>(stripe_bytes));
            }
        });
    }
    for (auto& worker : prefault) worker.join();
    const auto prefault_huge = anonymous_huge_bytes(base_, mapped_bytes_);
    const auto required_prefault_huge = static_cast<std::uint64_t>(
        static_cast<double>(logical_bytes) * kMinimumHugeBacking);
    stats_.hugepage_prefaulted = prefault_huge >= required_prefault_huge;
    if (!stats_.hugepage_prefaulted) {
        release();
        result.errors.emplace_back(
            "GLM-5.3 expert arena received insufficient anonymous huge-page "
            "backing during first touch (" +
            std::to_string(prefault_huge) + " of " +
            std::to_string(logical_bytes) + " bytes)");
        return result;
    }

    struct StageTensor {
        const Glm53ManifestTensor* tensor{};
        std::uint64_t offset{};
    };
    std::vector<StageTensor> staged;
    staged.reserve(tensors.size());
    extents_.reserve(tensors.size());
    std::uint64_t cursor = 0U;
    for (const auto* tensor : tensors) {
        staged.push_back(StageTensor{tensor, cursor});
        extents_.emplace(tensor->name,
                         Extent{cursor, tensor->source_bytes});
        std::uint64_t tensor_bytes = 0U;
        if (!align_up(tensor->source_bytes, kTensorAlignment, tensor_bytes)) {
            release();
            result.errors.emplace_back("GLM-5.3 expert arena layout overflows");
            return result;
        }
        cursor += tensor_bytes;
    }

    for (std::size_t shard_begin = 0U; shard_begin < staged.size();) {
        std::size_t shard_end = shard_begin + 1U;
        while (shard_end < staged.size() &&
               staged[shard_end].tensor->shard ==
                   staged[shard_begin].tensor->shard) {
            ++shard_end;
        }
        const auto task_count = shard_end - shard_begin;
        const auto worker_count = std::min({
            kMaximumReadWorkers, task_count,
            static_cast<std::size_t>(hardware.worker_threads(0.15))});
        std::atomic<std::size_t> next{shard_begin};
        std::vector<ValidationResult> loaded(task_count);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0U; worker < worker_count; ++worker) {
            workers.emplace_back([&] {
                while (true) {
                    const auto index = next.fetch_add(1U);
                    if (index >= shard_end) return;
                    const auto& task = staged[index];
                    loaded[index - shard_begin] = checkpoint.read_into(
                        task.tensor->name,
                        {base_ + task.offset,
                         static_cast<std::size_t>(task.tensor->source_bytes)});
                }
            });
        }
        for (auto& worker : workers) worker.join();
        for (auto& status : loaded) {
            if (!status.ok()) {
                static_cast<void>(checkpoint.discard_shard_pages(
                    staged[shard_begin].tensor->shard));
                result.errors.insert(result.errors.end(),
                                     std::make_move_iterator(
                                         status.errors.begin()),
                                     std::make_move_iterator(
                                         status.errors.end()));
            }
        }
        if (!result.ok()) {
            release();
            return result;
        }
        auto discarded = checkpoint.discard_shard_pages(
            staged[shard_begin].tensor->shard);
        if (!discarded.ok()) {
            result.errors = std::move(discarded.errors);
            release();
            return result;
        }
        ++stats_.shards;
        shard_begin = shard_end;
    }
    if (cursor != logical_bytes || extents_.size() != tensors.size()) {
        release();
        result.errors.emplace_back(
            "GLM-5.3 expert arena staging accounting mismatch");
        return result;
    }
    if (mprotect(base_, static_cast<std::size_t>(mapped_bytes_), PROT_READ) !=
        0) {
        const auto reason = std::strerror(errno);
        release();
        result.errors.emplace_back(
            "cannot seal GLM-5.3 expert arena: " + std::string(reason));
        return result;
    }

    stats_.logical_bytes = logical_bytes;
    stats_.mapped_bytes = mapped_bytes_;
    stats_.anonymous_huge_bytes = anonymous_huge_bytes(base_, mapped_bytes_);
    stats_.numa_bytes = numa_resident_bytes(
        base_, mapped_bytes_, hardware.numa.nodes);
    stats_.tensors = tensors.size();
    stats_.stage_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const auto required_huge = static_cast<std::uint64_t>(
        static_cast<double>(logical_bytes) * kMinimumHugeBacking);
    if (stats_.anonymous_huge_bytes < required_huge) {
        const auto observed = stats_.anonymous_huge_bytes;
        release();
        result.errors.emplace_back(
            "GLM-5.3 expert arena received insufficient anonymous huge-page "
            "backing (" + std::to_string(observed) + " of " +
            std::to_string(logical_bytes) + " bytes)");
        return result;
    }
    const auto numa_total = std::accumulate(stats_.numa_bytes.begin(),
                                            stats_.numa_bytes.end(),
                                            std::uint64_t{0U});
    if (stats_.numa_bytes.size() !=
            static_cast<std::size_t>(hardware.numa.nodes) ||
        numa_total < required_huge) {
        release();
        result.errors.emplace_back(
            "GLM-5.3 expert arena NUMA residency could not be verified");
        return result;
    }
    if (hardware.numa.multi_node()) {
        const auto expected = numa_total /
            static_cast<std::uint64_t>(hardware.numa.nodes);
        const auto tolerance = numa_total / 50U;  // two percentage points
        for (const auto node_bytes : stats_.numa_bytes) {
            const auto difference = node_bytes > expected
                ? node_bytes - expected : expected - node_bytes;
            if (difference > tolerance) {
                release();
                result.errors.emplace_back(
                    "GLM-5.3 expert arena is not balanced across NUMA nodes");
                return result;
            }
        }
    }
    complete_ = true;
    return result;
}

std::span<const std::byte> Glm53ExpertArena::find(
    std::string_view name) const noexcept {
    if (!complete_ || base_ == nullptr) return {};
    const auto found = extents_.find(std::string(name));
    if (found == extents_.end()) return {};
    return {base_ + found->second.offset,
            static_cast<std::size_t>(found->second.bytes)};
}

}  // namespace strata
