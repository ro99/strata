#include "strata/kimi_k3_expert_arena.hpp"

#include "strata/placement.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <sys/mman.h>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

namespace strata {
namespace {

// O_DIRECT requires offset, length, and buffer to be 512-byte aligned on most
// filesystems; 4096 is the safe figure on ext4 and what this path uses.
constexpr std::uint64_t kDirectAlignment = 4096U;
constexpr std::uint64_t kSectorBytes = 512U;

[[nodiscard]] std::uint64_t align_down(std::uint64_t value,
                                       std::uint64_t alignment) noexcept {
    return value - value % alignment;
}

[[nodiscard]] std::uint64_t align_up(std::uint64_t value,
                                     std::uint64_t alignment) noexcept {
    const auto remainder = value % alignment;
    return remainder == 0U ? value : value + (alignment - remainder);
}

[[nodiscard]] std::uint64_t expert_key(std::uint32_t layer,
                                       std::uint32_t expert) noexcept {
    return (static_cast<std::uint64_t>(layer) << 32U) | expert;
}

[[nodiscard]] ParseResult<std::uint64_t> disk_stat_field(const std::string& disk,
                                                         std::size_t field) {
    ParseResult<std::uint64_t> result;
    if (disk.empty() || disk.find('/') != std::string::npos) {
        result.errors.push_back("invalid block device name " + disk);
        return result;
    }
    const auto path = "/sys/block/" + disk + "/stat";
    std::ifstream input(path);
    if (!input) {
        result.errors.push_back("cannot read " + path);
        return result;
    }
    std::string line;
    std::getline(input, line);
    std::istringstream values(line);
    std::uint64_t value = 0U;
    for (std::size_t index = 0U; index <= field; ++index) {
        if (!(values >> value)) {
            result.errors.push_back(path + " has fewer fields than expected");
            return result;
        }
    }
    result.value = value;
    return result;
}

}  // namespace

ValidationResult kimi_apply_write_guard(const KimiWriteGuardConfig& config) {
    ValidationResult result;
    const auto forbidden = [&config](const std::string& disk) {
        return std::find(config.forbidden_disks.begin(),
                         config.forbidden_disks.end(),
                         disk) != config.forbidden_disks.end();
    };

    for (const auto& path : config.write_paths) {
        // Resolve the deepest existing ancestor: a path the run will create
        // inherits the backing device of the directory it lands in.
        std::filesystem::path probe(path);
        std::error_code code;
        while (!probe.empty() && !std::filesystem::exists(probe, code)) {
            const auto parent = probe.parent_path();
            if (parent == probe) break;
            probe = parent;
        }
        const auto storage = resolve_backing_storage(probe.string());
        if (!storage.resolved) {
            result.errors.push_back(
                "cannot resolve the backing block device of " + path +
                "; refusing to run rather than assume it is not protected");
            continue;
        }
        // A memory-backed path cannot write to any disk, so it is always
        // admitted regardless of what this run protects.
        if (!storage.memory_backed && forbidden(storage.disk)) {
            result.errors.push_back(
                path + " resolves to " + storage.disk +
                ", which this run may not write to");
        }
    }

    if (config.require_no_forbidden_swap) {
        std::ifstream swaps("/proc/swaps");
        std::string line;
        std::getline(swaps, line);  // header
        while (std::getline(swaps, line)) {
            std::istringstream fields(line);
            std::string name;
            if (!(fields >> name)) continue;
            const auto storage = resolve_backing_storage(name);
            const auto disk = storage.resolved
                ? storage.disk
                : std::filesystem::path(name).filename().string();
            if (!forbidden(disk)) continue;
            result.errors.push_back(
                "swap is active on " + disk + " (" + name +
                "): a large anonymous arena can be paged out to it, which "
                "writes model-derived bytes to a device this run protects. "
                "Run `swapoff " + name + "` and remove it from /etc/fstab");
        }
    }

    if (config.disable_core_dumps) {
        rlimit limit{0U, 0U};
        if (::setrlimit(RLIMIT_CORE, &limit) != 0) {
            result.errors.push_back(
                std::string("cannot disable core dumps: ") + std::strerror(errno) +
                "; a crash at this resident size would write a core file");
        }
    }
    return result;
}

namespace {

// One expert's six payloads sorted the way the shard stores them. `stage` reads
// them in this order and `kimi_expert_slot_layout` reports where each landed,
// so the writer and every reader of a slot share one ordering.
struct OrderedModule {
    std::string_view shard;
    std::uint64_t offset{};
    std::uint64_t bytes{};
    // Which of the six this is, so the layout can be reported by name after the
    // sort has shuffled them.
    std::uint8_t role{};
};

[[nodiscard]] std::array<OrderedModule, 6U> ordered_modules(
    const KimiExpertModules& modules) noexcept {
    std::array<OrderedModule, 6U> ordered{
        OrderedModule{modules.gate.shard, modules.gate.packed_offset,
                      modules.gate.packed_bytes, 0U},
        OrderedModule{modules.gate.shard, modules.gate.scale_offset,
                      modules.gate.scale_bytes, 1U},
        OrderedModule{modules.up.shard, modules.up.packed_offset,
                      modules.up.packed_bytes, 2U},
        OrderedModule{modules.up.shard, modules.up.scale_offset,
                      modules.up.scale_bytes, 3U},
        OrderedModule{modules.down.shard, modules.down.packed_offset,
                      modules.down.packed_bytes, 4U},
        OrderedModule{modules.down.shard, modules.down.scale_offset,
                      modules.down.scale_bytes, 5U}};
    std::sort(ordered.begin(), ordered.end(),
              [](const OrderedModule& left, const OrderedModule& right) {
                  if (left.shard != right.shard) return left.shard < right.shard;
                  return left.offset < right.offset;
              });
    return ordered;
}

}  // namespace

KimiExpertSlotLayout kimi_expert_slot_layout(
    const KimiExpertModules& modules) noexcept {
    const auto ordered = ordered_modules(modules);
    KimiExpertSlotLayout layout;
    std::uint64_t cursor = 0U;
    for (const auto& entry : ordered) {
        switch (entry.role) {
            case 0U: layout.gate_packed = cursor; break;
            case 1U: layout.gate_scale = cursor; break;
            case 2U: layout.up_packed = cursor; break;
            case 3U: layout.up_scale = cursor; break;
            case 4U: layout.down_packed = cursor; break;
            default: layout.down_scale = cursor; break;
        }
        cursor += entry.bytes;
    }
    layout.total_bytes = cursor;
    return layout;
}

ParseResult<std::uint64_t> kimi_disk_sectors_written(const std::string& disk) {
    // `/sys/block/<disk>/stat` field 7 (zero-based index 6) is cumulative
    // sectors written. It is field 10 of a `/proc/diskstats` row, which carries
    // three extra leading columns; reading index 9 here lands on `io_ticks`
    // instead and reports milliseconds of activity as if they were sectors.
    auto sectors = disk_stat_field(disk, 6U);
    if (!sectors.ok()) return sectors;
    sectors.value *= kSectorBytes;
    return sectors;
}

ParseResult<std::uint64_t> kimi_disk_sectors_read(const std::string& disk) {
    auto sectors = disk_stat_field(disk, 2U);
    if (!sectors.ok()) return sectors;
    sectors.value *= kSectorBytes;
    return sectors;
}

ParseResult<double> kimi_measure_idle_write_rate(const std::string& disk,
                                                 double seconds) {
    ParseResult<double> result;
    if (!(seconds > 0.0)) {
        result.errors.emplace_back("idle write sampling needs a positive window");
        return result;
    }
    auto before = kimi_disk_sectors_written(disk);
    if (!before.ok()) {
        result.errors = std::move(before.errors);
        return result;
    }
    const auto begin = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    const auto end = std::chrono::steady_clock::now();
    auto after = kimi_disk_sectors_written(disk);
    if (!after.ok()) {
        result.errors = std::move(after.errors);
        return result;
    }
    const auto elapsed = std::chrono::duration<double>(end - begin).count();
    result.value = static_cast<double>(after.value - before.value) / elapsed;
    return result;
}

// ------------------------------------------------------------------ arena

KimiExpertArena::~KimiExpertArena() { release(); }

KimiExpertArena::KimiExpertArena(KimiExpertArena&& other) noexcept {
    *this = std::move(other);
}

KimiExpertArena& KimiExpertArena::operator=(KimiExpertArena&& other) noexcept {
    if (this != &other) {
        release();
        capacity_bytes_ = other.capacity_bytes_;
        slot_bytes_ = other.slot_bytes_;
        clock_ = other.clock_;
        hits_ = other.hits_;
        misses_ = other.misses_;
        evictions_ = other.evictions_;
        locked_ = other.locked_;
        base_ = other.base_;
        slots_ = std::move(other.slots_);
        other.base_ = nullptr;
        other.capacity_bytes_ = 0U;
        other.locked_ = false;
    }
    return *this;
}

void KimiExpertArena::release() noexcept {
    if (base_ == nullptr) return;
    if (locked_) ::munlock(base_, capacity_bytes_);
    ::munmap(base_, capacity_bytes_);
    base_ = nullptr;
    locked_ = false;
}

void KimiExpertArena::clear() noexcept {
    release();
    capacity_bytes_ = 0U;
    slot_bytes_ = 0U;
    clock_ = 0U;
    hits_ = 0U;
    misses_ = 0U;
    evictions_ = 0U;
    slots_.clear();
}

ValidationResult KimiExpertArena::reset(const KimiArenaConfig& config) {
    ValidationResult result;
    clear();
    const auto slot = align_up(KimiCheckpointReader::expert_source_bytes(),
                               kDirectAlignment);
    if (config.capacity_bytes < slot) {
        result.errors.emplace_back(
            "Kimi-K3 expert arena must hold at least one expert");
        return result;
    }
    const auto count = config.capacity_bytes / slot;
    const auto bytes = count * slot;

    // Anonymous, so nothing on disk backs it; locked below so nothing pages it
    // out either.
    auto* memory = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (memory == MAP_FAILED) {
        result.errors.push_back(std::string("cannot reserve a ") +
                                format_bytes(bytes) + " expert arena: " +
                                std::strerror(errno));
        return result;
    }
    base_ = static_cast<std::byte*>(memory);
    capacity_bytes_ = bytes;
    slot_bytes_ = slot;
    slots_.assign(static_cast<std::size_t>(count), Entry{});
    for (std::size_t index = 0U; index < slots_.size(); ++index) {
        slots_[index].offset = static_cast<std::uint64_t>(index) * slot;
    }

    if (config.lock_pages) {
        if (::mlock(base_, capacity_bytes_) != 0) {
            const auto reason = std::strerror(errno);
            release();
            result.errors.push_back(
                std::string("cannot lock the ") + format_bytes(bytes) +
                " expert arena: " + reason +
                ". An unlocked arena can be paged to swap, which writes "
                "model-derived bytes to disk. Raise RLIMIT_MEMLOCK to at least " +
                std::to_string(bytes) + " bytes (ulimit -l) or run with "
                "locking disabled and accept that write path");
            return result;
        }
        locked_ = true;
    }
    return result;
}

std::int64_t KimiExpertArena::locate(std::uint64_t key) const noexcept {
    for (std::size_t index = 0U; index < slots_.size(); ++index) {
        if (slots_[index].published && slots_[index].key == key) {
            return static_cast<std::int64_t>(index);
        }
    }
    return -1;
}

std::span<const std::byte> KimiExpertArena::find(std::uint32_t layer,
                                                 std::uint32_t expert) noexcept {
    if (base_ == nullptr) return {};
    const auto index = locate(expert_key(layer, expert));
    if (index < 0) {
        ++misses_;
        return {};
    }
    auto& entry = slots_[static_cast<std::size_t>(index)];
    entry.stamp = ++clock_;
    ++hits_;
    return std::span<const std::byte>(base_ + entry.offset,
                                      static_cast<std::size_t>(entry.bytes));
}

std::span<std::byte> KimiExpertArena::reserve(std::uint32_t layer,
                                              std::uint32_t expert,
                                              std::uint64_t bytes) {
    if (base_ == nullptr || bytes > slot_bytes_) return {};
    const auto key = expert_key(layer, expert);
    const auto existing = locate(key);
    if (existing >= 0) {
        auto& entry = slots_[static_cast<std::size_t>(existing)];
        entry.stamp = ++clock_;
        return std::span<std::byte>(base_ + entry.offset,
                                    static_cast<std::size_t>(bytes));
    }
    // Prefer a never-used slot, else evict the least recently used published
    // one. A slot that is reserved but not yet published is in flight: handing
    // it out again would let two experts race into the same bytes and leave
    // whichever read finished last visible under both keys.
    std::size_t victim = slots_.size();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0U; index < slots_.size(); ++index) {
        if (slots_[index].stamp == 0U) {
            victim = index;
            break;
        }
        if (slots_[index].published && slots_[index].stamp < oldest) {
            oldest = slots_[index].stamp;
            victim = index;
        }
    }
    if (victim == slots_.size()) return {};
    auto& entry = slots_[victim];
    if (entry.published) ++evictions_;
    entry.key = key;
    entry.bytes = bytes;
    entry.stamp = ++clock_;
    entry.published = false;
    return std::span<std::byte>(base_ + entry.offset,
                                static_cast<std::size_t>(bytes));
}

void KimiExpertArena::publish(std::uint32_t layer,
                              std::uint32_t expert) noexcept {
    const auto key = expert_key(layer, expert);
    for (auto& entry : slots_) {
        if (entry.key == key && !entry.published) {
            entry.published = true;
            return;
        }
    }
}

// ----------------------------------------------------------------- reader

KimiExpertReader::~KimiExpertReader() { close(); }

KimiExpertReader::KimiExpertReader(KimiExpertReader&& other) noexcept {
    *this = std::move(other);
}

KimiExpertReader& KimiExpertReader::operator=(KimiExpertReader&& other) noexcept {
    if (this != &other) {
        close();
        queue_depth_ = other.queue_depth_;
        direct_ = other.direct_;
        stats_ = other.stats_;
        shards_ = std::move(other.shards_);
        other.shards_.clear();
    }
    return *this;
}

void KimiExpertReader::close() noexcept {
    for (auto& shard : shards_) {
        if (shard.descriptor >= 0) ::close(shard.descriptor);
        shard.descriptor = -1;
    }
    shards_.clear();
}

ValidationResult KimiExpertReader::open(const KimiCheckpointReader& checkpoint,
                                        const KimiReaderConfig& config) {
    ValidationResult result;
    close();
    if (config.queue_depth == 0U) {
        result.errors.emplace_back("expert reader needs a positive queue depth");
        return result;
    }
    queue_depth_ = config.queue_depth;
    direct_ = config.direct;
    stats_ = {};

    for (const auto& name : checkpoint.manifest().shards) {
        const auto path =
            (std::filesystem::path(checkpoint.model_directory()) / name).string();
        int flags = O_RDONLY | O_CLOEXEC;
        if (direct_) flags |= O_DIRECT;
        int descriptor = ::open(path.c_str(), flags);
        if (descriptor < 0 && direct_ && errno == EINVAL) {
            // The filesystem refused O_DIRECT. Say so rather than falling back
            // silently: buffered reads keep a second copy of every expert in
            // the page cache, which changes the memory budget.
            result.errors.push_back(
                path + " does not support O_DIRECT; buffered reads would "
                "duplicate every expert in the page cache");
            close();
            return result;
        }
        if (descriptor < 0) {
            result.errors.push_back("cannot open " + path + ": " +
                                    std::strerror(errno));
            close();
            return result;
        }
        shards_.push_back({name, descriptor});
    }
    return result;
}

int KimiExpertReader::descriptor_for(std::string_view shard) const noexcept {
    for (const auto& entry : shards_) {
        if (entry.name == shard) return entry.descriptor;
    }
    return -1;
}

ValidationResult KimiExpertReader::stage(
    const KimiCheckpointReader& checkpoint, KimiExpertArena& arena,
    std::span<const KimiReadRequest> requests) {
    ValidationResult result;
    if (shards_.empty()) {
        result.errors.emplace_back("expert reader was not opened");
        return result;
    }
    // A slot reserved and not yet published is in flight and cannot be reused,
    // so no more than the arena's capacity can be outstanding at once. Window
    // the requests rather than failing: a decode step asks for 16 experts per
    // layer over 92 layers and the arena holds a bounded subset of them.
    if (requests.size() > arena.slot_count()) {
        for (std::size_t begin = 0U; begin < requests.size();
             begin += arena.slot_count()) {
            const auto count =
                std::min<std::size_t>(arena.slot_count(), requests.size() - begin);
            auto window = stage(checkpoint, arena, requests.subspan(begin, count));
            if (!window.ok()) return window;
        }
        return result;
    }

    // One read descriptor per module; three modules per expert. Building the
    // whole list first is what lets the loop below keep `queue_depth_` reads
    // outstanding instead of stalling on each one in turn.
    struct Pending {
        std::uint32_t layer{};
        std::uint32_t expert{};
        int descriptor{-1};
        std::uint64_t offset{};
        std::uint64_t bytes{};
        std::byte* destination{};
        // O_DIRECT needs an aligned offset, so a read starts at the aligned
        // boundary below the payload and the payload is found at this skew.
        std::uint64_t skew{};
        std::uint64_t span{};
        std::uint64_t modules{};
    };
    std::vector<Pending> pending;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> reserved;
    std::vector<std::byte> bounce;

    for (const auto& request : requests) {
        ++stats_.requests;
        if (!arena.find(request.layer, request.expert).empty()) continue;
        KimiExpertModules modules{};
        if (!checkpoint.expert_modules(request.layer, request.expert, modules)) {
            result.errors.push_back(
                "Kimi-K3 expert " + std::to_string(request.expert) + " of layer " +
                std::to_string(request.layer) + " is not in the checkpoint");
            return result;
        }
        auto slot = arena.reserve(request.layer, request.expert,
                                  KimiCheckpointReader::expert_source_bytes());
        if (slot.empty()) {
            result.errors.emplace_back(
                "Kimi-K3 expert arena could not admit a slot");
            return result;
        }
        reserved.emplace_back(request.layer, request.expert);

        // The six modules in shard order, from the same ordering `fetch` uses
        // to find them again inside the slot.
        const auto ordered = ordered_modules(modules);

        std::uint64_t cursor = 0U;
        for (std::size_t index = 0U; index < ordered.size();) {
            // Extend the run while the next module begins exactly where this
            // one ends, in the same shard. On this checkpoint all six are one
            // 16.73 MiB extent, so this issues one read per expert instead of
            // six; a checkpoint that interleaves them still reads correctly,
            // just with more submissions.
            std::size_t last = index;
            std::uint64_t span = ordered[index].bytes;
            while (last + 1U < ordered.size() &&
                   ordered[last + 1U].shard == ordered[index].shard &&
                   ordered[last + 1U].offset ==
                       ordered[last].offset + ordered[last].bytes) {
                ++last;
                span += ordered[last].bytes;
            }
            Pending entry;
            entry.layer = request.layer;
            entry.expert = request.expert;
            entry.descriptor = descriptor_for(ordered[index].shard);
            if (entry.descriptor < 0) {
                result.errors.emplace_back(
                    "Kimi-K3 expert names a shard the reader did not open");
                return result;
            }
            entry.offset = ordered[index].offset;
            entry.bytes = span;
            entry.destination = slot.data() + cursor;
            entry.modules = last - index + 1U;
            if (direct_) {
                const auto aligned = align_down(entry.offset, kDirectAlignment);
                entry.skew = entry.offset - aligned;
                entry.offset = aligned;
                entry.span = align_up(entry.skew + span, kDirectAlignment);
            } else {
                entry.span = span;
            }
            cursor += span;
            pending.push_back(entry);
            index = last + 1U;
        }
    }

    if (pending.empty()) return result;

    // O_DIRECT needs an aligned destination too. Reading through an aligned
    // bounce buffer keeps the arena layout packed; the copy is host-to-host at
    // memory bandwidth, three orders of magnitude above the link being fed.
    std::uint64_t widest = 0U;
    for (const auto& entry : pending) widest = std::max(widest, entry.span);
    std::vector<std::byte> staging;
    if (direct_) {
        staging.resize(static_cast<std::size_t>(widest) * queue_depth_ +
                       kDirectAlignment);
    }
    auto* aligned_base =
        direct_ ? reinterpret_cast<std::byte*>(
                      align_up(reinterpret_cast<std::uintptr_t>(staging.data()),
                               kDirectAlignment))
                : nullptr;

    // Keep `queue_depth_` reads genuinely in flight. `pread` is synchronous, so
    // depth has to come from concurrent threads: batching synchronous calls
    // would issue them one at a time and deliver depth one while reporting
    // whatever depth was configured. The link saturates at depth four and runs
    // at roughly half speed at depth one, so this is a serialization fix and
    // not a volume one.
    std::atomic<std::size_t> next{0U};
    std::atomic<bool> failed{false};
    std::vector<std::string> failures(queue_depth_);
    const auto worker = [&](std::size_t lane) {
        auto* destination = direct_ ? aligned_base + lane * widest : nullptr;
        for (;;) {
            const auto index = next.fetch_add(1U, std::memory_order_relaxed);
            if (index >= pending.size() || failed.load(std::memory_order_relaxed)) {
                return;
            }
            auto& entry = pending[index];
            auto* target = direct_ ? destination : entry.destination;
            std::uint64_t done = 0U;
            while (done < entry.span) {
                const auto read = ::pread(
                    entry.descriptor, target + done,
                    static_cast<std::size_t>(entry.span - done),
                    static_cast<off_t>(entry.offset + done));
                if (read == 0) break;
                if (read < 0) {
                    if (errno == EINTR) continue;
                    failures[lane] = std::string("Kimi-K3 expert read failed: ") +
                                     std::strerror(errno);
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                done += static_cast<std::uint64_t>(read);
            }
            if (done < entry.skew + entry.bytes) {
                failures[lane] =
                    "Kimi-K3 expert read returned short of the module extent";
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (direct_) {
                std::memcpy(entry.destination, target + entry.skew,
                            static_cast<std::size_t>(entry.bytes));
            }
        }
    };

    const auto lanes = std::min<std::size_t>(queue_depth_, pending.size());
    stats_.peak_queue_depth = std::max<std::uint64_t>(stats_.peak_queue_depth,
                                                      lanes);
    std::vector<std::thread> threads;
    threads.reserve(lanes - 1U);
    for (std::size_t lane = 1U; lane < lanes; ++lane) {
        threads.emplace_back(worker, lane);
    }
    worker(0U);
    for (auto& thread : threads) thread.join();
    for (auto& failure : failures) {
        if (!failure.empty()) result.errors.push_back(std::move(failure));
    }
    if (!result.ok()) return result;
    stats_.submissions += pending.size();
    for (const auto& entry : pending) {
        stats_.bytes_read += entry.bytes;
        stats_.coalesced_modules += entry.modules;
    }

    for (const auto& [layer, expert] : reserved) arena.publish(layer, expert);
    return result;
}

}  // namespace strata
