#include "strata/deepseek_kv_cache.hpp"

#include "strata/model_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace strata {

namespace {

struct TableKey {
    Dsv4KvBlockKind kind{Dsv4KvBlockKind::Sliding};
    std::uint32_t layer{};

    friend bool operator==(const TableKey&, const TableKey&) = default;
};

struct TableKeyHash {
    std::size_t operator()(const TableKey& key) const noexcept {
        return (static_cast<std::size_t>(key.layer) << 3U) ^
               static_cast<std::size_t>(key.kind);
    }
};

[[nodiscard]] std::uint32_t required_ratio(Dsv4KvBlockKind kind) noexcept {
    switch (kind) {
        case Dsv4KvBlockKind::Sliding: return 1U;
        case Dsv4KvBlockKind::Csa: return 4U;
        case Dsv4KvBlockKind::Hca: return 128U;
        case Dsv4KvBlockKind::LearnedIndex: return 4U;
    }
    return 0U;
}

[[nodiscard]] std::uint32_t required_width(Dsv4KvBlockKind kind) noexcept {
    return kind == Dsv4KvBlockKind::LearnedIndex
        ? kDeepSeekV4ExecutionContract.index_head_dim
        : kDeepSeekV4ExecutionContract.head_dim;
}

}  // namespace

struct Dsv4KvCacheState {
    struct Table {
        std::vector<std::uint64_t> blocks;
        std::uint64_t minimum_row{};
        std::uint64_t end_row{};
        std::uint32_t compression_ratio{};
    };

    struct Sequence {
        std::uint64_t tokens{};
        std::unordered_map<TableKey, Table, TableKeyHash> tables;
    };

    struct Block {
        Dsv4KvBlockInfo info;
        std::vector<float> host;
        std::vector<CudaBuffer> devices;
        std::vector<std::uint64_t> device_last_use;
        std::vector<std::uint32_t> device_leases;
    };

    explicit Dsv4KvCacheState(Dsv4KvCacheConfig value, CudaBackend* backend)
        : config(std::move(value)), cuda(backend) {
        metrics.host_capacity_bytes = config.host_capacity_bytes;
        metrics.device_capacity_bytes = config.device_capacity_bytes;
        metrics.device_used_bytes.resize(config.devices.size());
        metrics.device_peak_bytes.resize(config.devices.size());
    }

    [[nodiscard]] ValidationResult validate() const {
        ValidationResult result;
        if (config.block_rows == 0U || config.sliding_window_rows == 0U) {
            result.errors.emplace_back(
                "DeepSeek KV block rows and sliding window must be positive");
        }
        if (config.host_capacity_bytes == 0U) {
            result.errors.emplace_back(
                "DeepSeek KV cache requires an explicit host capacity");
        }
        if (config.devices.size() != config.device_capacity_bytes.size()) {
            result.errors.emplace_back(
                "DeepSeek KV device ids and capacities must have equal lengths");
        }
        std::unordered_set<int> unique;
        for (const int device : config.devices) {
            if (device < 0 || !unique.insert(device).second) {
                result.errors.emplace_back(
                    "DeepSeek KV devices must be unique non-negative ids");
                break;
            }
        }
        return result;
    }

    [[nodiscard]] Sequence* sequence(Dsv4SequenceHandle handle) noexcept {
        const auto found = sequences.find(handle);
        return found == sequences.end() ? nullptr : &found->second;
    }

    [[nodiscard]] const Sequence* sequence(
        Dsv4SequenceHandle handle) const noexcept {
        const auto found = sequences.find(handle);
        return found == sequences.end() ? nullptr : &found->second;
    }

    [[nodiscard]] std::uint64_t total_leases(const Block& block) const noexcept {
        std::uint64_t total = 0U;
        for (const auto leases : block.device_leases) total += leases;
        return total;
    }

    void erase_block(std::uint64_t id) noexcept {
        const auto found = blocks.find(id);
        if (found == blocks.end() || found->second.info.refcount != 0U ||
            total_leases(found->second) != 0U) {
            return;
        }
        auto& block = found->second;
        const auto bytes = static_cast<std::uint64_t>(block.host.size()) *
                           sizeof(float);
        metrics.host_used_bytes -= bytes;
        for (std::size_t slot = 0U; slot < block.devices.size(); ++slot) {
            if (block.devices[slot].valid()) {
                metrics.device_used_bytes[slot] -=
                    block.devices[slot].device_bytes();
            }
        }
        blocks.erase(found);
    }

    void release_reference(std::uint64_t id) noexcept {
        const auto found = blocks.find(id);
        if (found == blocks.end() || found->second.info.refcount == 0U) return;
        --found->second.info.refcount;
        erase_block(id);
    }

    void release_tables(Sequence& target) noexcept {
        for (auto& [key, table] : target.tables) {
            static_cast<void>(key);
            for (const auto id : table.blocks) release_reference(id);
        }
        target.tables.clear();
        target.tokens = 0U;
    }

    [[nodiscard]] ValidationResult allocate_block(
        Dsv4SequenceHandle owner, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint32_t ratio, std::uint64_t start_row,
        const Block* source, std::uint64_t& output) {
        ValidationResult result;
        const auto width = required_width(kind);
        const auto elements = static_cast<std::uint64_t>(config.block_rows) * width;
        if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
            result.errors.emplace_back("DeepSeek KV block byte count overflows");
            return result;
        }
        const auto bytes = elements * sizeof(float);
        if (bytes > config.host_capacity_bytes -
                        std::min(config.host_capacity_bytes,
                                 metrics.host_used_bytes)) {
            result.errors.emplace_back(
                "DeepSeek KV host cache capacity is exhausted");
            return result;
        }
        if (next_block_id == std::numeric_limits<std::uint64_t>::max() ||
            start_row > std::numeric_limits<std::uint64_t>::max() / ratio) {
            result.errors.emplace_back("DeepSeek KV block identity overflows");
            return result;
        }
        const auto started = std::chrono::steady_clock::now();
        try {
            Block block;
            block.info.id = next_block_id++;
            block.info.owner_sequence = owner;
            block.info.logical_begin = start_row * ratio;
            block.info.used_rows = source == nullptr ? 0U : source->info.used_rows;
            block.info.capacity_rows = config.block_rows;
            block.info.row_width = width;
            block.info.layer = layer;
            block.info.compression_ratio = ratio;
            block.info.refcount = 1U;
            block.info.kind = kind;
            block.info.format = Dsv4KvFormat::F32;
            block.host.resize(static_cast<std::size_t>(elements));
            if (source != nullptr) block.host = source->host;
            block.devices.resize(config.devices.size());
            block.device_last_use.resize(config.devices.size());
            block.device_leases.resize(config.devices.size());
            output = block.info.id;
            blocks.emplace(output, std::move(block));
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back("cannot allocate DeepSeek KV block");
            return result;
        }
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        ++metrics.allocated_blocks;
        ++metrics.allocation_calls;
        metrics.allocation_nanoseconds += elapsed;
        metrics.host_used_bytes += bytes;
        metrics.host_peak_bytes = std::max(metrics.host_peak_bytes,
                                           metrics.host_used_bytes);
        return result;
    }

    [[nodiscard]] Block* find_block(Table& table,
                                    std::uint64_t logical_row) noexcept {
        const auto found_id = std::lower_bound(
            table.blocks.begin(), table.blocks.end(), logical_row,
            [this](std::uint64_t id, std::uint64_t row) {
                const auto& block = blocks.at(id);
                const auto begin = block.info.logical_begin /
                                   block.info.compression_ratio;
                return begin + block.info.capacity_rows <= row;
            });
        if (found_id == table.blocks.end()) return nullptr;
        auto& block = blocks.at(*found_id);
        const auto begin = block.info.logical_begin /
                           block.info.compression_ratio;
        return logical_row >= begin ? &block : nullptr;
    }

    [[nodiscard]] const Block* find_block(
        const Table& table, std::uint64_t logical_row) const noexcept {
        const auto found_id = std::lower_bound(
            table.blocks.begin(), table.blocks.end(), logical_row,
            [this](std::uint64_t id, std::uint64_t row) {
                const auto& block = blocks.at(id);
                const auto begin = block.info.logical_begin /
                                   block.info.compression_ratio;
                return begin + block.info.capacity_rows <= row;
            });
        if (found_id == table.blocks.end()) return nullptr;
        const auto& block = blocks.at(*found_id);
        const auto begin = block.info.logical_begin /
                           block.info.compression_ratio;
        return logical_row >= begin ? &block : nullptr;
    }

    void release_device_lease(std::uint64_t id, std::size_t slot) noexcept {
        const auto found = blocks.find(id);
        if (found == blocks.end() || slot >= found->second.device_leases.size() ||
            found->second.device_leases[slot] == 0U) {
            return;
        }
        --found->second.device_leases[slot];
        --found->second.info.in_flight;
        erase_block(id);
    }

    Dsv4KvCacheConfig config;
    CudaBackend* cuda{};
    std::unordered_map<Dsv4SequenceHandle, Sequence> sequences;
    std::unordered_map<std::uint64_t, Block> blocks;
    Dsv4KvCacheStats metrics;
    Dsv4SequenceHandle next_sequence_id{1U};
    std::uint64_t next_block_id{1U};
    std::uint64_t clock{};
};

Dsv4KvDeviceLease::Dsv4KvDeviceLease() = default;

Dsv4KvDeviceLease::Dsv4KvDeviceLease(
    std::shared_ptr<Dsv4KvCacheState> state, std::uint64_t block_id,
    std::size_t device_slot)
    : state_(std::move(state)), block_id_(block_id), device_slot_(device_slot) {}

Dsv4KvDeviceLease::~Dsv4KvDeviceLease() { release(); }

Dsv4KvDeviceLease::Dsv4KvDeviceLease(Dsv4KvDeviceLease&& other) noexcept
    : state_(std::move(other.state_)), block_id_(other.block_id_),
      device_slot_(other.device_slot_) {
    other.block_id_ = 0U;
}

Dsv4KvDeviceLease& Dsv4KvDeviceLease::operator=(
    Dsv4KvDeviceLease&& other) noexcept {
    if (this == &other) return *this;
    release();
    state_ = std::move(other.state_);
    block_id_ = other.block_id_;
    device_slot_ = other.device_slot_;
    other.block_id_ = 0U;
    return *this;
}

void Dsv4KvDeviceLease::release() noexcept {
    if (state_ != nullptr && block_id_ != 0U) {
        state_->release_device_lease(block_id_, device_slot_);
    }
    state_.reset();
    block_id_ = 0U;
}

bool Dsv4KvDeviceLease::valid() const noexcept {
    return buffer() != nullptr;
}

const CudaBuffer* Dsv4KvDeviceLease::buffer() const noexcept {
    if (state_ == nullptr || block_id_ == 0U) return nullptr;
    const auto found = state_->blocks.find(block_id_);
    if (found == state_->blocks.end() ||
        device_slot_ >= found->second.devices.size() ||
        !found->second.devices[device_slot_].valid()) {
        return nullptr;
    }
    return &found->second.devices[device_slot_];
}

Dsv4KvCache::Dsv4KvCache(Dsv4KvCacheConfig config, CudaBackend* cuda)
    : state_(std::make_shared<Dsv4KvCacheState>(std::move(config), cuda)) {}

ValidationResult Dsv4KvCache::validate() const { return state_->validate(); }

ParseResult<Dsv4SequenceHandle> Dsv4KvCache::create_sequence() {
    ParseResult<Dsv4SequenceHandle> result;
    result.errors = state_->validate().errors;
    if (!result.ok()) return result;
    if (state_->next_sequence_id == std::numeric_limits<std::uint64_t>::max()) {
        result.errors.emplace_back("DeepSeek KV sequence identity overflows");
        return result;
    }
    result.value = state_->next_sequence_id++;
    try {
        state_->sequences.emplace(result.value, Dsv4KvCacheState::Sequence{});
    } catch (const std::bad_alloc&) {
        result.value = 0U;
        result.errors.emplace_back("cannot allocate DeepSeek KV sequence metadata");
        return result;
    }
    ++state_->metrics.sequence_creations;
    return result;
}

ParseResult<Dsv4SequenceHandle> Dsv4KvCache::fork_sequence(
    Dsv4SequenceHandle source) {
    ParseResult<Dsv4SequenceHandle> result;
    const auto* original = state_->sequence(source);
    if (original == nullptr) {
        result.errors.emplace_back("DeepSeek KV source sequence does not exist");
        return result;
    }
    if (state_->next_sequence_id == std::numeric_limits<std::uint64_t>::max()) {
        result.errors.emplace_back("DeepSeek KV sequence identity overflows");
        return result;
    }
    for (const auto& [key, table] : original->tables) {
        static_cast<void>(key);
        for (const auto id : table.blocks) {
            const auto found = state_->blocks.find(id);
            if (found == state_->blocks.end() ||
                found->second.info.refcount ==
                    std::numeric_limits<std::uint32_t>::max()) {
                result.errors.emplace_back(
                    "DeepSeek KV block reference count overflows");
                return result;
            }
        }
    }
    Dsv4KvCacheState::Sequence copy;
    try {
        copy = *original;
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back("cannot allocate DeepSeek KV fork metadata");
        return result;
    }
    result.value = state_->next_sequence_id++;
    try {
        state_->sequences.emplace(result.value, std::move(copy));
    } catch (const std::bad_alloc&) {
        result.value = 0U;
        result.errors.emplace_back("cannot allocate DeepSeek KV fork metadata");
        return result;
    }
    auto* fork = state_->sequence(result.value);
    for (const auto& [key, table] : fork->tables) {
        static_cast<void>(key);
        for (const auto id : table.blocks) ++state_->blocks.at(id).info.refcount;
    }
    ++state_->metrics.sequence_creations;
    return result;
}

ValidationResult Dsv4KvCache::reset_sequence(Dsv4SequenceHandle sequence) {
    ValidationResult result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    state_->release_tables(*target);
    ++state_->metrics.sequence_resets;
    return result;
}

ValidationResult Dsv4KvCache::truncate_sequence(
    Dsv4SequenceHandle sequence, std::uint64_t tokens) {
    ValidationResult result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    if (tokens > target->tokens) {
        result.errors.emplace_back(
            "DeepSeek KV truncation exceeds the sequence length");
        return result;
    }
    for (const auto& [key, table] : target->tables) {
        if (key.kind == Dsv4KvBlockKind::Sliding &&
            tokens < table.minimum_row) {
            result.errors.emplace_back(
                "DeepSeek KV truncation precedes retained sliding state");
            return result;
        }
    }
    for (auto& [key, table] : target->tables) {
        const auto keep_rows = key.kind == Dsv4KvBlockKind::Sliding
            ? tokens : tokens / table.compression_ratio;
        table.end_row = std::min(table.end_row, keep_rows);
        while (!table.blocks.empty()) {
            const auto id = table.blocks.back();
            const auto& block = state_->blocks.at(id);
            const auto begin = block.info.logical_begin /
                               block.info.compression_ratio;
            if (begin < table.end_row) break;
            table.blocks.pop_back();
            state_->release_reference(id);
        }
        table.minimum_row = std::min(table.minimum_row, table.end_row);
    }
    target->tokens = tokens;
    ++state_->metrics.sequence_truncations;
    return result;
}

ValidationResult Dsv4KvCache::release_sequence(Dsv4SequenceHandle sequence) {
    ValidationResult result;
    const auto found = state_->sequences.find(sequence);
    if (found == state_->sequences.end()) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    state_->release_tables(found->second);
    state_->sequences.erase(found);
    ++state_->metrics.sequence_releases;
    return result;
}

ValidationResult Dsv4KvCache::append(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::uint32_t compression_ratio,
    std::uint64_t logical_row, std::span<const float> values) {
    ValidationResult result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    const auto wanted_ratio = required_ratio(kind);
    const auto width = required_width(kind);
    if (layer >= kDeepSeekV4ExecutionContract.layer_count ||
        compression_ratio != wanted_ratio || values.size() != width) {
        result.errors.emplace_back(
            "DeepSeek KV block kind, layer, ratio, or row width is invalid");
        return result;
    }
    if (logical_row == std::numeric_limits<std::uint64_t>::max() ||
        logical_row + 1U > std::numeric_limits<std::uint64_t>::max() /
                               compression_ratio) {
        result.errors.emplace_back("DeepSeek KV logical position overflows");
        return result;
    }
    const TableKey key{kind, layer};
    auto table_found = target->tables.find(key);
    if (table_found == target->tables.end()) {
        if (logical_row != 0U) {
            result.errors.emplace_back(
                "DeepSeek KV first row must start at logical row zero");
            return result;
        }
        try {
            table_found = target->tables.emplace(
                key,
                Dsv4KvCacheState::Table{{}, 0U, 0U, compression_ratio}).first;
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "cannot allocate DeepSeek KV block-table metadata");
            return result;
        }
    }
    auto& table = table_found->second;
    if (table.compression_ratio != compression_ratio ||
        logical_row != table.end_row) {
        result.errors.emplace_back(
            "DeepSeek KV append position is not contiguous");
        return result;
    }
    const auto block_begin =
        logical_row / state_->config.block_rows * state_->config.block_rows;
    auto* block = state_->find_block(table, logical_row);
    if (block != nullptr && block->info.refcount > 1U) {
        const auto old_id = block->info.id;
        std::uint64_t replacement = 0U;
        result = state_->allocate_block(sequence, kind, layer,
                                        compression_ratio, block_begin,
                                        block, replacement);
        if (!result.ok()) return result;
        const auto id = std::find(table.blocks.begin(), table.blocks.end(), old_id);
        *id = replacement;
        state_->release_reference(old_id);
        block = &state_->blocks.at(replacement);
        ++state_->metrics.copy_on_write_blocks;
    } else if (block == nullptr) {
        std::uint64_t id = 0U;
        result = state_->allocate_block(sequence, kind, layer,
                                        compression_ratio, block_begin,
                                        nullptr, id);
        if (!result.ok()) return result;
        try {
            table.blocks.push_back(id);
        } catch (const std::bad_alloc&) {
            state_->release_reference(id);
            result.errors.emplace_back(
                "cannot grow DeepSeek KV block-table metadata");
            return result;
        }
        block = &state_->blocks.at(id);
    }
    const auto offset = static_cast<std::size_t>(logical_row - block_begin) * width;
    std::copy(values.begin(), values.end(), block->host.begin() + offset);
    state_->metrics.host_write_bytes += values.size_bytes();
    block->info.used_rows = std::max(
        block->info.used_rows,
        static_cast<std::uint32_t>(logical_row - block_begin + 1U));
    ++table.end_row;
    target->tokens = std::max(target->tokens,
                              table.end_row * compression_ratio);
    if (kind == Dsv4KvBlockKind::Sliding) {
        table.minimum_row = table.end_row > state_->config.sliding_window_rows
            ? table.end_row - state_->config.sliding_window_rows : 0U;
        while (!table.blocks.empty()) {
            const auto id = table.blocks.front();
            const auto& first = state_->blocks.at(id);
            const auto begin = first.info.logical_begin;
            if (begin + first.info.capacity_rows > table.minimum_row) break;
            table.blocks.erase(table.blocks.begin());
            state_->release_reference(id);
        }
    }
    return result;
}

ParseResult<std::span<const float>> Dsv4KvCache::row(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::uint64_t logical_row) {
    ParseResult<std::span<const float>> result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        ++state_->metrics.misses;
        return result;
    }
    const auto table_found = target->tables.find(TableKey{kind, layer});
    if (table_found == target->tables.end() ||
        logical_row < table_found->second.minimum_row ||
        logical_row >= table_found->second.end_row) {
        result.errors.emplace_back("DeepSeek KV row is not retained");
        ++state_->metrics.misses;
        return result;
    }
    auto* block = state_->find_block(table_found->second, logical_row);
    if (block == nullptr) {
        result.errors.emplace_back("DeepSeek KV row block is unavailable");
        ++state_->metrics.misses;
        return result;
    }
    const auto block_begin = block->info.logical_begin /
                             block->info.compression_ratio;
    const auto offset = static_cast<std::size_t>(logical_row - block_begin) *
                        block->info.row_width;
    result.value = std::span<const float>(block->host).subspan(
        offset, block->info.row_width);
    ++state_->metrics.hits;
    state_->metrics.gather_bytes += result.value.size_bytes();
    return result;
}

ParseResult<std::vector<float>> Dsv4KvCache::gather(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::span<const std::uint32_t> logical_rows) {
    ParseResult<std::vector<float>> result;
    const auto width = required_width(kind);
    if (logical_rows.size() >
        std::numeric_limits<std::size_t>::max() / width) {
        result.errors.emplace_back("DeepSeek KV gather size overflows");
        return result;
    }
    try {
        result.value.reserve(logical_rows.size() * width);
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back("cannot allocate DeepSeek KV gather output");
        return result;
    }
    for (const auto logical_row : logical_rows) {
        auto fetched = row(sequence, kind, layer, logical_row);
        if (!fetched.ok()) {
            result.errors = std::move(fetched.errors);
            result.value.clear();
            return result;
        }
        result.value.insert(result.value.end(), fetched.value.begin(),
                            fetched.value.end());
    }
    return result;
}

ParseResult<Dsv4KvDeviceLease> Dsv4KvCache::acquire_device(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::uint64_t logical_row,
    std::size_t device_slot) {
    ParseResult<Dsv4KvDeviceLease> result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV device row is not retained");
        return result;
    }
    const auto table_found = target->tables.find(TableKey{kind, layer});
    if (table_found == target->tables.end() ||
        logical_row < table_found->second.minimum_row ||
        logical_row >= table_found->second.end_row) {
        result.errors.emplace_back("DeepSeek KV device row is not retained");
        return result;
    }
    auto* block = state_->find_block(table_found->second, logical_row);
    if (block == nullptr || device_slot >= state_->config.devices.size()) {
        result.errors.emplace_back("DeepSeek KV device slot or block is invalid");
        return result;
    }
    if (!block->devices[device_slot].valid()) {
        ++state_->metrics.misses;
        if (state_->cuda == nullptr ||
            state_->config.device_capacity_bytes[device_slot] == 0U) {
            result.errors.emplace_back(
                "DeepSeek KV device cache is not configured");
            return result;
        }
        const auto bytes = static_cast<std::uint64_t>(block->host.size()) *
                           sizeof(float);
        const auto capacity = state_->config.device_capacity_bytes[device_slot];
        if (bytes > capacity) {
            result.errors.emplace_back(
                "DeepSeek KV block exceeds the device cache capacity");
            return result;
        }
        while (state_->metrics.device_used_bytes[device_slot] >
               capacity - bytes) {
            Dsv4KvCacheState::Block* victim = nullptr;
            for (auto& [id, candidate] : state_->blocks) {
                static_cast<void>(id);
                if (!candidate.devices[device_slot].valid() ||
                    candidate.device_leases[device_slot] != 0U ||
                    &candidate == block) {
                    continue;
                }
                if (victim == nullptr ||
                    candidate.device_last_use[device_slot] <
                        victim->device_last_use[device_slot]) {
                    victim = &candidate;
                }
            }
            if (victim == nullptr) {
                result.errors.emplace_back(
                    "DeepSeek KV in-flight blocks exhaust the device cache");
                return result;
            }
            state_->metrics.device_used_bytes[device_slot] -=
                victim->devices[device_slot].device_bytes();
            victim->devices[device_slot] = {};
            ++state_->metrics.evictions;
        }
        CudaBuffer uploaded;
        const auto promotion_started = std::chrono::steady_clock::now();
        const auto uploaded_result = state_->cuda->upload_buffer(
            state_->config.devices[device_slot],
            std::as_bytes(std::span<const float>(block->host)), uploaded);
        if (!uploaded_result.ok()) {
            result.errors = uploaded_result.errors;
            return result;
        }
        block->devices[device_slot] = std::move(uploaded);
        state_->metrics.device_used_bytes[device_slot] += bytes;
        state_->metrics.device_peak_bytes[device_slot] = std::max(
            state_->metrics.device_peak_bytes[device_slot],
            state_->metrics.device_used_bytes[device_slot]);
        ++state_->metrics.promotions;
        state_->metrics.promotion_nanoseconds +=
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - promotion_started)
                    .count());
        state_->metrics.host_to_device_bytes += bytes;
    } else {
        ++state_->metrics.hits;
    }
    if (block->device_leases[device_slot] ==
            std::numeric_limits<std::uint32_t>::max() ||
        block->info.in_flight == std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back("DeepSeek KV in-flight count overflows");
        return result;
    }
    block->device_last_use[device_slot] = ++state_->clock;
    ++block->device_leases[device_slot];
    ++block->info.in_flight;
    result.value = Dsv4KvDeviceLease(state_, block->info.id, device_slot);
    return result;
}

ParseResult<std::vector<Dsv4KvBlockInfo>> Dsv4KvCache::block_table(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer) const {
    ParseResult<std::vector<Dsv4KvBlockInfo>> result;
    const auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    const auto table_found = target->tables.find(TableKey{kind, layer});
    if (table_found == target->tables.end()) return result;
    result.value.reserve(table_found->second.blocks.size());
    for (const auto id : table_found->second.blocks) {
        const auto found = state_->blocks.find(id);
        if (found == state_->blocks.end()) continue;
        auto info = found->second.info;
        info.device_resident.reserve(found->second.devices.size());
        for (const auto& device : found->second.devices) {
            info.device_resident.push_back(device.valid());
        }
        result.value.push_back(std::move(info));
    }
    return result;
}

Dsv4KvCacheStats Dsv4KvCache::stats() const noexcept {
    auto result = state_->metrics;
    result.used_blocks = state_->blocks.size();
    return result;
}

}  // namespace strata
