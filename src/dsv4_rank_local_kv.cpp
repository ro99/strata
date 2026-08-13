#include "strata/dsv4_rank_local_kv.hpp"

#include <algorithm>
#include <string>

namespace strata {

Dsv4RankLocalSelectionAgreement dsv4_rank_local_selection_agreement(
    std::span<const std::uint32_t> left,
    std::span<const std::uint32_t> right) noexcept {
    Dsv4RankLocalSelectionAgreement result;
    if (left.size() != right.size()) {
        result.first_mismatch = std::min(left.size(), right.size());
        result.left = static_cast<std::uint32_t>(left.size());
        result.right = static_cast<std::uint32_t>(right.size());
        return result;
    }
    for (std::size_t index = 0U; index < left.size(); ++index) {
        if (left[index] != right[index]) {
            result.first_mismatch = index;
            result.left = left[index];
            result.right = right[index];
            return result;
        }
    }
    result.agree = true;
    result.first_mismatch = left.size();
    return result;
}

Dsv4RankLocalKvTransaction::Dsv4RankLocalKvTransaction(
    Dsv4KvCache& cache, Dsv4SequenceHandle sequence,
    std::array<std::size_t, kDsv4RankLocalWorld> device_slots,
    std::uint64_t committed_tokens) noexcept
    : cache_(cache), sequence_(sequence), device_slots_(device_slots),
      committed_tokens_(committed_tokens) {}

Dsv4RankLocalKvTransaction::~Dsv4RankLocalKvTransaction() {
    // An uncommitted transaction must never survive its scope: an early
    // return would otherwise leave a half-appended position visible.
    if (!committed_) abort();
}

const Dsv4RankLocalLayerAppend* Dsv4RankLocalKvTransaction::find(
    std::uint32_t layer) const noexcept {
    const auto found = std::find_if(
        layers_.begin(), layers_.end(),
        [layer](const Dsv4RankLocalLayerAppend& entry) {
            return entry.layer == layer;
        });
    return found == layers_.end() ? nullptr : &*found;
}

Dsv4RankLocalLayerAppend* Dsv4RankLocalKvTransaction::find(
    std::uint32_t layer) noexcept {
    return const_cast<Dsv4RankLocalLayerAppend*>(
        static_cast<const Dsv4RankLocalKvTransaction*>(this)->find(layer));
}

ValidationResult Dsv4RankLocalKvTransaction::reserve_layer(
    std::uint32_t layer, std::uint32_t position, std::uint32_t ratio,
    Dsv4KvBlockKind compressed_kind, std::uint32_t index_ratio) {
    ValidationResult result;
    if (committed_ || aborted_) {
        result.errors.emplace_back(
            "rank-local KV transaction is closed and cannot reserve");
        return result;
    }
    if (find(layer) != nullptr) {
        result.errors.emplace_back(
            "rank-local KV layer " + std::to_string(layer) +
            " is already reserved for this token");
        return result;
    }
    if (device_slots_[0] == device_slots_[1]) {
        result.errors.emplace_back(
            "rank-local KV replication requires two distinct device slots");
        return result;
    }

    Dsv4RankLocalLayerAppend entry;
    entry.layer = layer;
    entry.position = position;
    entry.has_compressed = ratio != 0U && (position + 1U) % ratio == 0U;
    entry.has_index =
        index_ratio != 0U && (position + 1U) % index_ratio == 0U;

    // All-or-nothing across both ranks: a reservation that succeeds on one
    // rank and fails on the other must leave neither in place.
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        auto sliding = cache_.reserve_physical_append(
            sequence_, Dsv4KvBlockKind::Sliding, layer, 1U, position,
            device_slots_[rank]);
        if (!sliding.ok()) {
            result.errors = std::move(sliding.errors);
            return result;
        }
        entry.sliding[rank].emplace(std::move(sliding.value));
        if (!entry.has_compressed) continue;
        auto compressed = cache_.reserve_physical_append(
            sequence_, compressed_kind, layer, ratio, position / ratio,
            device_slots_[rank]);
        if (!compressed.ok()) {
            result.errors = std::move(compressed.errors);
            return result;
        }
        entry.compressed[rank].emplace(std::move(compressed.value));
    }
    for (std::size_t rank = 0U; entry.has_index && rank < kDsv4RankLocalWorld;
         ++rank) {
        auto index = cache_.reserve_physical_append(
            sequence_, Dsv4KvBlockKind::LearnedIndex, layer, index_ratio,
            position / index_ratio, device_slots_[rank]);
        if (!index.ok()) {
            result.errors = std::move(index.errors);
            return result;
        }
        entry.index[rank].emplace(std::move(index.value));
    }

    layers_.push_back(std::move(entry));
    return result;
}

ValidationResult Dsv4RankLocalKvTransaction::page_writes(
    std::uint32_t layer, std::size_t rank,
    std::vector<CudaDsv4AttentionPageWrite>& output) const {
    ValidationResult result;
    output.clear();
    const auto* entry = find(layer);
    if (entry == nullptr || rank >= kDsv4RankLocalWorld) {
        result.errors.emplace_back(
            "rank-local KV layer " + std::to_string(layer) +
            " has no reservation for rank " + std::to_string(rank));
        return result;
    }
    const auto add = [&output](const Dsv4KvPhysicalAppend& append) {
        output.push_back(CudaDsv4AttentionPageWrite{
            append.buffer(), append.data_offset(), append.scale_offset(),
            append.data_bytes(), append.scale_bytes()});
    };
    if (!entry->sliding[rank].has_value()) {
        result.errors.emplace_back(
            "rank-local KV sliding reservation is missing for rank " +
            std::to_string(rank));
        return result;
    }
    add(*entry->sliding[rank]);
    if (entry->has_compressed && entry->compressed[rank].has_value()) {
        add(*entry->compressed[rank]);
    }
    if (entry->has_index && entry->index[rank].has_value()) {
        add(*entry->index[rank]);
    }
    return result;
}

std::uint64_t Dsv4RankLocalKvTransaction::patch_bytes(
    std::uint32_t layer, std::size_t rank) const noexcept {
    const auto* entry = find(layer);
    if (entry == nullptr || rank >= kDsv4RankLocalWorld) return 0U;
    std::uint64_t bytes = 0U;
    if (entry->sliding[rank].has_value()) {
        bytes += entry->sliding[rank]->patch_bytes();
    }
    if (entry->has_compressed && entry->compressed[rank].has_value()) {
        bytes += entry->compressed[rank]->patch_bytes();
    }
    if (entry->has_index && entry->index[rank].has_value()) {
        bytes += entry->index[rank]->patch_bytes();
    }
    return bytes;
}

ValidationResult Dsv4RankLocalKvTransaction::commit_layer(
    std::uint32_t layer, std::size_t rank,
    std::span<const float> sliding_values,
    std::span<const float> compressed_values, std::span<std::byte> patch,
    std::span<const float> index_values) {
    ValidationResult result;
    if (committed_ || aborted_) {
        result.errors.emplace_back(
            "rank-local KV transaction is closed and cannot commit a layer");
        return result;
    }
    auto* entry = find(layer);
    if (entry == nullptr || rank >= kDsv4RankLocalWorld) {
        result.errors.emplace_back(
            "rank-local KV layer " + std::to_string(layer) +
            " has no reservation for rank " + std::to_string(rank));
        return result;
    }
    if (entry->committed[rank]) {
        result.errors.emplace_back(
            "rank-local KV layer " + std::to_string(layer) +
            " is already committed on rank " + std::to_string(rank));
        return result;
    }
    if (!entry->sliding[rank].has_value()) {
        result.errors.emplace_back(
            "rank-local KV sliding reservation is missing for rank " +
            std::to_string(rank));
        return result;
    }

    std::size_t cursor = 0U;
    const auto consume = [&](Dsv4KvPhysicalAppend& append,
                             std::span<const float> values) {
        const auto bytes = static_cast<std::size_t>(append.patch_bytes());
        if (bytes > patch.size() - cursor) {
            result.errors.emplace_back(
                "rank-local KV patch staging is truncated for layer " +
                std::to_string(layer) + " rank " + std::to_string(rank));
            return false;
        }
        auto committed = append.commit(values, patch.subspan(cursor, bytes));
        cursor += bytes;
        if (!committed.ok()) {
            result.errors.insert(result.errors.end(),
                                 committed.errors.begin(),
                                 committed.errors.end());
            return false;
        }
        return true;
    };

    if (!consume(*entry->sliding[rank], sliding_values)) return result;
    if (entry->has_compressed) {
        if (!entry->compressed[rank].has_value()) {
            result.errors.emplace_back(
                "rank-local KV compressed reservation is missing for rank " +
                std::to_string(rank));
            return result;
        }
        if (!consume(*entry->compressed[rank], compressed_values)) {
            return result;
        }
    }
    if (entry->has_index) {
        if (!entry->index[rank].has_value()) {
            result.errors.emplace_back(
                "rank-local KV learned-index reservation is missing for rank " +
                std::to_string(rank));
            return result;
        }
        if (!consume(*entry->index[rank], index_values)) return result;
    }
    if (cursor != patch.size()) {
        result.errors.emplace_back(
            "rank-local KV patch staging has " +
            std::to_string(patch.size() - cursor) +
            " unconsumed bytes for layer " + std::to_string(layer) +
            " rank " + std::to_string(rank));
        return result;
    }
    entry->committed[rank] = true;
    return result;
}

ValidationResult Dsv4RankLocalKvTransaction::commit() {
    ValidationResult result;
    if (aborted_) {
        result.errors.emplace_back(
            "rank-local KV transaction was aborted and cannot commit");
        return result;
    }
    if (committed_) {
        result.errors.emplace_back(
            "rank-local KV transaction is already committed");
        return result;
    }
    if (layers_.empty()) {
        result.errors.emplace_back(
            "rank-local KV transaction has no reserved layer to commit");
        return result;
    }
    // Publish only when every rank has committed every reserved layer. A
    // partially committed token is aborted, never published.
    for (const auto& entry : layers_) {
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            if (entry.committed[rank]) continue;
            result.errors.emplace_back(
                "rank-local KV layer " + std::to_string(entry.layer) +
                " is uncommitted on rank " + std::to_string(rank));
        }
    }
    if (!result.ok()) return result;

    for (auto& entry : layers_) {
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            const auto account = [&](std::optional<Dsv4KvPhysicalAppend>& slot) {
                if (!slot.has_value()) return;
                auto status = slot->account();
                if (!status.ok()) {
                    result.errors.insert(result.errors.end(),
                                         status.errors.begin(),
                                         status.errors.end());
                }
            };
            account(entry.sliding[rank]);
            if (entry.has_compressed) account(entry.compressed[rank]);
            if (entry.has_index) account(entry.index[rank]);
        }
    }
    if (!result.ok()) return result;

    committed_ = true;
    layers_.clear();
    return result;
}

void Dsv4RankLocalKvTransaction::abort() noexcept {
    if (committed_ || aborted_) return;
    aborted_ = true;
    // Drop every reservation first so no lease outlives the truncation, then
    // roll the sequence back to the last published token count. Truncation
    // covers KV, compressor and learned-index rows for this position.
    layers_.clear();
    auto truncated = cache_.truncate_sequence(sequence_, committed_tokens_);
    (void)truncated;
}

}  // namespace strata
