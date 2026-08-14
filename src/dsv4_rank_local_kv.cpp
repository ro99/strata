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

    // Reserve the logical row once on rank 0's slot, then lease the same block
    // on every other rank's device. Reserving per rank would advance the
    // table's end row once per rank and reject the second call as
    // non-contiguous: reservation belongs to the sequence, replication belongs
    // to the devices.
    const auto reserve = [&](Dsv4RankLocalRow& row, Dsv4KvBlockKind kind,
                             std::uint32_t row_ratio,
                             std::uint64_t logical_row) {
        auto reserved = cache_.reserve_physical_append(
            sequence_, kind, layer, row_ratio, logical_row, device_slots_[0]);
        if (!reserved.ok()) {
            result.errors = std::move(reserved.errors);
            return false;
        }
        row.append.emplace(std::move(reserved.value));
        for (std::size_t rank = 1U; rank < kDsv4RankLocalWorld; ++rank) {
            auto lease = cache_.acquire_device(
                sequence_, kind, layer, logical_row, device_slots_[rank]);
            if (!lease.ok()) {
                result.errors = std::move(lease.errors);
                return false;
            }
            row.peers[rank] = std::move(lease.value);
        }
        row.present = true;
        return true;
    };

    // All-or-nothing: a partial reservation leaves nothing behind, because the
    // entry is only published to layers_ once every row is in place.
    if (!reserve(entry.sliding, Dsv4KvBlockKind::Sliding, 1U, position)) {
        return result;
    }
    if (ratio != 0U && (position + 1U) % ratio == 0U) {
        if (!reserve(entry.compressed, compressed_kind, ratio,
                     position / ratio)) {
            return result;
        }
    }
    if (index_ratio != 0U && (position + 1U) % index_ratio == 0U) {
        if (!reserve(entry.index, Dsv4KvBlockKind::LearnedIndex, index_ratio,
                     position / index_ratio)) {
            return result;
        }
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
    // Offsets and extents come from the single reservation; only the target
    // buffer is rank-specific, because each rank owns its own device copy of
    // the same block.
    const auto add = [&output, rank](const Dsv4RankLocalRow& row) {
        if (!row.present) return;
        const auto& append = *row.append;
        output.push_back(CudaDsv4AttentionPageWrite{
            rank == 0U ? append.buffer() : row.peers[rank].buffer(),
            append.data_offset(), append.scale_offset(), append.data_bytes(),
            append.scale_bytes()});
    };
    if (!entry->sliding.present) {
        result.errors.emplace_back(
            "rank-local KV sliding reservation is missing for layer " +
            std::to_string(layer));
        return result;
    }
    add(entry->sliding);
    add(entry->compressed);
    add(entry->index);
    return result;
}

std::uint64_t Dsv4RankLocalKvTransaction::patch_bytes(
    std::uint32_t layer) const noexcept {
    const auto* entry = find(layer);
    if (entry == nullptr) return 0U;
    std::uint64_t bytes = 0U;
    if (entry->sliding.present) bytes += entry->sliding.append->patch_bytes();
    if (entry->compressed.present) {
        bytes += entry->compressed.append->patch_bytes();
    }
    if (entry->index.present) bytes += entry->index.append->patch_bytes();
    return bytes;
}

ValidationResult Dsv4RankLocalKvTransaction::commit_layer(
    std::uint32_t layer, std::span<const float> sliding_values,
    std::span<const float> compressed_values,
    std::array<std::span<std::byte>, kDsv4RankLocalWorld> patches,
    std::span<const float> index_values) {
    ValidationResult result;
    if (committed_ || aborted_) {
        result.errors.emplace_back(
            "rank-local KV transaction is closed and cannot commit a layer");
        return result;
    }
    auto* entry = find(layer);
    if (entry == nullptr) {
        result.errors.emplace_back(
            "rank-local KV layer " + std::to_string(layer) +
            " has no reservation");
        return result;
    }
    if (entry->committed) {
        result.errors.emplace_back(
            "rank-local KV layer " + std::to_string(layer) +
            " is already committed");
        return result;
    }
    if (!entry->sliding.present) {
        result.errors.emplace_back(
            "rank-local KV sliding reservation is missing for layer " +
            std::to_string(layer));
        return result;
    }
    const auto expected = static_cast<std::size_t>(patch_bytes(layer));
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        if (patches[rank].size() == expected) continue;
        result.errors.emplace_back(
            "rank-local KV patch staging for layer " + std::to_string(layer) +
            " rank " + std::to_string(rank) + " is " +
            std::to_string(patches[rank].size()) + " bytes, not " +
            std::to_string(expected));
        return result;
    }

    std::size_t cursor = 0U;
    const auto consume = [&](Dsv4RankLocalRow& row,
                             std::span<const float> values) {
        if (!row.present) return true;
        auto& append = *row.append;
        const auto bytes = static_cast<std::size_t>(append.patch_bytes());
        auto canonical = patches[0].subspan(cursor, bytes);
        auto committed = append.commit(values, canonical);
        if (!committed.ok()) {
            result.errors.insert(result.errors.end(),
                                 committed.errors.begin(),
                                 committed.errors.end());
            return false;
        }
        // One encode, copied out. Encoding per rank could only ever agree, and
        // if it ever disagreed the replicas would silently diverge.
        for (std::size_t rank = 1U; rank < kDsv4RankLocalWorld; ++rank) {
            std::copy(canonical.begin(), canonical.end(),
                      patches[rank].begin() +
                          static_cast<std::ptrdiff_t>(cursor));
        }
        cursor += bytes;
        return true;
    };

    if (!consume(entry->sliding, sliding_values)) return result;
    if (!consume(entry->compressed, compressed_values)) return result;
    if (!consume(entry->index, index_values)) return result;
    if (cursor != expected) {
        result.errors.emplace_back(
            "rank-local KV patch staging has " +
            std::to_string(expected - cursor) +
            " unconsumed bytes for layer " + std::to_string(layer));
        return result;
    }
    entry->committed = true;
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
    // Publish only when every reserved layer has been committed. A partially
    // committed token is aborted, never published.
    for (const auto& entry : layers_) {
        if (entry.committed) continue;
        result.errors.emplace_back(
            "rank-local KV layer " + std::to_string(entry.layer) +
            " is uncommitted");
    }
    if (!result.ok()) return result;

    for (auto& entry : layers_) {
        const auto account = [&](Dsv4RankLocalRow& row) {
            if (!row.present) return;
            auto status = row.append->account();
            if (!status.ok()) {
                for (auto& error : status.errors) {
                    result.errors.push_back(
                        "rank-local KV account layer " +
                        std::to_string(entry.layer) + ": " + error);
                }
            }
        };
        account(entry.sliding);
        account(entry.compressed);
        account(entry.index);
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
