#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/dsv4_rank_local_topology.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace strata {

// One layer's replicated physical reservations for a single token position.
// Both ranks reserve the same logical row on their own device slot; the two
// device copies are patched independently in each rank's stream order.
struct Dsv4RankLocalLayerAppend {
    std::array<std::optional<Dsv4KvPhysicalAppend>, kDsv4RankLocalWorld> sliding;
    std::array<std::optional<Dsv4KvPhysicalAppend>, kDsv4RankLocalWorld>
        compressed;
    // Learned-index row, present only in the sparse-indexer regime (above
    // kDsv4RankLocalSparseIndexerThreshold active tokens). Reserved here so the
    // index row is patched in the same stream order as the other two rather
    // than by a synchronous host append outside the queued chain.
    std::array<std::optional<Dsv4KvPhysicalAppend>, kDsv4RankLocalWorld> index;
    std::uint32_t layer{};
    std::uint32_t position{};
    bool has_compressed{};
    bool has_index{};
    std::array<bool, kDsv4RankLocalWorld> committed{};
};

// Result of comparing the two ranks' independently computed selections.
struct Dsv4RankLocalSelectionAgreement {
    bool agree{};
    std::size_t first_mismatch{};
    std::uint32_t left{};
    std::uint32_t right{};
};

// Both ranks run candidate selection independently over replicated state and
// must produce identical results. Disagreement means the replicas diverged and
// the token must be aborted, not reconciled.
[[nodiscard]] Dsv4RankLocalSelectionAgreement dsv4_rank_local_selection_agreement(
    std::span<const std::uint32_t> left,
    std::span<const std::uint32_t> right) noexcept;

// A decode token as a two-rank transaction over the physical KV cache.
//
// Reservation advances only host block-table metadata, so nothing is visible
// until commit. Each rank commits its own device patch; host-visible KV
// metadata is published only once every rank has committed every layer and
// the caller has confirmed the status collectives succeeded.
//
// Any failure aborts: every token-local KV, compressor and mHC mutation is
// truncated back to the committed token count, and the transaction refuses to
// publish. The destructor aborts an uncommitted transaction, so an early
// return can never leave a half-appended position visible.
class Dsv4RankLocalKvTransaction {
public:
    Dsv4RankLocalKvTransaction(
        Dsv4KvCache& cache, Dsv4SequenceHandle sequence,
        std::array<std::size_t, kDsv4RankLocalWorld> device_slots,
        std::uint64_t committed_tokens) noexcept;
    ~Dsv4RankLocalKvTransaction();

    Dsv4RankLocalKvTransaction(const Dsv4RankLocalKvTransaction&) = delete;
    Dsv4RankLocalKvTransaction& operator=(const Dsv4RankLocalKvTransaction&) =
        delete;
    Dsv4RankLocalKvTransaction(Dsv4RankLocalKvTransaction&&) = delete;
    Dsv4RankLocalKvTransaction& operator=(Dsv4RankLocalKvTransaction&&) = delete;

    // Reserves one logical position on both device slots for this layer. The
    // compressed row is reserved only on the compressor's boundary, matching
    // the centralized contract: ratio != 0 && (position + 1) % ratio == 0.
    // Reservation is all-or-nothing across both ranks.
    // `index_ratio` of zero leaves the learned-index row unreserved, which is
    // the regime below the sparse-indexer threshold.
    [[nodiscard]] ValidationResult reserve_layer(
        std::uint32_t layer, std::uint32_t position, std::uint32_t ratio,
        Dsv4KvBlockKind compressed_kind, std::uint32_t index_ratio = 0U);

    // Page writes for one rank's queued patch commands, in reservation order:
    // sliding first, then the compressed row when present. Matches the order
    // commit_layer consumes patch bytes in.
    [[nodiscard]] ValidationResult page_writes(
        std::uint32_t layer, std::size_t rank,
        std::vector<CudaDsv4AttentionPageWrite>& output) const;

    [[nodiscard]] std::uint64_t patch_bytes(
        std::uint32_t layer, std::size_t rank) const noexcept;

    // Commits one rank's rows for one layer into the canonical host page and
    // that rank's pinned patch staging. Both ranks write the same canonical
    // bytes; the patch spans are per-rank and disjoint.
    [[nodiscard]] ValidationResult commit_layer(
        std::uint32_t layer, std::size_t rank,
        std::span<const float> sliding_values,
        std::span<const float> compressed_values,
        std::span<std::byte> patch,
        std::span<const float> index_values = {});

    // Publishes the token. Legal only when every reserved layer has been
    // committed on every rank; the caller must have already confirmed both
    // ranks' status collectives succeeded.
    [[nodiscard]] ValidationResult commit();

    // Truncates every token-local mutation back to the committed token count.
    // Idempotent, and safe to call after a partial reserve or partial commit.
    void abort() noexcept;

    [[nodiscard]] bool committed() const noexcept { return committed_; }
    [[nodiscard]] std::size_t reserved_layers() const noexcept {
        return layers_.size();
    }

private:
    [[nodiscard]] const Dsv4RankLocalLayerAppend* find(
        std::uint32_t layer) const noexcept;
    [[nodiscard]] Dsv4RankLocalLayerAppend* find(std::uint32_t layer) noexcept;

    Dsv4KvCache& cache_;
    Dsv4SequenceHandle sequence_{};
    std::array<std::size_t, kDsv4RankLocalWorld> device_slots_{};
    std::uint64_t committed_tokens_{};
    std::vector<Dsv4RankLocalLayerAppend> layers_;
    bool committed_{};
    bool aborted_{};
};

}  // namespace strata
