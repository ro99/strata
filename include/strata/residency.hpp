#pragma once

#include "strata/types.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace strata {

struct ResidencyConfig {
    std::uint64_t vram_capacity_bytes{};
    std::uint64_t ram_capacity_bytes{};
    std::uint64_t expert_bytes{};
    std::uint64_t cold_read_budget_bytes{};
    std::uint64_t peer_activation_roundtrip_bytes{};
    std::uint32_t peer_resident_basis_points{};
    std::uint64_t lease_ticks{16};
    ReplacementPolicy policy{ReplacementPolicy::Lease};
    bool strict_cold_read_budget{};
};

struct ResidencyStats {
    std::uint64_t accesses{};
    std::uint64_t vram_hits{};
    std::uint64_t ram_hits{};
    std::uint64_t peer_hits{};
    std::uint64_t nvme_misses{};
    std::uint64_t nvme_read_bytes{};
    std::uint64_t nvme_prefetch_bytes{};
    std::uint64_t weight_h2d_bytes{};
    std::uint64_t peer_activation_bytes{};
    std::uint64_t prefetches{};
    std::uint64_t useful_prefetches{};
    std::uint64_t wasted_prefetches{};
    std::uint64_t vram_evictions{};
    std::uint64_t ram_evictions{};
    std::uint64_t cold_budget_violations{};
};

struct AccessResult {
    Tier tier{Tier::Nvme};
    bool budget_exceeded{};
};

class ResidencyManager {
public:
    explicit ResidencyManager(ResidencyConfig config);

    [[nodiscard]] AccessResult access(ExpertKey key, std::uint64_t tick);
    bool prefetch(ExpertKey key, std::uint64_t tick, double confidence);
    void finalize();

    [[nodiscard]] bool resident(ExpertKey key) const;
    [[nodiscard]] const ResidencyStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ResidencyConfig& config() const noexcept { return config_; }

private:
    struct Entry {
        ExpertKey key;
        std::uint64_t bytes{};
        std::uint64_t last_touch{};
        std::uint64_t frequency{};
        std::uint64_t lease_until{};
        bool prefetched{};
        bool used_after_prefetch{};
        double forecast_confidence{};
    };

    class Cache {
    public:
        explicit Cache(std::uint64_t capacity) : capacity_(capacity) {}

        [[nodiscard]] Entry* find(ExpertKey key);
        [[nodiscard]] const Entry* find(ExpertKey key) const;
        [[nodiscard]] std::optional<Entry> remove(ExpertKey key);
        [[nodiscard]] std::vector<Entry> insert(Entry entry, ReplacementPolicy policy,
                                                std::uint64_t tick);
        [[nodiscard]] std::uint64_t capacity() const noexcept { return capacity_; }
        [[nodiscard]] std::uint64_t used() const noexcept { return used_; }
        [[nodiscard]] std::uint64_t unused_prefetches() const noexcept;

    private:
        [[nodiscard]] ExpertKey choose_victim(ReplacementPolicy policy,
                                              std::uint64_t tick) const;

        std::uint64_t capacity_{};
        std::uint64_t used_{};
        std::unordered_map<ExpertKey, Entry, ExpertKeyHash> entries_;
    };

    [[nodiscard]] bool peer_resident(ExpertKey key) const;
    void mark_useful(Entry& entry);
    void account_evictions(const std::vector<Entry>& entries, Tier tier);
    void insert_ram(Entry entry, std::uint64_t tick);
    void insert_vram(Entry entry, std::uint64_t tick);

    ResidencyConfig config_;
    Cache vram_;
    Cache ram_;
    ResidencyStats stats_;
    bool finalized_{};
};

}  // namespace strata
