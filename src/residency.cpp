#include "strata/residency.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace strata {

ResidencyManager::Entry* ResidencyManager::Cache::find(ExpertKey key) {
    const auto found = entries_.find(key);
    return found == entries_.end() ? nullptr : &found->second;
}

const ResidencyManager::Entry* ResidencyManager::Cache::find(ExpertKey key) const {
    const auto found = entries_.find(key);
    return found == entries_.end() ? nullptr : &found->second;
}

std::optional<ResidencyManager::Entry> ResidencyManager::Cache::remove(ExpertKey key) {
    const auto found = entries_.find(key);
    if (found == entries_.end()) return std::nullopt;
    Entry entry = found->second;
    used_ -= entry.bytes;
    entries_.erase(found);
    return entry;
}

ExpertKey ResidencyManager::Cache::choose_victim(ReplacementPolicy policy,
                                                  std::uint64_t tick) const {
    auto selected = entries_.begin();
    for (auto current = std::next(entries_.begin()); current != entries_.end(); ++current) {
        const auto& candidate = current->second;
        const auto& incumbent = selected->second;
        bool choose = false;
        if (policy == ReplacementPolicy::Lru) {
            choose = std::tie(candidate.last_touch, candidate.key.layer, candidate.key.expert) <
                     std::tie(incumbent.last_touch, incumbent.key.layer, incumbent.key.expert);
        } else if (policy == ReplacementPolicy::Lfu) {
            choose = std::tie(candidate.frequency, candidate.last_touch,
                              candidate.key.layer, candidate.key.expert) <
                     std::tie(incumbent.frequency, incumbent.last_touch,
                              incumbent.key.layer, incumbent.key.expert);
        } else {
            const bool candidate_protected = candidate.lease_until > tick;
            const bool incumbent_protected = incumbent.lease_until > tick;
            if (candidate_protected != incumbent_protected) {
                choose = !candidate_protected;
            } else {
                const auto candidate_age = tick >= candidate.last_touch
                                               ? tick - candidate.last_touch + 1U
                                               : 1U;
                const auto incumbent_age = tick >= incumbent.last_touch
                                               ? tick - incumbent.last_touch + 1U
                                               : 1U;
                const double candidate_value =
                    candidate.forecast_confidence * 1024.0 +
                    static_cast<double>(candidate.frequency) /
                        static_cast<double>(candidate_age);
                const double incumbent_value =
                    incumbent.forecast_confidence * 1024.0 +
                    static_cast<double>(incumbent.frequency) /
                        static_cast<double>(incumbent_age);
                choose = candidate_value < incumbent_value ||
                         (candidate_value == incumbent_value &&
                          candidate.last_touch < incumbent.last_touch);
            }
        }
        if (choose) selected = current;
    }
    return selected->first;
}

std::vector<ResidencyManager::Entry> ResidencyManager::Cache::insert(
    Entry entry, ReplacementPolicy policy, std::uint64_t tick) {
    std::vector<Entry> evicted;
    if (entry.bytes == 0 || entry.bytes > capacity_) return evicted;
    if (const auto existing = remove(entry.key)) {
        entry.frequency = std::max(entry.frequency, existing->frequency);
        entry.prefetched = entry.prefetched || existing->prefetched;
        entry.used_after_prefetch = entry.used_after_prefetch || existing->used_after_prefetch;
    }
    while (used_ + entry.bytes > capacity_ && !entries_.empty()) {
        const auto victim = choose_victim(policy, tick);
        auto removed = remove(victim);
        if (removed) evicted.push_back(*removed);
    }
    if (used_ + entry.bytes <= capacity_) {
        used_ += entry.bytes;
        entries_.emplace(entry.key, std::move(entry));
    }
    return evicted;
}

std::uint64_t ResidencyManager::Cache::unused_prefetches() const noexcept {
    std::uint64_t count = 0;
    for (const auto& [key, entry] : entries_) {
        (void)key;
        if (entry.prefetched && !entry.used_after_prefetch) ++count;
    }
    return count;
}

ResidencyManager::ResidencyManager(ResidencyConfig config)
    : config_(config), vram_(config.vram_capacity_bytes), ram_(config.ram_capacity_bytes) {
    if (config_.expert_bytes == 0) throw std::invalid_argument("expert_bytes must be positive");
    if (config_.peer_resident_basis_points > 10000U) {
        throw std::invalid_argument("peer_resident_basis_points exceeds 10000");
    }
}

bool ResidencyManager::peer_resident(ExpertKey key) const {
    if (config_.peer_resident_basis_points == 0) return false;
    return ExpertKeyHash{}(key) % 10000U < config_.peer_resident_basis_points;
}

void ResidencyManager::mark_useful(Entry& entry) {
    if (entry.prefetched && !entry.used_after_prefetch) {
        entry.used_after_prefetch = true;
        ++stats_.useful_prefetches;
    }
}

void ResidencyManager::account_evictions(const std::vector<Entry>& entries, Tier tier) {
    if (tier == Tier::Vram) stats_.vram_evictions += entries.size();
    if (tier == Tier::Ram) stats_.ram_evictions += entries.size();
    if (tier == Tier::Ram) {
        for (const auto& entry : entries) {
            if (entry.prefetched && !entry.used_after_prefetch) ++stats_.wasted_prefetches;
        }
    }
}

void ResidencyManager::insert_ram(Entry entry, std::uint64_t tick) {
    if (ram_.capacity() < entry.bytes) {
        if (entry.prefetched && !entry.used_after_prefetch) ++stats_.wasted_prefetches;
        return;
    }
    const auto evicted = ram_.insert(std::move(entry), config_.policy, tick);
    account_evictions(evicted, Tier::Ram);
}

void ResidencyManager::insert_vram(Entry entry, std::uint64_t tick) {
    if (vram_.capacity() < entry.bytes) {
        insert_ram(std::move(entry), tick);
        return;
    }
    auto evicted = vram_.insert(std::move(entry), config_.policy, tick);
    account_evictions(evicted, Tier::Vram);
    for (auto& victim : evicted) insert_ram(std::move(victim), tick);
}

AccessResult ResidencyManager::access(ExpertKey key, std::uint64_t tick) {
    ++stats_.accesses;
    if (auto* entry = vram_.find(key)) {
        ++stats_.vram_hits;
        mark_useful(*entry);
        entry->last_touch = tick;
        ++entry->frequency;
        return AccessResult{Tier::Vram, false};
    }
    if (auto entry = ram_.remove(key)) {
        ++stats_.ram_hits;
        mark_useful(*entry);
        entry->last_touch = tick;
        ++entry->frequency;
        if (vram_.capacity() >= entry->bytes) {
            stats_.weight_h2d_bytes += entry->bytes;
            insert_vram(std::move(*entry), tick);
        } else {
            insert_ram(std::move(*entry), tick);
        }
        return AccessResult{Tier::Ram, false};
    }
    if (peer_resident(key)) {
        ++stats_.peer_hits;
        stats_.peer_activation_bytes += config_.peer_activation_roundtrip_bytes;
        return AccessResult{Tier::Peer, false};
    }

    const bool over_budget = config_.cold_read_budget_bytes > 0 &&
        stats_.nvme_read_bytes + config_.expert_bytes > config_.cold_read_budget_bytes;
    if (over_budget) {
        ++stats_.cold_budget_violations;
        if (config_.strict_cold_read_budget) return AccessResult{Tier::Nvme, true};
    }

    ++stats_.nvme_misses;
    stats_.nvme_read_bytes += config_.expert_bytes;
    Entry entry{key, config_.expert_bytes, tick, 1U, tick, false, false, 0.0};
    if (vram_.capacity() >= entry.bytes) {
        stats_.weight_h2d_bytes += entry.bytes;
        insert_vram(std::move(entry), tick);
    } else {
        insert_ram(std::move(entry), tick);
    }
    return AccessResult{Tier::Nvme, over_budget};
}

bool ResidencyManager::prefetch(ExpertKey key, std::uint64_t tick, double confidence) {
    if (resident(key) || peer_resident(key) || ram_.capacity() < config_.expert_bytes) {
        return false;
    }
    const bool over_budget = config_.cold_read_budget_bytes > 0 &&
        stats_.nvme_read_bytes + config_.expert_bytes > config_.cold_read_budget_bytes;
    if (over_budget) {
        ++stats_.cold_budget_violations;
        if (config_.strict_cold_read_budget) return false;
    }
    stats_.nvme_read_bytes += config_.expert_bytes;
    stats_.nvme_prefetch_bytes += config_.expert_bytes;
    ++stats_.prefetches;
    Entry entry{key,
                config_.expert_bytes,
                tick,
                0U,
                tick + config_.lease_ticks,
                true,
                false,
                std::clamp(confidence, 0.0, 1.0)};
    insert_ram(std::move(entry), tick);
    return true;
}

bool ResidencyManager::resident(ExpertKey key) const {
    return vram_.find(key) != nullptr || ram_.find(key) != nullptr;
}

void ResidencyManager::finalize() {
    if (finalized_) return;
    stats_.wasted_prefetches += vram_.unused_prefetches();
    stats_.wasted_prefetches += ram_.unused_prefetches();
    finalized_ = true;
}

}  // namespace strata
