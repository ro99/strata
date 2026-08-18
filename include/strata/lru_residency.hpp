#pragma once

// Recency tracking for a device-side weight cache.
//
// Three caches in this tree evict least-recently-used entries under a per
// device mutex: Dsv4WeightCache, GLM's WeightCache, and Laguna's ExpertCache.
// All three originally ranked every entry in the map to find one victim.
// DeepSeek measured what that costs -- "about 5,300 entries and decode evicts
// ~127 times a step, so the scan walked roughly 670,000 hash nodes a step,
// measured at 14.3 ms/step" -- and replaced it with an intrusive recency list.
// The other two were then fixed by writing that list out a second and third
// time, which is the argument for this header existing.
//
// It deliberately owns only the ordering, not the entries. The three caches
// key on different types, hold different payloads, and have different
// eviction eligibility rules (GLM has pinned entries, DeepSeek has prefetch
// leases). Trying to unify the cache itself would need every one of those
// differences as a template parameter; the part that is genuinely identical is
// this list and the two operations on it.

#include <list>
#include <utility>

namespace strata {

// One entry's position in a recency order. Store it beside the cached value.
template <typename Key>
struct LruPosition {
    typename std::list<Key>::iterator where{};
    bool linked{};
};

// Least-recently-used at the front. Callers hold their own lock; this type
// does no synchronization of its own, because the caches it serves already
// take a per-device mutex for reasons beyond recency.
template <typename Key>
class LruOrder {
public:
    // Moves a key to the most-recently-used end, inserting it if absent.
    void touch(const Key& key, LruPosition<Key>& position) {
        unlink(position);
        order_.push_back(key);
        position.where = std::prev(order_.end());
        position.linked = true;
    }

    void unlink(LruPosition<Key>& position) noexcept {
        if (!position.linked) return;
        order_.erase(position.where);
        position.linked = false;
    }

    // Least-recently-used first. Callers walk this and take the first entry
    // their own eligibility rule accepts -- an ineligible entry is skipped,
    // never unlinked, so releasing a lease does not make it newer than
    // entries used since.
    [[nodiscard]] auto begin() const noexcept { return order_.begin(); }
    [[nodiscard]] auto end() const noexcept { return order_.end(); }
    [[nodiscard]] bool empty() const noexcept { return order_.empty(); }
    void clear() noexcept { order_.clear(); }

private:
    std::list<Key> order_;
};

}  // namespace strata
