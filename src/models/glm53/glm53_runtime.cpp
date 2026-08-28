#include "strata/models/glm53/glm53_runtime.hpp"

#include "strata/engine/runtime_support.hpp"
#include "strata/models/common/tokenizer.hpp"
#include "strata/models/deepseek/deepseek_ops.hpp"
#include "strata/models/glm53/glm53_checkpoint.hpp"
#include "strata/models/kimi_k3/kimi_k3_ops.hpp"
#include "strata/platform/hardware_profile.hpp"
#include "strata/platform/numerics.hpp"
#include "strata/platform/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <list>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>

namespace strata {

std::vector<std::size_t> glm53_projection_slots(
    std::span<const std::string_view> keys,
    std::span<const std::uint64_t> costs,
    std::span<const std::uint64_t> capacities,
    std::size_t preferred_slot) {
    if (keys.empty() || keys.size() != costs.size() || capacities.empty() ||
        preferred_slot >= capacities.size() ||
        std::any_of(capacities.begin(), capacities.end(),
                    [](std::uint64_t value) { return value == 0U; })) {
        return {};
    }
    std::vector<std::size_t> order(keys.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                      std::size_t right) {
        if (costs[left] != costs[right]) return costs[left] > costs[right];
        if (keys[left] != keys[right]) return keys[left] < keys[right];
        return left < right;
    });
    std::vector<long double> loads(capacities.size(), 0.0L);
    std::vector<std::size_t> slots(keys.size());
    for (const auto index : order) {
        std::size_t best = 0U;
        for (std::size_t slot = 1U; slot < capacities.size(); ++slot) {
            const auto candidate = loads[slot] /
                static_cast<long double>(capacities[slot]);
            const auto incumbent = loads[best] /
                static_cast<long double>(capacities[best]);
            const auto candidate_distance =
                (slot + capacities.size() - preferred_slot) % capacities.size();
            const auto incumbent_distance =
                (best + capacities.size() - preferred_slot) % capacities.size();
            if (candidate < incumbent ||
                (candidate == incumbent &&
                 candidate_distance < incumbent_distance)) {
                best = slot;
            }
        }
        slots[index] = best;
        loads[best] += static_cast<long double>(std::max<std::uint64_t>(
            costs[index], 1U));
    }
    return slots;
}

namespace {

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kLayers = 45U;
constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kLinearHead = 128U;
constexpr std::uint32_t kLinearWidth = kHeads * kLinearHead;
constexpr std::uint32_t kMlaHead = 256U;
constexpr std::uint32_t kMlaWidth = kHeads * kMlaHead;
constexpr std::uint32_t kQueryRank = 1536U;
constexpr std::uint32_t kKvRank = 512U;
constexpr std::uint32_t kMhc = 4U;
constexpr std::uint32_t kVocabulary = 154880U;
constexpr std::uint32_t kExactSparseContext = 2048U;
constexpr std::uint64_t kDeviceWorkspaceReserve = 2ULL << 30U;
constexpr std::uint64_t kMinimumDeviceBudget = 2ULL << 30U;

[[nodiscard]] bool batched_projections_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_BATCHED_PROJECTIONS");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

[[nodiscard]] bool cross_gpu_projections_enabled(
    std::span<const int> devices) noexcept {
    static const int policy = [] {
        const char* value = std::getenv("STRATA_GLM53_CROSS_GPU_PROJECTIONS");
        if (value == nullptr) return -1;
        return std::string_view(value) != "0" &&
                       std::string_view(value) != "false" &&
                       std::string_view(value) != "off"
                   ? 1
                   : 0;
    }();
    if (policy >= 0) return policy != 0;
    for (std::size_t source = 0U; source < devices.size(); ++source) {
        for (std::size_t destination = source + 1U;
             destination < devices.size(); ++destination) {
            if (!CudaBackend::high_speed_peer_access_supported(
                    devices[source], devices[destination]) ||
                !CudaBackend::high_speed_peer_access_supported(
                    devices[destination], devices[source])) {
                return false;
            }
        }
    }
    return devices.size() > 1U;
}

[[nodiscard]] std::vector<int> projection_worker_cpus(
    std::span<const int> devices) {
    const auto& hardware = host_hardware_profile();
    std::vector<int> chosen;
    chosen.reserve(devices.size());
    const auto usable = [&](int cpu) {
        return std::find(hardware.usable_cpu_ids.begin(),
                         hardware.usable_cpu_ids.end(), cpu) !=
               hardware.usable_cpu_ids.end();
    };
    const auto available = [&](int cpu) {
        return usable(cpu) &&
               std::find(chosen.begin(), chosen.end(), cpu) == chosen.end();
    };
    for (const int device : devices) {
        const int node = CudaBackend::device_numa_node(device);
        const std::vector<int>* local = nullptr;
        if (node >= 0 && static_cast<std::size_t>(node) <
                             hardware.numa.node_primary_cpus.size() &&
            !hardware.numa.node_primary_cpus[static_cast<std::size_t>(node)]
                 .empty()) {
            local = &hardware.numa.node_primary_cpus[
                static_cast<std::size_t>(node)];
        } else if (node >= 0 && static_cast<std::size_t>(node) <
                                    hardware.numa.node_cpus.size()) {
            local = &hardware.numa.node_cpus[static_cast<std::size_t>(node)];
        }
        auto selected = hardware.usable_cpu_ids.end();
        if (local != nullptr) {
            const auto candidate = std::find_if(
                local->begin(), local->end(), available);
            if (candidate != local->end()) {
                selected = std::find(hardware.usable_cpu_ids.begin(),
                                     hardware.usable_cpu_ids.end(), *candidate);
            }
        }
        if (selected == hardware.usable_cpu_ids.end()) {
            selected = std::find_if(hardware.usable_cpu_ids.begin(),
                                    hardware.usable_cpu_ids.end(), available);
        }
        if (selected == hardware.usable_cpu_ids.end()) return {};
        chosen.push_back(*selected);
    }
    return chosen;
}

double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

float sigmoid(float value) noexcept {
    return value >= 0.0F ? 1.0F / (1.0F + std::exp(-value))
                         : std::exp(value) / (1.0F + std::exp(value));
}

void round_bf16(std::span<float> values) noexcept {
    for (auto& value : values) value = bf16_round_f32(value);
}

void append(std::vector<std::string>& destination,
            std::vector<std::string> source) {
    for (auto& error : source) destination.push_back(std::move(error));
}

[[nodiscard]] std::string projection_group_key(
    std::string_view base, Glm53TensorRole role) {
    const auto separator = base.find_last_of('.');
    const auto prefix = base.substr(0U, separator + 1U);
    const auto leaf = base.substr(separator + 1U);
    if (role == Glm53TensorRole::KdaAttention) {
        if (leaf == "q_proj" || leaf == "k_proj" || leaf == "v_proj" ||
            leaf == "f_a_proj" || leaf == "b_proj" || leaf == "g_a_proj") {
            return std::string(prefix) + "#kda-input";
        }
        if (leaf == "f_b_proj" || leaf == "g_b_proj") {
            return std::string(prefix) + "#kda-low-rank";
        }
    } else if (role == Glm53TensorRole::SparseAttention) {
        if (leaf == "q_a_proj" || leaf == "kv_a_proj_with_mqa") {
            return std::string(prefix) + "#mla-input";
        }
        if (leaf == "q_b_proj" || leaf == "kv_b_proj") {
            return std::string(prefix) + "#mla-expanded";
        }
    } else if (role == Glm53TensorRole::DenseMlp &&
               (leaf == "gate_proj" || leaf == "up_proj")) {
        return std::string(prefix) + "#dense-gate-up";
    }
    return std::string(base);
}

class Glm53WeightCache {
    struct Entry {
        CudaWeight weight;
        bool pinned{};
        std::uint32_t leases{};
        std::list<std::string>::iterator recency;
    };

    struct State {
        std::mutex mutex;
        std::unordered_map<std::string, Entry> entries;
        std::list<std::string> recency;
        std::uint64_t capacity{};
        std::uint64_t used{};
        std::uint64_t pinned{};
        std::uint64_t hits{};
        std::uint64_t misses{};
        std::uint64_t evictions{};
    };

public:
    struct LinearRequest {
        std::string_view base;
        std::uint64_t output_columns{};
        std::uint64_t input_columns{};
        std::span<const float> input;
        std::uint32_t rows{};
        std::span<float> output;
        bool bf16_output{};
    };

    struct Stats {
        std::vector<std::uint64_t> capacity;
        std::vector<std::uint64_t> used;
        std::vector<std::uint64_t> pinned;
        std::uint64_t hits{};
        std::uint64_t misses{};
        std::uint64_t evictions{};
    };

    Glm53WeightCache(Glm53CheckpointReader& checkpoint, CudaBackend& backend,
                     std::vector<int> devices,
                     std::vector<std::uint64_t> capacities)
        : checkpoint_(checkpoint), backend_(backend),
          devices_(std::move(devices)) {
        states_.reserve(capacities.size());
        for (const auto capacity : capacities) {
            auto state = std::make_unique<State>();
            state->capacity = capacity;
            states_.push_back(std::move(state));
        }
    }

    [[nodiscard]] ValidationResult preload(
        std::size_t slot, std::string_view base, std::uint64_t rows,
        std::uint64_t columns, bool& admitted) {
        admitted = false;
        const auto bytes = checkpoint_.cuda_linear_storage_bytes(base);
        if (slot >= states_.size() || bytes == 0U) {
            return {{"GLM-5.3 preload references an invalid CUDA linear"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        auto found = state.entries.find(std::string(base));
        if (found != state.entries.end()) {
            if (!found->second.pinned) {
                found->second.pinned = true;
                state.pinned += found->second.weight.device_bytes();
                state.recency.erase(found->second.recency);
            }
            admitted = true;
            ++state.hits;
            return {};
        }
        // A smaller or busier GPU may not fit its complete share of the
        // resident spine. Skipping residency changes only performance: the
        // exact weight is admitted through the demand/LRU path when needed.
        if (bytes > state.capacity - state.used) return {};
        Entry entry;
        auto loaded = checkpoint_.load_cuda_linear(
            base, rows, columns, devices_[slot], backend_, entry.weight);
        if (!loaded.ok()) return loaded;
        entry.pinned = true;
        const auto actual = entry.weight.device_bytes();
        if (actual > state.capacity - state.used) {
            return {{"GLM-5.3 resident linear exceeded its admitted CUDA cache"}};
        }
        state.used += actual;
        state.pinned += actual;
        state.entries.emplace(std::string(base), std::move(entry));
        admitted = true;
        ++state.misses;
        return {};
    }

    [[nodiscard]] ValidationResult matmul(
        std::size_t slot, std::string_view base, std::uint64_t output_columns,
        std::uint64_t input_columns, std::span<const float> input,
        std::uint32_t rows, std::span<float> output, bool bf16_output) {
        const LinearRequest request{base, output_columns, input_columns, input,
                                    rows, output, bf16_output};
        return matmul_batch(slot, std::span<const LinearRequest>(&request, 1U));
    }

    [[nodiscard]] ValidationResult matmul_batch(
        std::size_t slot, std::span<const LinearRequest> requests) {
        if (slot >= states_.size()) {
            return {{"GLM-5.3 linear targets an invalid CUDA cache slot"}};
        }
        if (requests.empty()) {
            return {{"GLM-5.3 linear batch is empty"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        struct BatchLeases {
            State& state;
            std::vector<std::string> keys;
            ~BatchLeases() {
                for (const auto& key : keys) {
                    const auto found = state.entries.find(key);
                    if (found != state.entries.end() &&
                        found->second.leases != 0U) {
                        --found->second.leases;
                    }
                }
            }
        } leases{state, {}};
        leases.keys.reserve(requests.size());
        std::vector<CudaMatmulBatchItem> batch;
        batch.reserve(requests.size());
        for (const auto& request : requests) {
            const std::string key(request.base);
            auto found = state.entries.find(key);
            if (found == state.entries.end()) {
                const auto bytes =
                    checkpoint_.cuda_linear_storage_bytes(request.base);
                if (bytes == 0U || bytes > state.capacity) {
                    return {{"GLM-5.3 linear is absent or exceeds its CUDA cache: " +
                             key}};
                }
                while (state.used + bytes > state.capacity) {
                    auto victim_position = state.recency.end();
                    for (auto candidate = state.recency.begin();
                         candidate != state.recency.end(); ++candidate) {
                        const auto entry = state.entries.find(*candidate);
                        if (entry != state.entries.end() &&
                            entry->second.leases == 0U) {
                            victim_position = candidate;
                            break;
                        }
                    }
                    if (victim_position == state.recency.end()) {
                        return {{"GLM-5.3 pinned spine leaves insufficient CUDA "
                                 "cache for an exact demand weight"}};
                    }
                    const auto victim_key = *victim_position;
                    state.recency.erase(victim_position);
                    auto victim = state.entries.find(victim_key);
                    if (victim == state.entries.end() || victim->second.pinned) {
                        return {{"GLM-5.3 CUDA cache recency bookkeeping is invalid"}};
                    }
                    state.used -= victim->second.weight.device_bytes();
                    state.entries.erase(victim);
                    ++state.evictions;
                }
                Entry entry;
                auto loaded = checkpoint_.load_cuda_linear(
                    request.base, request.output_columns, request.input_columns,
                    devices_[slot], backend_, entry.weight);
                if (!loaded.ok()) return loaded;
                const auto actual = entry.weight.device_bytes();
                if (actual > state.capacity - state.used) {
                    return {{"GLM-5.3 demand linear exceeded its admitted CUDA cache"}};
                }
                state.recency.push_back(key);
                entry.recency = std::prev(state.recency.end());
                state.used += actual;
                found = state.entries.emplace(key, std::move(entry)).first;
                ++state.misses;
            } else {
                ++state.hits;
                if (!found->second.pinned) {
                    state.recency.splice(state.recency.end(), state.recency,
                                         found->second.recency);
                }
            }
            ++found->second.leases;
            leases.keys.push_back(key);
            batch.push_back({&found->second.weight, request.input, request.rows,
                             request.output, request.bf16_output,
                             request.rows > 1U});
        }
        // One device-side event orders all deferred cache-miss uploads before
        // the consumer. This never blocks the host and is a no-op on a hit-only
        // path; matmul's output completion still protects the LRU entry.
        if (auto ordered = backend_.synchronize_uploads(devices_[slot]);
            !ordered.ok()) {
            return ordered;
        }
        return backend_.matmul_batch(batch);
    }

    [[nodiscard]] ValidationResult moe(
        std::size_t slot, std::string_view prefix,
        std::span<const KimiRoutedExpert> routed,
        std::span<const float> input, std::span<float> output) {
        ValidationResult result;
        if (slot >= states_.size() || routed.size() != 8U ||
            input.size() != kHidden || output.size() != kHidden) {
            result.errors.emplace_back("GLM-5.3 MoE command has an invalid shape");
            return result;
        }
        struct Projection {
            std::string key;
            std::uint64_t rows{};
            std::uint64_t columns{};
        };
        const auto make_modules = [](const std::string& base) {
            return std::array<Projection, 3U>{
                Projection{base + "gate_proj", 2048U, kHidden},
                Projection{base + "up_proj", 2048U, kHidden},
                Projection{base + "down_proj", kHidden, 2048U}};
        };
        struct DeviceGroup {
            std::vector<std::size_t> routes;
            bool has_shared{};
            bool enqueued{};
            std::vector<std::string> leased;
            std::vector<CudaMoeExpert> descriptors;
            CudaMoeExpert shared_descriptor;
            std::vector<float> routed_output;
            std::vector<float> shared_output;
        };
        std::vector<DeviceGroup> groups(states_.size());
        groups[slot].has_shared = true;

        // Routed and shared weights stay with the layer owner. A measured
        // expert-parallel variant issued these groups across every visible GPU,
        // but the extra cache duplication and PCIe/storage traffic regressed
        // both prefill and decode on a non-NVLink topology. Preserve locality;
        // layer splitting still distributes successive layers dynamically.
        for (std::size_t route_index = 0U; route_index < routed.size();
             ++route_index) {
            groups[slot].routes.push_back(route_index);
        }

        const auto release = [&](std::size_t group_slot) {
            auto& state = *states_[group_slot];
            std::scoped_lock lock(state.mutex);
            for (const auto& key : groups[group_slot].leased) {
                const auto found = state.entries.find(key);
                if (found != state.entries.end() && found->second.leases != 0U) {
                    --found->second.leases;
                }
            }
        };
        const auto ensure = [&](State& state, std::size_t target_slot,
                                const Projection& projection,
                                std::vector<std::string>& leased)
            -> ValidationResult {
            auto found = state.entries.find(projection.key);
            if (found == state.entries.end()) {
                const auto bytes =
                    checkpoint_.cuda_linear_storage_bytes(projection.key);
                if (bytes == 0U || bytes > state.capacity) {
                    return {{"GLM-5.3 MoE projection is absent or exceeds its "
                             "CUDA cache: " + projection.key}};
                }
                while (state.used + bytes > state.capacity) {
                    auto victim = state.recency.end();
                    for (auto candidate = state.recency.begin();
                         candidate != state.recency.end(); ++candidate) {
                        const auto entry = state.entries.find(*candidate);
                        if (entry != state.entries.end() &&
                            !entry->second.pinned && entry->second.leases == 0U) {
                            victim = candidate;
                            break;
                        }
                    }
                    if (victim == state.recency.end()) {
                        return {{"GLM-5.3 exact MoE expert set exceeds the "
                                 "available CUDA cache"}};
                    }
                    auto entry = state.entries.find(*victim);
                    state.used -= entry->second.weight.device_bytes();
                    state.entries.erase(entry);
                    state.recency.erase(victim);
                    ++state.evictions;
                }
                Entry entry;
                auto loaded = checkpoint_.load_cuda_linear(
                    projection.key, projection.rows, projection.columns,
                    devices_[target_slot], backend_, entry.weight);
                if (!loaded.ok()) return loaded;
                const auto actual = entry.weight.device_bytes();
                if (actual > state.capacity - state.used) {
                    return {{"GLM-5.3 MoE projection exceeded its admitted "
                             "CUDA cache"}};
                }
                state.recency.push_back(projection.key);
                entry.recency = std::prev(state.recency.end());
                state.used += actual;
                found = state.entries.emplace(projection.key,
                                              std::move(entry)).first;
                ++state.misses;
            } else {
                ++state.hits;
                if (!found->second.pinned) {
                    state.recency.splice(state.recency.end(), state.recency,
                                         found->second.recency);
                }
            }
            ++found->second.leases;
            leased.push_back(projection.key);
            return {};
        };

        std::vector<float> routed_output(routed.size() * kHidden);
        std::vector<float> shared_output(kHidden);
        for (std::size_t group_slot = 0U; group_slot < groups.size();
             ++group_slot) {
            auto& group = groups[group_slot];
            if (group.routes.empty() && !group.has_shared) continue;
            auto& state = *states_[group_slot];
            std::scoped_lock lock(state.mutex);
            std::vector<std::array<Projection, 3U>> modules;
            modules.reserve(group.routes.size());
            for (const auto route_index : group.routes) {
                modules.push_back(make_modules(
                    std::string(prefix) + "experts." +
                    std::to_string(routed[route_index].expert) + "."));
            }
            const auto shared_modules = make_modules(
                std::string(prefix) + "shared_experts.");
            group.leased.reserve(
                (modules.size() + (group.has_shared ? 1U : 0U)) * 3U);
            for (const auto& expert : modules) {
                for (const auto& projection : expert) {
                    auto loaded = ensure(state, group_slot, projection,
                                         group.leased);
                    if (!loaded.ok()) {
                        append(result.errors, std::move(loaded.errors));
                        break;
                    }
                }
                if (!result.ok()) break;
            }
            if (result.ok() && group.has_shared) {
                for (const auto& projection : shared_modules) {
                    auto loaded = ensure(state, group_slot, projection,
                                         group.leased);
                    if (!loaded.ok()) {
                        append(result.errors, std::move(loaded.errors));
                        break;
                    }
                }
            }
            if (!result.ok()) break;
            group.descriptors.resize(modules.size());
            for (std::size_t index = 0U; index < modules.size(); ++index) {
                const auto& expert = modules[index];
                group.descriptors[index] = {
                    &state.entries.at(expert[0].key).weight,
                    &state.entries.at(expert[1].key).weight,
                    &state.entries.at(expert[2].key).weight, 1.0F};
            }
            if (group.has_shared) {
                group.shared_descriptor = {
                    &state.entries.at(shared_modules[0].key).weight,
                    &state.entries.at(shared_modules[1].key).weight,
                    &state.entries.at(shared_modules[2].key).weight, 1.0F};
                group.shared_output.resize(kHidden);
            }
            // The whole routed-plus-shared set was admitted with deferred
            // copies. Order it once instead of synchronizing all 27 projection
            // uploads independently.
            auto ordered = backend_.synchronize_uploads(devices_[group_slot]);
            if (!ordered.ok()) {
                append(result.errors, std::move(ordered.errors));
                break;
            }
            group.routed_output.resize(group.routes.size() * kHidden);
            auto enqueued = backend_.enqueue_moe(
                devices_[group_slot], input, 1U, group.descriptors,
                group.has_shared ? &group.shared_descriptor : nullptr, 10.0F);
            if (!enqueued.ok()) {
                append(result.errors, std::move(enqueued.errors));
                break;
            }
            group.enqueued = true;
        }

        // Every active device has been enqueued before the first completion
        // boundary, so their expert projections and transfers overlap.
        for (std::size_t group_slot = 0U; group_slot < groups.size();
             ++group_slot) {
            auto& group = groups[group_slot];
            if (group.enqueued) {
                auto collected = backend_.collect_moe(
                    devices_[group_slot], group.routed_output,
                    group.has_shared ? std::span<float>(group.shared_output)
                                     : std::span<float>{});
                if (!collected.ok()) {
                    append(result.errors, std::move(collected.errors));
                } else {
                    for (std::size_t local = 0U; local < group.routes.size();
                         ++local) {
                        std::copy_n(
                            group.routed_output.begin() +
                                static_cast<std::ptrdiff_t>(local * kHidden),
                            kHidden,
                            routed_output.begin() + static_cast<std::ptrdiff_t>(
                                group.routes[local] * kHidden));
                    }
                    if (group.has_shared) {
                        std::copy(group.shared_output.begin(),
                                  group.shared_output.end(),
                                  shared_output.begin());
                    }
                }
            }
            if (!group.leased.empty()) release(group_slot);
        }
        if (!result.ok()) return result;
        std::copy(shared_output.begin(), shared_output.end(), output.begin());
        for (std::size_t expert = 0U; expert < routed.size(); ++expert) {
            const auto begin = expert * kHidden;
            for (std::size_t column = 0U; column < kHidden; ++column) {
                output[column] = bf16_round_f32(
                    output[column] + bf16_round_f32(
                        routed[expert].weight *
                        routed_output[begin + column]));
            }
        }
        return result;
    }

    [[nodiscard]] Stats stats() const {
        Stats result;
        for (const auto& state_ptr : states_) {
            auto& state = *state_ptr;
            std::scoped_lock lock(state.mutex);
            result.capacity.push_back(state.capacity);
            result.used.push_back(state.used);
            result.pinned.push_back(state.pinned);
            result.hits += state.hits;
            result.misses += state.misses;
            result.evictions += state.evictions;
        }
        return result;
    }

private:
    Glm53CheckpointReader& checkpoint_;
    CudaBackend& backend_;
    std::vector<int> devices_;
    std::vector<std::unique_ptr<State>> states_;
};

}  // namespace

struct Glm53Runtime::Impl {
    Glm53RuntimeConfig config;
    std::unique_ptr<Glm53CheckpointReader> checkpoint;
    ModelTokenizer tokenizer;
    CudaBackend cuda;
    std::vector<int> devices;
    std::vector<std::size_t> device_schedule;
    std::vector<std::uint64_t> weight_capacities;
    std::unique_ptr<Glm53WeightCache> weights;
    std::unique_ptr<HostWorkerPool> projection_workers;
    std::atomic<std::uint64_t> parallel_projection_batches{};
    std::atomic<std::uint64_t> parallel_projection_requests{};
    std::mutex host_tensor_mutex;
    std::unordered_map<std::string,
                       std::shared_ptr<const std::vector<float>>> host_tensors;
    std::array<std::vector<float>, kLayers> recurrent;
    std::array<std::array<std::vector<float>, 3U>, kLayers> convolution;
    std::array<std::vector<float>, kLayers> latents;
    ValidationResult warmup_result;
    bool ready{};
    std::thread warmup_thread;

    ~Impl() {
        if (warmup_thread.joinable()) warmup_thread.join();
    }

    [[nodiscard]] std::size_t slot_for(std::uint32_t layer) const noexcept {
        return device_schedule[layer % device_schedule.size()];
    }

    [[nodiscard]] int device_for(std::uint32_t layer) const noexcept {
        return devices[slot_for(layer)];
    }

    [[nodiscard]] ParseResult<std::shared_ptr<const std::vector<float>>>
    host_tensor(std::string_view name, std::uint64_t elements) {
        ParseResult<std::shared_ptr<const std::vector<float>>> result;
        const std::string key(name);
        {
            std::scoped_lock lock(host_tensor_mutex);
            const auto found = host_tensors.find(key);
            if (found != host_tensors.end()) {
                if (found->second->size() != elements) {
                    result.errors.push_back(
                        "GLM-5.3 cached host tensor has an invalid extent: " + key);
                } else {
                    result.value = found->second;
                }
                return result;
            }
        }
        auto loaded = checkpoint->read_f32(name, elements);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        auto value = std::make_shared<const std::vector<float>>(
            std::move(loaded.value));
        {
            std::scoped_lock lock(host_tensor_mutex);
            const auto [found, inserted] = host_tensors.emplace(key, value);
            result.value = inserted ? std::move(value) : found->second;
        }
        return result;
    }

    [[nodiscard]] ValidationResult wait_for_warmup() {
        if (warmup_thread.joinable()) warmup_thread.join();
        return warmup_result;
    }

    [[nodiscard]] ValidationResult warmup() {
        struct LinearTask {
            std::string base;
            std::string group;
            std::uint64_t rows{};
            std::uint64_t columns{};
            std::uint32_t layer{};
        };
        struct HostTask {
            std::string name;
            std::uint64_t elements{};
        };
        ValidationResult result;
        std::vector<LinearTask> linear_tasks;
        std::vector<std::vector<LinearTask>> device_tasks(devices.size());
        std::vector<HostTask> host_tasks;
        for (const auto& tensor : checkpoint->manifest().tensors) {
            if (tensor.role == Glm53TensorRole::Vision ||
                tensor.role == Glm53TensorRole::Mtp ||
                tensor.role == Glm53TensorRole::AttentionIndexer ||
                tensor.role == Glm53TensorRole::RoutedExpert ||
                tensor.role == Glm53TensorRole::Embedding ||
                tensor.component == Glm53TensorComponent::Scale) {
                continue;
            }
            const bool linear = tensor.name.ends_with(".weight") &&
                                tensor.source_shape.size() == 2U;
            if (linear) {
                const auto layer = tensor.layer >= 0
                    ? static_cast<std::uint32_t>(tensor.layer)
                    : kLayers - 1U;
                const auto base = tensor.name.substr(
                    0U, tensor.name.size() - 7U);
                linear_tasks.push_back({
                    base, projection_group_key(base, tensor.role),
                    tensor.source_shape[0], tensor.source_shape[1], layer});
                continue;
            }
            if (tensor.source_dtype != SafetensorsDtype::Bf16 &&
                tensor.source_dtype != SafetensorsDtype::F16 &&
                tensor.source_dtype != SafetensorsDtype::F32) {
                continue;
            }
            std::uint64_t elements = 1U;
            bool valid = !tensor.source_shape.empty();
            for (const auto dimension : tensor.source_shape) {
                if (dimension == 0U ||
                    elements > std::numeric_limits<std::uint64_t>::max() /
                                   dimension) {
                    valid = false;
                    break;
                }
                elements *= dimension;
            }
            if (valid) host_tasks.push_back({tensor.name, elements});
        }

        std::map<std::string, std::vector<LinearTask>> linear_groups;
        for (auto& task : linear_tasks) {
            linear_groups[task.group].push_back(std::move(task));
        }
        const bool parallel = projection_workers != nullptr &&
                              batched_projections_enabled() &&
                              cross_gpu_projections_enabled(devices) &&
                              devices.size() > 1U;
        for (auto& [group, tasks] : linear_groups) {
            static_cast<void>(group);
            if (!parallel || tasks.size() == 1U) {
                for (auto& task : tasks) {
                    device_tasks[slot_for(task.layer)].push_back(std::move(task));
                }
                continue;
            }
            std::vector<std::string_view> keys;
            std::vector<std::uint64_t> costs;
            keys.reserve(tasks.size());
            costs.reserve(tasks.size());
            for (const auto& task : tasks) {
                keys.push_back(task.base);
                costs.push_back(checkpoint->cuda_linear_storage_bytes(task.base));
            }
            const auto slots = glm53_projection_slots(
                keys, costs, weight_capacities, slot_for(tasks.front().layer));
            if (slots.size() != tasks.size()) {
                return {{"GLM-5.3 projection warmup assignment is invalid"}};
            }
            for (std::size_t index = 0U; index < tasks.size(); ++index) {
                device_tasks[slots[index]].push_back(std::move(tasks[index]));
            }
        }

        std::vector<ValidationResult> device_results(devices.size());
        std::vector<std::uint64_t> admitted(devices.size());
        std::vector<std::uint64_t> skipped(devices.size());
        std::atomic<std::size_t> next_slot{};
        const auto load_devices = [&] {
            for (;;) {
                const auto slot = next_slot.fetch_add(1U,
                                                       std::memory_order_relaxed);
                if (slot >= devices.size()) return;
                for (const auto& task : device_tasks[slot]) {
                    bool kept = false;
                    auto loaded = weights->preload(
                        slot, task.base, task.rows, task.columns, kept);
                    if (!loaded.ok()) {
                        append(device_results[slot].errors,
                               std::move(loaded.errors));
                        break;
                    }
                    kept ? ++admitted[slot] : ++skipped[slot];
                }
                if (device_results[slot].ok()) {
                    auto ordered = cuda.synchronize_uploads(devices[slot]);
                    if (!ordered.ok()) {
                        append(device_results[slot].errors,
                               std::move(ordered.errors));
                    }
                }
            }
        };
        const auto workers = std::min<std::size_t>(
            devices.size(), host_hardware_profile().worker_threads(0.1));
        std::vector<std::thread> loaders;
        loaders.reserve(workers);
        for (std::size_t worker = 0U; worker < workers; ++worker) {
            loaders.emplace_back(load_devices);
        }
        // Host-resident norms, convolution taps and mHC projections are only
        // 67 MiB for this checkpoint. Load them while independent PCIe links
        // receive their layer-split spine weights.
        for (const auto& task : host_tasks) {
            auto loaded = host_tensor(task.name, task.elements);
            if (!loaded.ok()) {
                append(result.errors, std::move(loaded.errors));
                break;
            }
        }
        for (auto& loader : loaders) loader.join();
        for (auto& device_result : device_results) {
            append(result.errors, std::move(device_result.errors));
        }
        if (config.verbose) {
            const auto stats = weights->stats();
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                std::cerr << "[glm53-load] cuda=" << devices[slot]
                          << " resident_linears=" << admitted[slot]
                          << " streamed_linears=" << skipped[slot]
                          << " pinned_bytes=" << stats.pinned[slot]
                          << " cache_capacity_bytes=" << stats.capacity[slot]
                          << '\n';
            }
        }
        return result;
    }

    [[nodiscard]] ValidationResult reset_sequence() {
        ValidationResult result;
        try {
            for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                if (glm53_kda_layer(layer)) {
                    recurrent[layer].assign(
                        static_cast<std::size_t>(kHeads) * kLinearHead *
                            kLinearHead,
                        0.0F);
                    for (auto& history : convolution[layer]) {
                        history.assign(static_cast<std::size_t>(kLinearWidth) * 3U,
                                       0.0F);
                    }
                } else {
                    latents[layer].assign(
                        static_cast<std::size_t>(config.maximum_context_tokens) *
                            kKvRank,
                        0.0F);
                }
            }
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "GLM-5.3 could not allocate recurrent and latent text state");
        }
        return result;
    }

    [[nodiscard]] ValidationResult linear(
        std::string_view base, std::span<const float> input,
        std::uint32_t rows, std::uint32_t columns,
        std::span<float> output, std::uint32_t layer,
        bool bf16_output = true) {
        ValidationResult result;
        if (input.size() != static_cast<std::size_t>(columns) * rows ||
            output.empty()) {
            result.errors.push_back("GLM-5.3 linear activation shape is invalid for " +
                                    std::string(base));
            return result;
        }
        const auto output_columns = output.size() / rows;
        if (output_columns * rows != output.size()) {
            result.errors.push_back("GLM-5.3 linear output shape is invalid for " +
                                    std::string(base));
            return result;
        }
        // PyTorch returns every published BF16/FP8 linear at the model's BF16
        // activation dtype. Keep that boundary even though the host-facing
        // CUDA API transports activations as float.
        return weights->matmul(slot_for(layer), base, output_columns, columns,
                               input, rows, output, bf16_output);
    }

    [[nodiscard]] ValidationResult linear_batch(
        std::span<const Glm53WeightCache::LinearRequest> requests,
        std::uint32_t layer) {
        if (requests.empty()) {
            return {{"GLM-5.3 linear projection batch is empty"}};
        }
        for (const auto& request : requests) {
            if (request.rows == 0U ||
                request.input.size() !=
                    static_cast<std::size_t>(request.input_columns) *
                        request.rows ||
                request.output.size() !=
                    static_cast<std::size_t>(request.output_columns) *
                        request.rows) {
                return {{"GLM-5.3 linear projection batch has an invalid shape"}};
            }
        }
        if (!batched_projections_enabled()) {
            for (const auto& request : requests) {
                auto projected = weights->matmul(
                    slot_for(layer), request.base, request.output_columns,
                    request.input_columns, request.input, request.rows,
                    request.output, request.bf16_output);
                if (!projected.ok()) return projected;
            }
            return {};
        }
        if (cross_gpu_projections_enabled(devices) &&
            projection_workers != nullptr &&
            devices.size() > 1U && requests.size() > 1U) {
            std::vector<std::string_view> keys;
            std::vector<std::uint64_t> costs;
            keys.reserve(requests.size());
            costs.reserve(requests.size());
            for (const auto& request : requests) {
                keys.push_back(request.base);
                costs.push_back(
                    checkpoint->cuda_linear_storage_bytes(request.base));
            }
            const auto slots = glm53_projection_slots(
                keys, costs, weight_capacities, slot_for(layer));
            if (slots.size() != requests.size()) {
                return {{"GLM-5.3 parallel projection assignment is invalid"}};
            }
            std::vector<std::vector<Glm53WeightCache::LinearRequest>> groups(
                devices.size());
            for (std::size_t index = 0U; index < requests.size(); ++index) {
                groups[slots[index]].push_back(requests[index]);
            }
            std::vector<ValidationResult> device_results(devices.size());
            auto dispatched = projection_workers->parallel_for_addressed(
                devices.size(), [&](std::size_t slot) {
                    if (!groups[slot].empty()) {
                        device_results[slot] =
                            weights->matmul_batch(slot, groups[slot]);
                    }
                });
            if (!dispatched.ok()) return dispatched;
            ValidationResult joined;
            std::uint64_t active_slots = 0U;
            for (std::size_t slot = 0U; slot < device_results.size(); ++slot) {
                if (!groups[slot].empty()) ++active_slots;
                append(joined.errors, std::move(device_results[slot].errors));
            }
            if (joined.ok() && active_slots > 1U) {
                parallel_projection_batches.fetch_add(
                    1U, std::memory_order_relaxed);
                parallel_projection_requests.fetch_add(
                    static_cast<std::uint64_t>(requests.size()),
                    std::memory_order_relaxed);
            }
            return joined;
        }
        return weights->matmul_batch(slot_for(layer), requests);
    }

    [[nodiscard]] ValidationResult norm(
        std::span<float> output, std::span<const float> input,
        std::string_view weight_name) {
        auto weight = host_tensor(weight_name, input.size());
        if (!weight.ok()) return {std::move(weight.errors)};
        auto result = kimi_rms_norm(output, input, *weight.value, 1.0e-5F);
        if (result.ok()) round_bf16(output);
        return result;
    }

    [[nodiscard]] ValidationResult norm_rows(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t columns,
        std::string_view weight_name) {
        ValidationResult result;
        if (rows == 0U || input.size() != output.size() ||
            input.size() != static_cast<std::size_t>(rows) * columns) {
            result.errors.emplace_back("GLM-5.3 RMSNorm page shape is invalid");
            return result;
        }
        auto weight = host_tensor(weight_name, columns);
        if (!weight.ok()) return {std::move(weight.errors)};
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto begin = static_cast<std::size_t>(row) * columns;
            auto normalized = kimi_rms_norm(
                output.subspan(begin, columns), input.subspan(begin, columns),
                *weight.value, 1.0e-5F);
            if (!normalized.ok()) return normalized;
            round_bf16(output.subspan(begin, columns));
        }
        return result;
    }

    [[nodiscard]] ValidationResult mhc_pre(
        std::span<float> collapsed, Dsv4MhcMix& mix,
        std::span<const float> streams, const std::string& prefix) {
        ValidationResult result;
        auto projection = host_tensor(prefix + "_fn", 24U * 16384U);
        auto base = host_tensor(prefix + "_base", 24U);
        auto scale = host_tensor(prefix + "_scale", 3U);
        if (!projection.ok() || !base.ok() || !scale.ok()) {
            append(result.errors, std::move(projection.errors));
            append(result.errors, std::move(base.errors));
            append(result.errors, std::move(scale.errors));
            return result;
        }
        double square_sum = 0.0;
        for (const auto value : streams) square_sum += static_cast<double>(value) * value;
        const auto reciprocal = 1.0F / std::sqrt(
            static_cast<float>(square_sum /
                               static_cast<double>(streams.size())) + 1.0e-5F);
        std::vector<float> projected(24U, 0.0F);
        for (std::size_t row = 0U; row < projected.size(); ++row) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < streams.size(); ++column) {
                sum += static_cast<double>((*projection.value)[row * streams.size() + column]) *
                       streams[column];
            }
            projected[row] = static_cast<float>(sum) * reciprocal;
        }
        auto split = dsv4_mhc_split_sinkhorn_f32(
            projected, *scale.value, *base.value, kMhc, 20U, 1.0e-6F);
        if (!split.ok()) return {std::move(split.errors)};
        mix = std::move(split.value);
        round_bf16(mix.post);
        round_bf16(mix.combination);
        std::fill(collapsed.begin(), collapsed.end(), 0.0F);
        for (std::size_t stream = 0U; stream < kMhc; ++stream) {
            for (std::size_t column = 0U; column < kHidden; ++column) {
                collapsed[column] += mix.pre[stream] *
                    streams[stream * kHidden + column];
            }
        }
        round_bf16(collapsed);
        return result;
    }

    [[nodiscard]] ValidationResult attention_kda(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, const std::string& attention) {
        ValidationResult result;
        std::vector<float> query(kLinearWidth), key(kLinearWidth),
            value(kLinearWidth), low(kLinearHead), beta(kHeads),
            gate_low(kLinearHead);
        const std::array<std::string, 6U> first_bases{
            attention + "q_proj", attention + "k_proj", attention + "v_proj",
            attention + "f_a_proj", attention + "b_proj",
            attention + "g_a_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 6U> first{
            {{first_bases[0], kLinearWidth, kHidden, input, 1U, query, true},
             {first_bases[1], kLinearWidth, kHidden, input, 1U, key, true},
             {first_bases[2], kLinearWidth, kHidden, input, 1U, value, true},
             {first_bases[3], kLinearHead, kHidden, input, 1U, low, true},
             {first_bases[4], kHeads, kHidden, input, 1U, beta, true},
             {first_bases[5], kLinearHead, kHidden, input, 1U, gate_low, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        for (std::uint32_t projection = 0U; projection < 3U; ++projection) {
            auto taps = host_tensor(
                attention + (projection == 0U ? "q_conv1d.weight"
                              : projection == 1U ? "k_conv1d.weight"
                                                 : "v_conv1d.weight"),
                static_cast<std::uint64_t>(kLinearWidth) * 4U);
            if (!taps.ok()) return {std::move(taps.errors)};
            auto& values = projection == 0U ? query : projection == 1U ? key : value;
            auto convolved = values;
            result = kimi_short_conv_step(convolved, values, *taps.value,
                                          convolution[layer][projection], 4U);
            if (!result.ok()) return result;
            values = std::move(convolved);
            round_bf16(values);
        }
        std::vector<float> forget(kLinearWidth), gate(kLinearWidth);
        const std::array<std::string, 2U> second_bases{
            attention + "f_b_proj", attention + "g_b_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kLinearWidth, kLinearHead, low, 1U, forget, true},
             {second_bases[1], kLinearWidth, kLinearHead, gate_low, 1U, gate,
              true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        for (auto& element : beta) {
            element = bf16_round_f32(sigmoid(element));
        }
        auto a_log = host_tensor(attention + "A_log", kHeads);
        auto dt_bias = host_tensor(attention + "dt_bias", kLinearWidth);
        auto o_norm = host_tensor(attention + "o_norm.weight", kLinearHead);
        if (!a_log.ok() || !dt_bias.ok() || !o_norm.ok()) {
            append(result.errors, std::move(a_log.errors));
            append(result.errors, std::move(dt_bias.errors));
            append(result.errors, std::move(o_norm.errors));
            return result;
        }
        std::vector<float> heads_out(kLinearWidth);
        const auto query_scale = 1.0F / std::sqrt(static_cast<float>(kLinearHead));
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            const auto begin = static_cast<std::size_t>(head) * kLinearHead;
            auto q = std::span<float>(query).subspan(begin, kLinearHead);
            auto k = std::span<float>(key).subspan(begin, kLinearHead);
            result = kimi_l2_normalize(q, 1.0e-6F);
            if (!result.ok()) return result;
            result = kimi_l2_normalize(k, 1.0e-6F);
            if (!result.ok()) return result;
            for (auto& element : q) element *= query_scale;
            std::vector<float> decay(kLinearHead);
            result = kimi_kda_log_decay(
                decay, std::span<const float>(forget).subspan(begin, kLinearHead),
                std::span<const float>(*dt_bias.value).subspan(begin, kLinearHead),
                (*a_log.value)[head], -5.0F);
            if (!result.ok()) return result;
            for (auto& element : decay) element = std::exp(element);
            std::vector<float> raw(kLinearHead);
            auto state = std::span<float>(recurrent[layer]).subspan(
                static_cast<std::size_t>(head) * kLinearHead * kLinearHead,
                static_cast<std::size_t>(kLinearHead) * kLinearHead);
            result = kimi_kda_step(
                raw, state, q, k,
                std::span<const float>(value).subspan(begin, kLinearHead),
                decay, beta[head], kLinearHead, kLinearHead);
            if (!result.ok()) return result;
            round_bf16(raw);
            result = kimi_kda_output_norm(
                std::span<float>(heads_out).subspan(begin, kLinearHead), raw,
                std::span<const float>(gate).subspan(begin, kLinearHead),
                *o_norm.value, 1.0e-5F);
            if (!result.ok()) return result;
            round_bf16(std::span<float>(heads_out).subspan(begin, kLinearHead));
        }
        return linear(attention + "o_proj", heads_out, 1U, kLinearWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult attention_kda_page(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t layer,
        const std::string& attention) {
        ValidationResult result;
        const auto wide_elements = static_cast<std::size_t>(rows) * kLinearWidth;
        std::vector<float> query(wide_elements), key(wide_elements),
            value(wide_elements),
            low(static_cast<std::size_t>(rows) * kLinearHead),
            beta(static_cast<std::size_t>(rows) * kHeads),
            gate_low(static_cast<std::size_t>(rows) * kLinearHead);
        const std::array<std::string, 6U> first_bases{
            attention + "q_proj", attention + "k_proj", attention + "v_proj",
            attention + "f_a_proj", attention + "b_proj",
            attention + "g_a_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 6U> first{
            {{first_bases[0], kLinearWidth, kHidden, input, rows, query, true},
             {first_bases[1], kLinearWidth, kHidden, input, rows, key, true},
             {first_bases[2], kLinearWidth, kHidden, input, rows, value, true},
             {first_bases[3], kLinearHead, kHidden, input, rows, low, true},
             {first_bases[4], kHeads, kHidden, input, rows, beta, true},
             {first_bases[5], kLinearHead, kHidden, input, rows, gate_low, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        for (std::uint32_t projection = 0U; projection < 3U; ++projection) {
            auto taps = host_tensor(
                attention + (projection == 0U ? "q_conv1d.weight"
                              : projection == 1U ? "k_conv1d.weight"
                                                 : "v_conv1d.weight"),
                static_cast<std::uint64_t>(kLinearWidth) * 4U);
            if (!taps.ok()) return {std::move(taps.errors)};
            auto& values = projection == 0U ? query : projection == 1U ? key : value;
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto begin = static_cast<std::size_t>(row) * kLinearWidth;
                std::vector<float> convolved(kLinearWidth);
                result = kimi_short_conv_step(
                    convolved,
                    std::span<const float>(values).subspan(begin, kLinearWidth),
                    *taps.value, convolution[layer][projection], 4U);
                if (!result.ok()) return result;
                round_bf16(convolved);
                std::copy(convolved.begin(), convolved.end(),
                          values.begin() + static_cast<std::ptrdiff_t>(begin));
            }
        }
        std::vector<float> forget(wide_elements);
        std::vector<float> gate(wide_elements);
        const std::array<std::string, 2U> second_bases{
            attention + "f_b_proj", attention + "g_b_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kLinearWidth, kLinearHead, low, rows, forget, true},
             {second_bases[1], kLinearWidth, kLinearHead, gate_low, rows, gate,
              true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        for (auto& element : beta) element = bf16_round_f32(sigmoid(element));
        auto a_log = host_tensor(attention + "A_log", kHeads);
        auto dt_bias = host_tensor(attention + "dt_bias", kLinearWidth);
        auto o_norm = host_tensor(attention + "o_norm.weight", kLinearHead);
        if (!a_log.ok() || !dt_bias.ok() || !o_norm.ok()) {
            append(result.errors, std::move(a_log.errors));
            append(result.errors, std::move(dt_bias.errors));
            append(result.errors, std::move(o_norm.errors));
            return result;
        }
        std::vector<float> heads_out(wide_elements);
        const auto query_scale = 1.0F / std::sqrt(static_cast<float>(kLinearHead));
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto row_begin = static_cast<std::size_t>(row) * kLinearWidth;
            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                const auto begin = row_begin +
                    static_cast<std::size_t>(head) * kLinearHead;
                auto q = std::span<float>(query).subspan(begin, kLinearHead);
                auto k = std::span<float>(key).subspan(begin, kLinearHead);
                result = kimi_l2_normalize(q, 1.0e-6F);
                if (!result.ok()) return result;
                result = kimi_l2_normalize(k, 1.0e-6F);
                if (!result.ok()) return result;
                for (auto& element : q) element *= query_scale;
                std::vector<float> decay(kLinearHead);
                result = kimi_kda_log_decay(
                    decay,
                    std::span<const float>(forget).subspan(begin, kLinearHead),
                    std::span<const float>(*dt_bias.value).subspan(
                        static_cast<std::size_t>(head) * kLinearHead,
                        kLinearHead),
                    (*a_log.value)[head], -5.0F);
                if (!result.ok()) return result;
                for (auto& element : decay) element = std::exp(element);
                std::vector<float> raw(kLinearHead);
                auto state = std::span<float>(recurrent[layer]).subspan(
                    static_cast<std::size_t>(head) * kLinearHead * kLinearHead,
                    static_cast<std::size_t>(kLinearHead) * kLinearHead);
                result = kimi_kda_step(
                    raw, state, q, k,
                    std::span<const float>(value).subspan(begin, kLinearHead),
                    decay, beta[static_cast<std::size_t>(row) * kHeads + head],
                    kLinearHead, kLinearHead);
                if (!result.ok()) return result;
                round_bf16(raw);
                result = kimi_kda_output_norm(
                    std::span<float>(heads_out).subspan(begin, kLinearHead), raw,
                    std::span<const float>(gate).subspan(begin, kLinearHead),
                    *o_norm.value, 1.0e-5F);
                if (!result.ok()) return result;
                round_bf16(std::span<float>(heads_out).subspan(
                    begin, kLinearHead));
            }
        }
        return linear(attention + "o_proj", heads_out, rows, kLinearWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult attention_mla(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, std::uint32_t position,
        const std::string& attention) {
        ValidationResult result;
        std::vector<float> q_rank(kQueryRank), query(kMlaWidth), latent(kKvRank);
        const std::array<std::string, 2U> first_bases{
            attention + "q_a_proj", attention + "kv_a_proj_with_mqa"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> first{
            {{first_bases[0], kQueryRank, kHidden, input, 1U, q_rank, true},
             {first_bases[1], kKvRank, kHidden, input, 1U, latent, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        result = norm(q_rank, q_rank, attention + "q_a_layernorm.weight");
        if (!result.ok()) return result;
        result = norm(latent, latent, attention + "kv_a_layernorm.weight");
        if (!result.ok()) return result;
        std::copy(latent.begin(), latent.end(),
                  latents[layer].begin() + static_cast<std::ptrdiff_t>(
                      static_cast<std::size_t>(position) * kKvRank));
        const auto history = position + 1U;
        std::vector<float> expanded(
            static_cast<std::size_t>(history) * kHeads * 2U * kMlaHead);
        const std::array<std::string, 2U> second_bases{
            attention + "q_b_proj", attention + "kv_b_proj"};
        const auto latent_history = std::span<const float>(latents[layer]).first(
            static_cast<std::size_t>(history) * kKvRank);
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kMlaWidth, kQueryRank, q_rank, 1U, query, true},
             {second_bases[1], kHeads * 2U * kMlaHead, kKvRank, latent_history,
              history, expanded, true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        std::vector<float> attended(kMlaWidth, 0.0F);
        const auto score_scale = 1.0F / std::sqrt(static_cast<float>(kMlaHead));
        std::vector<float> scores(history);
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            const auto* q = query.data() + static_cast<std::size_t>(head) * kMlaHead;
            float highest = -std::numeric_limits<float>::infinity();
            for (std::uint32_t token = 0U; token < history; ++token) {
                const auto* kv = expanded.data() +
                    (static_cast<std::size_t>(token) * kHeads + head) *
                        (2U * kMlaHead);
                float score = 0.0F;
                for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                    score += q[column] * kv[column];
                }
                scores[token] = score * score_scale;
                highest = std::max(highest, scores[token]);
            }
            float total = 0.0F;
            for (auto& score : scores) {
                score = std::exp(score - highest);
                total += score;
            }
            auto* destination = attended.data() +
                                static_cast<std::size_t>(head) * kMlaHead;
            for (std::uint32_t token = 0U; token < history; ++token) {
                const auto* values = expanded.data() +
                    (static_cast<std::size_t>(token) * kHeads + head) *
                        (2U * kMlaHead) + kMlaHead;
                const auto coefficient = bf16_round_f32(scores[token] / total);
                for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                    destination[column] += coefficient * values[column];
                }
            }
        }
        round_bf16(attended);
        return linear(attention + "o_proj", attended, 1U, kMlaWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult attention_mla_page(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t layer,
        const std::string& attention) {
        ValidationResult result;
        std::vector<float> q_rank(static_cast<std::size_t>(rows) * kQueryRank);
        std::vector<float> query(static_cast<std::size_t>(rows) * kMlaWidth);
        std::vector<float> latent(static_cast<std::size_t>(rows) * kKvRank);
        const std::array<std::string, 2U> first_bases{
            attention + "q_a_proj", attention + "kv_a_proj_with_mqa"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> first{
            {{first_bases[0], kQueryRank, kHidden, input, rows, q_rank, true},
             {first_bases[1], kKvRank, kHidden, input, rows, latent, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        result = norm_rows(q_rank, q_rank, rows, kQueryRank,
                           attention + "q_a_layernorm.weight");
        if (!result.ok()) return result;
        result = norm_rows(latent, latent, rows, kKvRank,
                           attention + "kv_a_layernorm.weight");
        if (!result.ok()) return result;
        std::copy(latent.begin(), latent.end(), latents[layer].begin());
        std::vector<float> expanded(
            static_cast<std::size_t>(rows) * kHeads * 2U * kMlaHead);
        const std::array<std::string, 2U> second_bases{
            attention + "q_b_proj", attention + "kv_b_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kMlaWidth, kQueryRank, q_rank, rows, query, true},
             {second_bases[1], kHeads * 2U * kMlaHead, kKvRank, latent, rows,
              expanded, true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        std::vector<float> attended(
            static_cast<std::size_t>(rows) * kMlaWidth, 0.0F);
        const auto score_scale = 1.0F / std::sqrt(static_cast<float>(kMlaHead));
        for (std::uint32_t row = 0U; row < rows; ++row) {
            std::vector<float> scores(row + 1U);
            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                const auto* q = query.data() +
                    (static_cast<std::size_t>(row) * kHeads + head) * kMlaHead;
                float highest = -std::numeric_limits<float>::infinity();
                for (std::uint32_t token = 0U; token <= row; ++token) {
                    const auto* kv = expanded.data() +
                        (static_cast<std::size_t>(token) * kHeads + head) *
                            (2U * kMlaHead);
                    float score = 0.0F;
                    for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                        score += q[column] * kv[column];
                    }
                    scores[token] = score * score_scale;
                    highest = std::max(highest, scores[token]);
                }
                float total = 0.0F;
                for (auto& score : scores) {
                    score = std::exp(score - highest);
                    total += score;
                }
                auto* destination = attended.data() +
                    (static_cast<std::size_t>(row) * kHeads + head) * kMlaHead;
                for (std::uint32_t token = 0U; token <= row; ++token) {
                    const auto* values = expanded.data() +
                        (static_cast<std::size_t>(token) * kHeads + head) *
                            (2U * kMlaHead) + kMlaHead;
                    const auto coefficient =
                        bf16_round_f32(scores[token] / total);
                    for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                        destination[column] += coefficient * values[column];
                    }
                }
            }
        }
        round_bf16(attended);
        return linear(attention + "o_proj", attended, rows, kMlaWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult swiglu_block(
        std::span<float> output, std::span<const float> input,
        const std::string& prefix, std::uint32_t inner,
        std::uint32_t layer) {
        ValidationResult result;
        std::vector<float> gate(inner), up(inner), activated(inner);
        const std::array<std::string, 2U> bases{
            prefix + "gate_proj", prefix + "up_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> projections{
            {{bases[0], inner, kHidden, input, 1U, gate, true},
             {bases[1], inner, kHidden, input, 1U, up, true}}};
        result = linear_batch(projections, layer);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < inner; ++index) {
            const auto g = std::min(gate[index], 10.0F);
            const auto u = std::clamp(up[index], -10.0F, 10.0F);
            activated[index] = g * sigmoid(g) * u;
        }
        round_bf16(activated);
        return linear(prefix + "down_proj", activated, 1U, inner,
                      output, layer);
    }

    [[nodiscard]] ValidationResult swiglu_block_page(
        std::span<float> output, std::span<const float> input,
        const std::string& prefix, std::uint32_t rows,
        std::uint32_t inner, std::uint32_t layer) {
        ValidationResult result;
        std::vector<float> gate(static_cast<std::size_t>(rows) * inner);
        std::vector<float> up(gate.size()), activated(gate.size());
        const std::array<std::string, 2U> bases{
            prefix + "gate_proj", prefix + "up_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> projections{
            {{bases[0], inner, kHidden, input, rows, gate, true},
             {bases[1], inner, kHidden, input, rows, up, true}}};
        result = linear_batch(projections, layer);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < gate.size(); ++index) {
            const auto g = std::min(gate[index], 10.0F);
            const auto u = std::clamp(up[index], -10.0F, 10.0F);
            activated[index] = g * sigmoid(g) * u;
        }
        round_bf16(activated);
        return linear(prefix + "down_proj", activated, rows, inner,
                      output, layer);
    }

    [[nodiscard]] ValidationResult feedforward(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, const std::string& prefix) {
        if (!glm53_moe_layer(layer)) {
            return swiglu_block(output, input, prefix + "mlp.", 12288U, layer);
        }
        ValidationResult result;
        std::vector<float> logits(288U);
        // The reference router explicitly promotes both operands to F32.
        result = linear(prefix + "mlp.gate", input, 1U, kHidden, logits,
                        layer, false);
        if (!result.ok()) return result;
        auto bias = host_tensor(
            prefix + "mlp.gate.e_score_correction_bias", 288U);
        if (!bias.ok()) return {std::move(bias.errors)};
        std::array<KimiRoutedExpert, 8U> selected{};
        result = kimi_route_topk(selected, logits, *bias.value, 2.5F);
        if (!result.ok()) return result;
        return weights->moe(slot_for(layer), prefix + "mlp.", selected,
                            input, output);
    }

    [[nodiscard]] ValidationResult feedforward_page(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t layer, const std::string& prefix) {
        if (!glm53_moe_layer(layer)) {
            return swiglu_block_page(output, input, prefix + "mlp.", rows,
                                     12288U, layer);
        }
        ValidationResult result;
        std::vector<float> logits(static_cast<std::size_t>(rows) * 288U);
        result = linear(prefix + "mlp.gate", input, rows, kHidden, logits,
                        layer, false);
        if (!result.ok()) return result;
        auto bias = host_tensor(
            prefix + "mlp.gate.e_score_correction_bias", 288U);
        if (!bias.ok()) return {std::move(bias.errors)};
        for (std::uint32_t row = 0U; row < rows; ++row) {
            std::array<KimiRoutedExpert, 8U> selected{};
            result = kimi_route_topk(
                selected,
                std::span<const float>(logits).subspan(
                    static_cast<std::size_t>(row) * 288U, 288U),
                *bias.value, 2.5F);
            if (!result.ok()) return result;
            result = weights->moe(
                slot_for(layer), prefix + "mlp.", selected,
                input.subspan(static_cast<std::size_t>(row) * kHidden, kHidden),
                output.subspan(static_cast<std::size_t>(row) * kHidden, kHidden));
            if (!result.ok()) return result;
        }
        return result;
    }

    [[nodiscard]] ValidationResult initialize_streams(
        std::uint32_t token, std::span<float> streams) {
        ValidationResult result;
        if (streams.size() != static_cast<std::size_t>(kMhc) * kHidden) {
            result.errors.emplace_back(
                "GLM-5.3 token streams have an invalid shape");
            return result;
        }
        auto embedding = checkpoint->read_f32_row(
            "model.language_model.embed_tokens.weight", token);
        if (!embedding.ok()) return {std::move(embedding.errors)};
        for (std::uint32_t stream = 0U; stream < kMhc; ++stream) {
            std::copy(embedding.value.begin(), embedding.value.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(
                          stream * kHidden));
        }
        return result;
    }

    [[nodiscard]] ValidationResult forward_layer(
        std::span<float> streams, std::uint32_t layer,
        std::uint32_t position) {
        ValidationResult result;
        if (streams.size() != static_cast<std::size_t>(kMhc) * kHidden ||
            layer >= kLayers) {
            result.errors.emplace_back(
                "GLM-5.3 layer command has an invalid shape");
            return result;
        }
        std::vector<float> collapsed(kHidden), normalized(kHidden), branch(kHidden);
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        Dsv4MhcMix mix;
        result = mhc_pre(collapsed, mix, streams, prefix + "hc_attn");
        if (!result.ok()) return result;
        result = norm(normalized, collapsed, prefix + "input_layernorm.weight");
        if (!result.ok()) return result;
        const auto attention = prefix + "self_attn.";
        result = glm53_kda_layer(layer)
            ? attention_kda(branch, normalized, layer, attention)
            : attention_mla(branch, normalized, layer, position, attention);
        if (!result.ok()) return result;
        std::vector<float> transitioned(streams.size());
        result = dsv4_mhc_post_f32(transitioned, branch, streams, mix, kMhc);
        if (!result.ok()) return result;
        round_bf16(transitioned);
        std::copy(transitioned.begin(), transitioned.end(), streams.begin());

        result = mhc_pre(collapsed, mix, streams, prefix + "hc_ffn");
        if (!result.ok()) return result;
        result = norm(normalized, collapsed,
                      prefix + "post_attention_layernorm.weight");
        if (!result.ok()) return result;
        result = feedforward(branch, normalized, layer, prefix);
        if (!result.ok()) return result;
        std::fill(transitioned.begin(), transitioned.end(), 0.0F);
        result = dsv4_mhc_post_f32(transitioned, branch, streams, mix, kMhc);
        if (!result.ok()) return result;
        round_bf16(transitioned);
        std::copy(transitioned.begin(), transitioned.end(), streams.begin());
        return result;
    }

    [[nodiscard]] ValidationResult forward_layer_page(
        std::span<float> streams, std::uint32_t rows, std::uint32_t layer) {
        ValidationResult result;
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        if (rows == 0U || layer >= kLayers ||
            streams.size() != static_cast<std::size_t>(rows) * stream_columns) {
            result.errors.emplace_back(
                "GLM-5.3 layer page has an invalid shape");
            return result;
        }
        const auto hidden_elements = static_cast<std::size_t>(rows) * kHidden;
        std::vector<float> collapsed(hidden_elements), normalized(hidden_elements),
            branch(hidden_elements);
        std::vector<Dsv4MhcMix> mixes(rows);
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = mhc_pre(
                std::span<float>(collapsed).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                mixes[row],
                std::span<const float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns),
                prefix + "hc_attn");
            if (!result.ok()) return result;
        }
        result = norm_rows(normalized, collapsed, rows, kHidden,
                           prefix + "input_layernorm.weight");
        if (!result.ok()) return result;
        const auto attention = prefix + "self_attn.";
        result = glm53_kda_layer(layer)
            ? attention_kda_page(branch, normalized, rows, layer, attention)
            : attention_mla_page(branch, normalized, rows, layer, attention);
        if (!result.ok()) return result;
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto stream_begin =
                static_cast<std::size_t>(row) * stream_columns;
            std::vector<float> transitioned(stream_columns);
            result = dsv4_mhc_post_f32(
                transitioned,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                std::span<const float>(streams).subspan(
                    stream_begin, stream_columns),
                mixes[row], kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            std::copy(transitioned.begin(), transitioned.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(stream_begin));
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = mhc_pre(
                std::span<float>(collapsed).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                mixes[row],
                std::span<const float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns),
                prefix + "hc_ffn");
            if (!result.ok()) return result;
        }
        result = norm_rows(normalized, collapsed, rows, kHidden,
                           prefix + "post_attention_layernorm.weight");
        if (!result.ok()) return result;
        result = feedforward_page(branch, normalized, rows, layer, prefix);
        if (!result.ok()) return result;
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto stream_begin =
                static_cast<std::size_t>(row) * stream_columns;
            std::vector<float> transitioned(stream_columns);
            result = dsv4_mhc_post_f32(
                transitioned,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                std::span<const float>(streams).subspan(
                    stream_begin, stream_columns),
                mixes[row], kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            std::copy(transitioned.begin(), transitioned.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(stream_begin));
        }
        return result;
    }

    [[nodiscard]] ValidationResult finish_streams(
        std::span<const float> streams, std::span<float> logits) {
        ValidationResult result;
        if (streams.size() != static_cast<std::size_t>(kMhc) * kHidden ||
            logits.empty()) {
            result.errors.emplace_back(
                "GLM-5.3 final text state has an invalid shape");
            return result;
        }
        std::vector<float> collapsed(kHidden), normalized(kHidden);
        for (std::size_t column = 0U; column < kHidden; ++column) {
            collapsed[column] = 0.25F *
                (streams[column] + streams[kHidden + column] +
                 streams[2U * kHidden + column] + streams[3U * kHidden + column]);
        }
        round_bf16(collapsed);
        result = norm(normalized, collapsed, "model.language_model.norm.weight");
        if (!result.ok()) return result;
        return linear("lm_head", normalized, 1U, kHidden, logits, kLayers - 1U);
    }

    [[nodiscard]] ValidationResult forward_token(
        std::uint32_t token, std::uint32_t position,
        std::span<float> logits) {
        ValidationResult result;
        std::vector<float> streams(static_cast<std::size_t>(kMhc) * kHidden);
        result = initialize_streams(token, streams);
        if (!result.ok()) return result;
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            result = forward_layer(streams, layer, position);
            if (!result.ok()) return result;
            if (config.load_progress) {
                std::cerr << "\r[glm53] layer " << (layer + 1U) << '/' << kLayers
                          << std::flush;
            }
        }
        if (config.load_progress) {
            std::cerr << '\r' << std::string(32U, ' ') << '\r';
        }
        return finish_streams(streams, logits);
    }

    [[nodiscard]] ValidationResult forward_prompt(
        std::span<const std::uint32_t> tokens, std::span<float> logits) {
        ValidationResult result;
        if (tokens.empty() || logits.empty()) {
            result.errors.emplace_back("GLM-5.3 prefill has an invalid shape");
            return result;
        }
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        std::vector<float> streams(tokens.size() * stream_columns);
        for (std::size_t position = 0U; position < tokens.size(); ++position) {
            result = initialize_streams(
                tokens[position],
                std::span<float>(streams).subspan(
                    position * stream_columns, stream_columns));
            if (!result.ok()) return result;
        }
        // Prompt rows are page/layer-major. Recurrent KDA and causal MLA state
        // still advance in token order inside each layer, while the active
        // layer's routed experts remain reusable in the bounded CUDA cache.
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            result = forward_layer_page(
                streams, static_cast<std::uint32_t>(tokens.size()), layer);
            if (!result.ok()) return result;
            if (config.load_progress) {
                std::cerr << "\r[glm53-prefill] layer " << (layer + 1U) << '/'
                          << kLayers << " rows " << tokens.size() << std::flush;
            }
        }
        if (config.load_progress) {
            std::cerr << '\r' << std::string(48U, ' ') << '\r';
        }
        return finish_streams(
            std::span<const float>(streams).last(stream_columns), logits);
    }
};

Glm53Runtime::Glm53Runtime() : impl_(std::make_unique<Impl>()) {}
Glm53Runtime::~Glm53Runtime() = default;
Glm53Runtime::Glm53Runtime(Glm53Runtime&&) noexcept = default;
Glm53Runtime& Glm53Runtime::operator=(Glm53Runtime&&) noexcept = default;

ValidationResult Glm53Runtime::initialize(
    const std::string& model_directory, const Glm53RuntimeConfig& config) {
    ValidationResult result;
    if (impl_->ready) {
        result.errors.emplace_back("GLM-5.3 runtime is already initialized");
        return result;
    }
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > kExactSparseContext) {
        result.errors.push_back(
            "GLM-5.3 text context must be within [1, 2048]; above 2048 the "
            "checkpoint's exact k-pool sparse indexer is required");
        return result;
    }
    impl_->config = config;
    impl_->devices = resolve_runtime_devices(config.devices);
    result = validate_common_runtime_config(
        impl_->devices, config.vram_cache_fraction,
        config.sampling_temperature, "GLM-5.3");
    if (!result.ok()) return result;
    auto device_plan = plan_runtime_devices(
        impl_->devices, config.vram_cache_fraction, kDeviceWorkspaceReserve,
        kMinimumDeviceBudget, "GLM-5.3");
    if (!device_plan.ok()) return {std::move(device_plan.errors)};
    auto tokenizer = ModelTokenizer::load(model_directory + "/tokenizer.json");
    if (!tokenizer.ok()) return {std::move(tokenizer.errors)};
    // The tokenizer has 154,820 base pieces plus 36 added special tokens.
    // The checkpoint pads its embedding and output matrices to 154,880 rows;
    // those 24 padding rows are deliberately not tokenizable.
    if (tokenizer.value.vocabulary_size() != 154856U) {
        result.errors.emplace_back(
            "GLM-5.3 tokenizer must expose 154856 usable token ids");
        return result;
    }
    auto checkpoint = Glm53CheckpointReader::open(model_directory);
    if (!checkpoint.ok()) return {std::move(checkpoint.errors)};
    result = impl_->cuda.initialize(impl_->devices);
    if (!result.ok()) return result;
    for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
        result = impl_->cuda.reserve_weight_arena(
            impl_->devices[slot], device_plan.value.weight_capacities[slot]);
        if (!result.ok()) return result;
    }
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->device_schedule = std::move(device_plan.value.weighted_schedule);
    impl_->weight_capacities = device_plan.value.weight_capacities;
    impl_->weights = std::make_unique<Glm53WeightCache>(
        *impl_->checkpoint, impl_->cuda, impl_->devices,
        impl_->weight_capacities);
    if (impl_->devices.size() > 1U) {
        auto worker_cpus = projection_worker_cpus(impl_->devices);
        if (worker_cpus.size() == impl_->devices.size()) {
            impl_->projection_workers = std::make_unique<HostWorkerPool>(
                std::move(worker_cpus));
        }
    }
    impl_->ready = true;
    try {
        // Keep API/server startup lazy-fast while warming independent device
        // spines in the background. The first generation joins this work; an
        // idle server usually reaches full residency before its first request.
        impl_->warmup_thread = std::thread([state = impl_.get()] {
            state->warmup_result = state->warmup();
        });
    } catch (const std::system_error& error) {
        impl_->ready = false;
        result.errors.push_back(
            "GLM-5.3 could not start background spine warmup: " +
            std::string(error.what()));
    }
    return result;
}

Glm53GenerationResult Glm53Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    Glm53GenerationResult result;
    if (!impl_->ready) {
        result.errors.emplace_back("GLM-5.3 runtime is not initialized");
        return result;
    }
    std::string error;
    if (!validate_sampling_options(sampling, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    if (!validate_chat_messages(messages, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    for (const auto& message : messages) {
        for (const auto& part : message.parts) {
            if (part.kind != ChatContentKind::Text) {
                result.errors.emplace_back(
                    "GLM-5.3 vision is not implemented; this runtime supports text-only messages");
                return result;
            }
        }
    }
    auto encoded = impl_->tokenizer.encode(
        render_glm53_chat_prompt(messages, "max", true));
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    if (encoded.value.empty() || encoded.value.size() + maximum_new_tokens >
            impl_->config.maximum_context_tokens) {
        result.errors.emplace_back(
            "GLM-5.3 prompt and requested generation exceed the admitted text context");
        return result;
    }
    auto warmup = impl_->wait_for_warmup();
    if (!warmup.ok()) {
        result.errors = std::move(warmup.errors);
        return result;
    }
    auto reset = impl_->reset_sequence();
    if (!reset.ok()) {
        result.errors = std::move(reset.errors);
        return result;
    }
    result.prompt_token_ids = encoded.value;
    result.metrics.prompt_tokens = encoded.value.size();
    std::vector<float> logits(kVocabulary);
    const auto prefill_started = now_seconds();
    auto prefill = impl_->forward_prompt(encoded.value, logits);
    if (!prefill.ok()) {
        result.errors = std::move(prefill.errors);
        return result;
    }
    result.metrics.prefill_tokens = encoded.value.size();
    result.metrics.prefill_seconds = now_seconds() - prefill_started;
    std::mt19937_64 generator(sampling.seed);
    std::vector<std::uint32_t> counts(kVocabulary, 0U);
    std::vector<std::uint32_t> sampled;
    StopSequenceBuffer streamed(stop);
    auto position = static_cast<std::uint32_t>(encoded.value.size());
    const auto decode_started = now_seconds();
    for (std::uint32_t index = 0U; index < maximum_new_tokens; ++index) {
        auto drawn = sample_logits(logits, sampling,
                                   SamplingHistory{counts, sampled}, generator);
        if (!drawn.ok()) {
            result.errors = std::move(drawn.errors);
            return result;
        }
        if (drawn.token == 154820U || drawn.token == 154827U ||
            drawn.token == 154829U) {
            result.stopped = true;
            break;
        }
        result.generated_token_ids.push_back(drawn.token);
        result.logprobs.push_back(drawn);
        sampled.push_back(drawn.token);
        ++counts[drawn.token];
        auto piece = impl_->tokenizer.decode_token(drawn.token);
        if (!piece.ok()) {
            result.errors = std::move(piece.errors);
            return result;
        }
        streamed.append(drawn.token, piece.value, on_token);
        if (streamed.stopped() || streamed.cancelled()) break;
        if (index + 1U == maximum_new_tokens) break;
        auto step = impl_->forward_token(drawn.token, position++, logits);
        if (!step.ok()) {
            result.errors = std::move(step.errors);
            return result;
        }
        ++result.metrics.decode_tokens;
    }
    result.metrics.decode_seconds = now_seconds() - decode_started;
    streamed.finish(on_token);
    result.text = streamed.text();
    result.stopped = result.stopped || streamed.stopped();
    if (impl_->config.verbose) {
        std::cerr << "[glm53-projection] parallel_batches="
                  << impl_->parallel_projection_batches.load(
                         std::memory_order_relaxed)
                  << " parallel_requests="
                  << impl_->parallel_projection_requests.load(
                         std::memory_order_relaxed)
                  << '\n';
    }
    return result;
}

}  // namespace strata
