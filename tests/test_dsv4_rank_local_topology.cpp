#include "test.hpp"

#include "strata/deepseek_runtime.hpp"
#include "strata/dsv4_rank_local_topology.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace {

// Two nodes, 24 CPUs each: the measured production shape.
[[nodiscard]] strata::NumaTopology two_node_topology(
    std::size_t cpus_per_node = strata::kDsv4RankLocalMinimumCpusPerRank) {
    strata::NumaTopology topology;
    topology.nodes = 2;
    topology.node_cpus.resize(2U);
    topology.cpu_node.clear();
    int cpu = 0;
    for (std::size_t node = 0U; node < 2U; ++node) {
        for (std::size_t index = 0U; index < cpus_per_node; ++index) {
            topology.node_cpus[node].push_back(cpu++);
            topology.cpu_node.push_back(static_cast<int>(node));
        }
    }
    return topology;
}

// An admissible request, so each test can perturb exactly one condition.
[[nodiscard]] strata::Dsv4RankLocalAdmissionRequest admissible_request() {
    strata::Dsv4RankLocalAdmissionRequest request;
    request.devices = {0, 1};
    request.kv_cache_mode = strata::Dsv4KvCacheMode::PhysicalDevice;
    request.nccl_available = true;
    request.supported_checkpoint = true;
    request.fp4_routed_experts = true;
    request.layer_count = strata::kDsv4RankLocalLayerCount;
    request.active_context_tokens = 256U;
    request.maximum_context_tokens = 1'048'576U;
    for (auto& device : request.device) {
        device.rank_local_weight_bytes = 8'190'558'208ULL;
        device.centralized_spine_bytes = 9'204'991'520ULL;
        device.workspace_bytes = 536'870'912ULL;
        device.kv_capacity_bytes = 7'236'928ULL;
        device.nccl_buffer_bytes = 64ULL << 20U;
        device.head_buffer_bytes = 16ULL << 20U;
        device.expert_cache_bytes = 20'000'000'000ULL;
    }
    request.host.routed_cpu_storage_bytes = 150ULL << 30U;
    request.host.host_parameter_bytes = 4ULL << 30U;
    request.host.host_workspace_bytes = 1ULL << 30U;
    return request;
}

[[nodiscard]] bool mentions(const std::vector<std::string>& errors,
                            std::string_view needle) {
    for (const auto& error : errors) {
        if (error.find(needle) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("rank-local CPU planning assigns one disjoint NUMA node per rank") {
    std::array<std::vector<int>, strata::kDsv4RankLocalWorld> cpus;
    const auto planned = strata::plan_dsv4_rank_local_cpus(
        two_node_topology(), strata::kDsv4RankLocalMinimumCpusPerRank, cpus);
    REQUIRE(planned.ok());
    REQUIRE(cpus[0].size() == strata::kDsv4RankLocalMinimumCpusPerRank);
    REQUIRE(cpus[1].size() == strata::kDsv4RankLocalMinimumCpusPerRank);
    // Disjoint: two pools must never contend for the same cores.
    for (const auto cpu : cpus[0]) {
        REQUIRE(std::find(cpus[1].begin(), cpus[1].end(), cpu) ==
                cpus[1].end());
    }
}

TEST_CASE("rank-local CPU planning rejects a single-node host") {
    strata::NumaTopology topology;
    topology.nodes = 1;
    topology.node_cpus.resize(1U);
    for (int cpu = 0; cpu < 48; ++cpu) topology.node_cpus[0].push_back(cpu);
    std::array<std::vector<int>, strata::kDsv4RankLocalWorld> cpus;
    const auto planned = strata::plan_dsv4_rank_local_cpus(
        topology, strata::kDsv4RankLocalMinimumCpusPerRank, cpus);
    REQUIRE(!planned.ok());
    REQUIRE(mentions(planned.errors, "at least 2 NUMA nodes"));
    REQUIRE(cpus[0].empty());
    REQUIRE(cpus[1].empty());
}

TEST_CASE("rank-local CPU planning rejects an underprovisioned node") {
    std::array<std::vector<int>, strata::kDsv4RankLocalWorld> cpus;
    const auto planned = strata::plan_dsv4_rank_local_cpus(
        two_node_topology(23U), strata::kDsv4RankLocalMinimumCpusPerRank, cpus);
    REQUIRE(!planned.ok());
    REQUIRE(mentions(planned.errors, "below the required"));
    // Fail closed: no rank keeps a partial CPU set.
    REQUIRE(cpus[0].empty());
    REQUIRE(cpus[1].empty());
}

TEST_CASE("rank-local admission accepts the production operating point") {
    const auto admitted = strata::admit_dsv4_rank_local(
        admissible_request(), two_node_topology());
    REQUIRE(admitted.ok());
    REQUIRE(admitted.rank_cpus[0].size() ==
            strata::kDsv4RankLocalMinimumCpusPerRank);
    for (std::size_t rank = 0U; rank < strata::kDsv4RankLocalWorld; ++rank) {
        REQUIRE(admitted.device_total_bytes[rank] <=
                strata::kDsv4RankLocalPerDeviceVramCeiling);
    }
}

TEST_CASE("rank-local admission caps the centralized prefill expert cache") {
    auto request = admissible_request();
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology());
    REQUIRE(admitted.ok());
    // The request asked for 20 GB of cache, which cannot coexist with the
    // fixed components; admission caps it rather than rejecting, because the
    // cache is a prefill term outside the measured decode window.
    const auto fixed = request.device[0].fixed_total();
    REQUIRE(admitted.expert_cache_capacity_bytes[0] <
            request.device[0].expert_cache_bytes);
    REQUIRE(admitted.expert_cache_capacity_bytes[0] ==
            strata::kDsv4RankLocalPerDeviceVramCeiling - fixed);
    REQUIRE(admitted.device_total_bytes[0] ==
            strata::kDsv4RankLocalPerDeviceVramCeiling);
}

TEST_CASE("rank-local admission fits the 1M decode set beside the prefill spine") {
    // This is what the per-device ceiling was raised for, so it is pinned
    // rather than left implicit in the constant. At the declared 1,048,576
    // context the rank-local decode set and the centralized prefill spine are
    // resident together; experiment 0082's 21,287,272,448 B gate refused that
    // combination by about 0.76 GiB.
    auto request = admissible_request();
    request.active_context_tokens = 1'048'576U;
    for (auto& device : request.device) {
        // Measured 1M KV and index state: 21 ratio-4 layers of 262,144
        // compressed and learned-index rows, 20 ratio-128 layers of 8,192
        // compressed rows, and the fixed 128-row sliding window.
        device.kv_capacity_bytes = 4'082'533'760ULL;
        // No prefill cache: the spine's own weights are the term under test.
        device.expert_cache_bytes = 0ULL;
    }
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology());
    REQUIRE(admitted.ok());
    for (std::size_t rank = 0U; rank < strata::kDsv4RankLocalWorld; ++rank) {
        REQUIRE(admitted.device_total_bytes[rank] ==
                request.device[rank].fixed_total());
        REQUIRE(admitted.device_total_bytes[rank] >
                21'287'272'448ULL);
        REQUIRE(admitted.device_total_bytes[rank] <=
                strata::kDsv4RankLocalPerDeviceVramCeiling);
    }

    // The headroom is finite: the ceiling still has to reject something, or it
    // is not a ceiling. One more gigabyte of KV does not fit.
    for (auto& device : request.device) {
        device.kv_capacity_bytes += 1ULL << 30U;
    }
    REQUIRE(!strata::admit_dsv4_rank_local(
        request, two_node_topology()).ok());
}

TEST_CASE("rank-local admission rejects the opt-in without NCCL") {
    auto request = admissible_request();
    request.nccl_available = false;
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology());
    REQUIRE(!admitted.ok());
    REQUIRE(mentions(admitted.errors, "no NCCL"));
    REQUIRE(mentions(admitted.errors, "STRATA_ENABLE_NCCL"));
}

TEST_CASE("rank-local admission rejects a device count other than two") {
    auto request = admissible_request();
    request.devices = {0};
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology());
    REQUIRE(!admitted.ok());
    REQUIRE(mentions(admitted.errors, "exactly 2 CUDA devices"));

    auto duplicated = admissible_request();
    duplicated.devices = {1, 1};
    const auto repeated =
        strata::admit_dsv4_rank_local(duplicated, two_node_topology());
    REQUIRE(!repeated.ok());
    REQUIRE(mentions(repeated.errors, "two distinct CUDA devices"));
}

TEST_CASE("rank-local admission requires physical-device KV and FP4 experts") {
    auto kv = admissible_request();
    kv.kv_cache_mode = strata::Dsv4KvCacheMode::Block;
    const auto kv_admitted =
        strata::admit_dsv4_rank_local(kv, two_node_topology());
    REQUIRE(!kv_admitted.ok());
    REQUIRE(mentions(kv_admitted.errors, "physical-device DSV4 KV mode"));

    auto experts = admissible_request();
    experts.fp4_routed_experts = false;
    const auto expert_admitted =
        strata::admit_dsv4_rank_local(experts, two_node_topology());
    REQUIRE(!expert_admitted.ok());
    REQUIRE(mentions(expert_admitted.errors, "FP4 routed experts"));

    auto layers = admissible_request();
    layers.layer_count = 42U;
    const auto layer_admitted =
        strata::admit_dsv4_rank_local(layers, two_node_topology());
    REQUIRE(!layer_admitted.ok());
    REQUIRE(mentions(layer_admitted.errors, "requires 43 layers"));
}

TEST_CASE("rank-local admission rejects a per-GPU ceiling breach and names it") {
    auto request = admissible_request();
    // Push the fixed components alone past the ceiling.
    request.device[1].rank_local_weight_bytes = 15ULL << 30U;
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology());
    REQUIRE(!admitted.ok());
    REQUIRE(mentions(admitted.errors, "rank 1 fixed residency"));
    REQUIRE(mentions(admitted.errors, "exceeds the per-GPU ceiling"));
    // The rejection must attribute the overrun by component.
    REQUIRE(mentions(admitted.errors, "rank-local weights"));
    REQUIRE(mentions(admitted.errors, "centralized spine"));
    // Fail closed: no usable plan survives a rejection.
    REQUIRE(admitted.device_total_bytes[0] == 0U);
    REQUIRE(admitted.expert_cache_capacity_bytes[0] == 0U);
    REQUIRE(admitted.rank_cpus[0].empty());
}

TEST_CASE("rank-local admission serves the full declared context") {
    auto request = admissible_request();
    // Well past the sparse-indexer threshold: the topology must admit the
    // indexer regime rather than cap the context.
    request.active_context_tokens =
        strata::kDsv4RankLocalSparseIndexerThreshold * 16U;
    REQUIRE(strata::admit_dsv4_rank_local(request, two_node_topology()).ok());

    // The model's own 1M declared maximum is admitted.
    request.active_context_tokens = 1'048'576U;
    REQUIRE(strata::admit_dsv4_rank_local(request, two_node_topology()).ok());

    // Beyond the model maximum is rejected.
    request.active_context_tokens = 1'048'577U;
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology());
    REQUIRE(!admitted.ok());
    REQUIRE(mentions(admitted.errors, "above the model maximum"));
}

TEST_CASE("rank-local admission rejects a host RSS ceiling breach") {
    auto request = admissible_request();
    request.host.routed_cpu_storage_bytes = 230ULL << 30U;
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology());
    REQUIRE(!admitted.ok());
    REQUIRE(mentions(admitted.errors, "exceeds the host ceiling"));
}

TEST_CASE("rank-local admission reports every unmet requirement at once") {
    auto request = admissible_request();
    request.nccl_available = false;
    request.fp4_routed_experts = false;
    request.kv_cache_mode = strata::Dsv4KvCacheMode::ScalarOracle;
    const auto admitted =
        strata::admit_dsv4_rank_local(request, two_node_topology(4U));
    REQUIRE(!admitted.ok());
    // One rejection should name all of them, so an operator fixes the
    // configuration in one pass rather than one condition per run.
    REQUIRE(mentions(admitted.errors, "no NCCL"));
    REQUIRE(mentions(admitted.errors, "FP4 routed experts"));
    REQUIRE(mentions(admitted.errors, "physical-device DSV4 KV mode"));
    REQUIRE(mentions(admitted.errors, "below the required"));
    REQUIRE(admitted.errors.size() >= 4U);
}

TEST_CASE("rank-local opt-in is rejected before any model load") {
    // The model directory does not exist. If the topology gate runs where it
    // must -- before the checkpoint is opened -- initialization fails with
    // topology errors and never reports a missing model.
    strata::DeepSeekV4Runtime runtime;
    strata::Dsv4RuntimeConfig config;
    config.decode_topology = strata::Dsv4DecodeTopology::RankLocalTp2;
    config.devices = {0};
    config.kv_cache_mode = strata::Dsv4KvCacheMode::ScalarOracle;
    const auto initialized = runtime.initialize(
        "/nonexistent/strata-rank-local-admission-probe", config);
    REQUIRE(!initialized.ok());
    REQUIRE(mentions(initialized.errors, "rank-local decode requires"));
    for (const auto& error : initialized.errors) {
        REQUIRE(error.find("checkpoint") == std::string::npos);
        REQUIRE(error.find("model directory") == std::string::npos);
        REQUIRE(error.find("manifest") == std::string::npos);
    }
}

TEST_CASE("decode topology names are stable") {
    REQUIRE(std::string(strata::dsv4_decode_topology_name(
                strata::Dsv4DecodeTopology::Centralized)) == "centralized");
    REQUIRE(std::string(strata::dsv4_decode_topology_name(
                strata::Dsv4DecodeTopology::RankLocalTp2)) == "rank_local_tp2");
}
