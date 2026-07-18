#include "test.hpp"

#include "strata/deepseek_manifest.hpp"
#include "strata/deepseek_checkpoint.hpp"
#include "strata/deepseek_admission.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

std::string shard_name(std::uint32_t ordinal) {
    std::ostringstream output;
    output << "model-" << std::setw(5) << std::setfill('0') << ordinal
           << "-of-00048.safetensors";
    return output.str();
}

strata::SafetensorsIndex synthetic_dsv4_index() {
    strata::SafetensorsIndex index;
    index.total_size = 166'878'536'440ULL;
    const auto add = [&index](std::string name, std::uint32_t shard) {
        index.entries.push_back({std::move(name), shard_name(shard)});
    };
    const auto add_pair = [&add](const std::string& base, std::uint32_t shard) {
        add(base + ".weight", shard);
        add(base + ".scale", shard);
    };

    add("embed.weight", 47U);
    add("head.weight", 48U);
    add("norm.weight", 48U);
    add("hc_head_base", 48U);
    add("hc_head_fn", 48U);
    add("hc_head_scale", 48U);

    const auto compression = strata::deepseek_v4_flash_dspark_spec()
                                 .deepseek_v4.compression_ratios;
    for (std::uint32_t block = 0U; block < 46U; ++block) {
        const bool dspark = block >= 43U;
        const std::uint32_t local = dspark ? block - 43U : block;
        const std::uint32_t shard = block + 1U;
        const std::string prefix =
            (dspark ? "mtp." : "layers.") + std::to_string(local) + ".";
        const std::string attention = prefix + "attn.";
        add(attention + "attn_sink", shard);
        add(attention + "q_norm.weight", shard);
        add(attention + "kv_norm.weight", shard);
        for (const auto* operation : {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}) {
            add_pair(attention + operation, shard);
        }
        add(prefix + "attn_norm.weight", shard);
        add(prefix + "ffn_norm.weight", shard);
        for (const auto* branch : {"attn", "ffn"}) {
            add(prefix + "hc_" + branch + "_base", shard);
            add(prefix + "hc_" + branch + "_fn", shard);
            add(prefix + "hc_" + branch + "_scale", shard);
        }
        add(prefix + "ffn.gate.weight", shard);
        if (!dspark && local < 3U) {
            add(prefix + "ffn.gate.tid2eid", shard);
        } else {
            add(prefix + "ffn.gate.bias", shard);
        }
        for (const auto* operation : {"w1", "w2", "w3"}) {
            add_pair(prefix + "ffn.shared_experts." + operation, shard);
        }
        for (std::uint32_t expert = 0U; expert < 256U; ++expert) {
            for (const auto* operation : {"w1", "w2", "w3"}) {
                add_pair(prefix + "ffn.experts." + std::to_string(expert) + "." +
                             operation,
                         shard);
            }
        }
        if (!dspark && compression[local] != 0U) {
            for (const auto* name : {"ape", "norm.weight", "wgate.weight", "wkv.weight"}) {
                add(attention + "compressor." + name, shard);
            }
            if (compression[local] == 4U) {
                for (const auto* name : {"ape", "norm.weight", "wgate.weight", "wkv.weight"}) {
                    add(attention + "indexer.compressor." + name, shard);
                }
                add_pair(attention + "indexer.wq_b", shard);
                add(attention + "indexer.weights_proj.weight", shard);
            }
        }
    }

    add("mtp.0.main_norm.weight", 44U);
    add_pair("mtp.0.main_proj", 44U);
    add("mtp.2.norm.weight", 46U);
    add("mtp.2.hc_head_base", 46U);
    add("mtp.2.hc_head_fn", 46U);
    add("mtp.2.hc_head_scale", 46U);
    add("mtp.2.markov_head.markov_w1.weight", 46U);
    add("mtp.2.markov_head.markov_w2.weight", 46U);
    add("mtp.2.confidence_head.proj.weight", 46U);

    for (std::uint32_t shard = 1U; shard <= 48U; ++shard) {
        index.shards.push_back(shard_name(shard));
    }
    return index;
}

}  // namespace

TEST_CASE("pinned DeepSeek V4 DSpark index classifies the native checkpoint") {
    const auto result = strata::build_deepseek_v4_flash_dspark_index_manifest(
        synthetic_dsv4_index());
    REQUIRE(result.ok());
    REQUIRE(result.manifest.tensors.size() == 72'317U);
    REQUIRE(result.manifest.shards.size() == 48U);
    REQUIRE(result.manifest.quantized_modules == 35'718U);
    REQUIRE(result.manifest.fp4_modules == 35'328U);
    REQUIRE(result.manifest.fp8_modules == 390U);
}

TEST_CASE("pinned DeepSeek V4 manifest rejects a missing FP4 scale") {
    auto index = synthetic_dsv4_index();
    index.entries.erase(std::find_if(
        index.entries.begin(), index.entries.end(), [](const auto& entry) {
            return entry.name == "layers.0.ffn.experts.0.w1.scale";
        }));
    const auto result = strata::build_deepseek_v4_flash_dspark_index_manifest(
        std::move(index));
    REQUIRE(!result.ok());
}

TEST_CASE("real DeepSeek V4 DSpark checkpoint opens without format conversion when available") {
    const auto model = std::filesystem::path(STRATA_SOURCE_DIR) /
                       "models/DeepSeek-V4-Flash-DSpark";
    if (!std::filesystem::exists(model / "model.safetensors.index.json")) {
        SKIP("pinned DeepSeek checkpoint fixture is absent");
    }
    const auto checkpoint = strata::Dsv4CheckpointReader::open(model.string());
    REQUIRE(checkpoint.ok());
    const auto* tensor = checkpoint.value->find("layers.0.ffn.experts.0.w1.weight");
    REQUIRE(tensor != nullptr);
    REQUIRE(tensor->source_dtype == strata::SafetensorsDtype::I8);
    REQUIRE(tensor->source_shape == std::vector<std::uint64_t>({2048U, 2048U}));
    const auto bytes = checkpoint.value->read_slice(tensor->name, 0U, 8U);
    REQUIRE(bytes.ok());
    constexpr std::array<std::uint8_t, 8> expected{
        0xacU, 0x54U, 0xa2U, 0x44U, 0xccU, 0x54U, 0x6cU, 0x55U};
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        REQUIRE(std::to_integer<std::uint8_t>(bytes.value[index]) == expected[index]);
    }
    std::vector<std::byte> direct(static_cast<std::size_t>(tensor->source_bytes));
    const auto before_direct = checkpoint.value->stats();
    strata::Dsv4CheckpointReadStats local_direct;
    const auto direct_read = checkpoint.value->read_into(
        tensor->name, direct, &local_direct);
    const auto after_direct = checkpoint.value->stats();
    REQUIRE(direct_read.ok());
    REQUIRE(after_direct.calls == before_direct.calls + 1U);
    REQUIRE(after_direct.bytes == before_direct.bytes + tensor->source_bytes);
    REQUIRE(local_direct.calls == 1U);
    REQUIRE(local_direct.bytes == tensor->source_bytes);
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        REQUIRE(std::to_integer<std::uint8_t>(direct[index]) == expected[index]);
    }
    direct.pop_back();
    REQUIRE(!checkpoint.value->read_into(tensor->name, direct).ok());
    strata::Dsv4AdmissionConfig config;
    config.host_memory_ceiling_bytes = 216ULL << 30U;
    config.vram_weight_budgets = {14ULL << 30U, 22ULL << 30U, 22ULL << 30U};
    config.maximum_context_tokens = 2048U;
    const auto admission = strata::plan_dsv4_resident_topology(
        checkpoint.value->manifest(), config);
    REQUIRE(admission.ok());
    REQUIRE(admission.plan.zero_nvme_decode);
    REQUIRE(admission.plan.steady_state_nvme_bytes == 0U);
    REQUIRE(admission.plan.routed_expert_host_bytes > (128ULL << 30U));
    REQUIRE(admission.plan.resident_spine_vram_bytes > (8ULL << 30U));
    REQUIRE(admission.plan.maximum_expert_bytes > (12ULL << 20U));

    config.maximum_context_tokens = 1'048'576U;
    const auto million_token_admission = strata::plan_dsv4_resident_topology(
        checkpoint.value->manifest(), config);
    REQUIRE(million_token_admission.ok());
    REQUIRE(million_token_admission.plan.maximum_context_tokens == 1'048'576U);
    REQUIRE(million_token_admission.plan.index_state_bytes > (2ULL << 30U));
    REQUIRE(million_token_admission.plan.kv_state_bytes > (13ULL << 30U));
    REQUIRE(million_token_admission.plan.required_host_bytes < (216ULL << 30U));

    config.maximum_context_tokens = 1'048'577U;
    REQUIRE(!strata::plan_dsv4_resident_topology(
                 checkpoint.value->manifest(), config).ok());
    config.maximum_context_tokens = 2048U;

    config.enable_dspark = true;
    const auto with_dspark = strata::plan_dsv4_resident_topology(
        checkpoint.value->manifest(), config);
    REQUIRE(with_dspark.ok());
    REQUIRE(with_dspark.plan.routed_expert_host_bytes >
            admission.plan.routed_expert_host_bytes);
    REQUIRE(with_dspark.plan.dspark_enabled);

    config.host_memory_ceiling_bytes = 64ULL << 30U;
    REQUIRE(!strata::plan_dsv4_resident_topology(
                 checkpoint.value->manifest(), config).ok());
}
