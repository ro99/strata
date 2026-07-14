#include "test.hpp"

#include "strata/glm_manifest.hpp"
#include "strata/safetensors.hpp"

#include <array>
#include <cstdio>
#include <set>
#include <string>
#include <utility>

namespace {

std::string shard_name(const char* prefix, int ordinal, int total) {
    char output[64]{};
    std::snprintf(output, sizeof(output), "%s-%05d-of-%05d.safetensors", prefix,
                  ordinal, total);
    return output;
}

strata::SafetensorsIndex synthetic_glm_index() {
    strata::SafetensorsIndex index;
    index.total_size = 405'459'090'304ULL;
    int main_cursor = 0;
    int mtp_cursor = 0;
    std::set<std::string> shards;
    auto next_shard = [&](int layer) {
        std::string shard;
        if (layer == 78) {
            shard = shard_name("mtp", mtp_cursor % 4 + 1, 4);
            ++mtp_cursor;
        } else {
            shard = shard_name("model", main_cursor % 124 + 1, 124);
            ++main_cursor;
        }
        shards.insert(shard);
        return shard;
    };
    auto add_plain = [&](std::string name, int layer) {
        index.entries.push_back({std::move(name), next_shard(layer)});
    };
    auto add_triplet = [&](const std::string& base, int layer) {
        const auto shard = next_shard(layer);
        index.entries.push_back({base + ".weight_packed", shard});
        index.entries.push_back({base + ".weight_scale", shard});
        index.entries.push_back({base + ".weight_shape", shard});
    };

    add_plain("model.embed_tokens.weight", -1);
    add_plain("lm_head.weight", -1);
    add_plain("model.norm.weight", -1);

    constexpr std::array<int, 22> full_indexer_layers{
        0, 1, 2, 3, 6, 10, 14, 18, 22, 26, 30,
        34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74};
    const auto has_full_indexer = [&](int layer) {
        for (const auto candidate : full_indexer_layers) {
            if (candidate == layer) return true;
        }
        return false;
    };
    constexpr std::array<const char*, 5> attention_modules{
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"};
    constexpr std::array<const char*, 3> projections{
        "gate_proj", "up_proj", "down_proj"};

    for (int layer = 0; layer <= 78; ++layer) {
        const auto prefix = "model.layers." + std::to_string(layer) + ".";
        add_plain(prefix + "input_layernorm.weight", layer);
        add_plain(prefix + "post_attention_layernorm.weight", layer);
        add_plain(prefix + "self_attn.q_a_layernorm.weight", layer);
        add_plain(prefix + "self_attn.kv_a_layernorm.weight", layer);
        for (const auto* module : attention_modules) {
            const auto base = prefix + "self_attn." + module;
            if (layer == 0) {
                add_plain(base + ".weight", layer);
            } else {
                add_triplet(base, layer);
            }
        }
        if (has_full_indexer(layer)) {
            add_plain(prefix + "self_attn.indexer.k_norm.bias", layer);
            add_plain(prefix + "self_attn.indexer.k_norm.weight", layer);
            add_plain(prefix + "self_attn.indexer.weights_proj.weight", layer);
            add_plain(prefix + "self_attn.indexer.wk.weight", layer);
            add_plain(prefix + "self_attn.indexer.wq_b.weight", layer);
        }
        if (layer < 3) {
            for (const auto* projection : projections) {
                const auto base = prefix + "mlp." + projection;
                if (layer == 0) {
                    add_plain(base + ".weight", layer);
                } else {
                    add_triplet(base, layer);
                }
            }
        } else {
            add_plain(prefix + "mlp.gate.weight", layer);
            add_plain(prefix + "mlp.gate.e_score_correction_bias", layer);
            for (const auto* projection : projections) {
                add_triplet(prefix + "mlp.shared_experts." + projection, layer);
            }
            for (int expert = 0; expert < 256; ++expert) {
                for (const auto* projection : projections) {
                    add_triplet(prefix + "mlp.experts." + std::to_string(expert) + "." +
                                    projection,
                                layer);
                }
            }
        }
        if (layer == 78) {
            add_plain(prefix + "eh_proj.weight", layer);
            add_plain(prefix + "enorm.weight", layer);
            add_plain(prefix + "hnorm.weight", layer);
            add_plain(prefix + "shared_head.norm.weight", layer);
        }
    }
    index.shards.assign(shards.begin(), shards.end());
    return index;
}

}  // namespace

TEST_CASE("Safetensors index preserves exact 64-bit sizes") {
    constexpr auto json = R"({
        "metadata":{"total_size":405459090304,"ignored":"value"},
        "weight_map":{"a.weight":"model-00001-of-00001.safetensors"}
    })";
    const auto result = strata::parse_safetensors_index(json);
    REQUIRE(result.ok());
    REQUIRE(result.value.total_size == 405'459'090'304ULL);
    REQUIRE(result.value.entries.size() == 1);
    REQUIRE(result.value.shards.size() == 1);
}

TEST_CASE("Safetensors index rejects duplicate tensor names") {
    constexpr auto json = R"({
        "metadata":{"total_size":8},
        "weight_map":{"a":"one.safetensors","a":"two.safetensors"}
    })";
    REQUIRE(!strata::parse_safetensors_index(json).ok());
}

TEST_CASE("Safetensors header validates dtype shape and contiguous extents") {
    constexpr auto header = R"({
        "a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},
        "b":{"dtype":"I64","shape":[2],"data_offsets":[8,24]},
        "__metadata__":{"format":"pt"}
    })";
    const auto data_start = 8U + sizeof(header) - 1U;
    const auto result = strata::parse_safetensors_header(header, data_start,
                                                          data_start + 24U);
    REQUIRE(result.ok());
    REQUIRE(result.value.tensors.size() == 2);
    REQUIRE(result.value.tensors[0].name == "a");
    REQUIRE(result.value.tensors[0].bytes() == 8);
    REQUIRE(result.value.tensors[1].absolute_begin == data_start + 8U);
}

TEST_CASE("Safetensors header rejects gaps and overlapping extents") {
    constexpr auto header = R"({
        "a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},
        "b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}
    })";
    REQUIRE(!strata::parse_safetensors_header(header, 128U, 140U).ok());
}

TEST_CASE("pinned GLM index classifies every tensor and quantization triplet") {
    const auto result = strata::build_quanttrio_glm52_index_manifest(synthetic_glm_index());
    REQUIRE(result.ok());
    REQUIRE(result.manifest.tensors.size() == 177'569);
    REQUIRE(result.manifest.shards.size() == 128);
    REQUIRE(result.manifest.quantized_modules == 58'992);
    REQUIRE(result.manifest.int4_modules == 57'600);
    REQUIRE(result.manifest.int8_group_modules == 616);
    REQUIRE(result.manifest.int8_channel_modules == 776);
}

TEST_CASE("pinned GLM manifest rejects a silently missing component") {
    auto index = synthetic_glm_index();
    index.entries.pop_back();
    const auto result = strata::build_quanttrio_glm52_index_manifest(std::move(index));
    REQUIRE(!result.ok());
}
