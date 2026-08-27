#include "test.hpp"

#include "strata/models/glm52/glm52_manifest.hpp"
#include "strata/platform/safetensors.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <unistd.h>

namespace {

std::string shard_name(const char* prefix, int ordinal, int total) {
    char output[64]{};
    std::snprintf(output, sizeof(output), "%s-%05d-of-%05d.safetensors", prefix,
                  ordinal, total);
    return output;
}

strata::SafetensorsIndex synthetic_glm_index() {
    strata::SafetensorsIndex index;
    index.total_size = 387'667'154'688ULL;
    int main_cursor = 0;
    std::set<std::string> shards;
    auto next_shard = [&](int layer) {
        static_cast<void>(layer);
        const auto shard = shard_name("model", main_cursor % 8 + 1, 8);
        ++main_cursor;
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

    constexpr std::array<const char*, 5> attention_modules{
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"};
    constexpr std::array<const char*, 3> projections{
        "gate_proj", "up_proj", "down_proj"};

    for (int layer = 0; layer < 78; ++layer) {
        const auto prefix = "model.layers." + std::to_string(layer) + ".";
        add_plain(prefix + "input_layernorm.weight", layer);
        add_plain(prefix + "post_attention_layernorm.weight", layer);
        add_plain(prefix + "self_attn.q_a_layernorm.weight", layer);
        add_plain(prefix + "self_attn.kv_a_layernorm.weight", layer);
        for (const auto* module : attention_modules) {
            const auto base = prefix + "self_attn." + module;
            add_triplet(base, layer);
        }
        add_plain(prefix + "self_attn.indexer.k_norm.bias", layer);
        add_plain(prefix + "self_attn.indexer.k_norm.weight", layer);
        add_plain(prefix + "self_attn.indexer.weights_proj.weight", layer);
        add_plain(prefix + "self_attn.indexer.wk.weight", layer);
        add_plain(prefix + "self_attn.indexer.wq_b.weight", layer);
        if (layer < 3) {
            for (const auto* projection : projections) {
                const auto base = prefix + "mlp." + projection;
                add_triplet(base, layer);
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
    }
    index.shards.assign(shards.begin(), shards.end());
    return index;
}

}  // namespace

TEST_CASE("Safetensors index preserves exact 64-bit sizes") {
    constexpr auto json = R"({
        "metadata":{"total_size":387667154688,"ignored":"value"},
        "weight_map":{"a.weight":"model-00001-of-00001.safetensors"}
    })";
    const auto result = strata::parse_safetensors_index(json);
    REQUIRE(result.ok());
    REQUIRE(result.value.total_size == 387'667'154'688ULL);
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

TEST_CASE("Safetensors header accepts unsigned 32-bit packed weights") {
    constexpr auto header = R"({
        "packed":{"dtype":"U32","shape":[2,4],"data_offsets":[0,32]}
    })";
    const auto data_start = 8U + sizeof(header) - 1U;
    const auto result = strata::parse_safetensors_header(
        header, data_start, data_start + 32U);
    REQUIRE(result.ok());
    REQUIRE(result.value.tensors.size() == 1U);
    REQUIRE(result.value.tensors[0].dtype == strata::SafetensorsDtype::U32);
    REQUIRE(strata::safetensors_dtype_bytes(strata::SafetensorsDtype::U32) == 4U);
    REQUIRE(strata::to_string(strata::SafetensorsDtype::U32) == "U32");
}

TEST_CASE("Safetensors header rejects gaps and overlapping extents") {
    constexpr auto header = R"({
        "a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},
        "b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}
    })";
    REQUIRE(!strata::parse_safetensors_header(header, 128U, 140U).ok());
}

TEST_CASE("Safetensors index is synthesized from a lone shard header") {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("strata-safetensors-" + std::to_string(getpid()));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{directory};

    const std::string header = R"({
        "a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},
        "b":{"dtype":"I64","shape":[2],"data_offsets":[8,24]}
    })";
    std::ofstream shard(directory / "model.safetensors", std::ios::binary);
    const auto header_size = static_cast<std::uint64_t>(header.size());
    std::array<unsigned char, 8> length{};
    for (std::size_t index = 0; index < length.size(); ++index) {
        length[index] = static_cast<unsigned char>(header_size >> (index * 8U));
    }
    shard.write(reinterpret_cast<const char*>(length.data()), length.size());
    shard.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::array<char, 24> payload{};
    shard.write(payload.data(), payload.size());
    shard.close();

    const auto result = strata::load_safetensors_index(directory.string());
    REQUIRE(result.ok());
    REQUIRE(result.value.total_size == 24U);
    REQUIRE(result.value.shards == std::vector<std::string>{"model.safetensors"});
    REQUIRE(result.value.entries.size() == 2U);
    REQUIRE(result.value.entries[0].name == "a");
    REQUIRE(result.value.entries[0].shard == "model.safetensors");
    REQUIRE(result.value.entries[1].name == "b");
}

TEST_CASE("Safetensors index file takes precedence over a lone shard") {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("strata-safetensors-index-" + std::to_string(getpid()));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{directory};

    std::ofstream index(directory / "model.safetensors.index.json");
    index << R"({"metadata":{"total_size":8},"weight_map":{"indexed":"part.safetensors"}})";
    index.close();
    std::ofstream lone_shard(directory / "model.safetensors", std::ios::binary);
    lone_shard << "not a Safetensors shard";
    lone_shard.close();

    const auto result = strata::load_safetensors_index(directory.string());
    REQUIRE(result.ok());
    REQUIRE(result.value.total_size == 8U);
    REQUIRE(result.value.shards == std::vector<std::string>{"part.safetensors"});
    REQUIRE(result.value.entries.size() == 1U);
    REQUIRE(result.value.entries[0].name == "indexed");
}

TEST_CASE("GLM W4A16 index classifies every tensor and quantization triplet") {
    const auto result = strata::build_glm52_w4a16_index_manifest(synthetic_glm_index());
    REQUIRE(result.ok());
    REQUIRE(result.manifest.tensors.size() == 175'527);
    REQUIRE(result.manifest.shards.size() == 8);
    REQUIRE(result.manifest.quantized_modules == 58'224);
    REQUIRE(result.manifest.int4_modules == 58'224);
}

TEST_CASE("pinned GLM manifest rejects a silently missing component") {
    auto index = synthetic_glm_index();
    index.entries.pop_back();
    const auto result = strata::build_glm52_w4a16_index_manifest(std::move(index));
    REQUIRE(!result.ok());
}
