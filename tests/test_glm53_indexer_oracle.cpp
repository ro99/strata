#include "test.hpp"

#include "../src/platform/json_cursor.hpp"
#include "strata/models/glm53/glm53_runtime.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// GLM-5.3's k-pool sparse indexer against the vendored `transformers`
// reference. The adapter's other exactness gates all run at or below
// `index_topk`, where selecting the top `index_topk` of at most `index_topk`
// candidates is the identity -- so they cannot see the sparse branch at all,
// and that blind spot is exactly why the indexer went unimplemented while every
// gate passed (record 0237, issue #43).
//
// The fixture is produced by `scripts/glm53_indexer_oracle.py`, which runs
// `Glm5NextTextIndexer` out of `experiments/references/glm5-next/` rather than
// reimplementing it. Inputs are not stored: both sides derive every weight and
// activation from the same 64-bit LCG, so the fixture holds only the reference's
// chosen positions.

namespace {

// Knuth's MMIX constants; the float uses bits 40..63 so it is exactly
// representable in binary32 and matches the Python generator bit for bit.
class Lcg {
public:
    explicit Lcg(std::uint64_t seed) noexcept : state_(seed) {}

    [[nodiscard]] float next() noexcept {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<double>(state_ >> 40U) /
                                      8388608.0 -
                                  1.0);
    }

    [[nodiscard]] std::vector<float> fill(std::size_t count) {
        std::vector<float> values(count);
        for (auto& value : values) value = next();
        return values;
    }

private:
    std::uint64_t state_{};
};

struct Probe {
    std::uint32_t row{};
    std::vector<std::uint32_t> selected;
};

struct Oracle {
    bool available{false};
    std::uint64_t seed{};
    std::uint32_t hidden_size{};
    std::uint32_t q_lora_rank{};
    std::uint32_t index_n_heads{};
    std::uint32_t index_head_dim{};
    std::uint32_t index_topk{};
    std::uint32_t index_kpool{};
    std::uint32_t sequence{};
    double selection_margin{};
    std::vector<Probe> probes;
};

[[nodiscard]] Oracle load_oracle() {
    Oracle oracle;
    const std::string path = std::string(STRATA_SOURCE_DIR) +
                             "/tests/fixtures/glm53/indexer-oracle.json";
    std::ifstream file(path, std::ios::binary);
    if (!file) return oracle;
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const auto text = buffer.str();

    strata::detail::JsonCursor cursor(text);
    cursor.expect('{');
    while (!cursor.consume('}')) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "seed") {
            oracle.seed = cursor.parse_uint64();
        } else if (key == "hidden_size") {
            oracle.hidden_size = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "q_lora_rank") {
            oracle.q_lora_rank = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "index_n_heads") {
            oracle.index_n_heads = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "index_head_dim") {
            oracle.index_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "index_topk") {
            oracle.index_topk = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "index_kpool") {
            oracle.index_kpool = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "sequence") {
            oracle.sequence = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "selection_margin") {
            oracle.selection_margin = cursor.parse_number();
        } else if (key == "probes") {
            cursor.expect('[');
            while (!cursor.consume(']')) {
                Probe probe;
                cursor.expect('{');
                while (!cursor.consume('}')) {
                    const auto field = cursor.parse_string();
                    cursor.expect(':');
                    if (field == "row") {
                        probe.row =
                            static_cast<std::uint32_t>(cursor.parse_uint64());
                    } else if (field == "selected") {
                        cursor.expect('[');
                        while (!cursor.consume(']')) {
                            probe.selected.push_back(
                                static_cast<std::uint32_t>(cursor.parse_uint64()));
                            static_cast<void>(cursor.consume(','));
                        }
                    } else {
                        cursor.skip_value();
                    }
                    static_cast<void>(cursor.consume(','));
                }
                oracle.probes.push_back(std::move(probe));
                static_cast<void>(cursor.consume(','));
            }
        } else {
            cursor.skip_value();
        }
        static_cast<void>(cursor.consume(','));
    }
    oracle.available = !oracle.probes.empty();
    return oracle;
}

}  // namespace

TEST_CASE("GLM-5.3 sparse indexer matches the transformers reference above index_topk") {
    using Parameters = strata::Glm53SparseIndexParameters;
    const auto oracle = load_oracle();
    if (!oracle.available) {
        SKIP("indexer oracle fixture is absent; run "
             "scripts/glm53_indexer_oracle.py");
    }
    // The fixture reduces only the projection widths. Everything the selection
    // semantics turn on must be the checkpoint's own geometry, or the oracle is
    // testing a different algorithm than the runtime runs.
    REQUIRE(oracle.index_n_heads == Parameters::heads);
    REQUIRE(oracle.index_head_dim == Parameters::head_dim);
    REQUIRE(oracle.index_topk == Parameters::top_k);
    REQUIRE(oracle.index_kpool == Parameters::pool);
    // An exact comparison against a differently associated float32 reference is
    // only meaningful while no probe row sits on the top-k boundary.
    REQUIRE(oracle.selection_margin > 1.0e-2);

    const auto hidden_size = oracle.hidden_size;
    const auto q_lora_rank = oracle.q_lora_rank;
    const auto sequence = oracle.sequence;
    constexpr auto kDim = Parameters::head_dim;
    constexpr auto kHeads = Parameters::heads;

    // Fill order is the contract with scripts/glm53_indexer_oracle.py.
    Lcg rng(oracle.seed);
    const auto wk = rng.fill(static_cast<std::size_t>(kDim) * hidden_size);
    const auto k_norm_weight = rng.fill(kDim);
    const auto k_norm_bias = rng.fill(kDim);
    const auto gate_weight = rng.fill(static_cast<std::size_t>(kDim) * hidden_size);
    const auto pool_ape = rng.fill(static_cast<std::size_t>(Parameters::pool) * kDim);
    const auto wq_b =
        rng.fill(static_cast<std::size_t>(kHeads) * kDim * q_lora_rank);
    const auto weights_proj = rng.fill(static_cast<std::size_t>(kHeads) * hidden_size);
    const auto hidden =
        rng.fill(static_cast<std::size_t>(sequence) * hidden_size);
    const auto q_resid = rng.fill(static_cast<std::size_t>(sequence) * q_lora_rank);

    // The per-token cache the runtime keeps: the normalized indexer key and the
    // k-pool gate, for every position.
    std::vector<float> keys(static_cast<std::size_t>(sequence) * kDim);
    std::vector<float> gates(keys.size());
    for (std::uint32_t token = 0U; token < sequence; ++token) {
        const auto row = std::span<const float>(hidden).subspan(
            static_cast<std::size_t>(token) * hidden_size, hidden_size);
        auto key = std::span<float>(keys).subspan(
            static_cast<std::size_t>(token) * kDim, kDim);
        strata::glm53_indexer_gate_for_test(key, row, wk);
        strata::glm53_indexer_layer_norm_for_test(key, k_norm_weight, k_norm_bias);
        strata::glm53_indexer_gate_for_test(
            std::span<float>(gates).subspan(
                static_cast<std::size_t>(token) * kDim, kDim),
            row, gate_weight);
    }

    std::vector<float> query(static_cast<std::size_t>(kHeads) * kDim);
    std::vector<float> head_weights(kHeads);
    std::vector<std::uint32_t> selected(Parameters::selection_width);
    for (const auto& probe : oracle.probes) {
        REQUIRE(probe.row < sequence);
        strata::glm53_indexer_gate_for_test(
            query,
            std::span<const float>(q_resid).subspan(
                static_cast<std::size_t>(probe.row) * q_lora_rank, q_lora_rank),
            wq_b);
        strata::glm53_indexer_gate_for_test(
            head_weights,
            std::span<const float>(hidden).subspan(
                static_cast<std::size_t>(probe.row) * hidden_size, hidden_size),
            weights_proj);
        const auto history = probe.row + 1U;
        const auto count = strata::glm53_sparse_index_select_for_test(
            selected, query, keys, gates, pool_ape, head_weights, history);
        REQUIRE(count == probe.selected.size());
        for (std::size_t index = 0U; index < count; ++index) {
            REQUIRE(selected[index] == probe.selected[index]);
        }
    }
}
