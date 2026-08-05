#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <map>
#include <span>
#include <string>
#include <vector>

// Shared reader for the fixtures `scripts/kimi_k3_reference_*.py` write, plus
// the agreement metric the Kimi gates report. Both gate 4 (one layer) and gate 5
// (the whole backbone) read the same format, and a second copy of this parser
// would be free to drift away from the writer.

namespace kimi_test {

inline std::string kimi_directory() {
    return (std::filesystem::path(STRATA_SOURCE_DIR) / "models/kimi-k3").string();
}

inline bool kimi_present() {
    return std::filesystem::exists(
        std::filesystem::path(kimi_directory()) / "model.safetensors.index.json");
}

// `results/` in the working tree is on the NVMe, and reference activations are
// derived from model weights, so the default lives on the SATA disk beside the
// checkpoint. See `docs/experiments/0048`.
inline std::string fixture_path(const char* name) {
    const auto* directory = std::getenv("STRATA_KIMI_FIXTURE_DIR");
    const std::filesystem::path base =
        directory != nullptr ? std::filesystem::path(directory)
                             : std::filesystem::path("/data/strata-results/kimi-k3-fixtures");
    return (base / name).string();
}

struct FixtureArray {
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

// The flat format the oracles write: a magic, a version, then one
// length-prefixed name, shape, and F32 payload per array.
class Fixture {
public:
    [[nodiscard]] bool load(const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) return false;
        char magic[4] = {};
        stream.read(magic, 4);
        if (std::memcmp(magic, "KMFX", 4) != 0) return false;
        std::uint32_t version = 0U;
        std::uint32_t count = 0U;
        read_raw(stream, version);
        read_raw(stream, count);
        if (version != 1U) return false;
        for (std::uint32_t index = 0U; index < count; ++index) {
            std::uint32_t name_length = 0U;
            read_raw(stream, name_length);
            std::string name(name_length, '\0');
            stream.read(name.data(), name_length);
            std::uint32_t rank = 0U;
            read_raw(stream, rank);
            FixtureArray array;
            array.shape.resize(rank);
            for (std::uint32_t axis = 0U; axis < rank; ++axis) {
                read_raw(stream, array.shape[axis]);
            }
            std::uint64_t elements = 0U;
            read_raw(stream, elements);
            array.values.resize(elements);
            stream.read(reinterpret_cast<char*>(array.values.data()),
                        static_cast<std::streamsize>(elements * sizeof(float)));
            if (!stream) return false;
            arrays_.emplace(std::move(name), std::move(array));
        }
        return true;
    }

    [[nodiscard]] const FixtureArray* find(const std::string& name) const {
        const auto entry = arrays_.find(name);
        return entry == arrays_.end() ? nullptr : &entry->second;
    }

private:
    template <typename T>
    static void read_raw(std::istream& stream, T& value) {
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    std::map<std::string, FixtureArray> arrays_;
};

struct Agreement {
    float relative_l2{};
    float cosine{};
};

inline Agreement compare(std::span<const float> measured,
                         std::span<const float> reference) {
    double difference = 0.0;
    double magnitude = 0.0;
    double dot = 0.0;
    double left = 0.0;
    double right = 0.0;
    for (std::size_t index = 0U; index < measured.size(); ++index) {
        const double a = measured[index];
        const double b = reference[index];
        difference += (a - b) * (a - b);
        magnitude += b * b;
        dot += a * b;
        left += a * a;
        right += b * b;
    }
    Agreement agreement;
    agreement.relative_l2 = static_cast<float>(std::sqrt(difference) /
                                               (std::sqrt(magnitude) + 1.0e-30));
    agreement.cosine = static_cast<float>(dot / (std::sqrt(left * right) + 1.0e-30));
    return agreement;
}

}  // namespace kimi_test
