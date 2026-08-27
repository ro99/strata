#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strata {

// One exact launch contract in a router catalog. The router never interprets
// model weights and never substitutes another entry: it renders these already
// validated arguments into an ordinary single-model strata-server child.
struct ModelCatalogEntry {
    std::string id;
    std::string name;
    std::string model_directory;
    std::string model_type;
    std::vector<std::string> launch_arguments;
    std::optional<std::uint32_t> maximum_new_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<std::uint64_t> seed;
    std::uint32_t stop_timeout_seconds{10U};
    bool load_on_startup{};
};

struct ModelCatalog {
    std::vector<ModelCatalogEntry> models;
};

struct ModelCatalogResult {
    ModelCatalog value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// Parse a version-1 INI catalog. Relative model and auxiliary paths resolve
// against the directory containing the catalog, not the caller's CWD.
[[nodiscard]] ModelCatalogResult load_model_catalog(const std::string& path);

}  // namespace strata
