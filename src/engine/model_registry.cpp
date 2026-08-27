#include "strata/engine/model_executor.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace strata {

namespace {

// Function-local rather than a namespace-scope global: registrars are static
// objects in other translation units, and their construction order relative to
// a namespace-scope container here is unspecified. A function-local static is
// constructed on first use, which is by definition the first registration.
[[nodiscard]] std::vector<ModelRegistration>& registry() {
    static std::vector<ModelRegistration> models;
    return models;
}

}  // namespace

void register_model(const ModelRegistration& registration) {
    auto& models = registry();
    const auto existing = std::find_if(
        models.begin(), models.end(), [&](const ModelRegistration& entry) {
            return entry.model == registration.model;
        });
    // Last registration wins rather than duplicating. Two registrars for one
    // RuntimeModel is a build mistake, but silently carrying both would make
    // find_model's answer depend on link order.
    if (existing != models.end()) {
        *existing = registration;
        return;
    }
    models.push_back(registration);
}

const ModelRegistration* find_model(RuntimeModel model) noexcept {
    const auto& models = registry();
    const auto entry = std::find_if(
        models.begin(), models.end(), [&](const ModelRegistration& candidate) {
            return candidate.model == model;
        });
    return entry == models.end() ? nullptr : &*entry;
}

const ModelRegistration* find_model_by_cli_name(
    std::string_view cli_name) noexcept {
    const auto& models = registry();
    const auto entry = std::find_if(
        models.begin(), models.end(), [&](const ModelRegistration& candidate) {
            return candidate.cli_name != nullptr &&
                   cli_name == candidate.cli_name;
        });
    return entry == models.end() ? nullptr : &*entry;
}

std::string registered_model_names() {
    std::string names;
    for (const auto& entry : registry()) {
        if (entry.cli_name == nullptr) continue;
        if (!names.empty()) names += "|";
        names += entry.cli_name;
    }
    return names;
}

std::span<const ModelRegistration> registered_models() noexcept {
    return registry();
}

}  // namespace strata
