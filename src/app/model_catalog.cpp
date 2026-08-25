#include "strata/model_catalog.hpp"

#include "strata/model_executor.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

namespace strata {
namespace {

using Options = std::map<std::string, std::string, std::less<>>;

constexpr std::uint64_t kMaximumCatalogBytes = 1ULL << 20U;

std::string trim(std::string_view value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t' ||
                              value.front() == '\r')) {
        value.remove_prefix(1U);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
                              value.back() == '\r')) {
        value.remove_suffix(1U);
    }
    return std::string(value);
}

bool parse_bool(std::string_view value, bool& output) {
    if (value == "true" || value == "yes" || value == "1") {
        output = true;
        return true;
    }
    if (value == "false" || value == "no" || value == "0") {
        output = false;
        return true;
    }
    return false;
}

template <typename Integer>
bool parse_unsigned(std::string_view value, Integer& output) {
    std::uint64_t parsed = 0U;
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed > static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
        return false;
    }
    output = static_cast<Integer>(parsed);
    return true;
}

bool parse_double(std::string_view value, double minimum, double maximum,
                  double& output) {
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), output);
    return result.ec == std::errc{} && result.ptr == value.data() + value.size() &&
        std::isfinite(output) && output >= minimum && output <= maximum;
}

bool valid_id(std::string_view id) {
    if (id.empty() || id.size() > 200U || id == "*") return false;
    return std::all_of(id.begin(), id.end(), [](unsigned char value) {
        return value >= 0x21U && value != 0x7FU;
    });
}

std::filesystem::path resolve_path(const std::filesystem::path& base,
                                   std::string_view value) {
    std::filesystem::path path(value);
    if (path.is_relative()) path = base / path;
    return std::filesystem::absolute(path).lexically_normal();
}

void push_value(std::vector<std::string>& arguments, std::string option,
                const std::string& value) {
    arguments.push_back(std::move(option));
    arguments.push_back(value);
}

bool known_key(std::string_view key) {
    static const std::set<std::string, std::less<>> keys{
        "model", "model-type", "name", "devices", "context-size", "max-new",
        "vram-fraction", "temperature", "top-p", "seed", "flash-attention",
        "block-kv-cache", "pin-resident-arena", "device-resident-runtime",
        "decode-topology", "prefill-page-tokens", "static-expert-plan",
        "static-expert-bytes", "plan-cache", "use-plan-cache", "replan",
        "load-on-startup", "stop-timeout"
    };
    return keys.contains(key);
}

bool model_only_key(std::string_view key) {
    return key == "model" || key == "model-type" || key == "name" ||
           key == "load-on-startup" || key == "stop-timeout";
}

std::optional<std::string> get(const Options& options, std::string_view key) {
    const auto found = options.find(key);
    if (found == options.end()) return std::nullopt;
    return found->second;
}

void validate_boolean_option(const Options& options, std::string_view key,
                             std::vector<std::string>& arguments,
                             std::vector<std::string>& errors,
                             const std::string& section) {
    const auto value = get(options, key);
    if (!value.has_value()) return;
    bool enabled = false;
    if (!parse_bool(*value, enabled)) {
        errors.push_back("catalog section [" + section + "]: " +
                         std::string(key) + " must be true or false");
        return;
    }
    if (enabled) arguments.push_back("--" + std::string(key));
}

ModelCatalogEntry build_entry(
    const std::string& id, const Options& options,
    const std::filesystem::path& catalog_directory,
    std::vector<std::string>& errors) {
    ModelCatalogEntry entry;
    entry.id = id;
    entry.name = get(options, "name").value_or(id);
    const auto model = get(options, "model");
    const auto model_type = get(options, "model-type");
    if (!model.has_value() || model->empty()) {
        errors.push_back("catalog section [" + id + "]: model is required");
    } else {
        entry.model_directory = resolve_path(catalog_directory, *model).string();
        std::error_code code;
        if (!std::filesystem::is_directory(entry.model_directory, code)) {
            errors.push_back("catalog section [" + id + "]: model directory does not exist: " +
                             entry.model_directory);
        }
    }
    if (!model_type.has_value() || model_type->empty()) {
        errors.push_back("catalog section [" + id + "]: model-type is required");
    } else {
        entry.model_type = *model_type;
        if (find_model_by_cli_name(entry.model_type) == nullptr) {
            errors.push_back("catalog section [" + id + "]: unknown model-type '" +
                             entry.model_type + "'");
        }
    }

    if (const auto value = get(options, "devices")) {
        push_value(entry.launch_arguments, "--devices", *value);
    }
    if (const auto value = get(options, "context-size")) {
        std::uint32_t parsed = 0U;
        if (!parse_unsigned(*value, parsed) || parsed == 0U) {
            errors.push_back("catalog section [" + id + "]: invalid context-size");
        } else {
            push_value(entry.launch_arguments, "--context-size", *value);
        }
    }
    if (const auto value = get(options, "max-new")) {
        std::uint32_t parsed = 0U;
        if (!parse_unsigned(*value, parsed) || parsed == 0U) {
            errors.push_back("catalog section [" + id + "]: invalid max-new");
        } else {
            entry.maximum_new_tokens = parsed;
            push_value(entry.launch_arguments, "--max-new", *value);
        }
    }
    if (const auto value = get(options, "vram-fraction")) {
        double parsed = 0.0;
        if (!parse_double(*value, 0.0, 0.95, parsed) || parsed == 0.0) {
            errors.push_back("catalog section [" + id + "]: vram-fraction must be within (0, 0.95]");
        } else {
            push_value(entry.launch_arguments, "--vram-fraction", *value);
        }
    }
    if (const auto value = get(options, "temperature")) {
        double parsed = 0.0;
        if (!parse_double(*value, 0.0, 2.0, parsed)) {
            errors.push_back("catalog section [" + id + "]: temperature must be within [0, 2]");
        } else {
            entry.temperature = parsed;
            push_value(entry.launch_arguments, "--temperature", *value);
        }
    }
    if (const auto value = get(options, "top-p")) {
        double parsed = 0.0;
        if (!parse_double(*value, 0.0, 1.0, parsed) || parsed == 0.0) {
            errors.push_back("catalog section [" + id + "]: top-p must be within (0, 1]");
        } else {
            entry.top_p = parsed;
            push_value(entry.launch_arguments, "--top-p", *value);
        }
    }
    if (const auto value = get(options, "seed")) {
        std::uint64_t parsed = 0U;
        if (!parse_unsigned(*value, parsed)) {
            errors.push_back("catalog section [" + id + "]: invalid seed");
        } else {
            entry.seed = parsed;
            push_value(entry.launch_arguments, "--seed", *value);
        }
    }

    validate_boolean_option(options, "flash-attention", entry.launch_arguments, errors, id);
    validate_boolean_option(options, "block-kv-cache", entry.launch_arguments, errors, id);
    validate_boolean_option(options, "pin-resident-arena", entry.launch_arguments, errors, id);
    validate_boolean_option(options, "device-resident-runtime", entry.launch_arguments, errors, id);
    validate_boolean_option(options, "replan", entry.launch_arguments, errors, id);

    if (const auto value = get(options, "decode-topology")) {
        if (*value != "centralized" && *value != "rank-local-tp2") {
            errors.push_back("catalog section [" + id + "]: invalid decode-topology");
        } else {
            push_value(entry.launch_arguments, "--decode-topology", *value);
        }
    }
    if (const auto value = get(options, "prefill-page-tokens")) {
        std::uint32_t parsed = 0U;
        if (!parse_unsigned(*value, parsed) || parsed == 0U) {
            errors.push_back("catalog section [" + id + "]: invalid prefill-page-tokens");
        } else {
            push_value(entry.launch_arguments, "--prefill-page-tokens", *value);
        }
    }
    if (const auto value = get(options, "static-expert-plan")) {
        push_value(entry.launch_arguments, "--static-expert-plan",
                   resolve_path(catalog_directory, *value).string());
    }
    if (const auto value = get(options, "static-expert-bytes")) {
        push_value(entry.launch_arguments, "--static-expert-bytes", *value);
    }
    if (const auto value = get(options, "plan-cache")) {
        push_value(entry.launch_arguments, "--plan-cache",
                   resolve_path(catalog_directory, *value).string());
    }
    if (const auto value = get(options, "use-plan-cache")) {
        bool enabled = false;
        if (!parse_bool(*value, enabled)) {
            errors.push_back("catalog section [" + id + "]: use-plan-cache must be true or false");
        } else if (!enabled) {
            entry.launch_arguments.push_back("--no-plan-cache");
        }
    }
    if (const auto value = get(options, "load-on-startup")) {
        if (!parse_bool(*value, entry.load_on_startup)) {
            errors.push_back("catalog section [" + id + "]: load-on-startup must be true or false");
        }
    }
    if (const auto value = get(options, "stop-timeout")) {
        if (!parse_unsigned(*value, entry.stop_timeout_seconds) ||
            entry.stop_timeout_seconds == 0U || entry.stop_timeout_seconds > 300U) {
            errors.push_back("catalog section [" + id + "]: stop-timeout must be within [1, 300]");
        }
    }

    const auto* registration = find_model_by_cli_name(entry.model_type);
    const bool deepseek_controls = get(options, "block-kv-cache").has_value() ||
        get(options, "device-resident-runtime").has_value() ||
        get(options, "decode-topology").has_value() ||
        get(options, "prefill-page-tokens").has_value() ||
        get(options, "static-expert-plan").has_value() ||
        get(options, "static-expert-bytes").has_value();
    if (registration != nullptr && deepseek_controls &&
        !registration->accepts_deepseek_controls) {
        errors.push_back("catalog section [" + id +
                         "]: DeepSeek controls are forbidden for model-type '" +
                         entry.model_type + "'");
    }
    return entry;
}

}  // namespace

ModelCatalogResult load_model_catalog(const std::string& path) {
    ModelCatalogResult result;
    std::error_code code;
    const auto bytes = std::filesystem::file_size(path, code);
    if (code) {
        result.errors.push_back("cannot inspect model catalog " + path + ": " + code.message());
        return result;
    }
    if (bytes > kMaximumCatalogBytes) {
        result.errors.push_back("model catalog exceeds 1 MiB: " + path);
        return result;
    }
    std::ifstream input(path);
    if (!input) {
        result.errors.push_back("cannot open model catalog: " + path);
        return result;
    }

    Options global;
    std::map<std::string, Options> sections;
    std::string current;
    std::string line;
    std::size_t line_number = 0U;
    bool version_seen = false;
    while (std::getline(input, line)) {
        ++line_number;
        const auto text = trim(line);
        if (text.empty() || text.front() == '#' || text.front() == ';') continue;
        if (text.front() == '[' && text.back() == ']') {
            current = trim(std::string_view(text).substr(1U, text.size() - 2U));
            if (current != "*" && !valid_id(current)) {
                result.errors.push_back("catalog line " + std::to_string(line_number) +
                                        ": invalid model section name");
            }
            continue;
        }
        const auto separator = text.find('=');
        if (separator == std::string::npos) {
            result.errors.push_back("catalog line " + std::to_string(line_number) +
                                    ": expected key = value");
            continue;
        }
        const auto key = trim(std::string_view(text).substr(0U, separator));
        const auto value = trim(std::string_view(text).substr(separator + 1U));
        if (current.empty()) {
            if (key != "version" || value != "1" || version_seen) {
                result.errors.push_back("catalog line " + std::to_string(line_number) +
                                        ": only one top-level 'version = 1' is allowed");
            } else {
                version_seen = true;
            }
            continue;
        }
        if (!known_key(key)) {
            result.errors.push_back("catalog line " + std::to_string(line_number) +
                                    ": unknown option '" + key + "'");
            continue;
        }
        if (current == "*" && model_only_key(key)) {
            result.errors.push_back("catalog line " + std::to_string(line_number) +
                                    ": option '" + key + "' is not allowed in [*]");
            continue;
        }
        auto& destination = current == "*" ? global : sections[current];
        if (!destination.emplace(key, value).second) {
            result.errors.push_back("catalog line " + std::to_string(line_number) +
                                    ": duplicate option '" + key + "'");
        }
    }
    if (!version_seen) result.errors.push_back("model catalog must declare 'version = 1'");
    if (sections.empty()) result.errors.push_back("model catalog contains no model sections");
    if (!result.errors.empty()) return result;

    const auto directory = std::filesystem::absolute(path).parent_path();
    for (const auto& [id, local] : sections) {
        Options merged = global;
        for (const auto& [key, value] : local) merged[key] = value;
        result.value.models.push_back(build_entry(id, merged, directory, result.errors));
    }
    return result;
}

}  // namespace strata
