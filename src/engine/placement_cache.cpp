#include "strata/placement.hpp"

#include "../platform/json_cursor.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>

namespace strata {
namespace {

constexpr std::uint64_t kMaximumPlanBytes = 4ULL << 20U;

void escape(std::ostringstream& output, std::string_view text) {
    output << '"';
    for (const unsigned char value : text) {
        switch (value) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (value < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned int>(value)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(value);
                }
        }
    }
    output << '"';
}

template <typename T>
void write_array(std::ostringstream& output, const std::vector<T>& values) {
    output << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << values[index];
    }
    output << ']';
}

[[nodiscard]] std::vector<std::uint64_t> read_uint64_array(
    detail::JsonCursor& cursor) {
    std::vector<std::uint64_t> values;
    cursor.expect('[');
    if (cursor.consume(']')) return values;
    while (true) {
        values.push_back(cursor.parse_uint64());
        if (cursor.consume(']')) break;
        cursor.expect(',');
    }
    return values;
}

[[nodiscard]] std::vector<std::size_t> read_size_array(detail::JsonCursor& cursor) {
    std::vector<std::size_t> values;
    for (const auto value : read_uint64_array(cursor)) {
        values.push_back(static_cast<std::size_t>(value));
    }
    return values;
}

[[nodiscard]] std::vector<int> read_int_array(detail::JsonCursor& cursor) {
    std::vector<int> values;
    for (const auto value : read_uint64_array(cursor)) {
        values.push_back(static_cast<int>(value));
    }
    return values;
}

[[nodiscard]] bool parse_placement_class(std::string_view text,
                                         PlacementClass& component) noexcept {
    for (std::uint8_t index = 0U;
         index < static_cast<std::uint8_t>(PlacementClass::Count); ++index) {
        const auto candidate = static_cast<PlacementClass>(index);
        if (to_string(candidate) == text) {
            component = candidate;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool parse_placement_tier(std::string_view text,
                                        PlacementTier& tier) noexcept {
    if (text == "device") { tier = PlacementTier::Device; return true; }
    if (text == "host") { tier = PlacementTier::Host; return true; }
    if (text == "storage") { tier = PlacementTier::Storage; return true; }
    return false;
}

// Each object member is dispatched by name; unknown members are skipped so a
// plan written by a newer build with the same version still parses.
template <typename Handler>
void read_object(detail::JsonCursor& cursor, Handler handler) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    while (true) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (!handler(key)) cursor.skip_value();
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
}

[[nodiscard]] std::string sanitize(std::string_view text) {
    std::string output;
    output.reserve(text.size());
    for (const char value : text) {
        const bool safe = (value >= 'a' && value <= 'z') ||
                          (value >= 'A' && value <= 'Z') ||
                          (value >= '0' && value <= '9') || value == '-' ||
                          value == '_' || value == '.';
        output.push_back(safe ? value : '-');
    }
    if (output.empty() || output == "." || output == "..") output = "model";
    if (output.size() > 64U) output.resize(64U);
    return output;
}

// Digest of everything about a request that changes the answer. Two runs whose
// digests match may share a plan; anything else gets its own file.
[[nodiscard]] std::string request_digest(const PlacementRequest& request) {
    std::ostringstream text;
    text << to_string(request.model) << '|'
         << std::filesystem::absolute(request.model_directory).lexically_normal().string()
         << '|' << std::fixed << std::setprecision(6) << request.vram_cache_fraction
         << '|' << request.maximum_context_tokens << '|'
         << (request.flash_attention ? 1 : 0) << '|'
         << (request.block_kv_cache ? 1 : 0) << '|';
    for (const int device : request.devices) text << device << ',';
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset;
    for (const unsigned char value : text.str()) {
        hash ^= value;
        hash *= prime;
    }
    std::array<char, 24> digest{};
    std::snprintf(digest.data(), digest.size(), "%016llx",
                  static_cast<unsigned long long>(hash));
    return digest.data();
}

}  // namespace

std::string encode_placement_plan(const PlacementPlan& plan) {
    std::ostringstream output;
    output << "{\n  \"version\": " << plan.version
           << ",\n  \"model_type\": ";
    escape(output, to_string(plan.request.model));
    output << ",\n  \"model_name\": ";
    escape(output, plan.model_name);
    output << ",\n  \"model_identity\": ";
    escape(output, plan.model_identity);
    output << ",\n  \"model_directory\": ";
    escape(output, plan.request.model_directory);
    output << ",\n  \"request\": {\"devices\": ";
    write_array(output, plan.request.devices);
    output << ", \"vram_cache_fraction\": " << std::fixed << std::setprecision(6)
           << plan.request.vram_cache_fraction
           << ", \"maximum_context_tokens\": " << plan.request.maximum_context_tokens
           << ", \"flash_attention\": "
           << (plan.request.flash_attention ? "true" : "false")
           << ", \"block_kv_cache\": "
           << (plan.request.block_kv_cache ? "true" : "false")
           << "}";
    output << ",\n  \"hardware\": {\"host_total_bytes\": "
           << plan.hardware.host_total_bytes << ", \"host_available_bytes\": "
           << plan.hardware.host_available_bytes << ", \"storage\": {\"path\": ";
    escape(output, plan.hardware.storage.path);
    output << ", \"device\": ";
    escape(output, plan.hardware.storage.device);
    output << ", \"disk\": ";
    escape(output, plan.hardware.storage.disk);
    output << ", \"nvme\": " << (plan.hardware.storage.nvme ? "true" : "false")
           << ", \"rotational\": "
           << (plan.hardware.storage.rotational ? "true" : "false")
           << ", \"memory_backed\": "
           << (plan.hardware.storage.memory_backed ? "true" : "false")
           << ", \"resolved\": "
           << (plan.hardware.storage.resolved ? "true" : "false") << '}';
    output << ", \"devices\": [";
    for (std::size_t index = 0U; index < plan.hardware.devices.size(); ++index) {
        const auto& device = plan.hardware.devices[index];
        if (index != 0U) output << ", ";
        output << "{\"id\": " << device.id << ", \"name\": ";
        escape(output, device.name);
        output << ", \"total_bytes\": " << device.total_bytes
               << ", \"free_bytes\": " << device.free_bytes << '}';
    }
    output << "]}";
    output << ",\n  \"components\": [";
    for (std::size_t index = 0U; index < plan.components.size(); ++index) {
        const auto& component = plan.components[index];
        output << (index == 0U ? "\n    " : ",\n    ") << "{\"component\": ";
        escape(output, to_string(component.component));
        output << ", \"tier\": ";
        escape(output, to_string(component.tier));
        output << ", \"bytes\": " << component.bytes
               << ", \"decode_read_bytes\": " << component.decode_read_bytes
               << ", \"device_bytes\": ";
        write_array(output, component.device_bytes);
        output << '}';
    }
    output << (plan.components.empty() ? "]" : "\n  ]");
    output << ",\n  \"device_budget_bytes\": ";
    write_array(output, plan.device_budget_bytes);
    output << ",\n  \"device_resident_bytes\": ";
    write_array(output, plan.device_resident_bytes);
    output << ",\n  \"device_expert_cache_bytes\": ";
    write_array(output, plan.device_expert_cache_bytes);
    output << ",\n  \"layer_device\": ";
    write_array(output, plan.layer_device);
    output << ",\n  \"weighted_schedule\": ";
    write_array(output, plan.weighted_schedule);
    output << ",\n  \"host_resident_bytes\": " << plan.host_resident_bytes
           << ",\n  \"storage_resident_bytes\": " << plan.storage_resident_bytes
           << ",\n  \"decode_device_read_bytes\": " << plan.decode_device_read_bytes
           << ",\n  \"decode_host_to_device_bytes\": "
           << plan.decode_host_to_device_bytes
           << ",\n  \"decode_storage_read_bytes\": "
           << plan.decode_storage_read_bytes
           << ",\n  \"maximum_context_tokens_that_fit\": "
           << plan.maximum_context_tokens_that_fit
           << ",\n  \"cross_device_activation_hops\": "
           << plan.cross_device_activation_hops
           << ",\n  \"prescriptive\": " << (plan.prescriptive ? "true" : "false")
           << ",\n  \"fits\": " << (plan.fits ? "true" : "false")
           << ",\n  \"io_dependent\": " << (plan.io_dependent ? "true" : "false")
           << ",\n  \"notes\": [";
    for (std::size_t index = 0U; index < plan.notes.size(); ++index) {
        if (index != 0U) output << ", ";
        escape(output, plan.notes[index]);
    }
    output << "]\n}\n";
    return output.str();
}

PlacementPlanResult decode_placement_plan(std::string_view text) {
    PlacementPlanResult result;
    auto& plan = result.value;
    try {
        detail::JsonCursor cursor(text);
        read_object(cursor, [&](const std::string& key) {
            if (key == "version") {
                plan.version = static_cast<std::uint32_t>(cursor.parse_uint64());
            } else if (key == "model_type") {
                if (!parse_placement_model(cursor.parse_string(),
                                           plan.request.model)) {
                    result.errors.emplace_back("placement plan names an unknown model");
                }
            } else if (key == "model_name") {
                plan.model_name = cursor.parse_string();
            } else if (key == "model_identity") {
                plan.model_identity = cursor.parse_string();
            } else if (key == "model_directory") {
                plan.request.model_directory = cursor.parse_string();
            } else if (key == "request") {
                read_object(cursor, [&](const std::string& field) {
                    if (field == "devices") {
                        plan.request.devices = read_int_array(cursor);
                    } else if (field == "vram_cache_fraction") {
                        plan.request.vram_cache_fraction = cursor.parse_number();
                    } else if (field == "maximum_context_tokens") {
                        plan.request.maximum_context_tokens =
                            static_cast<std::uint32_t>(cursor.parse_uint64());
                    } else if (field == "flash_attention") {
                        plan.request.flash_attention = cursor.parse_bool();
                    } else if (field == "block_kv_cache") {
                        plan.request.block_kv_cache = cursor.parse_bool();
                    } else {
                        return false;
                    }
                    return true;
                });
            } else if (key == "hardware") {
                read_object(cursor, [&](const std::string& field) {
                    if (field == "host_total_bytes") {
                        plan.hardware.host_total_bytes = cursor.parse_uint64();
                    } else if (field == "host_available_bytes") {
                        plan.hardware.host_available_bytes = cursor.parse_uint64();
                    } else if (field == "storage") {
                        read_object(cursor, [&](const std::string& entry) {
                            auto& storage = plan.hardware.storage;
                            if (entry == "path") {
                                storage.path = cursor.parse_string();
                            } else if (entry == "device") {
                                storage.device = cursor.parse_string();
                            } else if (entry == "disk") {
                                storage.disk = cursor.parse_string();
                            } else if (entry == "nvme") {
                                storage.nvme = cursor.parse_bool();
                            } else if (entry == "rotational") {
                                storage.rotational = cursor.parse_bool();
                            } else if (entry == "memory_backed") {
                                storage.memory_backed = cursor.parse_bool();
                            } else if (entry == "resolved") {
                                storage.resolved = cursor.parse_bool();
                            } else {
                                return false;
                            }
                            return true;
                        });
                    } else if (field == "devices") {
                        cursor.expect('[');
                        if (!cursor.consume(']')) {
                            while (true) {
                                PlacementDevice device;
                                read_object(cursor, [&](const std::string& entry) {
                                    if (entry == "id") {
                                        device.id = static_cast<int>(
                                            cursor.parse_uint64());
                                    } else if (entry == "name") {
                                        device.name = cursor.parse_string();
                                    } else if (entry == "total_bytes") {
                                        device.total_bytes = cursor.parse_uint64();
                                    } else if (entry == "free_bytes") {
                                        device.free_bytes = cursor.parse_uint64();
                                    } else {
                                        return false;
                                    }
                                    return true;
                                });
                                plan.hardware.devices.push_back(std::move(device));
                                if (cursor.consume(']')) break;
                                cursor.expect(',');
                            }
                        }
                    } else {
                        return false;
                    }
                    return true;
                });
            } else if (key == "components") {
                cursor.expect('[');
                if (!cursor.consume(']')) {
                    while (true) {
                        PlacementComponentTotals component;
                        read_object(cursor, [&](const std::string& field) {
                            if (field == "component") {
                                if (!parse_placement_class(cursor.parse_string(),
                                                           component.component)) {
                                    result.errors.emplace_back(
                                        "placement plan names an unknown component");
                                }
                            } else if (field == "tier") {
                                if (!parse_placement_tier(cursor.parse_string(),
                                                          component.tier)) {
                                    result.errors.emplace_back(
                                        "placement plan names an unknown tier");
                                }
                            } else if (field == "bytes") {
                                component.bytes = cursor.parse_uint64();
                            } else if (field == "decode_read_bytes") {
                                component.decode_read_bytes = cursor.parse_uint64();
                            } else if (field == "device_bytes") {
                                component.device_bytes = read_uint64_array(cursor);
                            } else {
                                return false;
                            }
                            return true;
                        });
                        plan.components.push_back(std::move(component));
                        if (cursor.consume(']')) break;
                        cursor.expect(',');
                    }
                }
            } else if (key == "device_budget_bytes") {
                plan.device_budget_bytes = read_uint64_array(cursor);
            } else if (key == "device_resident_bytes") {
                plan.device_resident_bytes = read_uint64_array(cursor);
            } else if (key == "device_expert_cache_bytes") {
                plan.device_expert_cache_bytes = read_uint64_array(cursor);
            } else if (key == "layer_device") {
                plan.layer_device = read_size_array(cursor);
            } else if (key == "weighted_schedule") {
                plan.weighted_schedule = read_size_array(cursor);
            } else if (key == "host_resident_bytes") {
                plan.host_resident_bytes = cursor.parse_uint64();
            } else if (key == "storage_resident_bytes") {
                plan.storage_resident_bytes = cursor.parse_uint64();
            } else if (key == "decode_device_read_bytes") {
                plan.decode_device_read_bytes = cursor.parse_uint64();
            } else if (key == "decode_host_to_device_bytes") {
                plan.decode_host_to_device_bytes = cursor.parse_uint64();
            } else if (key == "decode_storage_read_bytes") {
                plan.decode_storage_read_bytes = cursor.parse_uint64();
            } else if (key == "maximum_context_tokens_that_fit") {
                plan.maximum_context_tokens_that_fit =
                    static_cast<std::uint32_t>(cursor.parse_uint64());
            } else if (key == "cross_device_activation_hops") {
                plan.cross_device_activation_hops =
                    static_cast<std::uint32_t>(cursor.parse_uint64());
            } else if (key == "prescriptive") {
                plan.prescriptive = cursor.parse_bool();
            } else if (key == "fits") {
                plan.fits = cursor.parse_bool();
            } else if (key == "io_dependent") {
                plan.io_dependent = cursor.parse_bool();
            } else if (key == "notes") {
                cursor.expect('[');
                if (!cursor.consume(']')) {
                    while (true) {
                        plan.notes.push_back(cursor.parse_string());
                        if (cursor.consume(']')) break;
                        cursor.expect(',');
                    }
                }
            } else {
                return false;
            }
            return true;
        });
    } catch (const detail::JsonError& error) {
        result.errors.emplace_back(std::string("placement plan is not valid JSON: ") +
                                   error.what());
        return result;
    }
    if (!result.ok()) return result;
    if (plan.version != kPlacementPlanVersion) {
        result.errors.emplace_back("placement plan schema version is not current");
    }
    return result;
}

std::string placement_cache_directory(std::string_view override_directory) {
    if (!override_directory.empty()) return std::string(override_directory);
    if (const char* explicit_path = std::getenv("STRATA_PLAN_CACHE");
        explicit_path != nullptr && *explicit_path != '\0') {
        return explicit_path;
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME");
        xdg != nullptr && *xdg != '\0') {
        return (std::filesystem::path(xdg) / "strata" / "plans").string();
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return (std::filesystem::path(home) / ".cache" / "strata" / "plans").string();
    }
    return (std::filesystem::temp_directory_path() / "strata-plans").string();
}

std::string placement_plan_filename(const PlacementRequest& request,
                                    std::string_view model_identity) {
    const auto directory =
        std::filesystem::path(request.model_directory).filename().string();
    return std::string(to_string(request.model)) + '-' + sanitize(directory) + '-' +
           sanitize(model_identity) + '-' + request_digest(request) + ".plan.json";
}

ValidationResult store_placement_plan(const std::string& directory,
                                      const PlacementPlan& plan) {
    ValidationResult result;
    std::error_code code;
    std::filesystem::create_directories(directory, code);
    if (code) {
        result.errors.push_back("cannot create placement cache directory " +
                                directory + ": " + code.message());
        return result;
    }
    const auto path = std::filesystem::path(directory) /
        placement_plan_filename(plan.request, plan.model_identity);
    // Write and rename so a concurrent reader never sees a partial plan.
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            result.errors.push_back("cannot write placement plan " + temporary);
            return result;
        }
        output << encode_placement_plan(plan);
        if (!output) {
            result.errors.push_back("cannot write placement plan " + temporary);
            return result;
        }
    }
    std::filesystem::rename(temporary, path, code);
    if (code) {
        std::filesystem::remove(temporary, code);
        result.errors.push_back("cannot publish placement plan " + path.string());
    }
    return result;
}

ParseResult<PlacementResolution> resolve_placement_plan(
    const PlacementRequest& request, std::string_view cache_directory,
    bool use_cache, bool refresh) {
    ParseResult<PlacementResolution> result;
    auto hardware = probe_placement_hardware(request.devices,
                                             request.model_directory);
    if (!hardware.ok()) {
        result.errors = std::move(hardware.errors);
        return result;
    }
    auto identity = probe_model_identity(request.model_directory);
    if (!identity.ok()) {
        result.errors = std::move(identity.errors);
        return result;
    }
    const auto directory = placement_cache_directory(cache_directory);
    result.value.cache_path =
        (std::filesystem::path(directory) /
         placement_plan_filename(request, identity.value)).string();
    if (use_cache && !refresh) {
        auto cached = load_placement_plan(directory, request, hardware.value,
                                          identity.value);
        if (!cached.ok()) {
            result.errors = std::move(cached.errors);
            return result;
        }
        if (cached.value.version == kPlacementPlanVersion) {
            // Free VRAM moves between runs; the cached budget does not. Re-probe
            // and refresh rather than place against a stale reading.
            cached.value.hardware.host_available_bytes =
                hardware.value.host_available_bytes;
            for (std::size_t slot = 0U; slot < hardware.value.devices.size();
                 ++slot) {
                cached.value.hardware.devices[slot].free_bytes =
                    hardware.value.devices[slot].free_bytes;
            }
            result.value.plan = std::move(cached.value);
            result.value.from_cache = true;
            return result;
        }
    }
    auto planned = plan_model_placement(request, hardware.value);
    if (!planned.ok()) {
        result.errors = std::move(planned.errors);
        return result;
    }
    result.value.plan = std::move(planned.value);
    if (use_cache) {
        const auto stored = store_placement_plan(directory, result.value.plan);
        result.value.stored = stored.ok();
        for (const auto& error : stored.errors) {
            result.value.plan.notes.push_back(error);
        }
    }
    return result;
}

PlacementPlanResult load_placement_plan(const std::string& directory,
                                        const PlacementRequest& request,
                                        const PlacementHardware& hardware,
                                        std::string_view model_identity) {
    PlacementPlanResult result;
    result.value.version = 0U;
    const auto path = std::filesystem::path(directory) /
        placement_plan_filename(request, model_identity);
    std::error_code code;
    if (!std::filesystem::exists(path, code) || code) return result;
    const auto size = std::filesystem::file_size(path, code);
    if (code || size == 0U || size > kMaximumPlanBytes) return result;
    std::ifstream input(path, std::ios::binary);
    if (!input) return result;
    std::string text(static_cast<std::size_t>(size), '\0');
    input.read(text.data(), static_cast<std::streamsize>(size));
    if (!input) return result;

    auto decoded = decode_placement_plan(text);
    // A cached plan that no longer parses is a stale artifact, not a failure:
    // report the miss and let the caller compute a fresh one.
    if (!decoded.ok()) return result;
    const auto& plan = decoded.value;
    if (plan.model_identity != model_identity ||
        plan.request.model != request.model ||
        plan.request.devices != request.devices ||
        plan.request.maximum_context_tokens != request.maximum_context_tokens ||
        plan.request.flash_attention != request.flash_attention ||
        plan.request.block_kv_cache != request.block_kv_cache ||
        plan.hardware.storage.disk != hardware.storage.disk ||
        plan.hardware.storage.resolved != hardware.storage.resolved ||
        plan.hardware.devices.size() != hardware.devices.size()) {
        return result;
    }
    for (std::size_t slot = 0U; slot < hardware.devices.size(); ++slot) {
        if (plan.hardware.devices[slot].id != hardware.devices[slot].id ||
            plan.hardware.devices[slot].total_bytes !=
                hardware.devices[slot].total_bytes) {
            return result;
        }
    }
    return decoded;
}

}  // namespace strata
