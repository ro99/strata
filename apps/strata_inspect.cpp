#include "strata/glm_manifest.hpp"
#include "strata/compressed_tensors.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/safetensors.hpp"

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void usage() {
    std::cerr << "usage: strata-inspect --model DIR [--headers] [--allow-incomplete] [--json]\n"
                 "       strata-inspect --index FILE [--json]\n"
                 "       strata-inspect --shard FILE [--tensor NAME] [--json]\n"
                 "       strata-inspect --shard FILE --module BASE --bits 4|8\n"
                 "                      (--group-size N|--channel) [--cuda DEVICE]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string index_path;
    std::string model_directory;
    std::string shard_path;
    std::string tensor_name;
    std::string module_name;
    std::uint32_t module_bits = 0;
    std::uint32_t module_group_size = 0;
    int cuda_device = -1;
    bool module_channel = false;
    bool json = false;
    bool headers = false;
    bool allow_incomplete = false;
    bool require_read_only = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--json") {
            json = true;
        } else if (argument == "--headers") {
            headers = true;
        } else if (argument == "--allow-incomplete") {
            allow_incomplete = true;
        } else if (argument == "--require-read-only") {
            require_read_only = true;
        } else if (argument == "--shard" && index + 1 < argc) {
            shard_path = argv[++index];
        } else if (argument == "--tensor" && index + 1 < argc) {
            tensor_name = argv[++index];
        } else if (argument == "--module" && index + 1 < argc) {
            module_name = argv[++index];
        } else if (argument == "--bits" && index + 1 < argc) {
            module_bits = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--group-size" && index + 1 < argc) {
            module_group_size = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--channel") {
            module_channel = true;
        } else if (argument == "--cuda" && index + 1 < argc) {
            cuda_device = std::stoi(argv[++index]);
        } else if ((argument == "--model" || argument == "--index") && index + 1 < argc) {
            const bool model = argument == "--model";
            const std::string value = argv[++index];
            if (model) {
                model_directory = value;
                if (index_path.empty()) index_path = value + "/model.safetensors.index.json";
            } else {
                index_path = value;
            }
        } else {
            usage();
            return 2;
        }
    }
    if (!shard_path.empty()) {
        if (!index_path.empty()) {
            usage();
            return 2;
        }
        const auto shard = strata::load_safetensors_shard(shard_path);
        if (!shard.ok()) {
            for (const auto& error : shard.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        if (!tensor_name.empty()) {
            const auto found = std::lower_bound(
                shard.value.tensors.begin(), shard.value.tensors.end(), tensor_name,
                [](const auto& tensor, const std::string& name) { return tensor.name < name; });
            if (found == shard.value.tensors.end() || found->name != tensor_name) {
                std::cerr << "error: tensor not found: " << tensor_name << '\n';
                return 1;
            }
            std::cout << "status=ok\n"
                      << "name=" << found->name << '\n'
                      << "dtype=" << strata::to_string(found->dtype) << '\n'
                      << "absolute_begin=" << found->absolute_begin << '\n'
                      << "bytes=" << found->bytes() << '\n'
                      << "shape=";
            for (std::size_t dimension = 0; dimension < found->shape.size(); ++dimension) {
                if (dimension != 0U) std::cout << ',';
                std::cout << found->shape[dimension];
            }
            std::cout << '\n';
            if (found->dtype == strata::SafetensorsDtype::I64 && found->shape.size() == 1U &&
                found->shape[0] == 2U) {
                const auto encoded = strata::read_safetensors_tensor(shard_path, *found, 16U);
                if (!encoded.ok()) {
                    for (const auto& error : encoded.errors) std::cerr << "error: " << error << '\n';
                    return 1;
                }
                std::array<std::uint64_t, 2> logical_shape{};
                const auto decoded = strata::decode_compressed_logical_shape(encoded.value,
                                                                               logical_shape);
                if (!decoded.ok()) {
                    for (const auto& error : decoded.errors) std::cerr << "error: " << error << '\n';
                    return 1;
                }
                std::cout << "logical_shape=" << logical_shape[0] << ',' << logical_shape[1]
                          << '\n';
            }
            return 0;
        }
        if (!module_name.empty()) {
            if (module_bits != 4U && module_bits != 8U) {
                std::cerr << "error: --module requires --bits 4 or --bits 8\n";
                return 2;
            }
            if ((!module_channel && module_group_size == 0U) ||
                (module_channel && module_group_size != 0U)) {
                std::cerr << "error: use exactly one of --group-size N or --channel\n";
                return 2;
            }
            const auto find_tensor = [&](const std::string& name)
                -> const strata::SafetensorsTensor* {
                const auto found = std::lower_bound(
                    shard.value.tensors.begin(), shard.value.tensors.end(), name,
                    [](const auto& tensor, const std::string& wanted) {
                        return tensor.name < wanted;
                    });
                return found != shard.value.tensors.end() && found->name == name ? &*found
                                                                                 : nullptr;
            };
            const auto* packed = find_tensor(module_name + ".weight_packed");
            const auto* scales = find_tensor(module_name + ".weight_scale");
            const auto* shape = find_tensor(module_name + ".weight_shape");
            if (packed == nullptr || scales == nullptr || shape == nullptr) {
                std::cerr << "error: incomplete module triplet: " << module_name << '\n';
                return 1;
            }
            const auto packed_data = strata::read_safetensors_tensor(
                shard_path, *packed, 128ULL << 20U);
            const auto scale_data = strata::read_safetensors_tensor(
                shard_path, *scales, 32ULL << 20U);
            const auto shape_data = strata::read_safetensors_tensor(shard_path, *shape, 16U);
            if (!packed_data.ok() || !scale_data.ok() || !shape_data.ok()) {
                for (const auto* errors : {&packed_data.errors, &scale_data.errors,
                                           &shape_data.errors}) {
                    for (const auto& error : *errors) std::cerr << "error: " << error << '\n';
                }
                return 1;
            }
            std::array<std::uint64_t, 2> logical_shape{};
            const auto shape_result = strata::decode_compressed_logical_shape(
                shape_data.value, logical_shape);
            if (!shape_result.ok()) {
                for (const auto& error : shape_result.errors) std::cerr << "error: " << error << '\n';
                return 1;
            }
            if (packed->shape.size() != 2U || scales->shape.size() != 2U) {
                std::cerr << "error: packed module tensors must be rank two\n";
                return 1;
            }
            strata::CompressedTensorLayout layout;
            layout.logical_rows = logical_shape[0];
            layout.logical_columns = logical_shape[1];
            layout.packed_rows = packed->shape[0];
            layout.packed_columns = packed->shape[1];
            layout.scale_rows = scales->shape[0];
            layout.scale_columns = scales->shape[1];
            layout.packed_dtype = packed->dtype;
            layout.scale_dtype = scales->dtype;
            layout.shape_dtype = shape->dtype;
            const strata::QuantizedWeightSpec quantization{
                module_bits,
                module_channel ? strata::QuantizationGranularity::Channel
                               : strata::QuantizationGranularity::Group,
                module_group_size,
                true};
            std::vector<float> input(static_cast<std::size_t>(layout.logical_columns));
            for (std::size_t column = 0; column < input.size(); ++column) {
                input[column] = static_cast<float>(static_cast<int>(column % 31U) - 15) /
                                16.0F;
            }
            std::vector<float> output(static_cast<std::size_t>(layout.logical_rows));
            const auto matvec = strata::compressed_tensor_matvec_f32(
                output, input, packed_data.value, scale_data.value, layout, quantization);
            if (!matvec.ok()) {
                for (const auto& error : matvec.errors) std::cerr << "error: " << error << '\n';
                return 1;
            }
            std::vector<float> cuda_output;
            float cuda_maximum_absolute_error = 0.0F;
            if (cuda_device >= 0) {
                strata::CudaBackend backend;
                const std::array<int, 1> devices{cuda_device};
                const auto initialized = backend.initialize(devices);
                if (!initialized.ok()) {
                    for (const auto& error : initialized.errors) {
                        std::cerr << "error: " << error << '\n';
                    }
                    return 1;
                }
                strata::CudaWeightDescriptor descriptor;
                descriptor.encoding = module_bits == 4U
                                          ? strata::CudaWeightEncoding::OffsetPackedInt4
                                          : strata::CudaWeightEncoding::OffsetPackedInt8;
                descriptor.dtype = packed->dtype;
                descriptor.rows = layout.logical_rows;
                descriptor.columns = layout.logical_columns;
                descriptor.packed_columns = layout.packed_columns;
                descriptor.scale_columns = layout.scale_columns;
                descriptor.group_size = module_channel ? 0U : module_group_size;
                strata::CudaWeight weight;
                const auto uploaded = backend.upload(cuda_device, descriptor,
                                                     packed_data.value, scale_data.value,
                                                     weight);
                if (!uploaded.ok()) {
                    for (const auto& error : uploaded.errors) std::cerr << "error: " << error << '\n';
                    return 1;
                }
                cuda_output.resize(output.size());
                const auto executed = backend.matmul(weight, input, 1U, cuda_output);
                if (!executed.ok()) {
                    for (const auto& error : executed.errors) std::cerr << "error: " << error << '\n';
                    return 1;
                }
                for (std::size_t index = 0; index < output.size(); ++index) {
                    cuda_maximum_absolute_error = std::max(
                        cuda_maximum_absolute_error,
                        std::abs(cuda_output[index] - output[index]));
                }
            }
            double sum = 0.0;
            auto minimum = std::numeric_limits<float>::infinity();
            auto maximum = -std::numeric_limits<float>::infinity();
            std::uint64_t digest = 1469598103934665603ULL;
            for (const auto value : output) {
                sum += static_cast<double>(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                const auto bits = std::bit_cast<std::uint32_t>(value);
                for (std::uint32_t byte = 0; byte < 4U; ++byte) {
                    digest ^= (bits >> (byte * 8U)) & 0xFFU;
                    digest *= 1099511628211ULL;
                }
            }
            std::cout << std::setprecision(10)
                      << "status=ok\n"
                      << "module=" << module_name << '\n'
                      << "logical_shape=" << layout.logical_rows << ','
                      << layout.logical_columns << '\n'
                      << "output_sum=" << sum << '\n'
                      << "output_min=" << minimum << '\n'
                      << "output_max=" << maximum << '\n'
                      << "output_fnv1a64=" << digest << '\n'
                      << "cuda_device=" << cuda_device << '\n'
                      << "cuda_maximum_absolute_error=" << cuda_maximum_absolute_error << '\n'
                      << "output_head=";
            const auto head = std::min<std::size_t>(8U, output.size());
            for (std::size_t index = 0; index < head; ++index) {
                if (index != 0U) std::cout << ',';
                std::cout << output[index];
            }
            std::cout << '\n';
            return 0;
        }
        std::uint64_t payload_bytes = 0;
        for (const auto& tensor : shard.value.tensors) payload_bytes += tensor.bytes();
        if (json) {
            std::cout << "{\n"
                      << "  \"status\": \"ok\",\n"
                      << "  \"tensor_count\": " << shard.value.tensors.size() << ",\n"
                      << "  \"file_bytes\": " << shard.value.file_size << ",\n"
                      << "  \"header_bytes\": " << shard.value.header_size << ",\n"
                      << "  \"payload_bytes\": " << payload_bytes << "\n"
                      << "}\n";
        } else {
            std::cout << "status=ok\n"
                      << "tensor_count=" << shard.value.tensors.size() << '\n'
                      << "file_bytes=" << shard.value.file_size << '\n'
                      << "header_bytes=" << shard.value.header_size << '\n'
                      << "payload_bytes=" << payload_bytes << '\n';
        }
        return 0;
    }
    if (index_path.empty()) {
        usage();
        return 2;
    }

    const auto text = strata::load_bounded_text_file(index_path, 128ULL << 20U);
    if (!text.ok()) {
        for (const auto& error : text.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    auto parsed = strata::parse_safetensors_index(text.value);
    if (!parsed.ok()) {
        for (const auto& error : parsed.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    auto result = strata::build_quanttrio_glm52_index_manifest(std::move(parsed.value));
    if (!result.ok()) {
        for (const auto& error : result.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }

    if (headers) {
        if (model_directory.empty()) {
            std::cerr << "error: --headers requires --model DIR\n";
            return 2;
        }
        strata::GlmCheckpointOptions options;
        options.require_all_shards = !allow_incomplete;
        options.require_read_only = require_read_only;
        result = strata::validate_quanttrio_glm52_checkpoint(
            model_directory, std::move(result.manifest), options);
        if (!result.ok()) {
            for (const auto& error : result.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
    }

    const auto& manifest = result.manifest;
    if (json) {
        std::cout << "{\n"
                  << "  \"status\": \"ok\",\n"
                  << "  \"tensor_count\": " << manifest.tensors.size() << ",\n"
                  << "  \"indexed_tensor_bytes\": " << manifest.indexed_tensor_bytes << ",\n"
                  << "  \"shards\": " << manifest.shards.size() << ",\n"
                  << "  \"quantized_modules\": " << manifest.quantized_modules << ",\n"
                  << "  \"int4_group128_modules\": " << manifest.int4_modules << ",\n"
                  << "  \"int8_group128_modules\": " << manifest.int8_group_modules << ",\n"
                  << "  \"int8_channel_modules\": " << manifest.int8_channel_modules << ",\n"
                  << "  \"scanned_shards\": " << manifest.scanned_shards << ",\n"
                  << "  \"resolved_tensors\": " << manifest.resolved_tensors << ",\n"
                  << "  \"validated_layouts\": " << manifest.validated_layouts << ",\n"
                  << "  \"shard_file_bytes\": " << manifest.shard_file_bytes << ",\n"
                  << "  \"tensor_payload_bytes\": " << manifest.tensor_payload_bytes << "\n"
                  << "}\n";
    } else {
        std::cout << "status=ok\n"
                  << "tensor_count=" << manifest.tensors.size() << '\n'
                  << "indexed_tensor_bytes=" << manifest.indexed_tensor_bytes << '\n'
                  << "shards=" << manifest.shards.size() << '\n'
                  << "quantized_modules=" << manifest.quantized_modules << '\n'
                  << "int4_group128_modules=" << manifest.int4_modules << '\n'
                  << "int8_group128_modules=" << manifest.int8_group_modules << '\n'
                  << "int8_channel_modules=" << manifest.int8_channel_modules << '\n'
                  << "scanned_shards=" << manifest.scanned_shards << '\n'
                  << "resolved_tensors=" << manifest.resolved_tensors << '\n'
                  << "validated_layouts=" << manifest.validated_layouts << '\n'
                  << "shard_file_bytes=" << manifest.shard_file_bytes << '\n'
                  << "tensor_payload_bytes=" << manifest.tensor_payload_bytes << '\n';
        for (std::size_t role = 0; role < manifest.role_counts.size(); ++role) {
            std::cout << "role."
                      << strata::to_string(static_cast<strata::GlmTensorRole>(role)) << '='
                      << manifest.role_counts[role] << '\n';
        }
    }
    return 0;
}
