#include "strata/models/gemma4/gemma4_image.hpp"

#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/numerics.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace strata {
namespace {

constexpr std::size_t kMaximumEncodedBytes = 16U << 20U;
constexpr std::size_t kMaximumConvertOutput = 8U << 20U;

ParseResult<std::string> run_convert(
    std::string_view encoded, std::string_view mime_type,
    std::span<const std::string> arguments, std::size_t output_ceiling) {
    ParseResult<std::string> result;
    if (encoded.empty() || encoded.size() > kMaximumEncodedBytes) {
        result.errors.emplace_back("Gemma 4 image exceeds the 16 MiB input ceiling");
        return result;
    }
    const char* coder = mime_type == "image/png" ? "png"
        : mime_type == "image/jpeg" ? "jpeg"
        : mime_type == "image/webp" ? "webp" : nullptr;
    if (coder == nullptr) {
        result.errors.emplace_back("Gemma 4 image must be PNG, JPEG, or WebP");
        return result;
    }
    std::array<char, 34> path{"/tmp/strata-gemma4-image-XXXXXX"};
    const int file = mkstemp(path.data());
    if (file < 0) {
        result.errors.emplace_back("cannot create bounded Gemma 4 image input");
        return result;
    }
    const auto cleanup = [&]() {
        close(file);
        unlink(path.data());
    };
    std::size_t written = 0U;
    while (written < encoded.size()) {
        const auto count = write(file, encoded.data() + written,
                                 encoded.size() - written);
        if (count <= 0) {
            result.errors.emplace_back("cannot stage Gemma 4 image input");
            cleanup();
            return result;
        }
        written += static_cast<std::size_t>(count);
    }
    if (lseek(file, 0, SEEK_SET) < 0) {
        result.errors.emplace_back("cannot rewind Gemma 4 image input");
        cleanup();
        return result;
    }
    std::array<int, 2> output_pipe{};
    if (pipe(output_pipe.data()) != 0) {
        result.errors.emplace_back("cannot create Gemma 4 image decoder pipe");
        cleanup();
        return result;
    }
    const auto child = fork();
    if (child == 0) {
        dup2(output_pipe[1], STDOUT_FILENO);
        close(output_pipe[0]);
        close(output_pipe[1]);
        close(file);
        const std::string input = std::string(coder) + ':' + path.data();
        std::vector<std::string> owned{
            "/usr/bin/convert", "-limit", "memory", "512MiB",
            "-limit", "map", "1GiB", input};
        owned.insert(owned.end(), arguments.begin(), arguments.end());
        std::vector<char*> argv;
        argv.reserve(owned.size() + 1U);
        for (auto& value : owned) argv.push_back(value.data());
        argv.push_back(nullptr);
        execv(argv.front(), argv.data());
        _exit(127);
    }
    close(output_pipe[1]);
    if (child < 0) {
        close(output_pipe[0]);
        result.errors.emplace_back("cannot start the Gemma 4 image decoder");
        cleanup();
        return result;
    }
    std::array<char, 16U << 10U> buffer{};
    for (;;) {
        const auto count = read(output_pipe[0], buffer.data(), buffer.size());
        if (count == 0) break;
        if (count < 0) {
            if (errno == EINTR) continue;
            result.errors.emplace_back("cannot read Gemma 4 image decoder output");
            break;
        }
        if (static_cast<std::size_t>(count) > output_ceiling -
                std::min(output_ceiling, result.value.size())) {
            result.errors.emplace_back("Gemma 4 image decoder output exceeds its ceiling");
            break;
        }
        result.value.append(buffer.data(), static_cast<std::size_t>(count));
    }
    close(output_pipe[0]);
    if (!result.errors.empty()) kill(child, SIGKILL);
    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    cleanup();
    if (!result.errors.empty()) return result;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        result.errors.emplace_back(
            "ImageMagick could not decode the bounded Gemma 4 image");
        result.value.clear();
    }
    return result;
}

bool ppm_token(std::string_view ppm, std::size_t& offset, std::string_view& token) {
    while (offset < ppm.size()) {
        if (ppm[offset] == '#') {
            const auto end = ppm.find('\n', offset);
            if (end == std::string_view::npos) return false;
            offset = end + 1U;
        } else if (ppm[offset] == ' ' || ppm[offset] == '\n' ||
                   ppm[offset] == '\r' || ppm[offset] == '\t') {
            ++offset;
        } else break;
    }
    const auto begin = offset;
    while (offset < ppm.size() && ppm[offset] != ' ' && ppm[offset] != '\n' &&
           ppm[offset] != '\r' && ppm[offset] != '\t' && ppm[offset] != '#') {
        ++offset;
    }
    token = ppm.substr(begin, offset - begin);
    return !token.empty();
}

}  // namespace

ParseResult<std::pair<std::uint32_t, std::uint32_t>> gemma4_target_image_size(
    std::uint32_t width, std::uint32_t height) {
    ParseResult<std::pair<std::uint32_t, std::uint32_t>> result;
    if (width == 0U || height == 0U) {
        result.errors.emplace_back("Gemma 4 image dimensions must be positive");
        return result;
    }
    constexpr std::uint64_t max_patches =
        static_cast<std::uint64_t>(kGemma4ExecutionContract.default_image_tokens) *
        kGemma4ExecutionContract.vision_pooling_kernel *
        kGemma4ExecutionContract.vision_pooling_kernel;
    constexpr std::uint32_t side =
        kGemma4ExecutionContract.vision_patch_size *
        kGemma4ExecutionContract.vision_pooling_kernel;
    const double factor = std::sqrt(
        static_cast<double>(max_patches) *
        kGemma4ExecutionContract.vision_patch_size *
        kGemma4ExecutionContract.vision_patch_size /
        (static_cast<double>(width) * height));
    auto target_width = static_cast<std::uint64_t>(
        std::floor(factor * width / side)) * side;
    auto target_height = static_cast<std::uint64_t>(
        std::floor(factor * height / side)) * side;
    const auto maximum_side = max_patches /
        (kGemma4ExecutionContract.vision_pooling_kernel *
         kGemma4ExecutionContract.vision_pooling_kernel) * side;
    if (target_width == 0U && target_height == 0U) {
        result.errors.emplace_back("Gemma 4 image aspect ratio cannot fit one patch group");
        return result;
    }
    if (target_width == 0U) {
        target_width = side;
        target_height = std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(height) / width) * side, maximum_side);
    } else if (target_height == 0U) {
        target_height = side;
        target_width = std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(width) / height) * side, maximum_side);
    }
    const auto target_pixels = max_patches *
        kGemma4ExecutionContract.vision_patch_size *
        kGemma4ExecutionContract.vision_patch_size;
    if (target_width == 0U || target_height == 0U ||
        target_width > std::numeric_limits<std::uint32_t>::max() ||
        target_height > std::numeric_limits<std::uint32_t>::max() ||
        target_width * target_height > target_pixels) {
        result.errors.emplace_back("Gemma 4 image resize exceeds the patch budget");
        return result;
    }
    result.value = {static_cast<std::uint32_t>(target_width),
                    static_cast<std::uint32_t>(target_height)};
    return result;
}

ParseResult<Gemma4PreparedImage> prepare_gemma4_image(
    std::string_view encoded, std::string_view mime_type) {
    ParseResult<Gemma4PreparedImage> result;
    const std::array<std::string, 3> identify_args{
        "-format", "%w %h", "info:"};
    auto dimensions = run_convert(encoded, mime_type, identify_args, 128U);
    if (!dimensions.ok()) {
        result.errors = std::move(dimensions.errors);
        return result;
    }
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    const auto separator = dimensions.value.find(' ');
    if (separator == std::string::npos ||
        std::from_chars(dimensions.value.data(),
                        dimensions.value.data() + separator, width).ec != std::errc{} ||
        std::from_chars(dimensions.value.data() + separator + 1U,
                        dimensions.value.data() + dimensions.value.size(), height).ec !=
            std::errc{}) {
        result.errors.emplace_back("Gemma 4 image decoder returned invalid dimensions");
        return result;
    }
    auto target = gemma4_target_image_size(width, height);
    if (!target.ok()) {
        result.errors = std::move(target.errors);
        return result;
    }
    const auto geometry = std::to_string(target.value.first) + 'x' +
                          std::to_string(target.value.second) + '!';
    const std::array<std::string, 7> decode_args{
        "-filter", "Cubic", "-resize", geometry, "-depth", "8", "ppm:-"};
    auto decoded = run_convert(
        encoded, mime_type, decode_args, kMaximumConvertOutput);
    if (!decoded.ok()) {
        result.errors = std::move(decoded.errors);
        return result;
    }
    std::size_t offset = 0U;
    std::string_view token;
    if (!ppm_token(decoded.value, offset, token) || token != "P6" ||
        !ppm_token(decoded.value, offset, token)) {
        result.errors.emplace_back("Gemma 4 image decoder did not return RGB PPM");
        return result;
    }
    std::uint32_t decoded_width = 0U;
    if (std::from_chars(token.data(), token.data() + token.size(), decoded_width).ec !=
            std::errc{} || !ppm_token(decoded.value, offset, token)) {
        result.errors.emplace_back("Gemma 4 decoded image width is invalid");
        return result;
    }
    std::uint32_t decoded_height = 0U;
    if (std::from_chars(token.data(), token.data() + token.size(), decoded_height).ec !=
            std::errc{} || !ppm_token(decoded.value, offset, token) || token != "255" ||
        offset >= decoded.value.size()) {
        result.errors.emplace_back("Gemma 4 decoded image height or depth is invalid");
        return result;
    }
    ++offset;
    const auto pixels = static_cast<std::size_t>(decoded_width) * decoded_height * 3U;
    if (decoded_width != target.value.first || decoded_height != target.value.second ||
        decoded.value.size() - offset != pixels) {
        result.errors.emplace_back("Gemma 4 decoded image extent is invalid");
        return result;
    }
    constexpr auto patch = kGemma4ExecutionContract.vision_patch_size;
    result.value.patch_width = decoded_width / patch;
    result.value.patch_height = decoded_height / patch;
    const auto patch_rows = static_cast<std::size_t>(result.value.patch_width) *
                            result.value.patch_height;
    result.value.soft_tokens = static_cast<std::uint32_t>(
        patch_rows /
        (kGemma4ExecutionContract.vision_pooling_kernel *
         kGemma4ExecutionContract.vision_pooling_kernel));
    result.value.patches.resize(patch_rows * patch * patch * 3U);
    result.value.positions.resize(patch_rows * 2U);
    for (std::uint32_t py = 0U; py < result.value.patch_height; ++py) {
        for (std::uint32_t px = 0U; px < result.value.patch_width; ++px) {
            const auto row = static_cast<std::size_t>(py) * result.value.patch_width + px;
            result.value.positions[row * 2U] = static_cast<std::int32_t>(px);
            result.value.positions[row * 2U + 1U] = static_cast<std::int32_t>(py);
            auto destination = result.value.patches.begin() +
                static_cast<std::ptrdiff_t>(row * patch * patch * 3U);
            for (std::uint32_t y = 0U; y < patch; ++y) {
                for (std::uint32_t x = 0U; x < patch; ++x) {
                    const auto source = offset +
                        ((static_cast<std::size_t>(py * patch + y) * decoded_width +
                          px * patch + x) * 3U);
                    for (std::uint32_t channel = 0U; channel < 3U; ++channel) {
                        *destination++ = bf16_round_f32(
                            2.0F * (static_cast<unsigned char>(
                                decoded.value[source + channel]) / 255.0F - 0.5F));
                    }
                }
            }
        }
    }
    return result;
}

}  // namespace strata
