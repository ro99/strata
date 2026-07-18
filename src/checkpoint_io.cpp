#include "strata/checkpoint_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace strata {

CheckpointShardSet::~CheckpointShardSet() { close_all(); }

CheckpointShardSet::CheckpointShardSet(CheckpointShardSet&& other) noexcept
    : descriptors_(std::move(other.descriptors_)) {
    other.descriptors_.clear();
}

CheckpointShardSet& CheckpointShardSet::operator=(CheckpointShardSet&& other) noexcept {
    if (this == &other) return *this;
    close_all();
    descriptors_ = std::move(other.descriptors_);
    other.descriptors_.clear();
    return *this;
}

void CheckpointShardSet::close_all() noexcept {
    for (const auto& [name, descriptor_value] : descriptors_) {
        static_cast<void>(name);
        if (descriptor_value >= 0) static_cast<void>(close(descriptor_value));
    }
    descriptors_.clear();
}

ValidationResult CheckpointShardSet::open(
    const std::string& model_directory, const std::vector<std::string>& shards,
    std::string_view model_label) {
    ValidationResult result;
    if (!descriptors_.empty()) {
        result.errors.emplace_back("checkpoint shard set is already open");
        return result;
    }
    for (const auto& shard : shards) {
        const auto path = (std::filesystem::path(model_directory) / shard).string();
        const int descriptor_value = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_value < 0) {
            result.errors.emplace_back("cannot open " + std::string(model_label) +
                                       " checkpoint shard " + path + ": " +
                                       std::strerror(errno));
            close_all();
            return result;
        }
        descriptors_.emplace(shard, descriptor_value);
    }
    return result;
}

int CheckpointShardSet::descriptor(std::string_view shard) const noexcept {
    const auto found = descriptors_.find(std::string(shard));
    return found == descriptors_.end() ? -1 : found->second;
}

ValidationResult CheckpointShardSet::read(
    std::string_view shard, std::uint64_t absolute_offset,
    std::span<std::byte> destination, std::string_view tensor_name) const {
    ValidationResult result;
    const int descriptor_value = descriptor(shard);
    if (descriptor_value < 0) {
        result.errors.emplace_back("checkpoint shard is not open for " +
                                   std::string(tensor_name));
        return result;
    }
    if (absolute_offset > static_cast<std::uint64_t>(
                              std::numeric_limits<off_t>::max()) ||
        destination.size() > static_cast<std::uint64_t>(
                                 std::numeric_limits<off_t>::max()) -
                                 absolute_offset) {
        result.errors.emplace_back("tensor file range exceeds the platform range");
        return result;
    }
    std::size_t completed = 0U;
    while (completed < destination.size()) {
        const auto request = std::min<std::size_t>(
            destination.size() - completed,
            static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto count = pread(
            descriptor_value, destination.data() + completed, request,
            static_cast<off_t>(absolute_offset + completed));
        if (count < 0) {
            if (errno == EINTR) continue;
            result.errors.emplace_back("cannot read tensor " +
                                       std::string(tensor_name) + ": " +
                                       std::strerror(errno));
            return result;
        }
        if (count == 0) {
            result.errors.emplace_back("unexpected end of checkpoint shard reading " +
                                       std::string(tensor_name));
            return result;
        }
        completed += static_cast<std::size_t>(count);
    }
    return result;
}

}  // namespace strata
