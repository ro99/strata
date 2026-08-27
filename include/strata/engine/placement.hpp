#pragma once

#include "strata/platform/result.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

// Bump when the on-disk plan schema changes. A cached plan whose version does
// not match is discarded rather than reinterpreted.
//
// v2 renamed the overflow tier from `nvme` to `storage` and added the backing
// block device, because the tier is the read-only source file and its device is
// not necessarily an NVMe: on the Kimi-K3 target it is a SATA SSD, and calling
// it NVMe reported bytes as coming from somewhere they do not.
inline constexpr std::uint32_t kPlacementPlanVersion = 2U;

enum class PlacementModel : std::uint8_t {
    Glm52,
    DeepSeekV4,
    Gemma4,
    KimiK3,
    Laguna,
    Inkling,
};

enum class PlacementTier : std::uint8_t {
    Device,
    Host,
    // The checkpoint's own shards, wherever they live. Named for what it is
    // rather than for one machine's device class.
    Storage,
};

enum class PlacementClass : std::uint8_t {
    Attention,
    FeedForward,
    RoutedExpert,
    SharedExpert,
    Router,
    Norm,
    Embedding,
    OutputHead,
    Vision,
    KvCache,
    Workspace,
    Count,
};

// One sized, placeable piece of a checkpoint. Sizes are what the piece costs in
// each tier, not what it occupies on disk: a W8A16 linear is the same size in
// VRAM as in the shard, but a BF16 norm vector is decoded to F32 before upload.
struct PlacementItem {
    PlacementClass component{PlacementClass::Norm};
    // >= 0 pins the item to whichever device its layer is assigned. Items with
    // -1 are free to land anywhere, subject to `fixed_device_slot`.
    std::int32_t layer{-1};
    // >= 0 pins the item to one device slot regardless of the layer schedule.
    std::int32_t fixed_device_slot{-1};
    std::uint64_t device_bytes{};
    std::uint64_t host_bytes{};
    std::uint64_t source_bytes{};
    // Bytes this item contributes to one batch-1 decode step. This is the W_r
    // volume term of the governing cost model, not a duration: the planner
    // measures no bandwidth and converts nothing to milliseconds.
    std::uint64_t decode_read_bytes{};
    PlacementTier preferred_tier{PlacementTier::Device};
    // How far down the cascade this item may go. A densely read class can be
    // forced out of VRAM by capacity — Kimi-K3's BF16 spine is 103 GiB against
    // 64 GiB of VRAM, so roughly half of it must stream over PCIe every step —
    // but pushing it to storage would put the checkpoint's slowest resource on
    // the critical path of every token. A plan that would need to is an error,
    // not a spill: that is the charter's "dense larger than aggregate resident
    // memory is I/O dependent" case, which must be reported rather than
    // absorbed.
    PlacementTier deepest_tier{PlacementTier::Storage};
    // A spillable item may leave VRAM without changing any result. Sparse
    // classes qualify freely; a dense class only qualifies with a `deepest_tier`
    // of `Host`, because spilling it past that buys nothing under a max.
    bool spillable{};
    // The canonical copy lives outside VRAM and any device share is a cache on
    // top of it, not a move: MoE runtimes stage the whole expert set in host
    // memory and admit a subset to VRAM. Host residency therefore counts the
    // whole set, however much of it is also cached on a device.
    bool device_cache_only{};
};

// Model-derived sizing at one operating point. Contains no hardware facts, so
// it is reproducible from the checkpoint alone and testable without a GPU.
struct PlacementInventory {
    PlacementModel model{PlacementModel::Gemma4};
    std::string model_name;
    std::uint32_t layer_count{};
    std::uint32_t maximum_context_tokens{};
    std::uint64_t per_device_workspace_bytes{};
    std::uint64_t minimum_device_budget_bytes{2ULL << 30U};
    std::uint64_t host_workspace_bytes{};
    // Host memory the spill tier may not consume. Planning to the last free
    // byte reports a host-resident set the machine cannot actually hold once
    // activations, worker stacks, and page cache are counted. This is a
    // reserve, not a cost estimate: it predicts nothing and is printed with the
    // plan so the number is visible rather than buried.
    std::uint64_t host_reserve_bytes{8ULL << 30U};
    // Host ceiling a runtime enforces independently of how much the machine has
    // free. Zero means the free figure governs. Without it a plan can report a
    // fit the runtime's own admission then refuses.
    std::uint64_t host_capacity_bytes{};
    std::vector<PlacementItem> items;
    // False keeps a runtime's existing interleaved `layer % devices` schedule
    // instead of assigning contiguous blocks.
    bool contiguous_layer_blocks{true};
    // True when the runtime consumes this plan's layer assignment. False means
    // the plan describes and admits the runtime's own placement without
    // changing it.
    bool prescriptive{};
};

struct PlacementDevice {
    int id{};
    std::string name;
    std::uint64_t total_bytes{};
    std::uint64_t free_bytes{};
};

// The block device a path actually resolves to. The overflow tier is a file on
// a disk, and which disk it is decides both the bandwidth the run will see and
// whether the run is allowed at all.
struct PlacementStorage {
    std::string path;
    // Leaf device, e.g. `sda1`, and the whole disk it belongs to, e.g. `sda`.
    std::string device;
    std::string disk;
    bool nvme{};
    bool rotational{};
    // A tmpfs or ramfs has no block device behind it at all, which is a
    // resolved answer and the safest one: nothing written there reaches a
    // disk. `device` and `disk` stay empty.
    bool memory_backed{};
    bool resolved{};
};

struct PlacementHardware {
    std::vector<PlacementDevice> devices;
    std::uint64_t host_total_bytes{};
    std::uint64_t host_available_bytes{};
    // Where the checkpoint's shards live. Empty when the probe did not run.
    PlacementStorage storage;
};

// Everything about a run that changes the answer. Two requests with equal
// fields share a cached plan; anything else gets its own.
struct PlacementRequest {
    PlacementModel model{PlacementModel::Gemma4};
    std::string model_directory;
    std::vector<int> devices;
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    bool flash_attention{};
    bool block_kv_cache{};
};

struct PlacementComponentTotals {
    PlacementClass component{PlacementClass::Norm};
    PlacementTier tier{PlacementTier::Device};
    std::uint64_t bytes{};
    std::uint64_t decode_read_bytes{};
    std::vector<std::uint64_t> device_bytes;
};

struct PlacementPlan {
    std::uint32_t version{kPlacementPlanVersion};
    PlacementRequest request;
    PlacementHardware hardware;
    // Digest over shard names, sizes, and modification times. A checkpoint that
    // changed under a cached plan is rejected, not silently reused.
    std::string model_identity;
    std::string model_name;
    std::vector<PlacementComponentTotals> components;
    std::vector<std::uint64_t> device_budget_bytes;
    std::vector<std::uint64_t> device_resident_bytes;
    std::vector<std::uint64_t> device_expert_cache_bytes;
    std::vector<std::size_t> layer_device;
    std::vector<std::size_t> weighted_schedule;
    std::uint64_t host_resident_bytes{};
    // Bytes that stay in the checkpoint's shards and are read on demand.
    std::uint64_t storage_resident_bytes{};
    std::uint64_t decode_device_read_bytes{};
    std::uint64_t decode_host_to_device_bytes{};
    std::uint64_t decode_storage_read_bytes{};
    // Largest context the same placement admits. Zero when the search did not
    // run, which is every path except a full model plan.
    std::uint32_t maximum_context_tokens_that_fit{};
    // Times a decode step hands the activation from one device to another.
    std::uint32_t cross_device_activation_hops{};
    bool prescriptive{};
    bool fits{};
    bool io_dependent{};
    std::vector<std::string> notes;

    [[nodiscard]] std::uint64_t total_device_bytes() const noexcept;
};

using PlacementPlanResult = ParseResult<PlacementPlan>;

// Device bytes one component occupies on one device slot, zero when absent.
[[nodiscard]] std::uint64_t placement_component_bytes(
    const PlacementPlan& plan, PlacementClass component,
    std::size_t slot) noexcept;

[[nodiscard]] std::string_view to_string(PlacementModel model) noexcept;
[[nodiscard]] std::string_view to_string(PlacementTier tier) noexcept;
[[nodiscard]] std::string_view to_string(PlacementClass component) noexcept;
[[nodiscard]] bool parse_placement_model(std::string_view text,
                                         PlacementModel& model) noexcept;
[[nodiscard]] std::string format_bytes(std::uint64_t bytes);

// Resolves the block device backing `path` through `/sys/dev/block`. Reads no
// payload and writes nothing; a path that cannot be resolved comes back with
// `resolved == false` rather than a guess.
[[nodiscard]] PlacementStorage resolve_backing_storage(const std::string& path);

// Reads free and total VRAM per device plus host memory. Allocates nothing.
// `model_directory` resolves the storage tier's backing device; pass empty to
// skip that probe.
[[nodiscard]] ParseResult<PlacementHardware> probe_placement_hardware(
    std::span<const int> devices, const std::string& model_directory = {});

// Pure solver over an inventory. No file or device access, so tests drive it
// directly with synthetic inventories.
[[nodiscard]] PlacementPlanResult solve_placement(
    const PlacementInventory& inventory, const PlacementHardware& hardware,
    const PlacementRequest& request);

// Opens the checkpoint index and shard headers, sizes every component, and
// solves. Reads no tensor payload and uploads nothing.
// The planner seam. plan_model_placement below is a thin dispatcher in
// strata_engine; the implementation that knows how to open six different
// checkpoints and size six different inventories lives in strata_models and
// installs itself here at static-init time.
//
// This exists to invert one specific dependency. Before it, placement_cache.cpp
// (engine) called a symbol defined in placement_model.cpp, which references
// every model's checkpoint reader -- an engine->models link edge that
// check-symbols reported as a cycle and that only linked because ld happened
// to see the archives in a forgiving order. Engine now names no model symbol.
using PlacementPlanner = PlacementPlanResult (*)(const PlacementRequest&,
                                                 const PlacementHardware&);
void register_placement_planner(PlacementPlanner planner) noexcept;

struct PlacementPlannerRegistrar {
    explicit PlacementPlannerRegistrar(PlacementPlanner planner) noexcept {
        register_placement_planner(planner);
    }
};

[[nodiscard]] PlacementPlanResult plan_model_placement(
    const PlacementRequest& request, const PlacementHardware& hardware);

// Confirms a plan still holds against currently free VRAM and host memory.
// Exact mode has no hidden fallback: a plan that no longer fits is an error.
[[nodiscard]] ValidationResult verify_placement_plan(
    const PlacementPlan& plan, const PlacementHardware& hardware);

[[nodiscard]] std::string render_placement_report(const PlacementPlan& plan);
[[nodiscard]] std::string encode_placement_plan(const PlacementPlan& plan);
[[nodiscard]] PlacementPlanResult decode_placement_plan(std::string_view text);

// Digest of the checkpoint's shard set. Cheap: names, sizes, and mtimes only.
[[nodiscard]] ParseResult<std::string> probe_model_identity(
    const std::string& model_directory);

// `$STRATA_PLAN_CACHE`, else `$XDG_CACHE_HOME/strata/plans`, else
// `$HOME/.cache/strata/plans`. An explicit override wins over all of them.
[[nodiscard]] std::string placement_cache_directory(
    std::string_view override_directory);
[[nodiscard]] std::string placement_plan_filename(
    const PlacementRequest& request, std::string_view model_identity);

struct PlacementResolution {
    PlacementPlan plan;
    std::string cache_path;
    bool from_cache{};
    bool stored{};
};

// Probes the hardware, reuses a matching cached plan when there is one, and
// otherwise computes and stores a fresh plan. This is the single entry point
// both `--dry-run` and a real load go through, so a load never places weights
// by rules the dry run did not print.
[[nodiscard]] ParseResult<PlacementResolution> resolve_placement_plan(
    const PlacementRequest& request, std::string_view cache_directory,
    bool use_cache, bool refresh);

[[nodiscard]] ValidationResult store_placement_plan(
    const std::string& directory, const PlacementPlan& plan);
// Returns the cached plan when one matches this request, this checkpoint, this
// hardware, and this schema version. A miss is not an error: `value.version`
// is zero and `errors` is empty.
[[nodiscard]] PlacementPlanResult load_placement_plan(
    const std::string& directory, const PlacementRequest& request,
    const PlacementHardware& hardware, std::string_view model_identity);

}  // namespace strata
