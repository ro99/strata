#include "../common/executor_support.hpp"
#include "strata/runtime_support.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/deepseek_runtime.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/dsv4_rank_local_topology.hpp"

namespace strata {
namespace {

class Dsv4Executor final : public ModelExecutor {
public:
    ValidationResult initialize(const std::string& model_directory,
                                const RuntimeConfig& config,
                                const PlacementPlan*) override {
        Dsv4RuntimeConfig concrete;
        concrete.devices = resolve_runtime_devices(config.devices);
        concrete.vram_cache_fraction = config.vram_cache_fraction;
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling.temperature;
        concrete.sampling_seed = config.sampling.seed;
        concrete.verbose = config.verbose;
        concrete.enable_flash_attention = config.enable_flash_attention;
        concrete.enable_incremental_kv_continuation =
            config.enable_incremental_kv_continuation;
        concrete.pin_resident_arena = config.pin_resident_arena;
        concrete.prepack_mhc_projection = config.prepack_mhc_projection;
        concrete.kv_cache_mode = config.deepseek_device_resident_runtime
            ? Dsv4KvCacheMode::PhysicalDevice
            : config.deepseek_block_kv_cache ? Dsv4KvCacheMode::Block
                                             : Dsv4KvCacheMode::ScalarOracle;
        concrete.kv_block_rows = config.deepseek_device_resident_runtime
            ? kDsv4PhysicalKvBlockRows : kDsv4KvBlockRows;
        if (config.deepseek_prefill_page_tokens != 0U) {
            concrete.prefill_page_tokens = config.deepseek_prefill_page_tokens;
        }
        concrete.static_expert_plan_path = config.deepseek_static_expert_plan;
        concrete.static_expert_tier_bytes = config.deepseek_static_expert_bytes;
        if (config.deepseek_device_resident_runtime) {
            // The device-resident decode contract is a bundle, not a knob.
            // Leaving any member of it to the caller lets a run report the
            // accepted attention/mHC path while executing routed experts
            // somewhere else, which is what made this opt-in rather than a
            // default. These are the same implications strata-deepseek-run
            // applies to the flag.
            concrete.enable_flash_attention = true;
            concrete.enable_gpu_lightning_indexer = false;
            concrete.enable_host_routed_moe = true;
            concrete.prepack_mhc_projection = false;
        }
        concrete.decode_topology = config.deepseek_rank_local_decode
            ? Dsv4DecodeTopology::RankLocalTp2
            : Dsv4DecodeTopology::Centralized;
        return runtime_.initialize(model_directory, concrete);
    }

    GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages, const GenerationOptions& options,
        const TokenStreamCallback& on_token) override {
        GenerationResult result;
        auto concrete = runtime_.generate_chat_stream(
            messages, options.maximum_new_tokens, options.sampling,
            options.stop, on_token);
        detail::copy_common_generation(result, concrete);
        result.metrics.reused_prompt_tokens = concrete.metrics.reused_prompt_tokens;
        result.metrics.incremental_kv_continuation =
            concrete.metrics.incremental_kv_continuation;
        return result;
    }

private:
    DeepSeekV4Runtime runtime_;
};

// The only model that accepts the deepseek_* controls on RuntimeConfig. Every
// other registration declares false, and the facade rejects rather than
// ignores -- a request for rank-local decode that quietly ran a centralized
// GLM would report the accepted path while executing a different one.
const ModelRegistrar registrar{{
    RuntimeModel::DeepSeekV4, "DeepSeek-V4", "deepseek",
    PlacementModel::DeepSeekV4, true, false, false, true,
    [] { return std::unique_ptr<ModelExecutor>(new Dsv4Executor()); }}};

}  // namespace
}  // namespace strata
