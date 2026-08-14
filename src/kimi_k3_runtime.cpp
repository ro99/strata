#include "strata/kimi_k3_runtime.hpp"

#include "json_cursor.hpp"

#include "strata/kimi_k3_checkpoint.hpp"
#include "strata/kimi_k3_kv_cache.hpp"
#include "strata/kimi_k3_layer.hpp"
#include "strata/kimi_k3_ops.hpp"
#include "strata/model_adapter.hpp"
#include "strata/tokenizer.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>

namespace strata {
namespace {

constexpr auto& kContract = kKimiK3ExecutionContract;

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::uint64_t resident_set_bytes() {
    std::ifstream status("/proc/self/statm");
    std::uint64_t pages = 0U;
    std::uint64_t resident = 0U;
    if (status >> pages >> resident) {
        return resident * static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    }
    return 0U;
}

[[nodiscard]] std::uint64_t available_host_bytes() {
    std::ifstream info("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0U;
    std::string unit;
    while (info >> key >> value >> unit) {
        if (key == "MemAvailable:") return value * 1024U;
    }
    return 0U;
}

std::string layer_prefix(std::uint32_t layer) {
    return "language_model.model.layers." + std::to_string(layer) + ".";
}

// Weights held in host memory in their checkpoint encoding. The storage is
// node-stable so the spans handed to the layer graph stay valid as more
// tensors load.
class ResidentWeights {
public:
    [[nodiscard]] ValidationResult matrix(const KimiCheckpointReader& reader,
                                          const std::string& name,
                                          std::uint32_t rows,
                                          std::uint32_t columns,
                                          KimiBf16Matrix& out) {
        ValidationResult result;
        const auto elements = static_cast<std::uint64_t>(rows) * columns;
        auto raw = reader.read(name, elements * 2U);
        if (!raw.ok()) {
            result.errors = std::move(raw.errors);
            return result;
        }
        if (raw.value.size() != elements * 2U) {
            result.errors.push_back(name + " is not a BF16 tensor of the declared shape");
            return result;
        }
        bytes_ += raw.value.size();
        auto& stored = raw_.emplace_back(std::move(raw.value));
        out.values = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(stored.data()), elements);
        out.rows = rows;
        out.columns = columns;
        return result;
    }

    [[nodiscard]] ValidationResult vector(const KimiCheckpointReader& reader,
                                          const std::string& name,
                                          std::uint64_t elements,
                                          std::span<const float>& out) {
        ValidationResult result;
        auto raw = reader.read_f32(name, elements);
        if (!raw.ok()) {
            result.errors = std::move(raw.errors);
            return result;
        }
        if (raw.value.size() != elements) {
            result.errors.push_back(name + " has an unexpected element count");
            return result;
        }
        bytes_ += raw.value.size() * sizeof(float);
        auto& stored = floats_.emplace_back(std::move(raw.value));
        out = stored;
        return result;
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

private:
    std::deque<std::vector<std::byte>> raw_;
    std::deque<std::vector<float>> floats_;
    std::uint64_t bytes_{};
};

// Routed experts, served out of the locked host arena and read from the
// checkpoint's shards on a miss. Nothing here writes: a miss is a read.
class ArenaExpertSource final : public KimiExpertSource {
public:
    ArenaExpertSource(const KimiCheckpointReader& checkpoint,
                      KimiExpertArena& arena, KimiExpertReader& reader,
                      KimiRouteObserver observer = {})
        : checkpoint_(&checkpoint), arena_(&arena), reader_(&reader),
          observer_(std::move(observer)) {}

    void observe(std::uint32_t layer,
                 std::span<const std::uint32_t> experts) override {
        if (observer_) observer_(layer, experts);
    }

    ValidationResult prepare(std::uint32_t layer,
                             std::span<const std::uint32_t> experts) override {
        // Every expert the block will read, handed over together so the reader
        // can keep its queue depth. Issuing them one demand miss at a time runs
        // the link at under half speed for structural reasons.
        requests_.clear();
        requests_.reserve(experts.size());
        for (const auto expert : experts) {
            requests_.push_back(KimiReadRequest{layer, expert});
        }
        return reader_->stage(*checkpoint_, *arena_, requests_);
    }

    ValidationResult fetch(std::uint32_t layer, std::uint32_t expert,
                           KimiExpertWeights& weights) override {
        ValidationResult result;
        const auto slot = arena_->find(layer, expert);
        if (slot.empty()) {
            result.errors.push_back(
                "Kimi-K3 expert " + std::to_string(expert) + " of layer " +
                std::to_string(layer) + " is not resident after staging");
            return result;
        }
        KimiExpertModules modules{};
        if (!checkpoint_->expert_modules(layer, expert, modules)) {
            result.errors.push_back("Kimi-K3 expert " + std::to_string(expert) +
                                    " of layer " + std::to_string(layer) +
                                    " is not in the checkpoint");
            return result;
        }
        const auto layout = kimi_expert_slot_layout(modules);
        if (layout.total_bytes > slot.size()) {
            result.errors.emplace_back(
                "Kimi-K3 expert slot is smaller than its module layout");
            return result;
        }
        const auto inner = kContract.expert_intermediate_size;
        const auto latent = kContract.routed_expert_hidden_size;
        const auto* base = reinterpret_cast<const std::uint8_t*>(slot.data());
        const auto view = [&](std::uint64_t packed_offset,
                              std::uint64_t scale_offset, std::uint32_t rows,
                              std::uint32_t columns) {
            KimiExpertModuleView module;
            module.rows = rows;
            module.columns = columns;
            module.packed = std::span<const std::uint8_t>(
                base + packed_offset,
                static_cast<std::size_t>(rows) * (columns / 2U));
            module.scales = std::span<const std::uint8_t>(
                base + scale_offset,
                static_cast<std::size_t>(rows) * (columns / 32U));
            return module;
        };
        weights.gate = view(layout.gate_packed, layout.gate_scale, inner, latent);
        weights.up = view(layout.up_packed, layout.up_scale, inner, latent);
        weights.down = view(layout.down_packed, layout.down_scale, latent, inner);
        return result;
    }

private:
    const KimiCheckpointReader* checkpoint_{};
    KimiExpertArena* arena_{};
    KimiExpertReader* reader_{};
    std::vector<KimiReadRequest> requests_;
    KimiRouteObserver observer_;
};

// `generation_config.json` is the file the reference's `generate` reads, and
// it is the only place the stop token is stated for generation.
[[nodiscard]] ValidationResult read_generation_eos(
    const std::string& model_directory, std::uint32_t& eos) {
    ValidationResult result;
    const auto text = load_bounded_text_file(
        model_directory + "/generation_config.json", 1ULL << 20U);
    if (!text.ok()) {
        result.errors = text.errors;
        return result;
    }
    try {
        detail::JsonCursor cursor(text.value);
        bool found = false;
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "eos_token_id") {
                    const auto parsed = cursor.parse_uint64();
                    if (parsed >= kContract.vocabulary_size) {
                        throw std::runtime_error(
                            "eos_token_id is outside the vocabulary");
                    }
                    eos = static_cast<std::uint32_t>(parsed);
                    found = true;
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!found) throw std::runtime_error("no eos_token_id");
    } catch (const std::exception& error) {
        result.errors.emplace_back(std::string("generation_config.json: ") +
                                   error.what());
    }
    return result;
}

}  // namespace

struct KimiK3Runtime::Impl {
    KimiK3RuntimeConfig config;
    std::unique_ptr<KimiCheckpointReader> checkpoint;
    ResidentWeights resident;
    std::unique_ptr<HostWorkerPool> pool;

    KimiBf16Matrix embedding;
    KimiBf16Matrix output_head;
    std::span<const float> final_norm;
    std::span<const float> output_res_norm;
    std::span<const float> output_res_proj;
    std::vector<KimiLayerWeights> layers;

    KimiStateCache cache;
    KimiExpertArena arena;
    KimiExpertReader reader;
    std::unique_ptr<ArenaExpertSource> experts;

    ModelTokenizer tokenizer;
    std::uint32_t eos_token{};
    KimiK3RunMetrics metrics;
    std::uint32_t length{};
    bool ready{};

    [[nodiscard]] ValidationResult load_layer(std::uint32_t layer);
    [[nodiscard]] ValidationResult forward(std::span<const std::uint32_t> tokens,
                                           std::uint32_t position,
                                           std::span<float> logits);
};

ValidationResult KimiK3Runtime::Impl::load_layer(std::uint32_t layer) {
    ValidationResult result;
    const auto prefix = layer_prefix(layer);
    auto& out = layers[layer];
    const auto vector_of = [&](const std::string& name, std::uint64_t count,
                               std::span<const float>& destination) {
        return resident.vector(*checkpoint, prefix + name, count, destination);
    };
    const auto matrix_of = [&](const std::string& name, std::uint32_t rows,
                               std::uint32_t columns, KimiBf16Matrix& destination) {
        return resident.matrix(*checkpoint, prefix + name, rows, columns,
                               destination);
    };

    result = vector_of("input_layernorm.weight", kContract.hidden_size,
                       out.input_norm);
    if (!result.ok()) return result;
    result = vector_of("post_attention_layernorm.weight", kContract.hidden_size,
                       out.post_attention_norm);
    if (!result.ok()) return result;
    result = vector_of("self_attention_res_norm.weight", kContract.hidden_size,
                       out.attention_res_norm);
    if (!result.ok()) return result;
    result = vector_of("self_attention_res_proj.weight", kContract.hidden_size,
                       out.attention_res_proj);
    if (!result.ok()) return result;
    result = vector_of("mlp_res_norm.weight", kContract.hidden_size, out.mlp_res_norm);
    if (!result.ok()) return result;
    result = vector_of("mlp_res_proj.weight", kContract.hidden_size, out.mlp_res_proj);
    if (!result.ok()) return result;

    if (kimi_k3_kda_layer(layer)) {
        const auto projection =
            kContract.linear_attention_heads * kContract.linear_head_dim;
        auto& kda = out.kda;
        result = matrix_of("self_attn.q_proj.weight", projection,
                           kContract.hidden_size, kda.q_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.k_proj.weight", projection,
                           kContract.hidden_size, kda.k_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.v_proj.weight", projection,
                           kContract.hidden_size, kda.v_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.g_proj.weight", projection,
                           kContract.hidden_size, kda.g_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.o_proj.weight", kContract.hidden_size,
                           projection, kda.o_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.f_a_proj.weight", kContract.linear_head_dim,
                           kContract.hidden_size, kda.f_a_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.f_b_proj.weight", projection,
                           kContract.linear_head_dim, kda.f_b_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.b_proj.weight",
                           kContract.linear_attention_heads, kContract.hidden_size,
                           kda.b_proj);
        if (!result.ok()) return result;
        const auto taps =
            static_cast<std::uint64_t>(projection) * kContract.short_conv_kernel;
        result = vector_of("self_attn.q_conv1d.weight", taps, kda.q_conv);
        if (!result.ok()) return result;
        result = vector_of("self_attn.k_conv1d.weight", taps, kda.k_conv);
        if (!result.ok()) return result;
        result = vector_of("self_attn.v_conv1d.weight", taps, kda.v_conv);
        if (!result.ok()) return result;
        result = vector_of("self_attn.dt_bias", projection, kda.dt_bias);
        if (!result.ok()) return result;
        result = vector_of("self_attn.o_norm.weight", kContract.linear_head_dim,
                           kda.o_norm);
        if (!result.ok()) return result;
        // `A_log` is a per-head scalar the checkpoint zero-pads to head_dim.
        // Broadcasting it over the channel axis instead would apply exp(0) = 1
        // to a quarter of the channels and silently corrupt the decay, so the
        // padding is checked before it is dropped.
        std::span<const float> padded;
        result = vector_of("self_attn.A_log", kContract.linear_head_dim, padded);
        if (!result.ok()) return result;
        for (std::uint32_t index = kContract.linear_attention_heads;
             index < kContract.linear_head_dim; ++index) {
            if (padded[index] != 0.0F) {
                result.errors.push_back(
                    prefix + "self_attn.A_log has a non-zero value past head " +
                    std::to_string(kContract.linear_attention_heads) +
                    "; it is not the zero-padded per-head vector the contract "
                    "declares");
                return result;
            }
        }
        kda.a_log = padded.subspan(0U, kContract.linear_attention_heads);
    } else {
        const auto query_dim = kContract.nope_head_dim + kContract.rope_head_dim;
        auto& mla = out.mla;
        result = matrix_of("self_attn.q_a_proj.weight", kContract.query_lora_rank,
                           kContract.hidden_size, mla.q_a_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.q_b_proj.weight",
                           kContract.attention_heads * query_dim,
                           kContract.query_lora_rank, mla.q_b_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.kv_a_proj_with_mqa.weight",
                           kContract.kv_lora_rank + kContract.rope_head_dim,
                           kContract.hidden_size, mla.kv_a_proj);
        if (!result.ok()) return result;
        result = matrix_of(
            "self_attn.kv_b_proj.weight",
            kContract.attention_heads *
                (kContract.nope_head_dim + kContract.value_head_dim),
            kContract.kv_lora_rank, mla.kv_b_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.o_proj.weight", kContract.hidden_size,
                           kContract.attention_heads * kContract.value_head_dim,
                           mla.o_proj);
        if (!result.ok()) return result;
        result = matrix_of("self_attn.g_proj.weight",
                           kContract.attention_heads * kContract.value_head_dim,
                           kContract.hidden_size, mla.g_proj);
        if (!result.ok()) return result;
        result = vector_of("self_attn.q_a_layernorm.weight",
                           kContract.query_lora_rank, mla.q_a_norm);
        if (!result.ok()) return result;
        result = vector_of("self_attn.kv_a_layernorm.weight", kContract.kv_lora_rank,
                           mla.kv_a_norm);
        if (!result.ok()) return result;
    }

    if (kimi_k3_moe_layer(layer)) {
        const auto moe_prefix = std::string("block_sparse_moe.");
        auto& moe = out.moe;
        const auto shared_inner =
            kContract.expert_intermediate_size * kContract.shared_experts;
        result = matrix_of(moe_prefix + "gate.weight", kContract.routed_experts,
                           kContract.hidden_size, moe.router);
        if (!result.ok()) return result;
        result = vector_of(moe_prefix + "gate.e_score_correction_bias",
                           kContract.routed_experts, moe.router_bias);
        if (!result.ok()) return result;
        result = matrix_of(moe_prefix + "routed_expert_down_proj.weight",
                           kContract.routed_expert_hidden_size,
                           kContract.hidden_size, moe.latent_down);
        if (!result.ok()) return result;
        result = matrix_of(moe_prefix + "routed_expert_up_proj.weight",
                           kContract.hidden_size,
                           kContract.routed_expert_hidden_size, moe.latent_up);
        if (!result.ok()) return result;
        result = vector_of(moe_prefix + "routed_expert_norm.weight",
                           kContract.routed_expert_hidden_size, moe.latent_norm);
        if (!result.ok()) return result;
        result = matrix_of(moe_prefix + "shared_experts.gate_proj.weight",
                           shared_inner, kContract.hidden_size, moe.shared_gate);
        if (!result.ok()) return result;
        result = matrix_of(moe_prefix + "shared_experts.up_proj.weight",
                           shared_inner, kContract.hidden_size, moe.shared_up);
        if (!result.ok()) return result;
        result = matrix_of(moe_prefix + "shared_experts.down_proj.weight",
                           kContract.hidden_size, shared_inner, moe.shared_down);
        if (!result.ok()) return result;
    } else {
        auto& dense = out.dense;
        result = matrix_of("mlp.gate_proj.weight", kContract.dense_intermediate_size,
                           kContract.hidden_size, dense.gate);
        if (!result.ok()) return result;
        result = matrix_of("mlp.up_proj.weight", kContract.dense_intermediate_size,
                           kContract.hidden_size, dense.up);
        if (!result.ok()) return result;
        result = matrix_of("mlp.down_proj.weight", kContract.hidden_size,
                           kContract.dense_intermediate_size, dense.down);
        if (!result.ok()) return result;
    }
    return result;
}

ValidationResult KimiK3Runtime::Impl::forward(
    std::span<const std::uint32_t> tokens, std::uint32_t position,
    std::span<float> logits) {
    ValidationResult result;
    const auto count = static_cast<std::uint32_t>(tokens.size());
    const auto hidden = static_cast<std::size_t>(kContract.hidden_size);
    if (count == 0U) {
        result.errors.emplace_back("Kimi-K3 forward needs at least one token");
        return result;
    }
    if (!logits.empty() && logits.size() != kContract.vocabulary_size) {
        result.errors.emplace_back(
            "Kimi-K3 logit buffer must hold the whole vocabulary");
        return result;
    }
    for (const auto token : tokens) {
        if (token >= kContract.vocabulary_size) {
            result.errors.push_back("Kimi-K3 token id " + std::to_string(token) +
                                    " is outside the vocabulary");
            return result;
        }
    }

    std::vector<float> embeddings(static_cast<std::size_t>(count) * hidden);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto* row = embedding.values.data() +
                          static_cast<std::size_t>(tokens[index]) * hidden;
        for (std::size_t column = 0U; column < hidden; ++column) {
            const auto bits = static_cast<std::uint32_t>(row[column]) << 16U;
            float value = 0.0F;
            std::memcpy(&value, &bits, sizeof(value));
            embeddings[static_cast<std::size_t>(index) * hidden + column] = value;
        }
    }

    KimiResidualStream stream;
    result = stream.reset(count, kContract.hidden_size,
                          kContract.attention_residual_block_size);
    if (!result.ok()) return result;
    result = stream.begin(embeddings);
    if (!result.ok()) return result;

    KimiLayerScratch scratch;
    for (std::uint32_t layer = 0U; layer < kContract.layer_count; ++layer) {
        result = kimi_decoder_layer(stream, layers[layer], cache, *experts, layer,
                                    position, count, scratch, pool.get());
        if (!result.ok()) return result;
        if (config.layer_observer) config.layer_observer(layer, stream.prefix());
        if (config.load_progress && (layer + 1U) % 16U == 0U) {
            std::cerr << "\r[kimi-k3] layer " << (layer + 1U) << '/'
                      << kContract.layer_count << std::flush;
        }
    }
    if (config.load_progress) std::cerr << '\r' << std::string(40U, ' ') << '\r';

    // One commit per page, after every MLA layer has appended, so no layer can
    // read a row another has not written.
    result = cache.commit(position + count - 1U);
    if (!result.ok()) return result;
    length = position + count;

    if (logits.empty()) return result;

    // The final attention-residual site selects over the eight completed blocks
    // and the running prefix before the output norm.
    std::vector<float> mixed(static_cast<std::size_t>(count) * hidden);
    result = stream.mix(mixed, output_res_proj, output_res_norm,
                        kContract.rms_epsilon, pool.get());
    if (!result.ok()) return result;

    // One row, or every row. A prefill page only needs its last row, because a
    // page exists to build cache state and only its final token predicts
    // anything. Speculative verification needs all of them: accepting the
    // longest correct prefix means comparing the draft's token at position i
    // against this model's argmax at i, for every i.
    //
    // Selected by the span the caller passed rather than by a flag, so every
    // existing caller keeps the single-row behaviour without changing.
    const auto vocabulary = static_cast<std::size_t>(kContract.vocabulary_size);
    const auto rows = logits.size() >= static_cast<std::size_t>(count) * vocabulary
                          ? static_cast<std::size_t>(count)
                          : std::size_t{1U};
    const auto last = static_cast<std::size_t>(count - 1U) * hidden;

    std::vector<float> normalized(rows * hidden);
    for (std::size_t row = 0U; row < rows; ++row) {
        const auto source = rows == 1U ? last : row * hidden;
        result = kimi_rms_norm(
            std::span<float>(normalized).subspan(row * hidden, hidden),
            std::span<const float>(mixed).subspan(source, hidden), final_norm,
            kContract.rms_epsilon);
        if (!result.ok()) return result;
    }
    // The head is [163840, 7168] at 2.19 GiB and is read once per call whatever
    // `rows` is, so this is where a batch pays for itself rather than costing.
    const auto head_begin = std::chrono::steady_clock::now();
    result = kimi_bf16_matmul(logits.subspan(0U, rows * vocabulary), normalized,
                              output_head, static_cast<std::uint32_t>(rows),
                              pool.get());
    const auto head_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - head_begin).count();

    if (config.verbose) {
        // The per-phase breakdown of one pass, which is step 1 of the charter's
        // decision procedure and had been skipped for decode. Naming argmax_r is
        // the point: a mechanism that does not reduce it cannot improve the step.
        const auto ms = [](std::uint64_t nanoseconds) {
            return static_cast<double>(nanoseconds) / 1.0e6;
        };
        const auto total = ms(scratch.attention_ns + scratch.feedforward_ns +
                              scratch.residual_mix_ns) +
                           static_cast<double>(head_ns) / 1.0e6;
        std::cerr << "[phase] tokens " << count
                  << "  attention " << ms(scratch.attention_ns) << " ms"
                  << "  feedforward " << ms(scratch.feedforward_ns) << " ms"
                  << "  residual-mix " << ms(scratch.residual_mix_ns) << " ms"
                  << "  head " << static_cast<double>(head_ns) / 1.0e6 << " ms"
                  << "  accounted " << total << " ms\n";
    }
    return result;
}

KimiK3Runtime::KimiK3Runtime() : impl_(std::make_unique<Impl>()) {}
KimiK3Runtime::~KimiK3Runtime() = default;
KimiK3Runtime::KimiK3Runtime(KimiK3Runtime&&) noexcept = default;
KimiK3Runtime& KimiK3Runtime::operator=(KimiK3Runtime&&) noexcept = default;

ValidationResult KimiK3Runtime::initialize(const std::string& model_directory,
                                           const KimiK3RuntimeConfig& config) {
    ValidationResult result;
    if (impl_->ready) {
        result.errors.emplace_back("Kimi-K3 runtime is already initialized");
        return result;
    }
    impl_->config = config;
    const auto started = now_seconds();

    // The guard runs before anything is read. An enabled swap file on a
    // protected disk would page model bytes out through the back door however
    // carefully the read path is written, so this refuses rather than warns.
    if (config.apply_write_guard) {
        KimiWriteGuardConfig guard;
        guard.forbidden_disks = config.forbidden_disks;
        guard.write_paths = {model_directory};
        result = kimi_apply_write_guard(guard);
        if (!result.ok()) return result;
    }

    auto opened = KimiCheckpointReader::open(model_directory);
    if (!opened.ok()) {
        result.errors = std::move(opened.errors);
        return result;
    }
    impl_->checkpoint = std::move(opened.value);

    // Before the 106 GiB read, not after: a malformed vocabulary should cost a
    // second rather than five minutes.
    auto tokenizer = ModelTokenizer::load_kimi_k3(model_directory);
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }
    impl_->tokenizer = std::move(tokenizer.value);

    // Read rather than assumed. `config.json` also carries an `eos_token_id`,
    // but `generation_config.json` is what the reference generates against and
    // the two disagree here: 163586 (`<|end_of_msg|>`) against the tokenizer's
    // `[EOS]` at 163585. Hard-coding either would be a silent contract change.
    result = read_generation_eos(model_directory, impl_->eos_token);
    if (!result.ok()) return result;

    const auto workers = config.host_workers != 0U
        ? config.host_workers
        : std::max<std::size_t>(1U, std::thread::hardware_concurrency());
    impl_->pool = std::make_unique<HostWorkerPool>(workers);

    result = impl_->cache.reset(config.maximum_context_tokens);
    if (!result.ok()) return result;

    // Dense spine. 106.55 GiB of BF16 at roughly 400 MB/s of SATA is about
    // four and a half minutes, and it is paid once.
    result = impl_->resident.matrix(*impl_->checkpoint,
                                    "language_model.model.embed_tokens.weight",
                                    kContract.vocabulary_size,
                                    kContract.hidden_size, impl_->embedding);
    if (!result.ok()) return result;
    result = impl_->resident.matrix(*impl_->checkpoint,
                                    "language_model.lm_head.weight",
                                    kContract.vocabulary_size,
                                    kContract.hidden_size, impl_->output_head);
    if (!result.ok()) return result;
    result = impl_->resident.vector(*impl_->checkpoint,
                                    "language_model.model.norm.weight",
                                    kContract.hidden_size, impl_->final_norm);
    if (!result.ok()) return result;
    result = impl_->resident.vector(
        *impl_->checkpoint, "language_model.model.output_attn_res_norm.weight",
        kContract.hidden_size, impl_->output_res_norm);
    if (!result.ok()) return result;
    result = impl_->resident.vector(
        *impl_->checkpoint, "language_model.model.output_attn_res_proj.weight",
        kContract.hidden_size, impl_->output_res_proj);
    if (!result.ok()) return result;

    impl_->layers.resize(kContract.layer_count);
    for (std::uint32_t layer = 0U; layer < kContract.layer_count; ++layer) {
        result = impl_->load_layer(layer);
        if (!result.ok()) return result;
        if (config.load_progress) {
            std::fprintf(stderr, "\r[kimi-k3] loaded layer %u/%u, %.2f GiB",
                         layer + 1U, kContract.layer_count,
                         static_cast<double>(impl_->resident.bytes()) /
                             (1024.0 * 1024.0 * 1024.0));
            std::fflush(stderr);
        }
    }
    if (config.load_progress) std::fprintf(stderr, "\n");

    // Whatever is left after the spine goes to the routed-expert cache. It
    // cannot manufacture sparsity — 1.3 TiB of experts against ~130 GiB of
    // cache — so this is a hit-rate knob, not a residency plan.
    auto arena_bytes = config.expert_arena_bytes;
    if (arena_bytes == 0U) {
        const auto available = available_host_bytes();
        const auto reserve = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        arena_bytes = available > reserve ? available - reserve : 0U;
        if (config.host_budget_bytes != 0U) {
            const auto spine = impl_->resident.bytes();
            const auto budget = config.host_budget_bytes > spine
                ? config.host_budget_bytes - spine : 0U;
            arena_bytes = std::min(arena_bytes, budget);
        }
    }
    const auto slot = KimiCheckpointReader::expert_source_bytes();
    if (arena_bytes < slot * kContract.experts_per_token) {
        result.errors.push_back(
            "Kimi-K3 needs room for at least one step's experts: " +
            std::to_string(slot * kContract.experts_per_token) +
            " bytes, and only " + std::to_string(arena_bytes) + " are free");
        return result;
    }
    KimiArenaConfig arena;
    arena.capacity_bytes = arena_bytes;
    arena.lock_pages = config.lock_expert_arena;
    result = impl_->arena.reset(arena);
    if (!result.ok()) return result;

    KimiReaderConfig reader;
    reader.queue_depth = config.expert_queue_depth;
    reader.direct = config.direct_expert_reads;
    result = impl_->reader.open(*impl_->checkpoint, reader);
    if (!result.ok()) return result;
    impl_->experts = std::make_unique<ArenaExpertSource>(
        *impl_->checkpoint, impl_->arena, impl_->reader, config.route_observer);

    impl_->metrics = {};
    impl_->metrics.load_seconds = now_seconds() - started;
    impl_->metrics.resident_weight_bytes = impl_->resident.bytes();
    impl_->metrics.expert_arena_bytes = impl_->arena.capacity_bytes();
    impl_->metrics.expert_arena_locked = impl_->arena.locked();
    impl_->metrics.rss_bytes = resident_set_bytes();
    impl_->ready = true;
    if (config.verbose) {
        std::cerr << "[kimi-k3] resident spine "
                  << (impl_->resident.bytes() >> 30U) << " GiB, expert arena "
                  << (impl_->arena.capacity_bytes() >> 30U) << " GiB"
                  << (impl_->arena.locked() ? " (locked)" : " (NOT locked)")
                  << ", load " << impl_->metrics.load_seconds << " s\n";
    }
    return result;
}

ValidationResult KimiK3Runtime::reset_sequence() {
    ValidationResult result;
    if (!impl_->ready) {
        result.errors.emplace_back("Kimi-K3 runtime is not initialized");
        return result;
    }
    // The KDA half is recurrent and cannot be rewound, so a new sequence is a
    // full reset rather than a truncation.
    result = impl_->cache.reset(impl_->config.maximum_context_tokens);
    impl_->length = 0U;
    return result;
}

ValidationResult KimiK3Runtime::evaluate(std::span<const std::uint32_t> tokens,
                                         std::uint32_t position,
                                         std::span<float> logits) {
    ValidationResult result;
    if (!impl_->ready) {
        result.errors.emplace_back("Kimi-K3 runtime is not initialized");
        return result;
    }
    if (position != impl_->length) {
        result.errors.push_back(
            "Kimi-K3 evaluate must continue at position " +
            std::to_string(impl_->length) + ", not " + std::to_string(position) +
            "; the recurrent half cannot be rewound");
        return result;
    }
    result = impl_->forward(tokens, position, logits);
    impl_->metrics.expert_hits = impl_->arena.hits();
    impl_->metrics.expert_misses = impl_->arena.misses();
    impl_->metrics.expert_evictions = impl_->arena.evictions();
    impl_->metrics.storage_bytes_read = impl_->reader.stats().bytes_read;
    impl_->metrics.rss_bytes = resident_set_bytes();
    return result;
}

KimiK3GenerationResult KimiK3Runtime::generate_from_tokens(
    std::span<const std::uint32_t> prompt, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, const TokenStreamCallback& on_token) {
    KimiK3GenerationResult result;
    if (!impl_->ready) {
        result.errors.emplace_back("Kimi-K3 runtime is not initialized");
        return result;
    }
    if (prompt.empty()) {
        result.errors.emplace_back("Kimi-K3 needs a non-empty prompt");
        return result;
    }
    std::string error;
    if (!validate_sampling_options(sampling, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    auto reset = reset_sequence();
    if (!reset.ok()) {
        result.errors = std::move(reset.errors);
        return result;
    }
    result.prompt_token_ids.assign(prompt.begin(), prompt.end());
    result.metrics.prompt_tokens = prompt.size();

    std::vector<float> logits(kContract.vocabulary_size);
    const auto page = std::max<std::uint32_t>(1U, impl_->config.prefill_page_tokens);
    const auto prefill_started = now_seconds();
    std::uint32_t position = 0U;
    while (position < prompt.size()) {
        const auto count = std::min<std::uint32_t>(
            page, static_cast<std::uint32_t>(prompt.size()) - position);
        const auto last = position + count == prompt.size();
        auto step = evaluate(prompt.subspan(position, count), position,
                             last ? std::span<float>(logits) : std::span<float>{});
        if (!step.ok()) {
            result.errors = std::move(step.errors);
            return result;
        }
        position += count;
        result.metrics.prefill_tokens += count;
    }
    result.metrics.prefill_seconds = now_seconds() - prefill_started;

    std::mt19937_64 generator(sampling.seed);
    // The penalty stages read the tokens this run has produced, so the counts
    // and the order both have to be kept.
    std::vector<std::uint32_t> sampled_counts(kContract.vocabulary_size, 0U);
    std::vector<std::uint32_t> sampled_ids;
    const auto decode_started = now_seconds();
    for (std::uint32_t index = 0U; index < maximum_new_tokens; ++index) {
        auto drawn = sample_logits(logits, sampling,
                                   SamplingHistory{sampled_counts, sampled_ids},
                                   generator);
        if (!drawn.ok()) {
            result.errors = std::move(drawn.errors);
            return result;
        }
        if (drawn.token == impl_->eos_token) {
            result.stopped = true;
            break;
        }
        result.generated_token_ids.push_back(drawn.token);
        result.logprobs.push_back(drawn);
        ++sampled_counts[drawn.token];
        sampled_ids.push_back(drawn.token);
        if (on_token && !on_token(drawn.token, {})) break;
        if (index + 1U == maximum_new_tokens) break;
        const std::array<std::uint32_t, 1U> next{drawn.token};
        auto step = evaluate(next, position, logits);
        if (!step.ok()) {
            result.errors = std::move(step.errors);
            return result;
        }
        ++position;
        ++result.metrics.decode_tokens;
    }
    result.metrics.decode_seconds = now_seconds() - decode_started;
    result.metrics.load_seconds = impl_->metrics.load_seconds;
    result.metrics.resident_weight_bytes = impl_->metrics.resident_weight_bytes;
    result.metrics.expert_arena_bytes = impl_->metrics.expert_arena_bytes;
    result.metrics.expert_arena_locked = impl_->metrics.expert_arena_locked;
    result.metrics.expert_hits = impl_->arena.hits();
    result.metrics.expert_misses = impl_->arena.misses();
    result.metrics.expert_evictions = impl_->arena.evictions();
    result.metrics.storage_bytes_read = impl_->reader.stats().bytes_read;
    result.metrics.rss_bytes = resident_set_bytes();
    return result;
}

KimiK3GenerationResult KimiK3Runtime::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    const std::array messages{ChatMessage{ChatRole::User, std::string(prompt)}};
    SamplingOptions sampling;
    sampling.temperature = impl_->config.sampling_temperature;
    sampling.seed = impl_->config.sampling_seed;
    return generate_chat_stream(messages, maximum_new_tokens, sampling, {},
                                on_token);
}

KimiK3GenerationResult KimiK3Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    KimiK3GenerationResult result;
    if (!impl_->ready) {
        result.errors.emplace_back("Kimi-K3 runtime is not initialized");
        return result;
    }
    std::string error;
    if (!validate_chat_messages(messages, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    // Vision is not implemented. A message carrying an image must fail rather
    // than be rendered as text with the image dropped, which would answer a
    // question the caller did not ask. See ro99/strata#21.
    for (const auto& message : messages) {
        for (const auto& part : message.parts) {
            if (part.kind != ChatContentKind::Text) {
                result.errors.emplace_back(
                    "Kimi-K3 image input needs the MoonViT-V2 path, which is not "
                    "implemented; text-only messages are supported");
                return result;
            }
        }
    }

    auto encoded = impl_->tokenizer.encode(
        render_kimi_k3_chat_prompt(messages, true));
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    if (encoded.value.size() + maximum_new_tokens >
        impl_->config.maximum_context_tokens) {
        result.errors.emplace_back(
            "prompt and requested generation exceed the context ceiling");
        return result;
    }

    // The decoded text is assembled from the ids the sampler drew, so a stop
    // string that straddles two tokens is still found.
    auto generated = generate_from_tokens(encoded.value, maximum_new_tokens,
                                          sampling, on_token);
    result.errors = std::move(generated.errors);
    result.prompt_token_ids = std::move(generated.prompt_token_ids);
    result.generated_token_ids = std::move(generated.generated_token_ids);
    result.logprobs = std::move(generated.logprobs);
    result.metrics = generated.metrics;
    result.stopped = generated.stopped;
    if (!result.ok()) return result;

    auto text = impl_->tokenizer.decode(result.generated_token_ids);
    if (!text.ok()) {
        result.errors = std::move(text.errors);
        return result;
    }
    result.text = std::move(text.value);
    for (const auto& marker : stop) {
        if (marker.empty()) continue;
        if (const auto at = result.text.find(marker); at != std::string::npos) {
            result.text.resize(at);
            result.stopped = true;
        }
    }
    return result;
}

const KimiK3RunMetrics& KimiK3Runtime::metrics() const noexcept {
    return impl_->metrics;
}

std::uint32_t KimiK3Runtime::vocabulary_size() const noexcept {
    return kContract.vocabulary_size;
}

}  // namespace strata
