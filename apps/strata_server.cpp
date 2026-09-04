#include "strata/app/openai_protocol.hpp"
#include "strata/app/model_catalog.hpp"
#include "strata/app/runtime.hpp"
#include "strata/engine/runtime_support.hpp"
#include "strata/models/common/tokenizer.hpp"

#include "strata/app/cli.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string model_type;
    std::string model_id;
    std::string models_preset;
    std::string reasoning_effort;
    std::string host{"127.0.0.1"};
    std::vector<int> devices;
    std::uint32_t context_size{2048U};
    std::uint32_t max_new_tokens{256U};
    std::uint16_t port{8080U};
    std::uint32_t models_max{1U};
    double vram_fraction{0.85};
    double temperature{1.0};
    double top_p{1.0};
    std::optional<std::uint64_t> seed;
    bool devices_explicit{};
    bool models_autoload{true};
    bool flash_attention{};
    bool block_kv_cache{};
    bool device_resident_runtime{};
    bool rank_local_decode{};
    bool pin_resident_arena{};
    // Prompt rows per prefill page. The device-resident path executes a page
    // layer-major and groups its rows by expert, which is exact and faster;
    // 1 restores row-at-a-time prompt processing.
    std::uint32_t prefill_page_tokens{};
    std::string static_expert_plan;
    std::uint64_t static_expert_bytes{};
    std::string plan_cache;
    bool dry_run{};
    bool use_plan_cache{true};
    bool replan{};
};

struct HttpRequest {
    std::string method;
    std::string target;
    std::string body;
};

volatile std::sig_atomic_t stop_requested = 0;
int listening_socket = -1;

void stop_server(int) {
    stop_requested = 1;
    if (listening_socket >= 0) close(listening_socket);
}

void usage() {
    std::cerr
        << "usage: strata-server --model DIR --model-type "
           "gemma4|deepseek|glm|glm53|laguna|inkling|kimi-k3\n"
        << "                     [--model-id ID] [--host ADDRESS] [--port N]\n"
        << "                     [--context-size N] [--max-new N]\n"
        << "                     [--devices 0,1,2] [--vram-fraction F]\n"
        << "                     [--flash-attention] [--block-kv-cache]\n"
        << "                     [--pin-resident-arena]\n"
        << "                     [--device-resident-runtime]\n"
        << "                     [--decode-topology centralized|rank-local-tp2]\n"
        << "                     [--prefill-page-tokens N]\n"
        << "                     [--static-expert-plan PATH]\n"
        << "                     [--static-expert-bytes BYTES]\n"
        << "                     [--dry-run] [--replan]\n"
        << "                     [--plan-cache DIR] [--no-plan-cache]\n"
        << "                     [--temperature F] [--top-p F] [--seed N]\n"
        << "                     [--reasoning-effort E]\n"
        << "   or: strata-server --models-preset FILE [--models-max N]\n"
        << "                     [--no-models-autoload] [--host ADDRESS] [--port N]\n\n"
        << "--device-resident-runtime is the DeepSeek device-resident decode\n"
        << "contract as a whole: physical KV pages, device-resident mHC, CUDA\n"
        << "attention, the scalar lightning indexer, and routed experts in the\n"
        << "two NUMA-local CPU shards. --decode-topology rank-local-tp2 adds\n"
        << "rank-local decode on top of it, and needs a build with NCCL and\n"
        << "exactly two devices. It is admitted fail-closed before the\n"
        << "checkpoint is opened and supports at most 65,536 context tokens\n"
        << "(issue #22). Both are DeepSeek-only and are rejected by every other\n"
        << "--model-type. Each topology is deterministic and exact against its\n"
        << "own oracle, but they are not token-identical to each other: greedy\n"
        << "decode can diverge a dozen steps in, so switching topology changes\n"
        << "the text a served request returns, not just its speed.\n\n"
        << "--dry-run sizes every component against this machine, prints the\n"
        << "placement, caches it, and exits without reading a weight. Run it\n"
        << "before starting a service to see whether the configuration fits.\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        }
        if (argument == "--flash-attention") {
            options.flash_attention = true;
            continue;
        }
        if (argument == "--block-kv-cache") {
            options.block_kv_cache = true;
            options.device_resident_runtime = false;
            continue;
        }
        if (argument == "--device-resident-runtime") {
            options.block_kv_cache = true;
            options.device_resident_runtime = true;
            // Implication of the contract; the runtime enforces the rest for
            // embedders that never pass through this parser.
            options.flash_attention = true;
            continue;
        }
        if (argument == "--pin-resident-arena") {
            options.pin_resident_arena = true;
            continue;
        }
        if (argument == "--dry-run") {
            options.dry_run = true;
            continue;
        }
        if (argument == "--no-plan-cache") {
            options.use_plan_cache = false;
            continue;
        }
        if (argument == "--replan") {
            options.replan = true;
            continue;
        }
        if (argument == "--no-models-autoload") {
            options.models_autoload = false;
            continue;
        }
        if (argument == "--models-autoload") {
            options.models_autoload = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const auto next = [&]() { return std::string_view(argv[++index]); };
        if (argument == "--model") options.model = next();
        else if (argument == "--models-preset") options.models_preset = next();
        else if (argument == "--plan-cache") options.plan_cache = next();
        else if (argument == "--model-type") options.model_type = next();
        else if (argument == "--model-id") options.model_id = next();
        else if (argument == "--reasoning-effort") options.reasoning_effort = next();
        else if (argument == "--host") options.host = next();
        else if (argument == "--decode-topology") {
            const auto topology = next();
            if (topology == "centralized") {
                options.rank_local_decode = false;
            } else if (topology == "rank-local-tp2") {
                // Rank-local decode owns the same weights the device-resident
                // contract places, so the opt-in implies that contract rather
                // than silently running rank-local against a scalar cache.
                options.rank_local_decode = true;
                options.block_kv_cache = true;
                options.device_resident_runtime = true;
                options.flash_attention = true;
            } else {
                return false;
            }
        }
        else if (argument == "--static-expert-plan") {
            options.static_expert_plan = next();
        }
        else if (argument == "--static-expert-bytes") {
            const auto text = next();
            std::uint64_t multiplier = 1U;
            auto value = text;
            const char suffix = value.empty() ? '\0' : value.back();
            if (suffix == 'G' || suffix == 'g') { multiplier = 1ULL << 30U; value.remove_suffix(1U); }
            else if (suffix == 'M' || suffix == 'm') { multiplier = 1ULL << 20U; value.remove_suffix(1U); }
            std::uint64_t base = 0U;
            if (!strata::cli::parse_u64(value, base)) return false;
            options.static_expert_bytes = base * multiplier;
        }
        else if (argument == "--prefill-page-tokens") {
            if (!strata::cli::parse_positive_u32(
                    next(), options.prefill_page_tokens)) {
                return false;
            }
        }
        else if (argument == "--port") {
            std::uint32_t port = 0U;
            if (!strata::cli::parse_positive_u32(next(), port) ||
                port > std::numeric_limits<std::uint16_t>::max()) return false;
            options.port = static_cast<std::uint16_t>(port);
        } else if (argument == "--context-size") {
            if (!strata::cli::parse_positive_u32(next(), options.context_size)) return false;
        } else if (argument == "--models-max") {
            if (!strata::cli::parse_positive_u32(next(), options.models_max) ||
                options.models_max > 64U) return false;
        } else if (argument == "--max-new") {
            if (!strata::cli::parse_positive_u32(next(), options.max_new_tokens)) return false;
        } else if (argument == "--vram-fraction") {
            if (!strata::cli::parse_double(next(), options.vram_fraction, 0.0, 0.95)) return false;
        } else if (argument == "--temperature") {
            if (!strata::cli::parse_double(next(), options.temperature, 0.0, 2.0)) return false;
        } else if (argument == "--top-p") {
            if (!strata::cli::parse_double(next(), options.top_p, 0.0, 1.0) ||
                options.top_p == 0.0) return false;
        } else if (argument == "--seed") {
            std::uint64_t value = 0U;
            if (!strata::cli::parse_u64(next(), value)) return false;
            options.seed = value;
        } else if (argument == "--devices") {
            if (!strata::cli::parse_devices(next(), options.devices)) return false;
            options.devices_explicit = true;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    if (!options.models_preset.empty()) {
        return options.model.empty() && options.model_type.empty() &&
               !options.dry_run;
    }
    if (!options.devices_explicit) options.devices = {0, 1, 2};
    if (options.model_id.empty() && !options.model.empty()) {
        options.model_id = std::filesystem::path(options.model).filename().string();
    }
    // A request's prompt and its generation share one context window, so a
    // default --max-new that fills it would reject every request after the
    // model had already loaded.
    if (options.max_new_tokens >= options.context_size) {
        std::cerr << "error: --max-new " << options.max_new_tokens
                  << " leaves no room for a prompt in --context-size "
                  << options.context_size
                  << "; the two share the context window\n";
        return false;
    }
    // Resolved through the model registry rather than a hardcoded list, so a
    // new model needs no edit here.
    const auto* registration =
        strata::find_model_by_cli_name(options.model_type);
    if (registration == nullptr) return false;
    // The server default every request inherits unless it names its own, so an
    // unusable value has to fail here rather than on every request.
    if (!options.reasoning_effort.empty()) {
        if (!registration->reasoning.accepts_effort()) {
            std::cerr << "error: --reasoning-effort is not supported by "
                      << options.model_type << '\n';
            return false;
        }
        if (!strata::reasoning_effort_accepted(registration->reasoning,
                                               options.reasoning_effort)) {
            std::cerr << "error: unknown --reasoning-effort: "
                      << options.reasoning_effort << "; " << options.model_type
                      << " accepts " << registration->reasoning.efforts << '\n';
            return false;
        }
    }
    return !options.model.empty() && !options.model_id.empty();
}

std::string lower(std::string_view text) {
    std::string result(text);
    for (auto& value : result) {
        if (value >= 'A' && value <= 'Z') value = static_cast<char>(value - 'A' + 'a');
    }
    return result;
}

bool send_all(int socket, std::string_view data) {
    while (!data.empty()) {
        const auto sent = send(socket, data.data(), data.size(), MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (sent == 0) return false;
        data.remove_prefix(static_cast<std::size_t>(sent));
    }
    return true;
}

bool client_disconnected(int socket) {
    char byte{};
    for (;;) {
        const auto received = recv(socket, &byte, 1U, MSG_PEEK | MSG_DONTWAIT);
        if (received == 0) return true;
        if (received > 0) return false;
        if (errno == EINTR) continue;
        return errno != EAGAIN && errno != EWOULDBLOCK;
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

std::size_t message_bytes(std::span<const strata::ChatMessage> messages) {
    std::size_t bytes = 0U;
    for (const auto& message : messages) {
        bytes += message.content.size();
        for (const auto& part : message.parts) bytes += part.data.size();
    }
    return bytes;
}

bool send_response(int socket, int status, std::string_view reason,
                   std::string_view content_type, std::string_view body) {
    std::ostringstream header;
    header << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
           << "Content-Type: " << content_type << "\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n"
           << "Access-Control-Allow-Origin: *\r\n"
           << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
           << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n";
    return send_all(socket, header.str()) && send_all(socket, body);
}

std::string error_json(std::string_view message, std::string_view type,
                       std::string_view code = {}) {
    std::ostringstream output;
    output << "{\"error\":{\"message\":\""
           << strata::cli::json_escape(message) << "\",\"type\":\""
           << type << "\",\"param\":null,\"code\":";
    if (code.empty()) output << "null";
    else output << '"' << strata::cli::json_escape(code) << '"';
    output << "}}";
    return output.str();
}

bool send_error(int socket, int status, std::string_view reason,
                std::string_view message, std::string_view type,
                std::string_view code = {}) {
    return send_response(socket, status, reason, "application/json",
                         error_json(message, type, code));
}

bool read_request(int socket, HttpRequest& request, std::string& error) {
    std::string input;
    input.reserve(4096U);
    std::array<char, 8192U> buffer{};
    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        const auto received = recv(socket, buffer.data(), buffer.size(), 0);
        if (received <= 0) {
            error = "connection closed before HTTP headers";
            return false;
        }
        input.append(buffer.data(), static_cast<std::size_t>(received));
        if (input.size() > 64U * 1024U) {
            error = "HTTP headers exceed 64 KiB";
            return false;
        }
        header_end = input.find("\r\n\r\n");
    }
    const auto first_end = input.find("\r\n");
    const auto first_space = input.find(' ');
    const auto second_space = first_space == std::string::npos
        ? std::string::npos : input.find(' ', first_space + 1U);
    if (first_end == std::string::npos || first_space == std::string::npos ||
        second_space == std::string::npos || second_space > first_end) {
        error = "malformed HTTP request line";
        return false;
    }
    request.method = input.substr(0U, first_space);
    request.target = input.substr(first_space + 1U, second_space - first_space - 1U);
    std::size_t content_length = 0U;
    std::size_t line = first_end + 2U;
    while (line < header_end) {
        const auto end = input.find("\r\n", line);
        const auto colon = input.find(':', line);
        if (end == std::string::npos || colon == std::string::npos || colon > end) {
            error = "malformed HTTP header";
            return false;
        }
        const auto name = lower(std::string_view(input).substr(line, colon - line));
        auto value = std::string_view(input).substr(colon + 1U, end - colon - 1U);
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
            value.remove_prefix(1U);
        }
        if (name == "content-length") {
            const auto parsed = std::from_chars(
                value.data(), value.data() + value.size(), content_length);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
                error = "invalid Content-Length";
                return false;
            }
        } else if (name == "transfer-encoding" && lower(value) != "identity") {
            error = "chunked request bodies are not supported";
            return false;
        }
        line = end + 2U;
    }
    if (content_length > strata::maximum_chat_request_bytes) {
        error = "request body exceeds 16 MiB";
        return false;
    }
    const auto body_begin = header_end + 4U;
    while (input.size() - body_begin < content_length) {
        const auto received = recv(socket, buffer.data(), buffer.size(), 0);
        if (received <= 0) {
            error = "connection closed before request body";
            return false;
        }
        input.append(buffer.data(), static_cast<std::size_t>(received));
    }
    request.body = input.substr(body_begin, content_length);
    return true;
}

std::string finish_reason(const strata::GenerationResult& result,
                          const strata::GenerationOptions& options) {
    return result.generated_token_ids.size() >= options.maximum_new_tokens
        && !result.stopped ? "length" : "stop";
}

std::string token_logprobs_json(
    const strata::GenerationResult& result,
    const strata::ModelTokenizer& tokenizer) {
    std::ostringstream output;
    const auto bytes = [](std::ostream& stream, std::string_view text) {
        stream << '[';
        for (std::size_t index = 0U; index < text.size(); ++index) {
            if (index != 0U) stream << ',';
            stream << static_cast<unsigned int>(
                static_cast<unsigned char>(text[index]));
        }
        stream << ']';
    };
    output << "{\"content\":[";
    for (std::size_t index = 0U; index < result.logprobs.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& token = result.logprobs[index];
        const auto piece = tokenizer.decode_token(token.token);
        const std::string piece_text = piece.ok() ? piece.value : "";
        output << "{\"token\":\"" << strata::cli::json_escape(piece_text)
               << "\",\"logprob\":" << token.logprob << ",\"bytes\":";
        bytes(output, piece_text);
        output << ",\"top_logprobs\":[";
        for (std::size_t top = 0U; top < token.top.size(); ++top) {
            if (top != 0U) output << ',';
            const auto top_piece = tokenizer.decode_token(token.top[top].first);
            const std::string top_text = top_piece.ok() ? top_piece.value : "";
            output << "{\"token\":\"" << strata::cli::json_escape(top_text)
                   << "\",\"logprob\":" << token.top[top].second
                   << ",\"bytes\":";
            bytes(output, top_text);
            output << '}';
        }
        output << "]}";
    }
    output << "]}";
    return output.str();
}

class ApiServer {
public:
    ApiServer(const Options& options, strata::RuntimeSession& runtime,
              strata::ModelTokenizer tokenizer,
              strata::ReasoningFormat reasoning)
        : options_(options), runtime_(runtime), tokenizer_(std::move(tokenizer)),
          reasoning_(reasoning) {}

    void handle(int socket, const HttpRequest& http) {
        const auto query = http.target.find('?');
        const auto target = std::string_view(http.target).substr(0U, query);
        if (http.method == "OPTIONS") {
            send_response(socket, 204, "No Content", "text/plain", "");
        } else if (http.method == "GET" && target == "/v1/health") {
            send_response(socket, 200, "OK", "application/json",
                          "{\"status\":\"ok\"}");
        } else if (http.method == "GET" && target == "/v1/models") {
            models(socket);
        } else if (http.method == "POST" && target == "/v1/chat/completions") {
            completion(socket, http.body, true);
        } else if (http.method == "POST" && target == "/v1/completions") {
            completion(socket, http.body, false);
        } else if (http.method == "POST" && target == "/v1/tokenize") {
            tokenize(socket, http.body);
        } else if (http.method == "POST" && target == "/v1/embeddings") {
            send_error(socket, 400, "Bad Request",
                       "the loaded generative model does not support embeddings",
                       "invalid_request_error", "unsupported_endpoint");
        } else {
            send_error(socket, 404, "Not Found", "unknown endpoint",
                       "invalid_request_error", "not_found");
        }
    }

private:
    void models(int socket) const {
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::ostringstream body;
        body << "{\"object\":\"list\",\"data\":[{\"id\":\""
             << strata::cli::json_escape(options_.model_id)
             << "\",\"object\":\"model\",\"created\":" << created
             << ",\"owned_by\":\"strata\"}]}";
        send_response(socket, 200, "OK", "application/json", body.str());
    }

    void tokenize(int socket, std::string_view body) const {
        std::string model;
        std::string text;
        std::string error;
        if (!strata::parse_openai_tokenize_request(body, model, text, error)) {
            send_error(socket, 400, "Bad Request", error, "invalid_request_error");
            return;
        }
        if (model != options_.model_id) {
            send_error(socket, 404, "Not Found", "requested model is not loaded",
                       "invalid_request_error", "model_not_found");
            return;
        }
        const auto encoded = tokenizer_.encode(text);
        if (!encoded.ok()) {
            send_error(socket, 400, "Bad Request", encoded.errors.front(),
                       "invalid_request_error");
            return;
        }
        std::ostringstream response;
        response << "{\"object\":\"tokenization\",\"count\":"
                 << encoded.value.size() << ",\"tokens\":[";
        for (std::size_t index = 0U; index < encoded.value.size(); ++index) {
            if (index != 0U) response << ',';
            response << encoded.value[index];
        }
        response << "]}";
        send_response(socket, 200, "OK", "application/json", response.str());
    }

    void completion(int socket, std::string_view body, bool chat) {
        const auto request_started = std::chrono::steady_clock::now();
        const auto id = next_id(chat);
        std::cerr << "[request] id=" << id << " phase=received endpoint="
                  << (chat ? "chat.completions" : "completions")
                  << " body_bytes=" << body.size() << '\n';
        strata::OpenAiChatRequest request;
        std::string error;
        const bool parsed = chat
            ? strata::parse_openai_chat_request(body, request, error)
            : strata::parse_openai_completion_request(body, request, error);
        if (!parsed) {
            std::cerr << "[request] id=" << id
                      << " phase=rejected status=400 error=" << error << '\n';
            send_error(socket, 400, "Bad Request", error, "invalid_request_error");
            return;
        }
        if (request.model != options_.model_id) {
            std::cerr << "[request] id=" << id
                      << " phase=rejected status=404 error=model_not_found\n";
            send_error(socket, 404, "Not Found", "requested model is not loaded",
                       "invalid_request_error", "model_not_found");
            return;
        }
        if (request.has_tools) {
            std::cerr << "[request] id=" << id
                      << " phase=rejected status=400 error=unsupported_tools\n";
            send_error(socket, 400, "Bad Request",
                       "tool calling is not supported by this loaded base model",
                       "invalid_request_error", "unsupported_parameter");
            return;
        }
        // A budget the loaded model does not accept is a bad parameter, not a
        // server fault. The runtime rejects it too, but only after the request
        // has been admitted, which would report it as a 500.
        if (!request.generation.reasoning_effort.empty() &&
            !strata::reasoning_effort_accepted(
                reasoning_, request.generation.reasoning_effort)) {
            std::cerr << "[request] id=" << id
                      << " phase=rejected status=400 error=invalid_reasoning_effort\n";
            send_error(socket, 400, "Bad Request",
                       reasoning_.accepts_effort()
                           ? "reasoning_effort must be one of: " +
                                 std::string(reasoning_.efforts)
                           : "reasoning_effort is not supported by this "
                             "loaded model",
                       "invalid_request_error", "unsupported_parameter");
            return;
        }
        for (const auto& [token, bias] : request.generation.sampling.logit_bias) {
            static_cast<void>(bias);
            if (token >= tokenizer_.vocabulary_size()) {
                std::cerr << "[request] id=" << id
                          << " phase=rejected status=400 error=invalid_logit_bias\n";
                send_error(socket, 400, "Bad Request",
                           "logit_bias token id is outside the loaded vocabulary",
                           "invalid_request_error", "invalid_logit_bias");
                return;
            }
        }
        if (!request.has_max_tokens) {
            request.generation.maximum_new_tokens = options_.max_new_tokens;
        }
        if (!request.has_temperature) {
            request.generation.sampling.temperature = options_.temperature;
        }
        if (!request.has_top_p) {
            request.generation.sampling.top_p = options_.top_p;
        }
        if (!request.has_seed && options_.seed.has_value()) {
            request.generation.sampling.seed = *options_.seed;
            request.has_seed = true;
        }
        std::cerr << "[request] id=" << id << " phase=accepted stream="
                  << (request.stream ? "true" : "false")
                  << " messages=" << request.messages.size()
                  << " message_bytes=" << message_bytes(request.messages)
                  << " choices=" << request.n
                  << " max_tokens=" << request.generation.maximum_new_tokens << '\n';
        if (request.stream) stream(socket, request, chat, id, request_started);
        else complete(socket, request, chat, id, request_started);
    }

    std::string next_id(bool chat) {
        std::ostringstream id;
        id << (chat ? "chatcmpl-" : "cmpl-") << ++request_id_;
        return id.str();
    }

    void stream(int socket, const strata::OpenAiChatRequest& request, bool chat,
                const std::string& id,
                std::chrono::steady_clock::time_point request_started) {
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string header =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n";
        if (!send_all(socket, header)) {
            std::cerr << "[request] id=" << id
                      << " phase=cancelled stage=response_headers elapsed_ms="
                      << elapsed_ms(request_started) << '\n';
            return;
        }
        strata::GenerationMetrics aggregated;
        std::uint64_t returned_tokens = 0U;
        for (std::uint32_t index = 0U; index < request.n; ++index) {
            auto options = request.generation;
            options.sampling.seed = request.has_seed
                ? options.sampling.seed + index
                : static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count()) + index;
            if (chat) {
                std::ostringstream role;
                role << "data: {\"id\":\"" << id
                     << "\",\"object\":\"chat.completion.chunk\",\"created\":"
                     << created << ",\"model\":\""
                     << strata::cli::json_escape(request.model)
                     << "\",\"choices\":[{\"index\":" << index
                     << ",\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                        "\"logprobs\":null,\"finish_reason\":null}]}\n\n";
                if (!send_all(socket, role.str())) {
                    std::cerr << "[request] id=" << id
                              << " phase=cancelled stage=role_chunk elapsed_ms="
                              << elapsed_ms(request_started) << '\n';
                    return;
                }
            }
            bool connected = true;
            bool valid_utf8 = true;
            bool first_token = true;
            std::string utf8_pending;
            strata::ReasoningSplitter splitter(chat_reasoning(chat));
            // One delta object may carry both halves: the piece that completes
            // the closing tag ends the reasoning and opens the answer.
            const auto delta_json =
                [&](const strata::ReasoningSplitter::Delta& delta) {
                    std::ostringstream fields;
                    fields << "\"delta\":{";
                    bool written = false;
                    if (request.include_reasoning && !delta.reasoning.empty()) {
                        fields << "\"reasoning_content\":\""
                               << strata::cli::json_escape(delta.reasoning)
                               << '"';
                        written = true;
                    }
                    if (!delta.content.empty()) {
                        if (written) fields << ',';
                        fields << "\"content\":\""
                               << strata::cli::json_escape(delta.content) << '"';
                    }
                    fields << '}';
                    return fields.str();
                };
            const auto send_chunk = [&](const std::string& body) {
                std::ostringstream chunk;
                chunk << "data: {\"id\":\"" << id << "\",\"object\":\""
                      << (chat ? "chat.completion.chunk" : "text_completion")
                      << "\",\"created\":" << created << ",\"model\":\""
                      << strata::cli::json_escape(request.model)
                      << "\",\"choices\":[{\"index\":" << index << ','
                      << body
                      << ",\"logprobs\":null,\"finish_reason\":null}]}\n\n";
                return send_all(socket, chunk.str());
            };
            std::cerr << "[request] id=" << id
                      << " phase=generation_start choice=" << index << '\n';
            const strata::TokenStreamCallback callback =
                [&](std::uint32_t, std::string_view piece) {
                    if (!connected) return false;
                    if (client_disconnected(socket)) {
                        connected = false;
                        return false;
                    }
                    if (first_token) {
                        first_token = false;
                        std::cerr << "[request] id=" << id
                                  << " phase=first_token choice=" << index
                                  << " ttft_ms=" << elapsed_ms(request_started) << '\n';
                    }
                    if (request.json_object) return true;
                    utf8_pending.append(piece);
                    const auto complete = strata::complete_utf8_prefix(utf8_pending);
                    if (complete == std::string_view::npos) {
                        valid_utf8 = false;
                        return false;
                    }
                    if (complete == 0U) return true;
                    // Copied out before the buffer is trimmed: a view into
                    // `utf8_pending` would dangle the moment it is erased.
                    const std::string ready(utf8_pending, 0U, complete);
                    utf8_pending.erase(0U, complete);
                    if (!chat) {
                        connected = send_chunk(
                            "\"text\":\"" +
                            strata::cli::json_escape(ready) + '"');
                        return connected;
                    }
                    const auto delta = splitter.consume(ready);
                    // A piece held back as a possible delimiter produces no
                    // delta yet; sending an empty one would be a chunk that
                    // says nothing.
                    if (delta.empty()) return true;
                    connected = send_chunk(delta_json(delta));
                    return connected;
                };
            auto result = runtime_.generate_chat_stream(request.messages, options, callback);
            // Whatever the splitter still holds was real output: a generation
            // that stopped inside the closing tag -- the documented outcome
            // when --max-new cannot reach it -- must not silently lose it.
            if (result.ok() && connected && chat) {
                const auto tail = splitter.finish();
                if (!tail.empty()) connected = send_chunk(delta_json(tail));
            }
            if (!result.ok()) {
                std::cerr << "[request] id=" << id
                          << " phase=error stage=generation error="
                          << result.errors.front() << '\n';
                send_all(socket, "data: " + error_json(result.errors.front(), "server_error") + "\n\n");
                send_all(socket, "data: [DONE]\n\n");
                return;
            }
            aggregated.prompt_tokens += result.metrics.prompt_tokens;
            aggregated.prefill_tokens += result.metrics.prefill_tokens;
            aggregated.prefill_seconds += result.metrics.prefill_seconds;
            aggregated.decode_tokens += result.metrics.decode_tokens;
            aggregated.decode_seconds += result.metrics.decode_seconds;
            aggregated.reused_prompt_tokens =
                aggregated.reused_prompt_tokens.value_or(0U) +
                result.metrics.reused_prompt_tokens.value_or(0U);
            returned_tokens += result.generated_token_ids.size();
            if (!connected) {
                std::cerr << "[request] id=" << id
                          << " phase=cancelled stage=generation completion_tokens="
                          << result.generated_token_ids.size()
                          << " elapsed_ms=" << elapsed_ms(request_started) << '\n';
                return;
            }
            if (!valid_utf8 || !utf8_pending.empty()) {
                std::cerr << "[request] id=" << id
                          << " phase=error stage=utf8\n";
                send_all(socket, "data: " + error_json(
                    "generated text is not valid UTF-8", "server_error") + "\n\n");
                send_all(socket, "data: [DONE]\n\n");
                return;
            }
            if (request.json_object) {
                if (!strata::is_json_object(result.text)) {
                    send_all(socket, "data: " + error_json(
                        "model did not produce a valid JSON object", "server_error") + "\n\n");
                    send_all(socket, "data: [DONE]\n\n");
                    return;
                }
                std::ostringstream chunk;
                chunk << "data: {\"id\":\"" << id
                      << "\",\"object\":\"chat.completion.chunk\",\"created\":"
                      << created << ",\"model\":\""
                      << strata::cli::json_escape(request.model)
                      << "\",\"choices\":[{\"index\":" << index
                      << ",\"delta\":{\"content\":\""
                      << strata::cli::json_escape(result.text)
                      << "\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n";
                if (!send_all(socket, chunk.str())) return;
            }
            std::ostringstream final;
            final << "data: {\"id\":\"" << id << "\",\"object\":\""
                  << (chat ? "chat.completion.chunk" : "text_completion")
                  << "\",\"created\":" << created << ",\"model\":\""
                  << strata::cli::json_escape(request.model)
                  << "\",\"choices\":[{\"index\":" << index << ',';
            if (chat) final << "\"delta\":{}";
            else final << "\"text\":\"\"";
            final << ",\"logprobs\":null,\"finish_reason\":\""
                  << finish_reason(result, options) << "\"}]";
            if (index + 1U == request.n) {
                final << ",\"usage\":"
                      << strata::render_openai_usage(
                          aggregated.prompt_tokens, aggregated.decode_tokens,
                          aggregated.reused_prompt_tokens.value_or(0U))
                      << ",\"timings\":" << strata::render_openai_timings(aggregated);
            }
            final << "}\n\n";
            if (!send_all(socket, final.str())) return;
        }
        if (!send_all(socket, "data: [DONE]\n\n")) {
            std::cerr << "[request] id=" << id
                      << " phase=cancelled stage=done_chunk elapsed_ms="
                      << elapsed_ms(request_started) << '\n';
            return;
        }
        std::cerr << "[request] id=" << id
                  << " phase=completed status=200 prompt_tokens="
                  << aggregated.prompt_tokens << " completion_tokens="
                  << returned_tokens << " prefill_ms="
                  << aggregated.prefill_seconds * 1000.0 << " decode_ms="
                  << aggregated.decode_seconds * 1000.0 << " total_ms="
                  << elapsed_ms(request_started) << '\n';
    }

    void complete(int socket, const strata::OpenAiChatRequest& request, bool chat,
                  const std::string& id,
                  std::chrono::steady_clock::time_point request_started) {
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::vector<strata::GenerationResult> results;
        results.reserve(request.n);
        for (std::uint32_t index = 0U; index < request.n; ++index) {
            auto options = request.generation;
            options.sampling.seed = request.has_seed
                ? options.sampling.seed + index
                : static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count()) + index;
            std::cerr << "[request] id=" << id
                      << " phase=generation_start choice=" << index << '\n';
            auto result = runtime_.generate_chat_stream(request.messages, options);
            if (!result.ok()) {
                std::cerr << "[request] id=" << id
                          << " phase=error stage=generation error="
                          << result.errors.front() << '\n';
                send_error(socket, 500, "Internal Server Error", result.errors.front(),
                           "server_error");
                return;
            }
            // Judge the answer, not the scratchpad. A reasoning model's raw
            // text always opens with its reasoning, so checking that would fail
            // every JSON-mode request the model actually answered correctly.
            if (request.json_object &&
                !strata::is_json_object(
                    strata::split_reasoning(chat_reasoning(chat),
                                            result.text).content)) {
                send_error(socket, 500, "Internal Server Error",
                           "model did not produce a valid JSON object", "server_error");
                return;
            }
            results.push_back(std::move(result));
        }
        std::uint64_t completion_tokens = 0U;
        strata::GenerationMetrics aggregated;
        std::ostringstream response;
        response << "{\"id\":\"" << id << "\",\"object\":\""
                 << (chat ? "chat.completion" : "text_completion")
                 << "\",\"created\":" << created << ",\"model\":\""
                 << strata::cli::json_escape(request.model)
                 << "\",\"system_fingerprint\":\"strata-" STRATA_VERSION
                    "\",\"choices\":[";
        for (std::size_t index = 0U; index < results.size(); ++index) {
            if (index != 0U) response << ',';
            const auto& result = results[index];
            completion_tokens += result.generated_token_ids.size();
            aggregated.prompt_tokens += result.metrics.prompt_tokens;
            aggregated.prefill_tokens += result.metrics.prefill_tokens;
            aggregated.prefill_seconds += result.metrics.prefill_seconds;
            aggregated.decode_tokens += result.metrics.decode_tokens;
            aggregated.decode_seconds += result.metrics.decode_seconds;
            aggregated.reused_prompt_tokens =
                aggregated.reused_prompt_tokens.value_or(0U) +
                result.metrics.reused_prompt_tokens.value_or(0U);
            response << "{\"index\":" << index << ',';
            if (chat) {
                const auto split = strata::split_reasoning(chat_reasoning(true),
                                                           result.text);
                response << "\"message\":{\"role\":\"assistant\",";
                if (request.include_reasoning && !split.reasoning.empty()) {
                    response << "\"reasoning_content\":\""
                             << strata::cli::json_escape(split.reasoning)
                             << "\",";
                }
                response << "\"content\":\""
                         << strata::cli::json_escape(split.content)
                         << "\",\"refusal\":null,\"tool_calls\":null}";
            } else {
                response << "\"text\":\"" << strata::cli::json_escape(result.text) << '"';
            }
            response << ",\"logprobs\":";
            if (request.logprobs) response << token_logprobs_json(result, tokenizer_);
            else response << "null";
            response << ",\"finish_reason\":\""
                     << finish_reason(result, request.generation) << "\"}";
        }
        const auto prompt_tokens = aggregated.prompt_tokens;
        response << "],\"usage\":"
                 << strata::render_openai_usage(prompt_tokens, completion_tokens,
                                                aggregated.reused_prompt_tokens.value_or(0U))
                 << ",\"timings\":" << strata::render_openai_timings(aggregated)
                 << "}";
        double model_seconds = 0.0;
        double decode_seconds = 0.0;
        std::uint64_t decode_steps = 0U;
        for (const auto& result : results) {
            model_seconds += result.metrics.prefill_seconds + result.metrics.decode_seconds;
            decode_seconds += result.metrics.decode_seconds;
            decode_steps += result.metrics.decode_tokens;
        }
        const double request_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - request_started).count();
        const double overhead_ms = std::max(0.0, request_seconds - model_seconds) * 1000.0;
        const double decode_step_ms = decode_steps == 0U ? 0.0
            : decode_seconds * 1000.0 / static_cast<double>(decode_steps);
        std::cerr << "[request] id=" << id
                  << " phase=completed status=200 prompt_tokens=" << prompt_tokens
                  << " completion_tokens=" << completion_tokens
                  << " total_ms=" << request_seconds * 1000.0
                  << " serving_overhead_ms=" << overhead_ms
                  << " decode_step_ms=" << decode_step_ms
                  << " overhead_percent="
                  << (decode_step_ms == 0.0 ? 0.0 : overhead_ms * 100.0 / decode_step_ms)
                  << '\n';
        send_response(socket, 200, "OK", "application/json", response.str());
    }

    // How this model delimits reasoning, from its registration. Chat responses
    // return the two halves separately; the completions API has no field for
    // reasoning, so its text is passed through whole.
    [[nodiscard]] strata::ReasoningFormat chat_reasoning(bool chat) const noexcept {
        return chat ? reasoning_ : strata::ReasoningFormat{};
    }

    const Options& options_;
    strata::RuntimeSession& runtime_;
    strata::ModelTokenizer tokenizer_;
    strata::ReasoningFormat reasoning_{};
    std::atomic<std::uint64_t> request_id_{};
};

enum class RouterModelStatus : std::uint8_t {
    Unloaded,
    Loading,
    Loaded,
    Failed,
};

std::string_view router_status_name(RouterModelStatus status) {
    switch (status) {
        case RouterModelStatus::Unloaded: return "unloaded";
        case RouterModelStatus::Loading: return "loading";
        case RouterModelStatus::Loaded: return "loaded";
        case RouterModelStatus::Failed: return "failed";
    }
    return "failed";
}

struct RouterModelInstance {
    strata::ModelCatalogEntry catalog;
    RouterModelStatus status{RouterModelStatus::Unloaded};
    pid_t process{-1};
    std::uint16_t port{};
    std::uint64_t last_used{};
    std::uint32_t active_requests{};
    int exit_code{};
};

std::uint64_t monotonic_milliseconds() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

int connect_loopback(std::uint16_t port) {
    // Close-on-exec everywhere: the router forks and execs model children
    // while holding listening, client, and upstream sockets. An inherited
    // duplicate keeps peer connections open after the router closes its own
    // copy, so streaming clients would never observe end-of-stream.
    const int socket_fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket_fd < 0) return -1;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(socket_fd);
        return -1;
    }
    return socket_fd;
}

std::optional<std::uint16_t> unused_loopback_port() {
    const int socket_fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket_fd < 0) return std::nullopt;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(socket_fd);
        return std::nullopt;
    }
    socklen_t length = sizeof(address);
    if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        close(socket_fd);
        return std::nullopt;
    }
    const auto port = ntohs(address.sin_port);
    close(socket_fd);
    return port;
}

std::string current_executable(std::string_view argv0) {
    std::array<char, 4096U> path{};
    const auto length = readlink("/proc/self/exe", path.data(), path.size() - 1U);
    if (length > 0) return std::string(path.data(), static_cast<std::size_t>(length));
    return std::filesystem::absolute(std::filesystem::path(argv0)).string();
}

int process_exit_code(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
}

class RouterServer {
public:
    RouterServer(const Options& options, strata::ModelCatalog catalog,
                 std::string executable)
        : options_(options), executable_(std::move(executable)) {
        if (const char* override_path = std::getenv("STRATA_ROUTER_CHILD_EXECUTABLE")) {
            if (*override_path != '\0') executable_ = override_path;
        }
        models_.reserve(catalog.models.size());
        for (auto& entry : catalog.models) {
            RouterModelInstance instance;
            instance.catalog = std::move(entry);
            models_.push_back(std::move(instance));
        }
    }

    ~RouterServer() { shutdown(); }

    void load_startup_models() {
        for (auto& model : models_) {
            if (!model.catalog.load_on_startup) continue;
            std::string error;
            if (!ensure_loaded(model, error)) {
                std::cerr << "error: startup load for " << model.catalog.id
                          << " failed: " << error << '\n';
            }
        }
    }

    void shutdown() {
        if (shutdown_) return;
        shutdown_ = true;
        for (auto& model : models_) {
            if (model.process > 0) stop_child(model, false);
        }
    }

    void handle(int socket_fd, const HttpRequest& request) {
        refresh_children();
        const auto query = request.target.find('?');
        const auto target = std::string_view(request.target).substr(0U, query);
        if (request.method == "OPTIONS") {
            send_response(socket_fd, 204, "No Content", "text/plain", "");
        } else if (request.method == "GET" &&
                   (target == "/v1/health" || target == "/health")) {
            send_response(socket_fd, 200, "OK", "application/json",
                          "{\"status\":\"ok\",\"role\":\"router\"}");
        } else if (request.method == "GET" && target == "/props") {
            props(socket_fd);
        } else if (request.method == "GET" &&
                   (target == "/v1/models" || target == "/models")) {
            models(socket_fd);
        } else if (request.method == "POST" && target == "/models/load") {
            load(socket_fd, request.body);
        } else if (request.method == "POST" && target == "/models/unload") {
            unload(socket_fd, request.body);
        } else if (request.method == "POST") {
            route_post(socket_fd, request);
        } else {
            send_error(socket_fd, 404, "Not Found", "unknown router endpoint",
                       "invalid_request_error", "not_found");
        }
    }

private:
    RouterModelInstance* find(std::string_view id) {
        const auto found = std::find_if(models_.begin(), models_.end(),
            [&](const RouterModelInstance& model) { return model.catalog.id == id; });
        return found == models_.end() ? nullptr : &*found;
    }

    void refresh_children() {
        for (auto& model : models_) {
            if (model.process <= 0) continue;
            int status = 0;
            const auto waited = waitpid(model.process, &status, WNOHANG);
            if (waited != model.process) continue;
            model.exit_code = process_exit_code(status);
            model.process = -1;
            model.port = 0;
            if (model.status == RouterModelStatus::Loaded ||
                model.status == RouterModelStatus::Loading) {
                model.status = RouterModelStatus::Failed;
                std::cerr << "[router] model=" << model.catalog.id
                          << " child exited code=" << model.exit_code << '\n';
            }
        }
    }

    std::size_t running_count() const {
        return static_cast<std::size_t>(std::count_if(
            models_.begin(), models_.end(), [](const RouterModelInstance& model) {
                return model.status == RouterModelStatus::Loaded ||
                       model.status == RouterModelStatus::Loading;
            }));
    }

    bool make_capacity(RouterModelInstance& requested, std::string& error) {
        while (running_count() >= options_.models_max) {
            RouterModelInstance* victim = nullptr;
            for (auto& candidate : models_) {
                if (&candidate == &requested ||
                    candidate.status != RouterModelStatus::Loaded ||
                    candidate.active_requests != 0U) continue;
                if (victim == nullptr || candidate.last_used < victim->last_used) {
                    victim = &candidate;
                }
            }
            if (victim == nullptr) {
                error = "all model slots are busy";
                return false;
            }
            std::cerr << "[router] evicting model=" << victim->catalog.id
                      << " for model=" << requested.catalog.id << '\n';
            stop_child(*victim, true);
        }
        return true;
    }

    bool ensure_loaded(RouterModelInstance& model, std::string& error) {
        refresh_children();
        if (model.status == RouterModelStatus::Loaded) return true;
        if (!make_capacity(model, error)) return false;
        const auto port = unused_loopback_port();
        if (!port.has_value()) {
            error = "cannot allocate a loopback port";
            return false;
        }

        std::vector<std::string> arguments{
            executable_, "--model", model.catalog.model_directory,
            "--model-type", model.catalog.model_type,
            "--model-id", model.catalog.id,
            "--host", "127.0.0.1", "--port", std::to_string(*port)
        };
        arguments.insert(arguments.end(), model.catalog.launch_arguments.begin(),
                         model.catalog.launch_arguments.end());
        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1U);
        for (auto& argument : arguments) argv.push_back(argument.data());
        argv.push_back(nullptr);

        std::cerr << "[router] spawning model=" << model.catalog.id
                  << " port=" << *port << '\n';
        const auto child = fork();
        if (child < 0) {
            error = std::string("fork failed: ") + std::strerror(errno);
            return false;
        }
        if (child == 0) {
            execv(executable_.c_str(), argv.data());
            std::cerr << "error: cannot exec router child " << executable_
                      << ": " << std::strerror(errno) << '\n';
            _exit(127);
        }
        model.process = child;
        model.port = *port;
        model.status = RouterModelStatus::Loading;
        model.exit_code = 0;

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        while (stop_requested == 0 && std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            const auto waited = waitpid(child, &status, WNOHANG);
            if (waited == child) {
                model.exit_code = process_exit_code(status);
                model.process = -1;
                model.port = 0;
                model.status = RouterModelStatus::Failed;
                error = "model child exited with code " + std::to_string(model.exit_code);
                return false;
            }
            const int probe = connect_loopback(*port);
            if (probe >= 0) {
                close(probe);
                model.status = RouterModelStatus::Loaded;
                model.last_used = monotonic_milliseconds();
                std::cerr << "[router] model=" << model.catalog.id << " loaded\n";
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        error = stop_requested == 0 ? "model load timed out after one hour"
                                    : "router is shutting down";
        stop_child(model, false);
        model.status = RouterModelStatus::Failed;
        return false;
    }

    void stop_child(RouterModelInstance& model, bool clear_failure) {
        if (model.process <= 0) {
            model.status = RouterModelStatus::Unloaded;
            if (clear_failure) model.exit_code = 0;
            return;
        }
        const auto process = model.process;
        std::cerr << "[router] stopping model=" << model.catalog.id << '\n';
        static_cast<void>(kill(process, SIGTERM));
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(model.catalog.stop_timeout_seconds);
        int status = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto waited = waitpid(process, &status, WNOHANG);
            if (waited == process || (waited < 0 && errno == ECHILD)) {
                model.process = -1;
                model.port = 0;
                model.status = RouterModelStatus::Unloaded;
                if (clear_failure) model.exit_code = 0;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cerr << "warning: force-killing model=" << model.catalog.id << '\n';
        static_cast<void>(kill(process, SIGKILL));
        while (waitpid(process, &status, 0) < 0 && errno == EINTR) {}
        model.process = -1;
        model.port = 0;
        model.status = RouterModelStatus::Unloaded;
        if (clear_failure) model.exit_code = 0;
    }

    void props(int socket_fd) const {
        std::ostringstream body;
        body << "{\"role\":\"router\",\"max_instances\":"
             << options_.models_max << ",\"models_autoload\":"
             << (options_.models_autoload ? "true" : "false") << '}';
        send_response(socket_fd, 200, "OK", "application/json", body.str());
    }

    void models(int socket_fd) const {
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::ostringstream body;
        body << "{\"object\":\"list\",\"data\":[";
        for (std::size_t index = 0U; index < models_.size(); ++index) {
            if (index != 0U) body << ',';
            const auto& model = models_[index];
            body << "{\"id\":\"" << strata::cli::json_escape(model.catalog.id)
                 << "\",\"object\":\"model\",\"created\":" << created
                 << ",\"owned_by\":\"strata\",\"name\":\""
                 << strata::cli::json_escape(model.catalog.name)
                 << "\",\"model_type\":\""
                 << strata::cli::json_escape(model.catalog.model_type)
                 << "\",\"status\":{\"value\":\""
                 << router_status_name(model.status) << '"';
            if (model.status == RouterModelStatus::Failed) {
                body << ",\"failed\":true,\"exit_code\":" << model.exit_code;
            }
            body << "},\"defaults\":{";
            bool wrote_default = false;
            const auto write_default = [&](std::string_view key, const auto& value) {
                if (wrote_default) body << ',';
                body << '"' << key << "\":" << value;
                wrote_default = true;
            };
            if (model.catalog.maximum_new_tokens) {
                write_default("max_tokens", *model.catalog.maximum_new_tokens);
            }
            if (model.catalog.temperature) {
                write_default("temperature", *model.catalog.temperature);
            }
            if (model.catalog.top_p) write_default("top_p", *model.catalog.top_p);
            if (model.catalog.seed) write_default("seed", *model.catalog.seed);
            body << "}}";
        }
        body << "]}";
        send_response(socket_fd, 200, "OK", "application/json", body.str());
    }

    void load(int socket_fd, std::string_view body) {
        std::string id;
        std::string error;
        if (!strata::parse_openai_model_field(body, id, error)) {
            send_error(socket_fd, 400, "Bad Request", error, "invalid_request_error");
            return;
        }
        auto* model = find(id);
        if (model == nullptr) {
            send_error(socket_fd, 404, "Not Found", "model is not in the catalog",
                       "invalid_request_error", "model_not_found");
            return;
        }
        if (!ensure_loaded(*model, error)) {
            send_error(socket_fd, 503, "Service Unavailable", error, "server_error",
                       "model_load_failed");
            return;
        }
        send_response(socket_fd, 200, "OK", "application/json", "{\"success\":true}");
    }

    void unload(int socket_fd, std::string_view body) {
        std::string id;
        std::string error;
        if (!strata::parse_openai_model_field(body, id, error)) {
            send_error(socket_fd, 400, "Bad Request", error, "invalid_request_error");
            return;
        }
        auto* model = find(id);
        if (model == nullptr) {
            send_error(socket_fd, 404, "Not Found", "model is not in the catalog",
                       "invalid_request_error", "model_not_found");
            return;
        }
        if (model->active_requests != 0U) {
            send_error(socket_fd, 409, "Conflict", "model is serving a request",
                       "invalid_request_error", "model_busy");
            return;
        }
        stop_child(*model, true);
        send_response(socket_fd, 200, "OK", "application/json", "{\"success\":true}");
    }

    void route_post(int socket_fd, const HttpRequest& request) {
        std::string id;
        std::string error;
        if (!strata::parse_openai_model_field(request.body, id, error)) {
            send_error(socket_fd, 400, "Bad Request", error, "invalid_request_error");
            return;
        }
        auto* model = find(id);
        if (model == nullptr) {
            send_error(socket_fd, 404, "Not Found", "requested model is not in the catalog",
                       "invalid_request_error", "model_not_found");
            return;
        }
        if (model->status != RouterModelStatus::Loaded) {
            if (!options_.models_autoload) {
                send_error(socket_fd, 409, "Conflict", "requested model is not loaded",
                           "invalid_request_error", "model_not_loaded");
                return;
            }
            if (!ensure_loaded(*model, error)) {
                send_error(socket_fd, 503, "Service Unavailable", error, "server_error",
                           "model_load_failed");
                return;
            }
        }
        proxy(socket_fd, request, *model);
    }

    void proxy(int socket_fd, const HttpRequest& request, RouterModelInstance& model) {
        const int upstream = connect_loopback(model.port);
        if (upstream < 0) {
            refresh_children();
            send_error(socket_fd, 502, "Bad Gateway", "cannot connect to model child",
                       "server_error", "upstream_unavailable");
            return;
        }
        std::ostringstream head;
        head << request.method << ' ' << request.target << " HTTP/1.1\r\n"
             << "Host: 127.0.0.1:" << model.port << "\r\n"
             << "Content-Type: application/json\r\n"
             << "Accept: */*\r\n"
             << "Content-Length: " << request.body.size() << "\r\n"
             << "Connection: close\r\n\r\n";
        if (!send_all(upstream, head.str()) || !send_all(upstream, request.body)) {
            close(upstream);
            send_error(socket_fd, 502, "Bad Gateway", "cannot write to model child",
                       "server_error", "upstream_unavailable");
            return;
        }
        ++model.active_requests;
        model.last_used = monotonic_milliseconds();
        std::array<char, 64U * 1024U> buffer{};
        bool received_any = false;
        for (;;) {
            const auto received = recv(upstream, buffer.data(), buffer.size(), 0);
            if (received == 0) break;
            if (received < 0) {
                if (errno == EINTR) continue;
                if (!received_any) {
                    send_error(socket_fd, 502, "Bad Gateway", "cannot read from model child",
                               "server_error", "upstream_unavailable");
                }
                break;
            }
            received_any = true;
            if (!send_all(socket_fd, std::string_view(
                    buffer.data(), static_cast<std::size_t>(received)))) break;
        }
        close(upstream);
        --model.active_requests;
    }

    const Options& options_;
    std::string executable_;
    std::vector<RouterModelInstance> models_;
    bool shutdown_{};
};

int bind_server(const Options& options) {
    const int socket_fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket_fd < 0) return -1;
    int enabled = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(options.port);
    if (options.host == "0.0.0.0" || options.host == "*") {
        address.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (options.host == "localhost") {
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
        close(socket_fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        listen(socket_fd, 16) != 0) {
        close(socket_fd);
        return -1;
    }
    return socket_fd;
}

}  // namespace

int main(int argc, char** argv) {
    // Pin CUDA enumeration to PCI bus order, exactly as strata-deepseek-run
    // does, so `--devices 1,2` names the same two physical cards from either
    // binary. Without this the runtime takes CUDA's default "fastest first"
    // ordering, which on a mixed box does not match `nvidia-smi` and silently
    // selects different hardware than the operator asked for.
    //
    // On the development box the two orderings are:
    //   default:     0,1 = RTX 3090 (sm_86), 2 = RTX 5060 Ti (sm_120)
    //   PCI_BUS_ID:  0 = RTX 5060 Ti (sm_120), 1,2 = RTX 3090 (sm_86)
    // so `--devices 1,2` meant one 3090 paired with the 5060 Ti under the
    // server and the two 3090s under the runner. That put an sm_120 card into
    // a rank-local pair whose mHC contract requires sm_86, capped both ranks'
    // symmetric VRAM admission at the 16 GiB card, and left one 3090 idle.
    //
    // Set only when the operator has not chosen an order themselves.
    if (std::getenv("CUDA_DEVICE_ORDER") == nullptr) {
        static_cast<void>(setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0));
    }
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }
    if (!options.models_preset.empty()) {
        auto catalog = strata::load_model_catalog(options.models_preset);
        if (!catalog.ok()) {
            for (const auto& error : catalog.errors) {
                std::cerr << "error: " << error << '\n';
            }
            return 1;
        }
        const auto catalog_size = catalog.value.models.size();
        RouterServer router(options, std::move(catalog.value),
                            current_executable(argv[0]));
        listening_socket = bind_server(options);
        if (listening_socket < 0) {
            std::cerr << "error: cannot listen on " << options.host << ':' << options.port
                      << ": " << std::strerror(errno) << '\n';
            return 1;
        }
        std::signal(SIGINT, stop_server);
        std::signal(SIGTERM, stop_server);
        std::signal(SIGPIPE, SIG_IGN);
        std::cerr << "[ready] http://" << options.host << ':' << options.port
                  << " role=router models=" << catalog_size << "\n";
        router.load_startup_models();
        while (stop_requested == 0) {
            const int client = accept4(listening_socket, nullptr, nullptr, SOCK_CLOEXEC);
            if (client < 0) {
                if (errno == EINTR || stop_requested != 0) continue;
                std::cerr << "warning: accept failed: " << std::strerror(errno) << '\n';
                continue;
            }
            HttpRequest request;
            std::string error;
            if (!read_request(client, request, error)) {
                send_error(client, 400, "Bad Request", error, "invalid_request_error");
            } else {
                router.handle(client, request);
            }
            close(client);
        }
        listening_socket = -1;
        router.shutdown();
        std::cerr << "[shutdown] router stopped cleanly\n";
        return 0;
    }
    // parse_options already rejected an unregistered --model-type.
    const auto* registration =
        strata::find_model_by_cli_name(options.model_type);
    strata::RuntimeConfig config;
    config.model = registration->model;
    config.devices = options.devices;
    config.maximum_context_tokens = options.context_size;
    config.vram_cache_fraction = options.vram_fraction;
    config.enable_flash_attention =
        options.flash_attention || registration->flash_attention_by_default;
    config.deepseek_block_kv_cache = options.block_kv_cache;
    config.deepseek_device_resident_runtime = options.device_resident_runtime;
    config.deepseek_prefill_page_tokens = options.prefill_page_tokens;
    config.deepseek_rank_local_decode = options.rank_local_decode;
    config.deepseek_static_expert_plan = options.static_expert_plan;
    config.deepseek_static_expert_bytes = options.static_expert_bytes;
    config.pin_resident_arena = options.pin_resident_arena;
    config.verbose = registration->verbose_by_default;
    config.load_progress = registration->progress_by_default;
    config.placement_cache_directory = options.plan_cache;
    config.use_placement_cache = options.use_plan_cache;
    config.refresh_placement_plan = options.replan;
    config.report_placement_plan = true;
    config.reasoning_effort = options.reasoning_effort;

    if (options.dry_run) {
        const auto resolved = strata::resolve_placement_plan(
            strata::placement_request_for(options.model, config),
            options.plan_cache, options.use_plan_cache, options.replan);
        if (!resolved.ok()) {
            for (const auto& error : resolved.errors) {
                std::cerr << "error: " << error << '\n';
            }
            return 1;
        }
        std::cout << strata::render_placement_report(resolved.value.plan);
        std::cerr << "[dry-run] plan at " << resolved.value.cache_path << '\n';
        return resolved.value.plan.fits ? 0 : 1;
    }

    auto tokenizer = strata::ModelTokenizer::load(
        (std::filesystem::path(options.model) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        for (const auto& error : tokenizer.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    strata::RuntimeSession runtime;
    std::cerr << "[startup] loading " << options.model_id << "...\n";
    const auto initialized = runtime.initialize(options.model, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    listening_socket = bind_server(options);
    if (listening_socket < 0) {
        std::cerr << "error: cannot listen on " << options.host << ':' << options.port
                  << ": " << std::strerror(errno) << '\n';
        return 1;
    }
    std::signal(SIGINT, stop_server);
    std::signal(SIGTERM, stop_server);
    std::signal(SIGPIPE, SIG_IGN);
    std::cerr << "[ready] http://" << options.host << ':' << options.port << "\n";
    ApiServer server(options, runtime, std::move(tokenizer.value),
                     registration->reasoning);
    const bool concurrent_requests =
        registration->model == strata::RuntimeModel::Glm53;
    // The GLM scheduler multiplexes generation iterations, so its HTTP front
    // end must not serialize independently arriving streams. Bound detached
    // connection workers from the CPUs the host actually exposes instead of
    // accumulating one joinable thread object for the lifetime of the server.
    const auto discovered_cpus = std::max(1U, std::thread::hardware_concurrency());
    const auto maximum_clients = std::min<std::uint32_t>(128U,
        std::max<std::uint32_t>(4U, discovered_cpus * 2U));
    std::atomic<std::uint32_t> active_clients{};
    std::mutex clients_mutex;
    std::condition_variable clients_drained;
    while (stop_requested == 0) {
        const int client = accept4(listening_socket, nullptr, nullptr, SOCK_CLOEXEC);
        if (client < 0) {
            if (errno == EINTR || stop_requested != 0) continue;
            std::cerr << "warning: accept failed: " << std::strerror(errno) << '\n';
            continue;
        }
        const auto serve = [&server, &active_clients, &clients_drained](
                               int socket) {
            HttpRequest request;
            std::string error;
            if (!read_request(socket, request, error)) {
                send_error(socket, 400, "Bad Request", error,
                           "invalid_request_error");
            } else {
                server.handle(socket, request);
            }
            close(socket);
            active_clients.fetch_sub(1U, std::memory_order_acq_rel);
            clients_drained.notify_all();
        };
        if (concurrent_requests) {
            if (active_clients.load(std::memory_order_acquire) >=
                maximum_clients) {
                send_error(client, 503, "Service Unavailable",
                           "GLM-5.3 request admission is full",
                           "server_overloaded");
                close(client);
                continue;
            }
            active_clients.fetch_add(1U, std::memory_order_acq_rel);
            try {
                std::thread(serve, client).detach();
            } catch (const std::system_error& error) {
                active_clients.fetch_sub(1U, std::memory_order_acq_rel);
                send_error(client, 503, "Service Unavailable",
                           std::string("cannot start request worker: ") +
                               error.what(),
                           "server_overloaded");
                close(client);
            }
        } else {
            active_clients.fetch_add(1U, std::memory_order_acq_rel);
            serve(client);
        }
    }
    listening_socket = -1;
    std::unique_lock clients_lock(clients_mutex);
    clients_drained.wait(clients_lock, [&] {
        return active_clients.load(std::memory_order_acquire) == 0U;
    });
    std::cerr << "[shutdown] stopped cleanly\n";
    return 0;
}
