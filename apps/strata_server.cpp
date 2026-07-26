#include "strata/openai_protocol.hpp"
#include "strata/runtime.hpp"
#include "strata/runtime_support.hpp"
#include "strata/tokenizer.hpp"

#include "cli_common.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string model_type;
    std::string model_id;
    std::string host{"127.0.0.1"};
    std::vector<int> devices;
    std::uint32_t context_size{2048U};
    std::uint32_t max_new_tokens{256U};
    std::uint16_t port{8080U};
    double vram_fraction{0.85};
    bool devices_explicit{};
    bool flash_attention{};
    bool block_kv_cache{};
    bool pin_resident_arena{};
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
        << "usage: strata-server --model DIR --model-type deepseek|glm\n"
        << "                     [--model-id ID] [--host ADDRESS] [--port N]\n"
        << "                     [--context-size N] [--max-new N]\n"
        << "                     [--devices 0,1,2] [--vram-fraction F]\n"
        << "                     [--flash-attention] [--block-kv-cache]\n"
        << "                     [--pin-resident-arena]\n";
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
            continue;
        }
        if (argument == "--pin-resident-arena") {
            options.pin_resident_arena = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const auto next = [&]() { return std::string_view(argv[++index]); };
        if (argument == "--model") options.model = next();
        else if (argument == "--model-type") options.model_type = next();
        else if (argument == "--model-id") options.model_id = next();
        else if (argument == "--host") options.host = next();
        else if (argument == "--port") {
            std::uint32_t port = 0U;
            if (!strata::cli::parse_positive_u32(next(), port) ||
                port > std::numeric_limits<std::uint16_t>::max()) return false;
            options.port = static_cast<std::uint16_t>(port);
        } else if (argument == "--context-size") {
            if (!strata::cli::parse_positive_u32(next(), options.context_size)) return false;
        } else if (argument == "--max-new") {
            if (!strata::cli::parse_positive_u32(next(), options.max_new_tokens)) return false;
        } else if (argument == "--vram-fraction") {
            if (!strata::cli::parse_double(next(), options.vram_fraction, 0.0, 0.95)) return false;
        } else if (argument == "--devices") {
            if (!strata::cli::parse_devices(next(), options.devices)) return false;
            options.devices_explicit = true;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    if (!options.devices_explicit) options.devices = {0, 1, 2};
    if (options.model_id.empty() && !options.model.empty()) {
        options.model_id = std::filesystem::path(options.model).filename().string();
    }
    return !options.model.empty() && !options.model_id.empty() &&
        (options.model_type == "glm" || options.model_type == "deepseek");
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

bool send_response(int socket, int status, std::string_view reason,
                   std::string_view content_type, std::string_view body) {
    std::ostringstream header;
    header << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
           << "Content-Type: " << content_type << "\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n"
           << "Access-Control-Allow-Origin: *\r\n\r\n";
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
              strata::ModelTokenizer tokenizer)
        : options_(options), runtime_(runtime), tokenizer_(std::move(tokenizer)) {}

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
        strata::OpenAiChatRequest request;
        std::string error;
        const bool parsed = chat
            ? strata::parse_openai_chat_request(body, request, error)
            : strata::parse_openai_completion_request(body, request, error);
        if (!parsed) {
            send_error(socket, 400, "Bad Request", error, "invalid_request_error");
            return;
        }
        if (request.model != options_.model_id) {
            send_error(socket, 404, "Not Found", "requested model is not loaded",
                       "invalid_request_error", "model_not_found");
            return;
        }
        if (request.has_tools) {
            send_error(socket, 400, "Bad Request",
                       "tool calling is not supported by this loaded base model",
                       "invalid_request_error", "unsupported_parameter");
            return;
        }
        for (const auto& [token, bias] : request.generation.sampling.logit_bias) {
            static_cast<void>(bias);
            if (token >= tokenizer_.vocabulary_size()) {
                send_error(socket, 400, "Bad Request",
                           "logit_bias token id is outside the loaded vocabulary",
                           "invalid_request_error", "invalid_logit_bias");
                return;
            }
        }
        if (!request.has_max_tokens) {
            request.generation.maximum_new_tokens = options_.max_new_tokens;
        }
        if (request.stream) stream(socket, request, chat);
        else complete(socket, request, chat, request_started);
    }

    std::string next_id(bool chat) {
        std::ostringstream id;
        id << (chat ? "chatcmpl-" : "cmpl-") << ++request_id_;
        return id.str();
    }

    void stream(int socket, const strata::OpenAiChatRequest& request, bool chat) {
        const auto id = next_id(chat);
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string header =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";
        if (!send_all(socket, header)) return;
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
                if (!send_all(socket, role.str())) return;
            }
            bool connected = true;
            std::string utf8_pending;
            const strata::TokenStreamCallback callback =
                [&](std::uint32_t, std::string_view piece) {
                    if (!connected) return;
                    if (request.json_object) return;
                    utf8_pending.append(piece);
                    const auto complete = strata::complete_utf8_prefix(utf8_pending);
                    if (complete == std::string_view::npos) {
                        connected = false;
                        return;
                    }
                    if (complete == 0U) return;
                    piece = std::string_view(utf8_pending).substr(0U, complete);
                    std::ostringstream chunk;
                    chunk << "data: {\"id\":\"" << id << "\",\"object\":\""
                          << (chat ? "chat.completion.chunk" : "text_completion")
                          << "\",\"created\":" << created << ",\"model\":\""
                          << strata::cli::json_escape(request.model)
                          << "\",\"choices\":[{\"index\":" << index << ',';
                    if (chat) chunk << "\"delta\":{\"content\":\""
                                    << strata::cli::json_escape(piece) << "\"}";
                    else chunk << "\"text\":\"" << strata::cli::json_escape(piece) << '"';
                    chunk << ",\"logprobs\":null,\"finish_reason\":null}]}\n\n";
                    connected = send_all(socket, chunk.str());
                    utf8_pending.erase(0U, complete);
                };
            auto result = runtime_.generate_chat_stream(request.messages, options, callback);
            if (!result.ok()) {
                send_all(socket, "data: " + error_json(result.errors.front(), "server_error") + "\n\n");
                send_all(socket, "data: [DONE]\n\n");
                return;
            }
            if (!connected || !utf8_pending.empty()) {
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
                  << finish_reason(result, options) << "\"}]}\n\n";
            if (!send_all(socket, final.str())) return;
        }
        send_all(socket, "data: [DONE]\n\n");
    }

    void complete(int socket, const strata::OpenAiChatRequest& request, bool chat,
                  std::chrono::steady_clock::time_point request_started) {
        const auto id = next_id(chat);
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
            auto result = runtime_.generate_chat_stream(request.messages, options);
            if (!result.ok()) {
                send_error(socket, 500, "Internal Server Error", result.errors.front(),
                           "server_error");
                return;
            }
            if (request.json_object && !strata::is_json_object(result.text)) {
                send_error(socket, 500, "Internal Server Error",
                           "model did not produce a valid JSON object", "server_error");
                return;
            }
            results.push_back(std::move(result));
        }
        std::uint64_t completion_tokens = 0U;
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
            response << "{\"index\":" << index << ',';
            if (chat) {
                response << "\"message\":{\"role\":\"assistant\",\"content\":\""
                         << strata::cli::json_escape(result.text)
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
        const auto prompt_tokens = results.empty() ? 0U : results.front().metrics.prompt_tokens;
        response << "],\"usage\":{\"prompt_tokens\":" << prompt_tokens
                 << ",\"completion_tokens\":" << completion_tokens
                 << ",\"total_tokens\":" << prompt_tokens + completion_tokens
                 << "}}";
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
        std::cerr << "[request] serving_overhead_ms=" << overhead_ms
                  << " decode_step_ms=" << decode_step_ms
                  << " overhead_percent="
                  << (decode_step_ms == 0.0 ? 0.0 : overhead_ms * 100.0 / decode_step_ms)
                  << '\n';
        send_response(socket, 200, "OK", "application/json", response.str());
    }

    const Options& options_;
    strata::RuntimeSession& runtime_;
    strata::ModelTokenizer tokenizer_;
    std::uint64_t request_id_{};
};

int bind_server(const Options& options) {
    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
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
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }
    auto tokenizer = strata::ModelTokenizer::load(
        (std::filesystem::path(options.model) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        for (const auto& error : tokenizer.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    strata::RuntimeConfig config;
    config.model = options.model_type == "glm"
        ? strata::RuntimeModel::Glm52 : strata::RuntimeModel::DeepSeekV4;
    config.devices = options.devices;
    config.maximum_context_tokens = options.context_size;
    config.vram_cache_fraction = options.vram_fraction;
    config.enable_flash_attention = options.flash_attention;
    config.deepseek_block_kv_cache = options.block_kv_cache;
    config.pin_resident_arena = options.pin_resident_arena;
    config.verbose = options.model_type == "deepseek";
    config.load_progress = options.model_type == "glm";
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
    ApiServer server(options, runtime, std::move(tokenizer.value));
    while (stop_requested == 0) {
        const int client = accept(listening_socket, nullptr, nullptr);
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
            server.handle(client, request);
        }
        close(client);
    }
    listening_socket = -1;
    std::cerr << "[shutdown] stopped cleanly\n";
    return 0;
}
