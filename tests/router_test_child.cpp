#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace {

volatile std::sig_atomic_t stopping = 0;
int listener = -1;

void stop(int) {
    stopping = 1;
    if (listener >= 0) close(listener);
}

bool send_all(int socket_fd, std::string_view data) {
    while (!data.empty()) {
        const auto sent = send(socket_fd, data.data(), data.size(), MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (sent == 0) return false;
        data.remove_prefix(static_cast<std::size_t>(sent));
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::uint16_t port = 0U;
    std::string model;
    for (int index = 1; index + 1 < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--port") {
            port = static_cast<std::uint16_t>(std::strtoul(argv[++index], nullptr, 10));
        } else if (argument == "--model-id") {
            model = argv[++index];
        }
    }
    if (port == 0U || model.empty()) return 2;

    listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) return 1;
    int enabled = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        listen(listener, 8) != 0) return 1;
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    std::signal(SIGPIPE, SIG_IGN);

    while (stopping == 0) {
        const int client = accept(listener, nullptr, nullptr);
        if (client < 0) {
            if (errno == EINTR || stopping != 0) continue;
            return 1;
        }
        std::array<char, 16U * 1024U> buffer{};
        const auto received = recv(client, buffer.data(), buffer.size(), 0);
        if (received <= 0) {
            close(client);
            continue;
        }
        const std::string_view request(buffer.data(), static_cast<std::size_t>(received));
        if (request.starts_with("GET /v1/health ")) {
            send_all(client,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Content-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}");
        } else if (request.starts_with("POST /v1/chat/completions ")) {
            const std::string payload =
                "data: {\"object\":\"chat.completion.chunk\",\"model\":\"" + model +
                "\",\"choices\":[{\"delta\":{\"content\":\"" + model +
                " ready\"},\"finish_reason\":null}]}\n\n"
                "data: [DONE]\n\n";
            const std::string header =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                "Connection: close\r\n\r\n";
            send_all(client, header);
            send_all(client, payload);
        } else {
            send_all(client,
                "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n"
                "Content-Length: 2\r\nConnection: close\r\n\r\n{}");
        }
        close(client);
    }
    listener = -1;
    return 0;
}
