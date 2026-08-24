#pragma once

// Terminal front-end helpers shared by the interactive Strata applications:
// colour, a small line editor with history, a spinner for the phases that
// print nothing, and an incremental UTF-8 assembler so a multi-byte codepoint
// split across two token pieces is never written out half-formed.
//
// Header-only and dependency-free on purpose. Nothing here knows about a
// model, a runtime, or a protocol.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace strata::cli::term {

// ---------------------------------------------------------------- colour ---

inline bool& colour_enabled() {
    static bool enabled = false;
    return enabled;
}

// Honours NO_COLOR and a non-tty stream, so redirected output stays plain.
// Callers pass whether the *diagnostic* stream is a terminal: colour never
// reaches the generated text on stdout.
inline void detect_colour(bool stream_is_tty) {
    const char* no_colour = std::getenv("NO_COLOR");
    const char* term = std::getenv("TERM");
    colour_enabled() = stream_is_tty && (no_colour == nullptr || no_colour[0] == '\0') &&
                       (term == nullptr || std::strcmp(term, "dumb") != 0);
}

inline const char* paint(const char* code) {
    return colour_enabled() ? code : "";
}

inline const char* reset()  { return paint("\033[0m");  }
inline const char* bold()   { return paint("\033[1m");  }
inline const char* dim()    { return paint("\033[2m");  }
inline const char* red()    { return paint("\033[31m"); }
inline const char* green()  { return paint("\033[32m"); }
inline const char* yellow() { return paint("\033[33m"); }
inline const char* blue()   { return paint("\033[34m"); }
inline const char* cyan()   { return paint("\033[36m"); }
inline const char* grey()   { return paint("\033[90m"); }

// ------------------------------------------------------------------ tty ----

inline bool stdout_is_tty() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    return isatty(STDOUT_FILENO) != 0;
#else
    return false;
#endif
}

inline bool stdin_is_tty() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    return isatty(STDIN_FILENO) != 0;
#else
    return false;
#endif
}

inline bool stderr_is_tty() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    return isatty(STDERR_FILENO) != 0;
#else
    return false;
#endif
}

inline unsigned int terminal_columns() {
#if defined(TIOCGWINSZ)
    struct winsize size {};
    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) {
        return size.ws_col;
    }
#endif
    return 80U;
}

// ----------------------------------------------------------------- utf-8 ---

// Byte length of the sequence starting at `lead`, or 1 for a stray byte.
inline std::size_t utf8_sequence_length(unsigned char lead) {
    if (lead < 0x80U) return 1U;
    if ((lead & 0xE0U) == 0xC0U) return 2U;
    if ((lead & 0xF0U) == 0xE0U) return 3U;
    if ((lead & 0xF8U) == 0xF0U) return 4U;
    return 1U;
}

inline bool utf8_is_continuation(unsigned char byte) {
    return (byte & 0xC0U) == 0x80U;
}

// Start of the codepoint that ends at `position`.
inline std::size_t utf8_previous(std::string_view text, std::size_t position) {
    if (position == 0U) return 0U;
    --position;
    while (position > 0U &&
           utf8_is_continuation(static_cast<unsigned char>(text[position]))) {
        --position;
    }
    return position;
}

// Start of the codepoint after the one at `position`.
inline std::size_t utf8_next(std::string_view text, std::size_t position) {
    if (position >= text.size()) return text.size();
    position += utf8_sequence_length(static_cast<unsigned char>(text[position]));
    return position > text.size() ? text.size() : position;
}

// Codepoint count, used as a display-width estimate. Wide and combining
// characters are counted as one; that is wrong for CJK and emoji, and the only
// consequence is a redraw that wraps a column early.
inline std::size_t utf8_width(std::string_view text) {
    std::size_t width = 0U;
    for (std::size_t index = 0U; index < text.size(); ++index) {
        if (!utf8_is_continuation(static_cast<unsigned char>(text[index]))) ++width;
    }
    return width;
}

// Visible width of a string that may contain SGR escape sequences.
inline std::size_t visible_width(std::string_view text) {
    std::size_t width = 0U;
    for (std::size_t index = 0U; index < text.size();) {
        if (text[index] == '\033') {
            while (index < text.size() && text[index] != 'm') ++index;
            if (index < text.size()) ++index;
            continue;
        }
        if (!utf8_is_continuation(static_cast<unsigned char>(text[index]))) ++width;
        ++index;
    }
    return width;
}

// Streams token pieces out as complete UTF-8 only. A model emits a codepoint
// across two pieces often enough that writing each piece straight through
// produces mojibake mid-word, and a JSON encoder given half a sequence emits
// an invalid string.
class Utf8Assembler {
public:
    // Feeds `piece` and hands every complete, valid run to `sink`. Invalid
    // bytes are replaced individually so one bad byte does not eat the rest.
    template <typename Sink>
    void push(std::string_view piece, Sink&& sink) {
        pending_.append(piece.data(), piece.size());
        drain(static_cast<Sink&&>(sink));
    }

    // Ends the stream. Anything still held back was truncated mid-codepoint
    // and becomes one replacement character.
    template <typename Sink>
    void finish(Sink&& sink) {
        drain(sink);
        if (!pending_.empty()) {
            sink(std::string_view(replacement, 3U));
            pending_.clear();
        }
    }

    [[nodiscard]] bool holding() const noexcept { return !pending_.empty(); }

private:
    static constexpr const char replacement[] = "\xEF\xBF\xBD";

    // >0: a complete valid sequence of that length. 0: incomplete, wait for
    // more bytes. -1: invalid at this position.
    [[nodiscard]] int classify(std::size_t position) const {
        const auto lead = static_cast<unsigned char>(pending_[position]);
        const std::size_t remaining = pending_.size() - position;
        const auto trail = [&](std::size_t offset) {
            return static_cast<unsigned char>(pending_[position + offset]);
        };
        const auto continuation = [&](std::size_t offset) {
            return utf8_is_continuation(trail(offset));
        };

        if (lead <= 0x7FU) return 1;
        // C0 and C1 are overlong two-byte forms; F5..FF is beyond U+10FFFF.
        if (lead < 0xC2U || lead > 0xF4U) return -1;

        const std::size_t length = utf8_sequence_length(lead);
        if (remaining < length) {
            // Validate what we do have, so an invalid tail is not mistaken for
            // a sequence still in flight.
            for (std::size_t offset = 1U; offset < remaining; ++offset) {
                if (!continuation(offset)) return -1;
            }
            return 0;
        }
        for (std::size_t offset = 1U; offset < length; ++offset) {
            if (!continuation(offset)) return -1;
        }
        // Overlong forms and the UTF-16 surrogate range.
        if (lead == 0xE0U && trail(1) < 0xA0U) return -1;
        if (lead == 0xEDU && trail(1) > 0x9FU) return -1;
        if (lead == 0xF0U && trail(1) < 0x90U) return -1;
        if (lead == 0xF4U && trail(1) > 0x8FU) return -1;
        return static_cast<int>(length);
    }

    template <typename Sink>
    void drain(Sink&& sink) {
        std::size_t position = 0U;
        while (position < pending_.size()) {
            const int length = classify(position);
            if (length > 0) {
                position += static_cast<std::size_t>(length);
                continue;
            }
            if (length == 0) break;
            if (position > 0U) {
                sink(std::string_view(pending_.data(), position));
                pending_.erase(0U, position);
                position = 0U;
            }
            sink(std::string_view(replacement, 3U));
            pending_.erase(0U, 1U);
        }
        if (position > 0U) {
            sink(std::string_view(pending_.data(), position));
            pending_.erase(0U, position);
        }
    }

    std::string pending_;
};

// --------------------------------------------------------------- spinner ---

// Marks a phase that produces no output of its own. Writes to stderr so a
// redirected transcript on stdout stays clean.
class Spinner {
public:
    Spinner() = default;
    Spinner(const Spinner&) = delete;
    Spinner& operator=(const Spinner&) = delete;
    ~Spinner() { stop(); }

    void start(std::string label) {
        if (running_ || !colour_enabled()) {
            // Without a tty there is nothing to animate; say it once instead.
            if (!running_ && !label.empty()) {
                std::fprintf(stderr, "  %s...\n", label.c_str());
                std::fflush(stderr);
            }
            return;
        }
        label_ = std::move(label);
        running_ = true;
        stop_.store(false);
        worker_ = std::thread([this]() {
            static const char* frames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
            std::size_t frame = 0U;
            while (!stop_.load()) {
                std::fprintf(stderr, "\r\033[K  %s%s%s %s", grey(),
                             frames[frame % 10U], reset(), label_.c_str());
                std::fflush(stderr);
                ++frame;
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }
            std::fprintf(stderr, "\r\033[K");
            std::fflush(stderr);
        });
    }

    void stop() {
        if (!running_) return;
        stop_.store(true);
        if (worker_.joinable()) worker_.join();
        running_ = false;
    }

private:
    std::atomic<bool> stop_{false};
    bool running_{};
    std::string label_;
    std::thread worker_;
};

// ----------------------------------------------------------- quiet input ---

// Suppresses terminal echo for its lifetime. While an answer streams the tty
// is back in cooked mode, so anything typed is echoed by the line discipline
// straight into the middle of the model's output -- most visibly the ^C that
// stops the generation. Signals stay enabled: Ctrl+C must still be delivered.
class QuietInput {
public:
    QuietInput() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (!stdin_is_tty()) return;
        if (tcgetattr(STDIN_FILENO, &saved_) != 0) return;
        struct termios quiet = saved_;
        quiet.c_lflag &= ~(static_cast<tcflag_t>(ECHO | ECHOE | ECHOK | ECHONL));
#if defined(ECHOCTL)
        quiet.c_lflag &= ~static_cast<tcflag_t>(ECHOCTL);
#endif
        active_ = tcsetattr(STDIN_FILENO, TCSANOW, &quiet) == 0;
#endif
    }

    ~QuietInput() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (!active_) return;
        // Drop whatever was typed while the answer streamed. It was never
        // echoed, so leaving it queued would push text the user cannot see
        // into the next prompt.
        tcflush(STDIN_FILENO, TCIFLUSH);
        tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
#endif
    }

    QuietInput(const QuietInput&) = delete;
    QuietInput& operator=(const QuietInput&) = delete;

private:
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct termios saved_ {};
#endif
    bool active_{};
};

// ----------------------------------------------------------- line editor ---

// A small readline: history, in-line editing, and the emacs keys people
// reflexively press. std::getline gives none of that, and an arrow key in a
// cooked-mode prompt lands in the prompt text as an escape sequence.
class LineEditor {
public:
    enum class Status : std::uint8_t {
        Line,       // a line was submitted (possibly empty)
        Eof,        // Ctrl+D on an empty line, or stdin closed
        Interrupt,  // Ctrl+C
    };

    Status read(const std::string& prompt, std::string& line) {
        line.clear();
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (!stdin_is_tty() || !stderr_is_tty()) return read_plain(prompt, line);
        return read_raw(prompt, line);
#else
        return read_plain(prompt, line);
#endif
    }

    void remember(const std::string& line) {
        if (line.empty()) return;
        if (!history_.empty() && history_.back() == line) return;
        history_.push_back(line);
        if (history_.size() > 500U) history_.erase(history_.begin());
    }

private:
    Status read_plain(const std::string& prompt, std::string& line) {
        std::fputs(prompt.c_str(), stderr);
        std::fflush(stderr);
        int character = 0;
        while ((character = std::fgetc(stdin)) != EOF) {
            if (character == '\n') return Status::Line;
            line.push_back(static_cast<char>(character));
        }
        return line.empty() ? Status::Eof : Status::Line;
    }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    // RAII raw mode. ISIG is cleared as well, so Ctrl+C arrives as a byte we
    // can turn into a cancelled line rather than a signal that kills the
    // process mid-prompt.
    class RawMode {
    public:
        RawMode() {
            if (tcgetattr(STDIN_FILENO, &saved_) != 0) return;
            struct termios raw = saved_;
            raw.c_lflag &= ~(static_cast<tcflag_t>(ICANON | ECHO | ISIG));
            raw.c_iflag &= ~(static_cast<tcflag_t>(IXON | ICRNL));
            raw.c_cc[VMIN] = 1;
            raw.c_cc[VTIME] = 0;
            active_ = tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == 0;
        }
        ~RawMode() {
            if (active_) tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved_);
        }
        RawMode(const RawMode&) = delete;
        RawMode& operator=(const RawMode&) = delete;
        [[nodiscard]] bool active() const noexcept { return active_; }

    private:
        struct termios saved_ {};
        bool active_{};
    };

    Status read_raw(const std::string& prompt, std::string& line) {
        RawMode raw;
        if (!raw.active()) return read_plain(prompt, line);

        const std::size_t prompt_width = visible_width(prompt);
        std::size_t cursor = 0U;
        std::size_t history_index = history_.size();
        std::string stashed;
        rendered_rows_ = 0U;
        render(prompt, prompt_width, line, cursor);

        for (;;) {
            int byte = std::fgetc(stdin);
            if (byte == EOF) {
                finish_line(prompt, prompt_width, line, cursor);
                return line.empty() ? Status::Eof : Status::Line;
            }
            const auto value = static_cast<unsigned char>(byte);

            if (value == '\r' || value == '\n') {
                finish_line(prompt, prompt_width, line, cursor);
                return Status::Line;
            }
            if (value == 3U) {  // Ctrl+C
                finish_line(prompt, prompt_width, line, cursor);
                return Status::Interrupt;
            }
            if (value == 4U) {  // Ctrl+D
                if (line.empty()) {
                    finish_line(prompt, prompt_width, line, cursor);
                    return Status::Eof;
                }
                if (cursor < line.size()) {
                    line.erase(cursor, utf8_next(line, cursor) - cursor);
                }
            } else if (value == 127U || value == 8U) {  // Backspace
                if (cursor > 0U) {
                    const std::size_t start = utf8_previous(line, cursor);
                    line.erase(start, cursor - start);
                    cursor = start;
                }
            } else if (value == 1U) {  // Ctrl+A
                cursor = 0U;
            } else if (value == 5U) {  // Ctrl+E
                cursor = line.size();
            } else if (value == 21U) {  // Ctrl+U
                line.erase(0U, cursor);
                cursor = 0U;
            } else if (value == 11U) {  // Ctrl+K
                line.erase(cursor);
            } else if (value == 23U) {  // Ctrl+W
                std::size_t start = cursor;
                while (start > 0U && line[start - 1U] == ' ') --start;
                while (start > 0U && line[start - 1U] != ' ') --start;
                line.erase(start, cursor - start);
                cursor = start;
            } else if (value == 12U) {  // Ctrl+L
                std::fputs("\033[H\033[2J", stderr);
                rendered_rows_ = 0U;
            } else if (value == 27U) {  // escape sequence
                const int next = std::fgetc(stdin);
                if (next != '[' && next != 'O') continue;
                const int code = std::fgetc(stdin);
                if (code == EOF) continue;
                if (code >= '0' && code <= '9') {
                    // Extended form: ESC [ <n> ~
                    const int terminator = std::fgetc(stdin);
                    if (terminator == '~') {
                        if (code == '3' && cursor < line.size()) {  // Delete
                            line.erase(cursor, utf8_next(line, cursor) - cursor);
                        } else if (code == '1' || code == '7') {
                            cursor = 0U;
                        } else if (code == '4' || code == '8') {
                            cursor = line.size();
                        }
                    }
                } else if (code == 'D') {
                    cursor = utf8_previous(line, cursor);
                } else if (code == 'C') {
                    cursor = utf8_next(line, cursor);
                } else if (code == 'H') {
                    cursor = 0U;
                } else if (code == 'F') {
                    cursor = line.size();
                } else if (code == 'A') {  // history back
                    if (history_index > 0U) {
                        if (history_index == history_.size()) stashed = line;
                        --history_index;
                        line = history_[history_index];
                        cursor = line.size();
                    }
                } else if (code == 'B') {  // history forward
                    if (history_index < history_.size()) {
                        ++history_index;
                        line = history_index == history_.size()
                                   ? stashed : history_[history_index];
                        cursor = line.size();
                    }
                }
            } else if (value >= 32U) {
                // A printable byte, or the lead of a multi-byte codepoint;
                // pull the continuation bytes so the cursor never lands inside
                // a character.
                std::string character(1U, static_cast<char>(value));
                for (std::size_t remaining = utf8_sequence_length(value) - 1U;
                     remaining > 0U; --remaining) {
                    const int continuation = std::fgetc(stdin);
                    if (continuation == EOF) break;
                    character.push_back(static_cast<char>(continuation));
                }
                line.insert(cursor, character);
                cursor += character.size();
            }
            render(prompt, prompt_width, line, cursor);
        }
    }

    // Redraws prompt and buffer in place. The whole block is repainted rather
    // than patched, which costs nothing at these lengths and is the only
    // approach that survives a line wrapping past the terminal width.
    void render(const std::string& prompt, std::size_t prompt_width,
                const std::string& line, std::size_t cursor) {
        const std::size_t columns = terminal_columns();
        const std::size_t before = prompt_width + utf8_width(
            std::string_view(line).substr(0U, cursor));
        const std::size_t total = prompt_width + utf8_width(line);

        if (rendered_rows_ > 0U && rendered_cursor_row_ + 1U < rendered_rows_) {
            std::fprintf(stderr, "\033[%zuB", rendered_rows_ - rendered_cursor_row_ - 1U);
        }
        for (std::size_t row = 1U; row < rendered_rows_; ++row) {
            std::fputs("\r\033[K\033[A", stderr);
        }
        std::fputs("\r\033[K", stderr);

        std::fputs(prompt.c_str(), stderr);
        std::fwrite(line.data(), 1U, line.size(), stderr);

        rendered_rows_ = total / columns + 1U;
        rendered_cursor_row_ = before / columns;
        // A cursor sitting exactly at the wrap column has not been flushed to
        // the next row by the terminal yet; force it so the position we
        // compute and the one it shows agree.
        if (total > 0U && total % columns == 0U) {
            std::fputs(" \r", stderr);
        }

        const std::size_t up = rendered_rows_ - 1U - rendered_cursor_row_;
        if (up > 0U) std::fprintf(stderr, "\033[%zuA", up);
        std::fputs("\r", stderr);
        const std::size_t column = before % columns;
        if (column > 0U) std::fprintf(stderr, "\033[%zuC", column);
        std::fflush(stderr);
    }

    // Leaves the cursor after the last rendered row so what follows does not
    // overwrite the line the user just typed.
    void finish_line(const std::string& prompt, std::size_t prompt_width,
                     const std::string& line, std::size_t cursor) {
        render(prompt, prompt_width, line, cursor);
        if (rendered_rows_ > 0U && rendered_cursor_row_ + 1U < rendered_rows_) {
            std::fprintf(stderr, "\033[%zuB", rendered_rows_ - rendered_cursor_row_ - 1U);
        }
        std::fputs("\r\n", stderr);
        std::fflush(stderr);
        rendered_rows_ = 0U;
    }
#endif

    std::vector<std::string> history_;
    std::size_t rendered_rows_{};
    std::size_t rendered_cursor_row_{};
};

}  // namespace strata::cli::term
