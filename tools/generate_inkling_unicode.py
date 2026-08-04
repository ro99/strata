#!/usr/bin/env python3
"""Regenerates src/inkling_unicode.hpp from the running Python's Unicode tables.

The Inkling pre-tokenizer splits letter runs by case, which no other pinned
tokenizer in this repository does:

    [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|...)?
    |[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|...)?

So it needs the two composite classes the pattern actually names, rather than
the single \\p{L} class the Laguna tables provide. `Upper` is Lu|Lt|Lm|Lo|M and
`Lower` is Ll|Lm|Lo|M; they deliberately overlap on Lm, Lo and M, which is what
makes a run of caseless letters match either alternative.

Regenerate with: python3 tools/generate_inkling_unicode.py > src/inkling_unicode.hpp
"""
import unicodedata

HEADER = '''#pragma once

// Generated from Unicode {version} category data by
// tools/generate_inkling_unicode.py. Do not edit by hand.
//
// The Inkling pre-tokenizer splits letter runs by case, so it needs the two
// composite classes its regex names rather than a single letter class:
//   Upper = \\p{{Lu}}|\\p{{Lt}}|\\p{{Lm}}|\\p{{Lo}}|\\p{{M}}
//   Lower = \\p{{Ll}}|\\p{{Lm}}|\\p{{Lo}}|\\p{{M}}
// The classes overlap on Lm, Lo and M by construction: a run of caseless
// letters satisfies either alternative, which is what keeps CJK and Devanagari
// from being split character by character.

#include "laguna_unicode.hpp"

#include <array>
#include <cstdint>

namespace strata::detail {{
'''

FOOTER = '''
[[nodiscard]] constexpr bool inkling_upper_letter(std::uint32_t codepoint) noexcept {
    return in_ranges(kInklingUpperRanges, codepoint);
}

[[nodiscard]] constexpr bool inkling_lower_letter(std::uint32_t codepoint) noexcept {
    return in_ranges(kInklingLowerRanges, codepoint);
}

static_assert(ranges_are_ordered(kInklingUpperRanges));
static_assert(ranges_are_ordered(kInklingLowerRanges));

}  // namespace strata::detail
'''


def ranges(pred, hi=0x110000):
    out, start = [], None
    for cp in range(hi):
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, hi - 1))
    return out


def emit(name, rs):
    lines = [f"inline constexpr std::array<CodepointRange, {len(rs)}> {name}{{{{"]
    row = "    "
    for a, b in rs:
        piece = f"{{0x{a:04X}U, 0x{b:04X}U}}, "
        if len(row) + len(piece) > 92:
            lines.append(row.rstrip())
            row = "    "
        row += piece
    if row.strip():
        lines.append(row.rstrip().rstrip(","))
    lines.append("}};")
    return "\n".join(lines)


UPPER = {"Lu", "Lt", "Lm", "Lo"}
LOWER = {"Ll", "Lm", "Lo"}


def main():
    upper = ranges(
        lambda c: unicodedata.category(chr(c)) in UPPER
        or unicodedata.category(chr(c)).startswith("M")
    )
    lower = ranges(
        lambda c: unicodedata.category(chr(c)) in LOWER
        or unicodedata.category(chr(c)).startswith("M")
    )
    print(HEADER.format(version=unicodedata.unidata_version))
    print(emit("kInklingUpperRanges", upper))
    print()
    print(emit("kInklingLowerRanges", lower))
    print(FOOTER)


if __name__ == "__main__":
    main()
