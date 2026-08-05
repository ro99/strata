#!/usr/bin/env python3
"""Regenerates src/laguna_unicode.hpp from the running Python's Unicode tables.

The Laguna pre-tokenizer pattern tests \\p{L}, \\p{N} and \\s directly. The
shared heuristic rune classifier in src/tokenizer.cpp treats unknown non-ASCII
as a letter, which puts U+00BB and friends on the wrong side of the
[^\\s\\p{L}\\p{N}]+ alternative. These tables are the exact ranges.
"""
import unicodedata


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


def main():
    letters = ranges(lambda c: unicodedata.category(chr(c)).startswith("L"))
    numbers = ranges(lambda c: unicodedata.category(chr(c)).startswith("N"))
    print(f"Unicode {unicodedata.unidata_version}: "
          f"{len(letters)} letter ranges, {len(numbers)} number ranges")
    print(emit("kLagunaLetterRanges", letters)[:80] + " ...")


if __name__ == "__main__":
    main()
