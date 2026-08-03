#!/usr/bin/env python3
"""Emit a Kimi-K3 tokenizer fixture from the checkpoint's own tokenizer.

Gate 7. The reference is `tokenization_kimi.TikTokenTokenizer` driving a real
`tiktoken.Encoding` with the checkpoint's `tiktoken.model` and `pat_str`; this
script records what it produces so the C++ tokenizer is compared against the
tokenizer, not against a reading of its regex.

The cases are chosen to separate the alternatives of that `pat_str`, because a
pretokenizer that is wrong on one alternative is usually right on all the
others and a corpus of ordinary English would not notice:

    [\\p{Han}]+                       Han runs, and only Han — kana are not Han
    upper* lower+ (contraction)?      the two case alternatives, which is what
    upper+ lower*                     splits CamelCase into Camel + Case
    \\p{N}{1,3}                       digits in groups of at most three
     ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*      punctuation runs with a leading space
    \\s*[\\r\\n]+, \\s+(?!\\S), \\s+     the three whitespace alternatives

plus the special-token ids and a rendered chat conversation, which is the only
thing that exercises the XTML segment builder end to end.

The fixture is text, small, and derived from the tokenizer rather than the
weights; it still goes to `/data` beside the other fixtures so that one rule
covers every artifact this project generates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kimi_k3_reference_fixture import refuse_forbidden_disk

CASES: list[tuple[str, str]] = [
    ("ascii", "Hello world"),
    ("leading-space", " leading space"),
    ("contraction-s", "it's Kimi's turn"),
    ("contraction-all", "I'm don't you're we've they'll he'd can't"),
    ("contraction-upper", "IT'S KIMI'S TURN"),
    # The two case alternatives: `upper* lower+` and `upper+ lower*`. A
    # pretokenizer with only `\p{L}+` keeps these as one piece.
    ("camel-case", "CamelCaseIdentifier"),
    ("screaming-case", "HTTPServerError"),
    ("mixed-case", "aBcDeF"),
    ("single-upper", "A"),
    ("han", "你好世界"),
    # Kana are \p{L} but not \p{Han}, so they take the letter alternatives, not
    # the Han one. A pretokenizer that treats all CJK alike splits these wrong.
    ("hiragana", "こんにちは"),
    ("katakana", "カタカナ"),
    ("han-kana-mixed", "漢字とかなとカナ"),
    ("han-ascii-mixed", "中文mixed中文"),
    ("digits", "0 1 12 123 1234 12345 1234567890"),
    ("digits-in-words", "abc123def4567"),
    ("punctuation", "!!! ??? ...  ,,, ;;;"),
    ("punctuation-newline", "end.\n\nnext!\n"),
    ("symbols", "a+b=c && d||e -> f"),
    ("newlines", "one\ntwo\n\nthree\r\nfour"),
    ("tabs", "a\tb\t\tc"),
    ("trailing-space", "trailing   "),
    ("only-space", "   "),
    ("emoji", "hello \U0001f600 world \U0001f680\U0001f680"),
    ("accents", "naïve café über ÉLÈVE"),
    ("cyrillic", "Привет мир"),
    ("combining-mark", "école à côté"),
    ("code", "def f(x):\n    return x ** 2  # square\n"),
    ("json", '{"key": "value", "n": [1, 2, 3]}'),
    ("url", "https://example.com/path?q=1&r=2#frag"),
    ("empty", ""),
    ("single-space-word", " a"),
    ("long-mixed",
     "The quick brown fox jumps over 13 lazy dogs; 中文测试, "
     "and it's 100% done!\n\tTabbed line.\n"),
]

CHATS: list[tuple[str, list[dict], dict]] = [
    ("chat-simple",
     [{"role": "user", "content": "Hello"}],
     {"thinking": False}),
    ("chat-system",
     [{"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Hi"}],
     {"thinking": False}),
    ("chat-multi-turn",
     [{"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"},
      {"role": "user", "content": "And 3+3?"}],
     {"thinking": False}),
    ("chat-thinking",
     [{"role": "user", "content": "Think about it."}],
     {"thinking": True}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/kimi-k3", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("/data/strata-results/kimi-k3-fixtures"))
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.model))
    from transformers import AutoTokenizer  # noqa: E402

    # Loaded the way a caller loads it. `TikTokenTokenizer.__init__` expects
    # `added_tokens_decoder` to hold `AddedToken` objects, which is what
    # `from_pretrained` builds out of the JSON; constructing it directly from
    # the raw config would exercise a path nobody uses.
    tokenizer = AutoTokenizer.from_pretrained(str(arguments.model),
                                              trust_remote_code=True)
    config = json.loads((arguments.model / "tokenizer_config.json").read_text())
    decoder = {
        int(key): value["content"]
        for key, value in config.get("added_tokens_decoder", {}).items()
    }

    fixture: dict = {
        "vocabulary_size": tokenizer.n_words,
        "base_tokens": tokenizer.n_words - tokenizer.num_reserved_special_tokens,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id,
        "unk_id": tokenizer.unk_id,
        "special_tokens": {name: identifier
                           for name, identifier in tokenizer.special_tokens.items()
                           if not name.startswith("<|reserved_token_")},
        "added_tokens_decoder": decoder,
        "cases": [],
        "chats": [],
    }

    for name, text in CASES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        fixture["cases"].append({
            "name": name,
            "text": text,
            "ids": ids,
            # The round trip is recorded too: a tokenizer that encodes correctly
            # and decodes wrong still produces garbage at the other end.
            "decoded": tokenizer.decode(ids),
        })

    for name, messages, options in CHATS:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True, **options)
        # `name`, then the text, then `ids`: the C++ reader walks the file in
        # that order, so a record that interleaves them differently would make
        # it pick up the next record's ids.
        fixture["chats"].append({
            "name": name,
            "rendered": tokenizer.decode(ids),
            "ids": ids,
            "messages": messages,
            "options": options,
        })

    arguments.out.mkdir(parents=True, exist_ok=True)
    refuse_forbidden_disk(arguments.out)
    destination = arguments.out / "kimi-k3-tokenizer.json"
    destination.write_text(json.dumps(fixture, ensure_ascii=False, indent=1))
    print(f"wrote {destination} ({destination.stat().st_size} bytes): "
          f"{len(fixture['cases'])} cases, {len(fixture['chats'])} chats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
