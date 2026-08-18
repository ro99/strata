# Sampling

Sampler options, their exact semantics, and the future-entropy lookahead.
Reproducibility rules are part of the contract here, not a footnote: a run is
reproducible when nothing stochastic is enabled, and the runtime says which
mode it is in.

See [`cli.md`](cli.md) for how to pass these, and [`../README.md`](../README.md)
for what Strata is.


The sampler runs as a fixed pipeline. Penalties rewrite the logits, every
truncation stage then reads the model's own distribution, and temperature
rescales only the surviving candidates before a seeded Gumbel-max draw:

```
presence/frequency/repetition → DRY → n-gram ban → logit bias
  → top_k → top_p → min_p → typical_p → XTC → future entropy
  → temperature → draw
```

Truncating on the natural distribution is what makes the thresholds mean what
they say: `--min-p 0.05` is "at least 5% as likely as the best token according
to the model" at any temperature.

**Just want temperature?** Pass `--temperature` with no `--preset` and nothing
else runs — no truncation, no penalties, no XTC:

```bash
./build/strata-chat --model models/dsv4f --model-type deepseek --temperature 0.9
```

The startup banner echoes exactly what's active, e.g. `[sampler]
temperature=0.9`; if other stages were silently on, they'd show there too. In
the TUI, this means leaving the `SAMPLER` field on `PRECISE` — it writes no
stages of its own — and editing `TEMPERATURE` directly; an edited temperature
always overrides whatever the preset last wrote there.

Four presets bundle the stages that are otherwise on separate flags:

| Preset | Stages | What to expect |
|---|---|---|
| `--preset precise` | none (temperature only, default `0`) | Deterministic at temperature 0; with temperature raised, plain softmax sampling over the full vocabulary — the most expensive path per token, ~4.6 ms/token at DeepSeek's vocabulary, and the one most prone to picking an implausible tail token at high temperature since nothing truncates it |
| `--preset balanced` | `min_p 0.05`, `repetition_penalty 1.05` over the last 256 tokens | Cuts tokens under 5% as likely as the best one, so the tail can't get picked; mild pushback on restating the same tokens. Closest to "temperature sampling, but safe" |
| `--preset creative` | `min_p 0.02`, XTC 50% at threshold 0.1, DRY, `repetition_penalty 1.03` over the last 512 tokens | Half the time, removes whichever safe/obvious token was in reach and forces a still-plausible alternative instead — the mechanism aimed at "goes to the mean" prose. DRY leans against the loops that removing the safe choice tends to invite. Expect more variance run to run; verify on your own material before trusting it unsupervised |
| `--preset future-entropy` | `min_p 0.05`, 20-candidate lookahead over the top-30 future, `alpha 0`, DRY, `repetition_penalty 1.03` over the last 512 tokens | Scores each candidate by how much future choice it unlocks. **Costs 21 forward passes per token**, so it decodes roughly 21× slower than the same settings without it — this is a quality knob, not a throughput one |

A preset writes defaults; any flag after it overrides them — including
`--temperature`, which is how you'd run e.g. `--preset creative --temperature
0.7` to keep XTC/DRY but sample less aggressively.

| Flag | Purpose |
|---|---|
| `--temperature F` | Rescales survivors before the draw; `0` is greedy |
| `--top-k N` | Keep the `N` most likely tokens |
| `--top-p F` | Keep the smallest set carrying mass `F` |
| `--min-p F` | Keep tokens at least `F` times as likely as the best one |
| `--typical-p F` | Keep the tokens whose surprisal is nearest the distribution's entropy |
| `--xtc-probability F` `--xtc-threshold F` | With probability `F`, drop every candidate above the threshold except the least likely of them |
| `--presence-penalty F` `--frequency-penalty F` | Subtractive repetition penalties |
| `--repetition-penalty F` `--penalty-window N` | Multiplicative penalty, optionally bounded to the last `N` tokens |
| `--dry-multiplier F` `--dry-base F` `--dry-allowed-length N` `--dry-window N` | Penalize the token extending the longest repeated suffix |
| `--no-repeat-ngram N` | Hard ban on completing an `N`-gram already in the output |
| `--future-entropy N` `--future-entropy-top-n N` | Look ahead one step past the `N` likeliest survivors and score them by the entropy of the top-`n` future |
| `--alpha F` `--future-entropy-curve NAME` | Crossfade between probability and future entropy; which exponent mapping to use |
| `--alpha-wave-amplitude F` `--alpha-wave-period F` | Oscillate alpha over the generated tokens |

XTC is the stage aimed squarely at flat prose: it removes the safe continuation
and leaves a plausible one in its place. It draws from the generator even at
temperature zero, so a run using it is seeded-reproducible but not greedy, and
the startup banner reports it as sampled rather than exact.

## Future entropy

Prefers tokens that keep the next step's options open, after
[Count Bayesie, *Making LLMs better at creative writing using
entropy*](https://www.countbayesie.com/blog/2026/7/1/making-llms-better-at-creative-writing-using-entropy).
The model is run one step past each surviving candidate `w` to get
`q_w = p(V | c + w)`; the normalized entropy of that distribution's top-`n`,
`H(w) = [-Σ q̃ ln q̃] / ln n ∈ [0, 1]`, reweights the candidate:

```
s(w) = p(w | c)^a · H(w)^b
```

and the draw is made from `s` renormalized over the candidates, so
`--temperature` rescales the blended score rather than the raw probability.
`--alpha` sets both exponents at once: `-1` is ordinary sampling and the stage
becomes a no-op, `+1` scores on the future alone. Two mappings are available
because the article and the reference implementation disagree between the
endpoints — `--future-entropy-curve article` (default) uses
`a = 1 − max(0, α)`, `b = 1 − max(0, −α)`, so `α = 0` is exactly the article's
headline `s = p · H`; `crossfade` uses `a = 1 − t`, `b = t` for `t = (α+1)/2`,
so `α = 0` is `√(p · H)`. They agree at `α = ±1` and nowhere else.

Two things are load-bearing:

- **It costs forward passes.** One per candidate, sequentially, because the KV
  cache holds a single sequence — `--future-entropy 20` makes a token 21 decode
  steps instead of one. Every other stage in the pipeline is arithmetic on
  logits that are already in hand; this one is not. The batched form the
  reference implementation uses needs `k` forked sequences decoded at one
  position, which the GLM cache cannot express today.
- **A relative-plausibility cut in front of it is not optional.** Broken
  word-fragments have maximally uncertain futures, so entropy selects them
  unless something has already removed them. Keep `--min-p` at 0.05 or above;
  the preset does. The lookahead runs last, on the survivors only, which is
  also what keeps the cost proportional to what the cheaper stages accepted.

The lookahead is exact: each speculative pass is rolled back to a
bit-identical cache before the next candidate runs, so the emitted token is
decoded from the same state it would have been without the stage. It draws
nothing from the generator, so `--temperature 0` with future entropy is still
reported as exact greedy decoding — the argmax of `s` rather than of `p`.

Every knob is also accepted by the OpenAI-compatible server, under the same
names, on both `/v1/chat/completions` and `/v1/completions`.

Reported `logprobs` are the model's natural log probabilities, computed from the
unmodified logits before penalties, truncation, and temperature. They describe
the model rather than the sampler settings, so they stay comparable across
requests that used different knobs.

