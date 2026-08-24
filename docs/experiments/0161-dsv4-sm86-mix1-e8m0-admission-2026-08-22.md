# Experiment 0161 — MIX-1 admission: E8M0 scales are checked at checkpoint load

Status: **MIX-1 ADMISSION WIRED.** `dsv4_admit_e8m0_scales` now runs on every
FP4 region as it is loaded, and an inadmissible E8M0 code fails the load with
the tensor name, the offending code and its byte offset. A gating test proves
it fires on both bad codes and admits the real checkpoint's range untouched.

This closes the admission item owed since experiment 0155 and listed as
outstanding in 0160.

## What was wired, and where

The check lives in `include/strata/deepseek_admission.hpp` alongside the rest of
DeepSeek admission, not in the probe where it was prototyped:

- `dsv4_admit_e8m0_scales(span<const byte>)` returns a count of inadmissible
  codes, split by code 0 and code 255, plus the first offending offset and code.
- `dsv4_admit_e8m0_scales_for(name, span)` wraps it into a `ValidationResult`
  that names the tensor.

It is called in `load_dsv4_cuda_linear` at the point the FP4 branch binds its
scale data, so it runs **before** the descriptor is built and before anything is
uploaded to the device. A rejected region therefore never reaches a kernel.

## Why codes 0 and 255 are inadmissible

Both FP4 decoders map the E8M0 code straight into a BF16 exponent field, which
is exact for 1–254 and silently wrong outside it:

| Code | Means | Encodes as | Result |
|---:|---|---|---|
| 0 | 2^-127 | `0x0000` | **+0** — subnormal in BF16, whose smallest normal is 2^-126 |
| 255 | E8M0 NaN | `0x7F80` | **+inf** |

Neither is a rounding error; both are silent substitutions of a different value,
which the contract forbids in exact mode.

## Gate

The test asserts the shape of the check rather than only its happy path:

- a 4,096-byte region spanning codes 119–125, the real checkpoint's measured
  range, is admitted with zero inadmissible;
- the window **boundaries** 1 and 254 are admitted, so the check is not
  off-by-one at either end;
- a single injected code 255 among 4,096 is rejected, counted as `code_255`,
  and reported with offset 1234 and code 255 in the message;
- a single injected code 0 is rejected and counted as `code_zero`;
- an empty region is vacuously admissible rather than a failure.

`make check` passes: 315/329 unit tests, up one from 0160's 314, 14 skipped,
all three ctest suites green.

## What this does not establish

- **No real checkpoint has been loaded through this path yet.** The test uses
  synthetic spans. Experiment 0156 measured the real checkpoint's scale range
  independently — 1.41 billion bytes spanning codes [119, 125] — so the check is
  expected to admit it silently, but that is a prediction until a real load runs.
- Only the **FP4** path is wired. The FP8 block-128 branch has no equivalent
  admission, and whether E4M3/E8M0 block-128 needs one is a separate question.
- This is one MIX-1 requirement. The accepted register-fed paths are still not
  substituted, and prepack, VRAM accounting, graph integration and fixtures are
  untouched.

## Exact next action

Dump the matmul route census from a real DeepSeek V4 run, which also exercises
this admission path on 156 GB of real scales for the first time.
