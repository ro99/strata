# Experiment 0138 — the 1605 MHz clock lock, and a corrected campaign operating point

Status: **CORRECTION TO EXPERIMENT 0136, AND AN OWNER AMENDMENT TO THE
OPERATING POINT.** This machine hard-locks both RTX 3090s to 1605 MHz against a
2100 MHz maximum, roughly a 24% underclock. Experiment 0136 measured the FP4
decoder at that locked clock and reported a clean rejection. **At stock clocks
the same decoder clears the gate** — 621.71 and 604.09 GB/s against >600.0 —
so 0136's measurement was correct but its scope was overstated. **F4-1 is
alive.** The owner has moved the experimentation operating point to a single
unlocked 3090 so that campaign numbers are silicon numbers rather than tuning
artefacts.

## Who asked the question, and why it mattered

The owner asked whether this machine's power and clock caps were interfering
with the gate result. A first pass tested only the power cap and found it was
not the cause; the owner corrected that the question was really about the
**clock cap**, and it was. That correction is the reason this record exists.
The failure it caught is a specific one: **a campaign gate was being held
against a candidate whose bottleneck resource was throttled while the resource
that sets the gate was not.**

## The tuning that was in force

Discovered at `/etc/systemd/system/apply-3090-tuning.service`, which runs
`/usr/local/sbin/apply-3090-tuning.sh` at boot for every RTX 3090:

```bash
nvidia-smi -pm 1 -i "$index"
nvidia-smi -pl 250 -i "$index"                 # 250 W, against a 350 W default
nvidia-smi -lgc 1605,1605 --mode=1 -i "$index" # SM clock locked, min = max
```

The SM clock maximum is 2100 MHz, so the lock is about a 24% underclock. The
**memory clock is not locked**: 810 MHz idle, 9751 MHz maximum.

That asymmetry is the whole problem. The campaign's gates derive from the
842 GB/s **memory** ruler, which the lock does not touch. A candidate's ability
to meet them is set by **ALU** throughput, which the lock cuts by roughly a
quarter. The campaign was therefore holding a full-speed memory target against
three-quarter-speed compute.

## Measurements

Two configurations were changed, one at a time, on a single RTX 3090
(`nvidia-smi` index 1, CUDA index 0). The second 3090 stayed locked, capped,
and idle at 0% utilisation and 51.9–52.5 W throughout every run, which is the
owner's stated safety condition — the electrical constraint binds only when
both cards are heavily loaded at once.

### Cold probe, the F4-2 gate protocol, median of three interleaved processes

| Shape | Arm | 250 W / 1605 | 350 W / 1605 | 350 W / unlocked |
|---|---|---:|---:|---:|
| `gate_up_w1` | `read_only` | 831.59 | 847.79 | 847.79 |
| `gate_up_w1` | 0135 decoder | 514.02 | 520.16 | **621.71** |
| `gate_up_w1` | PRMT successor | 810.93 | 826.33 | 831.59 |
| `down_w2` | `read_only` | 831.59 | 843.34 | 847.79 |
| `down_w2` | 0135 decoder | 512.00 | 520.16 | **604.09** |
| `down_w2` | PRMT successor | 810.93 | 826.33 | 831.59 |

**Removing the power cap alone was worth about 1.2%. Removing the clock lock
was worth about 20%.** The owner's suspicion was correct and the first
diagnosis, which stopped at the power cap, was incomplete.

### Sustained load, 12–14 s continuous

| Arm | Config | GB/s | SM clock | Power | `SW Power Cap` |
|---|---|---:|---:|---:|---|
| `read_only` | 250 W / 1605 | 869.7 | 1605 | 232.6 W | Not Active |
| `read_only` | 350 W / 1605 | 874.0 | 1605 | 233.2 W | Not Active |
| `read_only` | 350 W / unlocked | 876.3 | 1965 | 317.3 W | Active |
| 0135 decoder | 250 W / 1605 | 470.6 | 1440 | 249.4 W | Active |
| 0135 decoder | 350 W / 1605 | 522.4 | 1605 | 292.8 W | Not Active |
| 0135 decoder | 350 W / unlocked | 568.5 | 1755 | 349.1 W | Active |
| PRMT successor | 250 W / 1605 | 802.8 | 1065 | 250.0 W | Active |
| PRMT successor | 350 W / 1605 | 850.7 | 1605 | 296.0 W | Active |
| PRMT successor | 350 W / unlocked | 857.3 | 1755 | 346.0 W | Active |

### The cleanest `argmax` attribution in the campaign

The clock changes are a natural controlled experiment, and they confirm the
bottleneck far better than experiment 0136's `decode_x2` arm did:

| Arm | Clock change | Throughput change | Scaling |
|---|---|---|---|
| 0135 decoder | 1440 → 1605 MHz, +11.5% | +11.0% | **0.96 : 1** |
| PRMT successor | 1065 → 1605 MHz, +50.7% | +6.0% | **0.12 : 1** |

An ALU-bound kernel must scale about 1:1 with SM clock; a DRAM-bound one must
barely scale at all. Both predictions hold exactly. The 0135 decoder is
ALU-bound and the successor is DRAM-bound, now established by two independent
methods.

### The campaign ruler, re-measured

The contract requires re-measurement when clocks change, so the 0134 baseline
probe was re-run at the new operating point:

| Shape | Arm | 1605 locked / 250 W | unlocked / 350 W | Delta |
|---|---|---:|---:|---:|
| `gate_up_w1` | roofline, cold | 840.71 | 845.63 | +0.6% |
| `gate_up_w1` | production, cold | 87.04 | 90.67 | +4.2% |
| `gate_up_w1` | N64 WMMA, cold | 161.37 | 174.08 | +7.9% |
| `down_w2` | roofline, cold | 845.63 | 845.63 | 0.0% |
| `down_w2` | production, cold | 87.04 | 90.67 | +4.2% |
| `down_w2` | N64 WMMA, cold | 174.08 | 181.33 | +4.2% |

**The ruler is unchanged, because it is memory-bound.** The 842-class figure
stands and **the >600.0 parity and 632 surpass gates do not move.** Unlocking
makes those gates honestly *meetable*; it does not lower them. That distinction
is load-bearing — moving a threshold so a rejected candidate survives is
exactly what the charter forbids, and this is not that.

## A defect in the gate protocol itself

The cold probe boosts to **1935 MHz** at only 150–200 W, because its arms are
160–260 µs with a 256 MiB scrub between them. Sustained load settles at
**1755 MHz** pinned against 350 W. The same 0135 decoder therefore measures
**621.71 GB/s cold** and **568.5 GB/s sustained**, a 9.4% gap.

**The F4-2 cold protocol overstates an ALU-bound candidate by about 9% and a
DRAM-bound one not at all.** This is independent of the owner's tuning and is a
property of the measurement itself. Any ALU-bound candidate that clears the
gate only on the cold protocol has not really cleared it, and every future
result must report its sustained clock alongside the cold number.

## Correction to experiment 0136

Experiment 0136 recorded: *"REJECTED: the 0135 shift/rebias decoder cannot
support the F4-2 parity gate on SM86."*

That is **too strong, and the scope was wrong**. The corrected statement:

- Its **measurements were correct** at the operating point they were taken at,
  and its internal reasoning — the ALU attribution, the 13.1 ops-per-code-pair
  budget, the two probe defects it caught — all stand.
- The 0135 decoder **fails** the gate at the locked 1605 MHz operating point
  (514.02 / 512.00) and **passes** it at stock clocks (621.71 / 604.09).
- The rejection was therefore **operating-point-dependent** and was reported as
  though it were a property of the decoder.

The earlier row is preserved unedited, per the contract. This record is its
correction.

**What survives the correction, and why the campaign's direction does not
change.** The 0135 decoder clears `down_w2` by 0.7% as a *decoder-only ceiling*
— before the fragment prepack, activation feed, output publication and split-K
are paid — and gives back 9.4% under sustained load. It has no usable margin at
either operating point. The PRMT successor sits at 831.59 GB/s against a
read floor of 847.79, is clock-independent at 0.12:1, and is unchanged across
all three configurations to within 2.5%. **F4-1 continues on the successor, not
because 0135 fails the gate, but because 0135 has no margin and the successor
has all of it.**

## Retired and amended constants

- **`B_ALU` = 10.35 Tops/s (experiment 0136) is RETIRED.** It was measured at a
  locked 1605 MHz and describes no operating point the campaign now uses.
- The **842-class ruler stands**, re-measured at 845.63 GB/s cold.
- The **600.0 / 632 gates stand**, unchanged.
- Experiment 0136's **13.1 ALU-ops-per-code-pair budget** was derived at
  1605 MHz. It is retained only as the screen that correctly predicted the
  successor's behaviour, and must be re-derived before it is used to reject a
  future decoder.

## Owner amendment: two operating points, never to be conflated

- **Experimentation:** a single RTX 3090, 350 W, stock unlocked clocks, with the
  second 3090 idle. Chosen by the owner so that campaign numbers are silicon
  numbers rather than tuning artefacts. Not reboot-persistent — the tuning
  service reapplies 250 W and the 1605 lock at boot.
- **Production:** both RTX 3090s under TP2 load, 250 W each, SM clock locked at
  1605 MHz. Required by the owner's electrical installation, which is
  constrained only when both cards are heavily loaded simultaneously.

Every future result must state which point it was taken at. An ALU-bound kernel
differs by 32% between them (621.71 against 470.6); a DRAM-bound one differs by
2.5%. **A production claim may not be made from an experimentation number**, and
the eventual MIX-2 end-to-end result must be measured at the production point.

## Gate verdict

| Gate | Required | Result | Verdict |
|---|---|---|---|
| Ruler stability across operating points | re-measured, not assumed | 845.63 GB/s, +0.0–0.6% | PASS |
| Gates unmoved | 600.0 / 632 unchanged | unchanged | PASS |
| Second 3090 idle throughout | owner safety condition | 0%, 51.9–52.5 W, all runs | PASS |
| Tuning restored after measurement | caps back on both cards | verified 250 W / 1605 both | PASS |
| Oracles across all configurations | 0 mismatches | 0, every run | PASS |

## What this does not establish

- No new candidate, kernel, or dispatch. This is an operating-point and
  correction record.
- Nothing about FP8, E4M3, block-128 scales, or D-F8-GATE.
- It does **not** re-open the 0135 decoder as the basis for F4-1. It removes the
  false claim that the decoder is incapable, while leaving the real reason to
  prefer the successor intact.
- The sustained figures were taken with a standalone harness, not the gate
  probe. They are attribution evidence, not headline numbers.

## Exact next action

F4-1 continues on the PRMT successor at the experimentation operating point, in
the order established by experiment 0137:

1. **Re-measure whether the MMA is still free**, with a `decode_mma` arm built
   on the successor rather than the 0135 decoder. The successor now sits within
   1.9% of the read floor (831.59 against 847.79), so there is even less room
   than experiment 0137 recorded, and 0136's free-MMA result may not be
   inherited.
2. Build the E2M1/E8M0 group-32 fragment prepack for both production shapes and
   prove the scale-to-K binding across a group boundary.
3. Add an admission check for E8M0 codes 0 and 255.
4. Time a full candidate step against the unmoved >600.0 GB/s gate, reporting
   **both** the cold number and the sustained clock, so the 9.4% cold-protocol
   inflation cannot hide inside the result.

**F8-0 remains open, unblocked, and independent. D-F8-GATE remains an open
owner decision.**
