# Orchestrated experiment protocol

How to run Strata performance work as an **orchestrator** (a reasoning agent that
decides direction) driving an **executor** (a coding agent in a terminal pane that
measures and builds). Written after the August 2026 DSV4 prefill session, which
produced a validated 1.53x by following this shape — and which produced it
*despite* the orchestrator being wrong about seven successive premises.

This document is the operating manual. `CLAUDE.md` is the research charter and
outranks it. Where this document adds rules, they are rules the charter implies
but that were still violated in practice until written down.

---

## 0. The one lesson that generated the rest

> Seven of the orchestrator's premises were wrong. The executor caught most of
> them before code was written. What produced the result was not hypotheses —
> it was instrumentation, the reference stack's documented recipe, and a
> protocol that made stopping cheaper than guessing.

Every rule below exists because skipping it cost time in that session. The
orchestrator's job is **not** to be right about mechanisms. It is to keep the
cost of being wrong low, and to make sure the wrongness is discovered by a
measurement rather than by a merged regression.

---

## 1. Roles

**Orchestrator.** Chooses the next question, writes briefs, sets gates, decides
what lands. Does not write production code. Its scarce resource is judgement
about *what to measure next*, not about *what the answer is*.

**Executor.** Reads source, builds instrumentation, runs measurements,
implements, gates. Has full context of the repository and is expected to
**falsify the orchestrator's premise** when source contradicts it. This is its
most valuable function, not an exception path.

The executor stopping to say "your premise is wrong" is a success. Budget for it.

---

## 2. Environment

Run the executor in a **separate git worktree** on an experiment branch:

```bash
herdr worktree create              # or an existing worktree
herdr pane split --current --direction right --cwd "$PWD" --no-focus
herdr agent start <name> --kind codex --pane <returned-pane-id>
```

A worktree gives a real mechanical guard: git refuses to check out `main` in two
worktrees, so the executor **cannot** switch to `main` even by mistake. Do not
rely on that alone — see §4 — but do take it.

### Herdr mechanics that are not obvious

These cost real time to learn:

- **The executor cannot message you.** It is a separate process in a terminal.
  The only signal is its lifecycle state.
- **Keep a watch armed at all times, and re-arm it every time it fires.**
  Otherwise the executor finishes into silence and sits idle while you believe
  it is working.
  ```bash
  herdr agent wait <name> --timeout 3600000
  ```
- **Never pass `--until idle`.** Herdr's `idle` means settled *and the tab has
  been seen in the focused UI*; an unfocused background tab settles as `done`.
  `--until idle --until blocked` therefore never fires. Confusingly
  `herdr agent get` still prints `agent_status: idle`. Use the **bare** wait,
  whose default set is idle|done|blocked. Cost of learning this: 18 minutes of a
  finished agent waiting unnoticed.
- **`prompt --wait` is not a standing watch.** It returns on the first settle;
  after that you are blind.
- **A CLI timeout is not a failure.** `{"error":{"code":"timeout"}}` just means
  the window expired. Check `herdr agent get` before concluding anything.
- **Require a durable report file.** Terminal scrollback is lost on the
  alternate screen and `--lines` cannot recover it. Have the executor overwrite
  a fixed path with its full check-in *every time it stops*, before ending its
  turn.

---

## 3. The executor's standing protocol

Give this to the executor once, at spawn, and restate it in every brief. It is
not implied by the charter; the charter's "do not merge failed runtime code" was
read as "delete it" and cost a working bit-exact candidate.

1. **Destructive actions require explicit sign-off for that specific action.**
   No `reset --hard`, `checkout -- <path>`, `clean`, `stash drop`, `branch -D`,
   ref deletion, force update, history rewrite, or killing processes it did not
   start.
2. **Never merge, push, or move a ref without per-action sign-off.** This is
   separate from (1) — publishing is not "destructive" but is the higher risk.
3. **"Do not merge failed runtime code" means do not merge it to `main`. It does
   not mean destroy it.** Commit rejected candidates on the experiment branch
   with a message marking them rejected. A rejected result whose code is gone
   cannot be re-examined — and rejected candidates are sometimes good candidates
   with one identified defect.
4. **Preserve first, decide second.** Commit intermediate states. Keep result
   JSON, logs, counter dumps, probe sources.
5. **Stop and report at gates and decisions — in either direction — before
   acting on the consequence.** Also when a number contradicts the design, when
   a budget is about to be exceeded, or when the brief's premise looks wrong.
6. **Execute steps already authorized in the current brief without checking
   back.** Building instrumentation and then running the arm it was built for is
   one unit of work, not two. Stop at gates, not at every intermediate
   checkpoint.
7. **Report honestly, including partial and negative results.** A 1.31x against
   a 1.5x gate is reported as exactly that — not as a failure with no number,
   and not as a success.
8. **Never infer approval.** Approval for one action is not approval for the
   next.

---

## 4. Protecting `main`

- All work on `exp/…` branches in the worktree. `main` is only ever touched by
  the orchestrator, from the primary checkout, after explicit user authorization.
- **The branch tip is not a landing candidate.** A long experiment branch is a
  linear stack that deliberately contains rejected work. Merging it wholesale
  lands rejected experiments. Landing must be curated commit by commit with a
  stated rationale for each.
- **But the validated artifact is the tree, not the commits.** The measured
  result was produced by one specific tree. Cherry-picking a subset produces a
  combination nobody measured. Reconcile these two facts explicitly at landing
  time: either land the whole tree because inspection shows nothing in it should
  be excluded, or re-measure whatever subset you land.
- Before pushing, verify the push is a fast-forward and that no remote work is
  discarded.
- State in the merge message which rejections in the ancestry were later
  overturned, or a future reader will see "reject X" three commits before X
  lands and conclude something went wrong.

---

## 5. Designing gates

This is where the orchestrator did the most damage in the reference session.

- **Do not set thresholds on sub-terms.** Two experiments passed every
  structural gate and were rejected on invented numbers for one bucket. Gate on
  what actually matters.
- **Gate on noise-immune counters when the wall clock is contaminated.** Device
  kernel time, dispatch counts, kernel launches, bytes moved. These are
  deterministic and cannot be perturbed by host-side placement luck. Use the
  wall clock for value, but only after §6.
- **A structural or numerical pass with a partial timing win is
  report-and-discuss, not automatic rejection.** Say this in the brief. Without
  it, working code gets binned.
- **When a gate fires, check whether its stated premise survived.** One
  experiment's kill criterion said "if the clock does not move, the cost is in
  the kernels" — and the same run's device counter showed the kernels were 4 s
  of 14 s. The gate fired on a premise its own data disproved. That is a
  different situation from a real falsification and must be handled differently.
- **Distinguish a second attempt from a rescue.** Fixing a defect the experiment
  itself identified and re-running the *unchanged* gate at the *unchanged*
  operating point is a second attempt. Moving the threshold, changing the prompt
  length, or hunting for a friendlier regime is laundering. Say which one you
  are doing, in the brief, explicitly.

---

## 6. Measure the noise floor before setting any threshold

The charter says never reuse a measured constant across operating points. The
reference session violated it with a *variance* figure and it suppressed two
finished results.

- A variance figure measured at one operating point does not describe another.
  A 15.5/16.1/32.4 s spread measured at page 64 moving 468 GB says nothing about
  page 8192 moving 74 GB.
- **Measure the spread at the operating point you are gating on**, with runs
  that differ only in noise, before you assert any threshold.
- Report **every run and the range**, not just a median. The reference session's
  promotion campaign found a 13.5 s baseline range and a 2.6 s candidate range —
  both useful, and neither visible in a median.
- If the effect is far outside the measured band in every pair, say so; that is
  a stronger claim than a median ratio.

---

## 7. Sequencing an investigation

The order that worked:

1. **Read the reference implementation first.** For DSV4 that is
   `/home/rodrigo/Developer/Lvllmds4-x` — `bench/launch.sh`, `bench/results.txt`,
   the sparse indexer, the MLA kernels, and the `SM80_DEEPSEEK_V4_NOTES.md`
   porting notes. Three separate times in one session, reading it would have
   saved a full cycle; twice the answer was in a file already opened. The
   reference has usually already solved the problem *on this hardware* and
   documented the numerical contract.
2. **Attribute before hypothesising.** The single most productive action of the
   session was an instrumentation-only experiment that split a 10 s
   synchronization term and two projection buckets by subsystem, with nothing
   over 0.4 s unexplained. It killed two hypotheses and located the real one.
3. **Microbenchmark the mechanism before the system.** Minutes, no model load.
   A negative screen kills the idea cheaply.
4. **Validate the probe against production.** A probe that reproduces the
   production counter to within ~1% is trustworthy; one that does not is
   measuring something else. In the reference session an "8x discrepancy"
   between probe and production turned out to be the probe exercising a
   different kernel entirely.
5. **Only then implement**, with correctness gated first and value second.

---

## 8. Verifying a claim before acting on it

The orchestrator's failure mode was consistent and is worth naming: **reasoning
from aggregate counters and skimmed code instead of verifying the specific fact
the claim depends on.** Concretely, in one session:

- Divided a *semantic* matmul counter by rows and layers and concluded the
  projections were dispatched per row. They had been batched for several
  experiments already.
- Benchmarked a BF16 kernel path for tensors that are FP8 E4M3 block-128 — after
  reading a reference document that stated the encoding plainly.
- Repeated a handover's "no NUMA policy" claim in six briefs; the production
  arena had explicit, stable, verified binding.
- Asserted a noise floor from the wrong operating point, twice, in both
  directions.

Rules that follow:

- **Check the encoding, dtype and dispatch path your claim depends on, in
  source, before writing the brief.**
- **Never derive a dispatch count from a semantic counter.** Confirm what a
  counter counts before dividing by anything.
- **Do not repeat a handover or prior record as fact.** Prior documents are
  hypotheses with provenance. Verify the specific claim you are about to act on,
  and correct the document when it is wrong — in place, naming the experiment
  that disproved it.
- **Hand the executor your arithmetic explicitly flagged as unverified**, and
  ask it to correct you. It will.

---

## 9. Budgets

- State expected wall time per arm and in total, **before launching**.
- Prefer the cheapest falsifying measurement. A microbenchmark that decides the
  question in minutes beats a six-arm matrix.
- **One cheap pair that already falsifies the hypothesis ends the matrix.** In
  the reference session an eight-arm campaign spent an hour confirming a null
  result that the first pair had shown in eight minutes.
- Instrumentation runs need no baseline arm — a measurement is not an A/B.
- Variance gates legitimately need three or more identical arms. That is the one
  case where a larger budget is the point; authorize it explicitly.

---

## 10. Recording

- One experiment number per bounded hypothesis, including falsifications and
  pre-implementation stops. `docs/experiments/NNNN-slug-YYYY-MM-DD.md`.
- A rejection record is a first-class deliverable. State the premise, what
  disproved it, and what the numbers were.
- **Do not relabel an earlier failure when a later experiment reframes it.** Keep
  the original result visible and add the correction. One session's FP32
  numerical failure stayed on the record while a later experiment proved the
  differing bits did not survive the production rounding boundary — both facts
  are true and both matter.
- When a mechanism is rejected but its structural counters passed, say so
  precisely. "Rejected on the declared timing gate; structurally valid at 338x
  fewer dispatches and bit-exact" is a useful record. "Rejected" alone is not.

---

## 11. Checklist for a new investigation

```
[ ] Read the reference implementation for this subsystem
[ ] Read the last few experiment records and verify, do not inherit, their claims
[ ] Instantiate the cost model: measure B_r and W_r, name argmax_r
[ ] Measure the noise floor at THIS operating point
[ ] Spawn executor in a worktree; give it the standing protocol (§3)
[ ] Arm a bare `herdr agent wait`; require a durable report file
[ ] Brief: hypothesis, primary metric, correctness gate, memory ceiling,
    rollback condition, arm budget, and which of your claims are unverified
[ ] Gate on noise-immune counters first, value second
[ ] Re-arm the watch after every fire
[ ] Curate the landing; never merge a branch tip unexamined
```

---

## 12. Worked outcome

The August 2026 DSV4 prefill session, run this way: eleven experiments
(0097–0107), six of them falsifications, three of those caught before any code
was written. Landed **1.53x** median total prefill at 677 tokens — 81.24 s /
8.33 tok/s to 52.96 s / 12.78 tok/s — via row-batched page attention (29,111
dispatches to 86; 6.99 GB of page reads to 30.6 MB) and SM86 FP8 tensor
projections (7.21 s to 0.33 s of device kernel), both capability-gated with
decode untouched.

The two landed mechanisms came from an attribution run and from the reference
stack's documented Ampere recipe. Neither came from an orchestrator hypothesis.
