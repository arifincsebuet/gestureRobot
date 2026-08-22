# Reproducibility artifact: Risk-Asymmetric Gesture Authorization (RAGA)

This repository accompanies the paper **"Risk-Asymmetric Gesture
Authorization: Decoupling Stop and Start Authority in IoT and Human-Robot
Environments,"** submitted for double-anonymous review. It reproduces every
numeric claim, table, and figure in the paper's Experimentation section
(Sec. VI, E1-E13), and includes a runnable implementation of the physical
three-node testbed described there.

> **Anonymity note.** This repository is anonymized for review: no author
> names, affiliations, or identifying paths appear anywhere in it. Do not add
> any before the review period ends. A camera-ready revision can restore
> attribution and add a LICENSE file (not included here, to keep the review
> copy minimal — MIT or Apache-2.0 are common choices for this kind of
> artifact).

## What's here

```
raga_experiments.py      Monte Carlo simulation for E1-E13 (~3 min on a laptop core)
merge_figures.py          combines per-experiment figures into the paper's multi-panel figures
verify_paper_claims.py    checks every number the paper states against regenerated data
dataset/                  reference CSVs from a verified run (see "Reference data" below)
requirements.txt          numpy, matplotlib (simulation only)
testbed/                  physical three-node UDP testbed (pure standard library)
```

## 1. Reproducing the simulation results (E1-E13)

```bash
pip install -r requirements.txt
python raga_experiments.py       # writes ./figures/*.png and ./dataset/*.csv
python merge_figures.py          # combines pairs of figures into the paper's panel figures
python verify_paper_claims.py    # checks every number above against the manuscript
```

`verify_paper_claims.py` is the artifact the paper itself refers to ("Every
numeric claim in the paper is checked automatically against regenerated data
by a script released with the artifact; it also encodes structural guards
that fail if any headline finding stops holding," Sec. IV-D). It hardcodes
every number as printed in the manuscript, regenerates the data independently
from `raga_experiments.py`'s output, and compares the two — it does not read
its expected values from the same run it's checking. A clean run prints:

```
========================================================================
Verified 126 numeric claims against dataset/
========================================================================

All paper claims are consistent with the generated data.
```

Beyond simple number matching, it also checks the paper's *structural*
claims — for example, that the polarity-only ablation is actually worse than
full RAGA on the joint objective (Sec. VI, E2), that the cost-optimal and
robustness-optimal threshold separations are actually different (E12), and
that conjunctive quorum actually dominates noisy-OR fusion on both security
and friction (E13). These are the claims most likely to silently break if
the simulation is ever refactored, which is exactly why the script exists —
see its module docstring for the history of why it was written.

All Monte Carlo randomness is explicitly seeded (see `RNG_MASTER_SEED` and
the per-experiment seed offsets in `raga_experiments.py`); a fresh run should
reproduce the reference data in `dataset/` exactly. The one exception is E9
(decision throughput), which is a wall-clock timing measurement — the paper
reports it as an order-of-magnitude comparison for exactly this reason, and
`verify_paper_claims.py` checks only the order of magnitude, not the exact
number.

### Reference data

`dataset/` contains the CSVs from one verified run (all 126 checks passed
against the current manuscript). This is included so a reviewer can inspect
the exact numbers behind every table without spending the ~3 minutes to
regenerate them, and to make it easy to spot if a future code change causes
the data to drift from what's committed here. Regenerating is still
recommended before trusting the artifact — see above.

## 2. The physical testbed

The paper's Sec. VI ("Physical Testbed") describes implementing the decision
logic "as a single stdlib-only Python node on three Android phones running
Termux, over UDP with HMAC-SHA256-signed envelopes (one generator, two device
agents: CRITICAL robot, LOW lamp)." That code lives in `testbed/`:

```
testbed/common.py               shared config, threshold math (Eqs. 2-4), envelope sign/verify
testbed/generator.py            Tier 2 node: builds and broadcasts a signed authority envelope
testbed/device_agent.py         Tier 3 node: Algorithm 2, run independently per device
testbed/trigger_hazard_event.py stands in for a device's own hazard detection, for testing ERLC
testbed/run_local_demo.sh       end-to-end demo on localhost, no phones required
```

Every file in `testbed/` uses only the Python standard library (`socket`,
`hmac`, `hashlib`, `json`, `argparse`, `threading`) — no `pip install`
needed, so it runs unmodified under Termux's stock `python3` exactly as
described in the paper.

### Quick check (no hardware required)

```bash
cd testbed
bash run_local_demo.sh
```

This starts two device-agent processes on localhost (the lamp and the robot
arm) and drives them through the paper's three reported testbed
observations, printing each decision and appending it to a `.jsonl` log:

1. An unenrolled visitor (`c = 0.001`) halts the **CRITICAL** robot arm →
   `EXECUTE` (`T_eff = 0.0001`, matching Table I's `T_res` for that class).
2. The same visitor cannot halt the **LOW** lamp → `WITHHOLD`
   (`T_eff = 0.33333`, again matching Table I).
3. The same halt on the same lamp, ~0.4 s after a hazard event is registered
   on that lamp's own timeline, falls inside the ERLC reflex band →
   `EXECUTE` at `T_eff = 0.00033` — reproducing the exact threshold value the
   paper reports for this trial.

### Running on real hardware (e.g. three Termux phones)

Run one `device_agent.py` per device on its own phone, and `generator.py` on
a third, pointing `--agents` at the device phones' LAN IP addresses instead
of `127.0.0.1`. See each script's `--help` for the full option list; the
options and defaults match the parameters in Table I and Eqs. (2)-(4) of the
paper.

The reflex-band leniency multiplier (`epsilon` in Eq. (4)) is set to
`c_floor = 1e-3` in `common.py`; this value is back-computed from the paper's
own reported measurement (a halt threshold of `T = 0.00033` on the lamp,
`T_res = 0.33333 * epsilon`), not independently asserted — see the comment
in `common.py` for the derivation. The `TemporalWindow` staleness bounds are
implementation choices the paper explicitly does not fix numerically
(Sec. V-F: "we neither evaluate it separately nor claim it as a
contribution"); the defaults here are documented as such and are safe to
override with `--res-window` / `--perm-window`.

Conjunctive quorum (`--c-second` on `generator.py`) is included as an
extension for anyone who wants to exercise the corrected quorum design
(Sec. VI, E13) on physical hardware too — the paper's own three-phone trial
tested only the threshold inversion and ERLC, not quorum.

## Threats to validity (carried over from the paper)

As the paper states (Sec. VI, "Threats to Validity"): every input
distribution in the simulation (identity confidence, gesture error, the
reaction band, the cost parameters) is assumed, not measured, and every
headline multiplier is a property of this simulation, not externally
validated. The physical testbed validates *distributed decision behaviour*,
not sensing: identity confidence is injected on the command line, with no
gesture recognition in the loop. Human-subject measurement of the reaction
band and gesture error on deployed hardware is future work.
