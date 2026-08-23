#!/usr/bin/env python3
"""
test_matches_parameters.py
==========================
The testbed restates the cost matrix literally in common.py rather than
importing ../parameters.py, so that this directory runs on a phone with
nothing but the Python standard library. That duplication is a place where
the two copies can silently drift apart, so this test checks they agree.

Run it from the repository root (it needs both modules importable):

    python testbed/test_matches_parameters.py

Exits non-zero if any value disagrees. Unlike the rest of testbed/, this
script does import the simulation package, because comparing the two is the
whole point; it is a development check, not part of the deployed node.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import common                    # noqa: E402  (testbed copy)
import parameters as P           # noqa: E402  (cited source of truth)

TOL = 1e-6
problems = []


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


if not close(common.C_FLOOR, P.C_FLOOR):
    problems.append(f"c_floor: testbed={common.C_FLOOR} parameters={P.C_FLOOR}")

for dev in P.DEVICES_IN_HAZARD_ORDER:
    hazard = P.HAZARD_OF_DEVICE[dev]
    want = P.cost_triple(dev)
    got = common.COST_TABLE[hazard]
    for name, w, g in zip(("Lambda", "alpha", "phi"), want, got):
        if not close(w, g):
            problems.append(f"{hazard} {name}: testbed={g} parameters={w}")

want_estop = P.cost_triple("arm", P.T_STOP_S_ARM_EMERGENCY_STOP)
got_estop = common.COST_CRITICAL_EMERGENCY_STOP
for name, w, g in zip(("Lambda", "alpha", "phi"), want_estop, got_estop):
    if not close(w, g):
        problems.append(f"CRITICAL/e-stop {name}: testbed={g} parameters={w}")

for name, a, b in [("rho_min", common.RHO_MIN, P.RHO_MIN_S),
                   ("rho_max", common.RHO_MAX, P.RHO_MAX_S),
                   ("g_reflex", common.G_REFLEX, P.G_REFLEX)]:
    if not close(a, b):
        problems.append(f"{name}: testbed={a} parameters={b}")

print("=" * 66)
if problems:
    print(f"{len(problems)} DISAGREEMENTS between testbed/common.py and parameters.py:")
    for p in problems:
        print(f"  - {p}")
    print("\nUpdate testbed/common.py to match parameters.py.")
    sys.exit(1)
print("testbed/common.py agrees with parameters.py on every value.")
print("=" * 66)
sys.exit(0)
