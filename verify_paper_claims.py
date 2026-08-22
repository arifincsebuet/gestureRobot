#!/usr/bin/env python3
"""
verify_paper_claims.py
======================
Checks every numeric claim in the paper against the CSVs produced by
raga_experiments.py. Exits non-zero if any claim disagrees with the data.

This exists because during preparation of this paper the experiment script was
refactored, which changed the RNG draw order and therefore every result by a
small amount -- while the manuscript still carried the older numbers. Nine
values were stale before this checker was written. Run it after any change to
raga_experiments.py, and before every submission.

Usage:
    python raga_experiments.py      # regenerate dataset/
    python verify_paper_claims.py   # check the paper against it
"""

import csv
import os
import sys

DATA_DIR = "dataset"

# ── Claims as printed in the manuscript ──────────────────────────────────
# Update these ONLY by copying from the paper, never from the data. The whole
# point is to catch the two drifting apart.

PAPER_CLAIMS = {
    "E1_safety_jump_at_cliff": 11388,          # E1
    # Table III: (safety, security, nuisance)
    "E2": {
        "uniform_loose":     (0.0,    1850.3, 753.4),
        "uniform_mid":       (2865.6, 921.3,  124.1),
        "uniform_strict":    (3053.4, 607.0,  83.2),
        "polarity_only":     (191.5,  55.1,   753.4),
        "raga":              (213.3,  66.6,   433.1),
        "raga+quorum":       (185.4,  217.7,  433.1),
        "raga+quorum+live":  (185.8,  186.1,  404.6),
    },
    # E2: the coupling must beat polarity_only on the 3-metric objective
    "E2_paired_mean_diff":    223.4,
    "E2_cohens_d":            8.94,
    # Table VIII: stranger-spread robustness
    "E11": {0.0: 3493, 0.1: 1649, 0.25: 1002, 0.5: 578, 1.0: 302},
    # Table IX: phase transition (gap -> sec+nui, safety@2% err)
    "E12": {
        0.00: (709.9, 1613.5),
        0.50: (759.1, 809.1),
        0.80: (592.3, 811.8),
        0.85: (604.3, 36.6),
        0.90: (509.7, 38.5),
        0.95: (409.9, 40.1),
        1.00: (334.0, 185.8),
        1.50: (320.1, 185.8),
    },
    "E12_crossing_gap":       0.817,   # bisection, E12
    "E12_fragility_g095":     4.6,     # abstract + contributions
    "E12_step_at_crossing":   22.2,    # 811.8 -> 36.6
    # Table X: four-quadrant accounting (safety, security, nuisance, friction)
    "E13": {
        "uniform_loose": (0.0,   3751.0, 513.0, 0.0),
        "polarity_only": (164.0, 55.0,   513.0, 13793.0),
        "raga":          (190.0, 67.0,   316.0, 8696.0),
        "raga+quorum":   (169.0, 377.0,  306.0, 7640.0),
    },
    # Table XI: quorum fusion semantics
    "E13_quorum": {
        "none":        (65.0,  8557.0),
        "noisy_or":    (371.0, 7500.0),
        "conjunctive": (74.0,  6288.0),
    },
    "E3": {                                     # Table V (naive_safety, failsafe_security)
        0.00: (0.0,   44.0),
        0.01: (63.3,  43.5),
        0.02: (125.2, 43.3),
        0.05: (323.2, 41.9),
        0.10: (653.2, 39.4),
    },
    "E4_inversion_held": (15, 15),              # E4
    "E5": {                                     # Table IV
        "naive":         1.0000,
        "opportunistic": 1.0000,
        "informed":      0.9206,
        "expert":        0.5468,
    },
    "E6": {                                     # E6, percent
        1:   1.2,
        10:  10.0,
        50:  31.1,
        500: 49.0,
    },
    "E8_regret_bound":        20000.2,          # Cor. 1
    "E8_operator_in_band":    True,
    # Throughput is hardware-dependent. Within a session CV is ~0.9%, but it
    # varies by tens of percent across sessions on shared virtualised hardware.
    # The paper therefore reports an ORDER OF MAGNITUDE, and we check only that.
    "E9_throughput_order":    5,                # i.e. ~10^5 decisions/s
    "E10": {                                    # Table VI
        "full (raga+quorum+live)": (113.7, 115.5),
        "minus liveness":          (113.5, 135.9),
        "minus quorum":            (129.9, 21.1),
        "minus both (raga core)":  (129.7, 43.1),
    },
    "aggregate_reduction_x":  3.4,              # Abstract + E2
}

TOL = 0.05        # absolute tolerance for one-decimal figures
TOL_AUC = 0.0001  # AUC printed to 4 dp

problems = []
checks = 0


def read(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        problems.append(f"MISSING DATASET: {fname} (run raga_experiments.py first)")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


def check(label, paper, data, tol=TOL):
    global checks
    checks += 1
    if not close(paper, data, tol):
        problems.append(f"{label}: paper={paper} data={data}")


# ── E1 ───────────────────────────────────────────────────────────────────
rows = read("exp1_trap.csv")
if rows:
    jump = max(int(r["safety"]) for r in rows
               if float(r["T"]) < 0.002 and int(r["safety"]) > 0)
    check("E1 safety jump at cliff", PAPER_CLAIMS["E1_safety_jump_at_cliff"], jump, tol=0)

# ── E2 ───────────────────────────────────────────────────────────────────
rows = read("exp2_policy_summary.csv")
if rows:
    d = {r["policy"]: r for r in rows}
    for pol, (saf, sec, nui) in PAPER_CLAIMS["E2"].items():
        if pol not in d:
            problems.append(f"E2: policy '{pol}' missing from data")
            continue
        check(f"E2 {pol} safety",   saf, round(float(d[pol]["safety_mean"]), 1))
        check(f"E2 {pol} security", sec, round(float(d[pol]["security_mean"]), 1))
        check(f"E2 {pol} nuisance", nui, round(float(d[pol]["nuisance_mean"]), 1))

    # aggregate reduction claim (abstract headline), now on the 3-metric total
    best_uni = float(d["uniform_loose"]["total_mean"])
    full     = float(d["raga+quorum+live"]["total_mean"])
    check("aggregate reduction factor",
          PAPER_CLAIMS["aggregate_reduction_x"], round(best_uni / full, 1), tol=0.05)

    # STRUCTURAL: the coupling must beat polarity_only on the 3-metric total.
    # If this inverts, E2's central argument is false.
    checks += 1
    po_total = float(d["polarity_only"]["total_mean"])
    rg_total = float(d["raga+quorum+live"]["total_mean"])
    if not po_total > rg_total:
        problems.append(
            f"E2 STRUCTURAL: polarity_only ({po_total}) should be WORSE than "
            f"raga+quorum+live ({rg_total}) on the 3-metric objective. "
            f"E2's central claim is contradicted by the data.")

rows = read("exp2_paired_test.csv")
if rows:
    r = rows[0]
    check("E2 paired mean difference", PAPER_CLAIMS["E2_paired_mean_diff"],
          float(r["mean_difference"]), tol=0.5)
    check("E2 Cohen's d", PAPER_CLAIMS["E2_cohens_d"],
          float(r["cohens_d"]), tol=0.02)

# ── E3 ───────────────────────────────────────────────────────────────────
rows = read("exp3_gesture_error.csv")
if rows:
    d = {round(float(r["gesture_error"]), 2): r for r in rows}
    for err, (naive, fs_sec) in PAPER_CLAIMS["E3"].items():
        if err not in d:
            problems.append(f"E3: error rate {err} missing from data")
            continue
        check(f"E3 {err:.0%} naive safety",     naive,  round(float(d[err]["naive_safety"]), 1))
        check(f"E3 {err:.0%} failsafe security", fs_sec, round(float(d[err]["failsafe_security"]), 1))
        # the central structural claim: fail-safe must be exactly zero everywhere
        checks += 1
        if float(d[err]["failsafe_safety"]) != 0.0:
            problems.append(f"E3 {err:.0%}: fail-safe safety should be 0, "
                            f"got {d[err]['failsafe_safety']}")

# ── E4 ───────────────────────────────────────────────────────────────────
rows = read("exp4_sensitivity.csv")
if rows:
    held = sum(1 for r in rows if r["inversion_holds"].strip().lower() == "true")
    want_held, want_total = PAPER_CLAIMS["E4_inversion_held"]
    check("E4 inversion held (count)", want_held, held, tol=0)
    check("E4 configurations tested",  want_total, len(rows), tol=0)

# ── E5 ───────────────────────────────────────────────────────────────────
rows = read("exp5_erlc_adversarial.csv")
if rows:
    d = {r["adversary"]: float(r["auc"]) for r in rows}
    for adv, auc in PAPER_CLAIMS["E5"].items():
        if adv not in d:
            problems.append(f"E5: adversary '{adv}' missing from data")
            continue
        check(f"E5 {adv} AUC", auc, round(d[adv], 4), tol=TOL_AUC)

# ── E6 ───────────────────────────────────────────────────────────────────
rows = read("exp6_clock_skew.csv")
if rows:
    d = {int(r["clock_skew_ms"]): float(r["disagreement_rate"]) * 100 for r in rows}
    for ms, pct in PAPER_CLAIMS["E6"].items():
        if ms not in d:
            problems.append(f"E6: skew {ms}ms missing from data")
            continue
        check(f"E6 {ms}ms disagreement %", pct, round(d[ms], 1), tol=0.15)

# ── E8 ───────────────────────────────────────────────────────────────────
rows = read("exp8_calibration_regret.csv")
if rows:
    r = rows[0]
    check("E8 regret bound", PAPER_CLAIMS["E8_regret_bound"],
          float(r["regret_bound"]), tol=1.0)
    checks += 1
    inside = r["operator_inside_band"].strip().lower() == "true"
    if inside != PAPER_CLAIMS["E8_operator_in_band"]:
        problems.append(f"E8 operator-inside-band: paper="
                        f"{PAPER_CLAIMS['E8_operator_in_band']} data={inside}")

# ── E9 ───────────────────────────────────────────────────────────────────
rows = read("exp9_throughput_comparison.csv")
if rows:
    raga = [r for r in rows if r["system"].startswith("RAGA")]
    if raga:
        import math
        got = float(raga[0]["throughput"])
        checks += 1
        order = int(math.log10(got))
        if order != PAPER_CLAIMS["E9_throughput_order"]:
            problems.append(
                f"E9 throughput order of magnitude: paper=10^"
                f"{PAPER_CLAIMS['E9_throughput_order']} data=10^{order} "
                f"({got:,.0f}/s). The paper's claim is an order of magnitude, so "
                f"this only fails on a genuine order-of-magnitude change.")

# ── E10 ──────────────────────────────────────────────────────────────────
rows = read("exp10_ablation.csv")
if rows:
    d = {r["configuration"]: r for r in rows}
    for cfg, (saf, sec) in PAPER_CLAIMS["E10"].items():
        if cfg not in d:
            problems.append(f"E10: configuration '{cfg}' missing from data")
            continue
        check(f"E10 {cfg} safety",   saf, round(float(d[cfg]["safety_mean"]), 1))
        check(f"E10 {cfg} security", sec, round(float(d[cfg]["security_mean"]), 1))

    # the paper's two-sided claim: quorum removal must move BOTH axes,
    # security down and safety up. If this ever stops holding, the prose is wrong.
    if "full (raga+quorum+live)" in d and "minus quorum" in d:
        checks += 1
        full_s = float(d["full (raga+quorum+live)"]["safety_mean"])
        noq_s  = float(d["minus quorum"]["safety_mean"])
        full_c = float(d["full (raga+quorum+live)"]["security_mean"])
        noq_c  = float(d["minus quorum"]["security_mean"])
        if not (noq_s > full_s and noq_c < full_c):
            problems.append(
                "E10 two-sided trade-off claim no longer holds: removing quorum "
                f"changed safety {full_s}->{noq_s} and security {full_c}->{noq_c}. "
                "E2 prose must be revised.")

# ── E11 ──────────────────────────────────────────────────────────────────
rows = read("exp11_stranger_spread.csv")
if rows:
    d = {float(r["stranger_spread_log10sd"]): r for r in rows}
    for sp, jump in PAPER_CLAIMS["E11"].items():
        if sp not in d:
            problems.append(f"E11: spread {sp} missing from data")
            continue
        check(f"E11 spread={sp} jump", jump, int(d[sp]["jump"]), tol=0)
    # STRUCTURAL: under ANY spread > 0 the loose policy must stop reaching zero.
    checks += 1
    nonzero = [d[s] for s in d if s > 0]
    if nonzero and any(int(r["safety_at_loose"]) == 0 for r in nonzero):
        problems.append(
            "E11 STRUCTURAL: with stranger spread > 0 the loose uniform policy "
            "should NOT reach zero safety failures. E11's argument that "
            "the trap deepens under spread is contradicted.")

# ── E12 ──────────────────────────────────────────────────────────────────
rows = read("exp12_phase_transition.csv")
if rows:
    d = {round(float(r["gap"]), 2): r for r in rows}
    for gap, (secnui, saf) in PAPER_CLAIMS["E12"].items():
        if gap not in d:
            problems.append(f"E12: gap {gap} missing from data")
            continue
        check(f"E12 g={gap} sec+nui",   secnui, round(float(d[gap]["sec_plus_nui"]), 1))
        check(f"E12 g={gap} safety@2%", saf,    round(float(d[gap]["safety_at_2pct"]), 1))

    # derived claims used in the abstract and contributions
    if 1.00 in d and 0.95 in d:
        ratio = float(d[1.00]["safety_at_2pct"]) / float(d[0.95]["safety_at_2pct"])
        check("E12 fragility ratio (g=1 vs g=0.95)",
              PAPER_CLAIMS["E12_fragility_g095"], round(ratio, 1), tol=0.05)
    if 0.80 in d and 0.85 in d:
        step = float(d[0.80]["safety_at_2pct"]) / float(d[0.85]["safety_at_2pct"])
        check("E12 step at crossing", PAPER_CLAIMS["E12_step_at_crossing"],
              round(step, 1), tol=0.15)

    # STRUCTURAL: cost-optimal must NOT be robustness-optimal. If this stops
    # holding, E12's central claim is false.
    checks += 1
    best = min(rows, key=lambda r: float(r["safety_at_2pct"]))
    if abs(float(best["gap"]) - 1.0) < 1e-9:
        problems.append(
            "E12 STRUCTURAL: the robustness optimum coincides with the "
            "cost-optimal g=1. E12's claim that they differ is "
            "contradicted by the data.")

rows = read("exp12_robustness_valley.csv")
if rows:
    check("E12 crossing gap", PAPER_CLAIMS["E12_crossing_gap"],
          float(rows[0]["crossing_gap"]), tol=0.005)

# ── E13 ──────────────────────────────────────────────────────────────────
rows = read("exp13_counts.csv")
if rows:
    d = {r["policy"]: r for r in rows}
    for pol, (sa, se, nu, fr) in PAPER_CLAIMS["E13"].items():
        if pol not in d:
            problems.append(f"E13: policy '{pol}' missing"); continue
        check(f"E13 {pol} safety",   sa, round(float(d[pol]["safety"]), 1), tol=0.5)
        check(f"E13 {pol} security", se, round(float(d[pol]["security"]), 1), tol=0.5)
        check(f"E13 {pol} nuisance", nu, round(float(d[pol]["nuisance"]), 1), tol=0.5)
        check(f"E13 {pol} friction", fr, round(float(d[pol]["friction"]), 1), tol=0.5)

rows = read("exp13_ranking.csv")
if rows:
    checks += 1
    if rows[0]["rankings_disagree"].strip().lower() != "true":
        problems.append(
            "E13 STRUCTURAL: unweighted and cost-weighted rankings now AGREE. "
            "E13's central claim that they disagree is contradicted.")

rows = read("exp13_quorum_semantics.csv")
if rows:
    d = {r["quorum_mode"]: r for r in rows}
    for mode, (se, fr) in PAPER_CLAIMS["E13_quorum"].items():
        if mode not in d:
            problems.append(f"E13 quorum: mode '{mode}' missing"); continue
        check(f"E13 quorum {mode} security", se, round(float(d[mode]["security"]), 1), tol=0.5)
        check(f"E13 quorum {mode} friction", fr, round(float(d[mode]["friction"]), 1), tol=0.5)
    # STRUCTURAL: conjunctive must beat noisy-OR on security AND friction
    checks += 1
    if "conjunctive" in d and "noisy_or" in d:
        cj, no = d["conjunctive"], d["noisy_or"]
        if not (float(cj["security"]) < float(no["security"]) and
                float(cj["friction"]) < float(no["friction"])):
            problems.append(
                "E13 STRUCTURAL: conjunctive quorum no longer dominates noisy-OR "
                "on both security and friction. E13's fix is contradicted.")

# ── COMPONENT SUMS: do the parts add up to the reported totals? ──────────
# The paper prints per-seed means rounded independently, so a row can differ
# from its total in the last digit. That is expected and footnoted in the
# paper; what is NOT acceptable is a total that disagrees with the sum of the
# UNROUNDED components. This checks the unrounded arithmetic.
rows = read("exp2_policy_summary.csv")
if rows:
    for r in rows:
        checks += 1
        parts = (float(r["safety_mean"]) + float(r["security_mean"])
                 + float(r["nuisance_mean"]))
        total = float(r["total_mean"])
        if abs(parts - total) > 0.05:
            problems.append(
                f"E2 SUM {r['policy']}: components sum to {parts:.2f} but total_mean "
                f"is {total:.2f} (unrounded mismatch, not a rounding artifact)")

rows = read("exp13_counts.csv")
if rows:
    for r in rows:
        checks += 1
        parts = (float(r["safety"]) + float(r["security"])
                 + float(r["nuisance"]) + float(r["friction"]))
        total = float(r["unweighted_total"])
        if abs(parts - total) > 0.05:
            problems.append(
                f"E13 SUM {r['policy']}: components sum to {parts:.2f} but "
                f"unweighted_total is {total:.2f} (unrounded mismatch)")

# ── ANALYTIC: quorum reachability at CRITICAL ────────────────────────────
# The paper once claimed a single operator (c=0.9993) plus a second attester
# (c2=0.98) clears T_perm(CRITICAL). It does not: 0.999986 < 0.9999900001.
# This guard recomputes the requirement so the claim cannot silently rot.
checks += 1
_L, _a, _f = 2000000, 200, 20
_T = _L / (_L + _f)
_c, _c2 = 0.9993, 0.98
_eff = 1 - (1 - _c) * (1 - _c2)
_required_c2 = 1 - (1 - _T) / (1 - _c)
if _eff >= _T:
    problems.append(
        f"QUORUM: c2={_c2} now clears T_perm={_T:.9f} (eff={_eff:.9f}). The paper "
        f"states it falls short; prose must be revised.")
checks += 1
if not (0.985 < _required_c2 < 0.986):
    problems.append(
        f"QUORUM: required c2 is {_required_c2:.6f}, but the paper states 0.9857. "
        f"E2 prose must be updated.")

# ── report ───────────────────────────────────────────────────────────────
print("=" * 72)
print(f"Verified {checks} numeric claims against {DATA_DIR}/")
print("=" * 72)
if problems:
    print(f"\n{len(problems)} INCONSISTENCIES FOUND:\n")
    for p in problems:
        print(f"  - {p}")
    print("\nFix the manuscript (or regenerate the data) before submitting.")
    sys.exit(1)
else:
    print("\nAll paper claims are consistent with the generated data.")
    sys.exit(0)
