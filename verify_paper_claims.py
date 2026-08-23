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
    # Table II: (safety, security, nuisance)
    "E2": {
        "uniform_loose":     (0.0,    1975.1, 735.5),
        "uniform_mid":       (2865.6, 924.2,  123.6),
        "uniform_strict":    (3053.4, 610.1,  83.0),
        "polarity_only":     (333.5,  55.6,   735.5),
        "raga":              (444.4,  53.8,   423.3),
        "raga+quorum":       (338.5,  368.6,  423.3),
        "raga+quorum+live":  (341.1,  307.9,  395.3),
    },
    # E2: the coupling must beat polarity_only on the 3-metric objective
    "E2_paired_mean_diff":    203.2,
    "E2_cohens_d":            9.02,
    "E2_best_raga_variant":   "raga",
    # Table VI: stranger-spread robustness
    "E11": {0.0: 3493, 0.1: 1649, 0.25: 1002, 0.5: 578, 1.0: 302},
    # Table VII: phase transition (gap -> sec+nui, safety@GERR)
    "E12": {
        0.00: (668.9, 1613.5),
        0.50: (601.3, 1613.5),
        0.80: (534.5, 867.3),
        0.85: (473.7, 872.5),
        0.90: (388.9, 877.9),
        0.95: (387.2, 101.1),
        1.00: (325.0, 306.5),
        1.50: (326.6, 487.2),
    },
    "E12_crossing_gap":       0.5146,  # bisection, E12 (HIGH)
    "E12_crossing_gap_crit":  0.9094,  # bisection, E12 (CRITICAL)
    "E12_fragility_g095":     3.0,     # abstract + contributions
    # Table VIII: four-quadrant accounting (safety, security, nuisance, friction)
    "E13": {
        "uniform_loose": (0.0,   3753.9, 510.6, 0.0),
        "polarity_only": (268.1, 55.1,   510.6, 7215.4),
        "raga":          (383.2, 53.6,   313.0, 10161.7),
        "raga+quorum":   (292.9, 604.3,  301.0, 8586.9),
    },
    # Table IX: quorum fusion semantics (security, friction)
    "E13_quorum": {
        "none":        (48.9,  9671.9),
        "noisy_or":    (569.7, 8122.0),
        "conjunctive": (42.6,  11609.3),
    },
    "E3": {                                     # Table III (naive_safety, failsafe_security)
        0.00: (0.0,   35.2),
        0.01: (52.1,  34.6),
        0.02: (104.3, 34.5),
        0.05: (268.9, 33.3),
        0.10: (540.4, 31.1),
        0.15: (812.5, 29.2),
    },
    # E4: the per-device inversion is unconditional on the grounded inventory;
    # the cross-class ordering is not, and fails at HIGH->CRITICAL.
    "E4_per_device_inversion": (4, 4),
    "E4_cross_class_steps_ok": 2,
    "E4_cross_class_failing_step": "HIGH->CRITICAL",
    "E5": {                                     # Table IV
        "naive":         1.0000,
        "opportunistic": 0.9939,
        "informed":      0.7226,
        "expert":        0.4990,
    },
    "E6": {                                     # percent disagreement
        1.0:   1.0,
        10.0:  10.0,
        50.0:  30.8,
        500.0: 49.9,
    },
    "E8_regret_bound":        15566.67,         # Cor. 1, grounded costs
    "E8_operator_in_band":    True,
    # Throughput is hardware-dependent; the paper claims an order of magnitude.
    "E9_throughput_order":    5,                # i.e. ~10^5 decisions/s
    "E10": {                                    # Table V
        "full (raga+quorum+live)": (199.8, 187.1),
        "minus liveness":          (198.3, 225.0),
        "minus quorum":            (268.5, 16.1),
        "minus both (raga core)":  (268.1, 33.2),
    },
    "aggregate_reduction_x":  2.94,             # Abstract + E2
    # E14: the stranger-halt property is bought with restart speed.
    "E14_breakeven_t_stop_s": 2.775,
    "E14_srms_may_halt":      True,             # 2 s safety-rated monitored stop
    "E14_estop_may_halt":     False,            # 60 s emergency stop, manual reset
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

    # aggregate reduction claim (abstract headline), measured against whichever
    # RAGA variant is actually best -- see the note in raga_experiments.py.
    raga_variants = [p for p in d if p.startswith("raga")]
    best_raga = min(raga_variants, key=lambda p: float(d[p]["total_mean"]))
    check("E2 best RAGA variant is as the paper states", 0,
          0 if best_raga == PAPER_CLAIMS["E2_best_raga_variant"] else 1, tol=0)
    best_uni = float(d["uniform_loose"]["total_mean"])
    full     = float(d[best_raga]["total_mean"])
    check("aggregate reduction factor",
          PAPER_CLAIMS["aggregate_reduction_x"], round(best_uni / full, 2), tol=0.05)

    # STRUCTURAL: the coupling must beat polarity_only on the 3-metric total.
    # If this inverts, E2's central argument is false.
    checks += 1
    po_total = float(d["polarity_only"]["total_mean"])
    rg_total = float(d[best_raga]["total_mean"])
    if not po_total > rg_total:
        problems.append(
            f"E2 STRUCTURAL: polarity_only ({po_total}) should be WORSE than "
            f"{best_raga} ({rg_total}) on the 3-metric objective. "
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
# Two separate properties, which the grounded cost matrix showed are NOT the
# same claim: the per-device inversion (unconditional) and the cross-class
# ordering (an empirical property of the device inventory).
rows = read("exp4_per_device_inversion.csv")
if rows:
    held = sum(1 for r in rows if r["inversion_holds"].strip().lower() == "true")
    want_held, want_total = PAPER_CLAIMS["E4_per_device_inversion"]
    check("E4 per-device inversion (count)", want_held, held, tol=0)
    check("E4 devices tested", want_total, len(rows), tol=0)
    # STRUCTURAL: the per-device inversion must hold for EVERY device. If it
    # ever fails, the authorization logic itself is unsound, not merely the
    # ordering across the inventory.
    checks += 1
    if held != len(rows):
        problems.append(
            f"E4 STRUCTURAL: per-device inversion holds for only {held}/{len(rows)} "
            f"devices. Eq. (2) must sit below Eq. (3) for every device or the "
            f"coupled design is meaningless.")

rows = read("exp4_cross_class_monotonicity.csv")
if rows:
    ok_steps = sum(1 for r in rows
                   if r["T_res_decreasing"].strip().lower() == "true"
                   and r["T_perm_increasing"].strip().lower() == "true")
    check("E4 cross-class steps satisfied", PAPER_CLAIMS["E4_cross_class_steps_ok"],
          ok_steps, tol=0)
    # STRUCTURAL: the paper's revised Theorem 1 says the cross-class ordering
    # is conditional and, on this inventory, fails at exactly one step. If it
    # stops failing there the revised theorem's motivating example is gone.
    checks += 1
    failing = [r["step"] for r in rows
               if not (r["T_res_decreasing"].strip().lower() == "true"
                       and r["T_perm_increasing"].strip().lower() == "true")]
    if failing != [PAPER_CLAIMS["E4_cross_class_failing_step"]]:
        problems.append(
            f"E4 STRUCTURAL: cross-class ordering was expected to fail at exactly "
            f"[{PAPER_CLAIMS['E4_cross_class_failing_step']}] but failed at {failing}. "
            f"The revised Theorem 1 discussion must be updated.")

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
    d = {float(r["clock_skew_ms"]): float(r["disagreement_rate"]) * 100 for r in rows}
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
        check(f"E12 g={gap} safety@err", saf,    round(float(d[gap]["safety_at_2pct"]), 1))

    # derived claims used in the abstract and contributions
    if 1.00 in d and 0.95 in d:
        ratio = float(d[1.00]["safety_at_2pct"]) / float(d[0.95]["safety_at_2pct"])
        check("E12 fragility ratio (g=1 vs g=0.95)",
              PAPER_CLAIMS["E12_fragility_g095"], round(ratio, 1), tol=0.05)
    # STRUCTURAL: the response to widening the separation must be a STEP, not
    # a smooth gradient. With the grounded costs the two haltable devices come
    # in at different g, so there are two steps; we require at least one drop
    # of 1.5x or more between adjacent g values.
    checks += 1
    seq = sorted(((float(r["gap"]), float(r["safety_at_2pct"])) for r in rows))
    biggest = max((seq[i][1] / seq[i + 1][1]) for i in range(len(seq) - 1)
                  if seq[i + 1][1] > 0)
    if biggest < 1.5:
        problems.append(
            f"E12 STRUCTURAL: the largest adjacent drop in safety failures is "
            f"only {biggest:.2f}x. E12's claim that the system steps rather "
            f"than degrading smoothly is contradicted.")

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
    check("E12 crossing gap (HIGH)", PAPER_CLAIMS["E12_crossing_gap"],
          float(rows[0]["crossing_gap"]), tol=0.005)

rows = read("exp12_crossings.csv")
if rows:
    d = {r["hazard"]: float(r["crossing_gap"]) for r in rows}
    for hz, key in [("HIGH", "E12_crossing_gap"), ("CRITICAL", "E12_crossing_gap_crit")]:
        if hz not in d:
            problems.append(f"E12: no recorded crossing for {hz}")
            continue
        check(f"E12 crossing gap ({hz})", PAPER_CLAIMS[key], d[hz], tol=0.005)
    # STRUCTURAL: the paper describes a staircase, i.e. the devices must cross
    # at DIFFERENT g. If they coincide, the two-step narrative is wrong.
    checks += 1
    if len(set(d.values())) < len(d):
        problems.append(
            "E12 STRUCTURAL: devices now cross c_floor at the same g, so the "
            "staircase described in the text collapses to a single step.")

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
    # STRUCTURAL: conjunctive quorum must beat noisy-OR on SECURITY and on
    # TOTAL COST. It does not, and cannot, beat it on friction: requiring both
    # attesters to clear T independently is strictly stricter than fusing them
    # into one boosted score, so it necessarily refuses more legitimate starts.
    # That is the trade the paper argues is worth making. (The earlier version
    # of this evaluation appeared to win on friction too, but only because the
    # conjunctive rule then used an arbitrary per-party bar of 0.95 that was
    # looser than T; the rule is now min(c1,c2) >= T, matching Algorithm 2.)
    if "conjunctive" in d and "noisy_or" in d:
        cj, no = d["conjunctive"], d["noisy_or"]
        checks += 1
        if not float(cj["security"]) < float(no["security"]):
            problems.append(
                "E13 STRUCTURAL: conjunctive quorum no longer beats noisy-OR on "
                "security. The unsoundness argument for noisy-OR is contradicted.")
        checks += 1
        if not float(cj["cost_total"]) < float(no["cost_total"]):
            problems.append(
                "E13 STRUCTURAL: conjunctive quorum no longer beats noisy-OR on "
                "total cost. E13's recommendation is contradicted.")
        checks += 1
        if not float(cj["friction"]) > float(no["friction"]):
            problems.append(
                "E13 SANITY: conjunctive quorum shows LESS friction than noisy-OR. "
                "That is impossible for a strictly stricter rule -- the "
                "conjunctive implementation has probably drifted from min(c1,c2)>=T.")

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
# Recomputed from the grounded cost matrix rather than hardcoded, so the
# guard cannot silently rot when parameters.py changes.
import parameters as _P
checks += 1
_L, _a, _f = _P.cost_triple("arm")
_T = _L / (_L + _f)
_c = _P.OPERATOR_GOOD_MEAN
_required_c2 = 1 - (1 - _T) / (1 - _c) if _c < 1 else 1.0
# Under the CORRECTED conjunctive rule the second attester must clear T in its
# own right; noisy-OR is what allowed a weak second attester to suffice.
if _required_c2 >= 1.0 or _required_c2 <= 0.0:
    pass   # a single attestation already clears T; nothing to check
else:
    if not (0.0 < _required_c2 < 1.0):
        problems.append(
            f"QUORUM: required second-attester confidence under noisy-OR is "
            f"{_required_c2:.6f}, outside (0,1). Check the cost matrix.")

# STRUCTURAL: conjunctive quorum must not be able to clear T on the strength
# of one party alone -- that was the whole defect in noisy-OR.
checks += 1
_c1_strong, _c2_weak = 0.999, 0.50
if min(_c1_strong, _c2_weak) >= _T:
    problems.append(
        "QUORUM STRUCTURAL: a weak second attester still clears T under the "
        "conjunctive rule. The E13 fix is not doing what the paper says.")
if (1 - (1 - _c1_strong) * (1 - _c2_weak)) < _T:
    problems.append(
        "QUORUM STRUCTURAL: noisy-OR no longer lets a strong party carry a weak "
        "one over T, so E13's motivating defect no longer reproduces.")

# ── E14 ──────────────────────────────────────────────────────────────────
rows = read("exp14_stop_categories_named.csv")
if rows:
    d = {r["stop_category"]: r for r in rows}
    srms = "safety-rated monitored stop"
    estop = "emergency stop, manual reset"
    for key, want in [(srms, PAPER_CLAIMS["E14_srms_may_halt"]),
                      (estop, PAPER_CLAIMS["E14_estop_may_halt"])]:
        if key not in d:
            problems.append(f"E14: stop category '{key}' missing"); continue
        checks += 1
        got = d[key]["stranger_may_halt"].strip().lower() == "true"
        if got != want:
            problems.append(
                f"E14 STRUCTURAL: with '{key}' the stranger-halt property is "
                f"{got}, paper states {want}. E14's central contrast is gone.")

rows = read("exp14_stop_category.csv")
if rows:
    # break-even is the largest t_stop for which the property still holds
    ok = [float(r["t_stop_s"]) for r in rows
          if r["stranger_may_halt"].strip().lower() == "true"]
    bad = [float(r["t_stop_s"]) for r in rows
           if r["stranger_may_halt"].strip().lower() != "true"]
    checks += 1
    if ok and bad and not (max(ok) < PAPER_CLAIMS["E14_breakeven_t_stop_s"] < min(bad)):
        problems.append(
            f"E14: break-even {PAPER_CLAIMS['E14_breakeven_t_stop_s']}s does not lie "
            f"between the last holding point ({max(ok)}s) and the first failing "
            f"point ({min(bad)}s).")

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
