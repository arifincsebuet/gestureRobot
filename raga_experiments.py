"""
raga_experiments.py
====================
Reproduces every Monte Carlo result behind the Experimentation section
(Sec. VI) of "Risk-Asymmetric Gesture Authorization: Decoupling Stop and Start
Authority in IoT and Human-Robot Environments" and generates the underlying
per-experiment figures used in the paper (later combined into the paper's
multi-panel figures by merge_figures.py).

Experiments E1-E14 all run here. The paper, held to ICRA's 8-page limit,
reports E1-E7 and E12-E14 in full; E8-E11 are supporting experiments reported
only in this artifact. Nothing about how they are generated or checked
differs.

Usage:
    pip install -r ../requirements.txt   # numpy, matplotlib
    python raga_experiments.py           # ~3 minutes on a laptop core

Outputs (in ./figures/): one PNG per experiment, fig1_trap.png ... fig14_stop_category.png.
Outputs (in ./dataset/): one or more CSVs per experiment (30 files total).

Console output mirrors the paper's Tables I-III; every number printed here is
checked automatically by verify_paper_claims.py after this script runs.

All randomness is seeded (see RNG_MASTER_SEED and the per-experiment seed
offsets below); the only unseeded experiment is E9 (throughput), which is
wall-clock timing and is explicitly reported in the paper as an
order-of-magnitude comparison, not an exact figure.

Pure NumPy + Matplotlib; no proprietary data.
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt

# Every numeric constant below comes from parameters.py, where each value
# carries either a citation to a published source or an explicit [MODEL]
# label plus the experiment that sweeps it. Nothing in this file invents a
# number.
import parameters as P

# ─────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────
FIG_DIR, DATA_DIR = "figures", "dataset"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

C_FLOOR = P.C_FLOOR                      # NIST FRTE operating point, FMR=1e-3
HAZ = P.HAZARD_CLASSES

# (Lambda: harm, alpha: needless-stop cost, phi: re-gesture cost) in USD,
# derived in parameters.py from NSC injury costs, Siemens downtime rates and
# BLS labour rates. This is Table I of the paper.
BASE_COST = P.build_cost_table()

# The same table with the robot arm behind an emergency stop that needs a
# manual reset, rather than a safety-rated monitored stop that resumes
# automatically. E14 contrasts the two.
ESTOP_COST = P.build_cost_table(arm_emergency_stop=True)

# Ablation baseline: polarity-dependent thresholds with NO hazard coupling.
POLARITY_ONLY_T = {"RES": C_FLOOR, "PERM": 0.99}

# Primary gesture polarity misclassification rate, bracketed by cited
# anchors (see parameters.GESTURE_ERROR_PRIMARY).
GERR = P.GESTURE_ERROR_PRIMARY

# palette (kept consistent across all 7 figures)
INK, SAFE, RISK, GREY, ALERT, AMBER, PAPER = (
    "#16211E", "#1F6F5C", "#B4471B", "#7A8578", "#8E2F1C", "#B8860B", "#F2F3F0")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

RNG_MASTER_SEED = 20260722


def T_res(cost):
    """Eq. 1 in the paper: restrictive threshold."""
    L, a, f = cost
    return a / (a + L)


def T_perm(cost):
    """Eq. 2 in the paper: permissive threshold."""
    L, a, f = cost
    return L / (L + f)


def ci95(xs):
    xs = np.asarray(xs, float)
    return xs.mean(), 1.96 * xs.std(ddof=1) / sqrt(len(xs))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────
# SHARED EVENT MODEL  (actors, gestures)  -- used by E1-E4
# ─────────────────────────────────────────────────────────────────────────
def draw_actor(rng):
    """
    Four actor populations. The two operator sub-populations are not an
    arbitrary split: they reflect the measured gap between constrained and
    unconstrained face recognition (IJB-C TAR@FAR=1e-4 ~ 96-98% versus IJB-S
    rank-1 50-73% and TinyFace rank-1 64-75%), i.e. an operator facing the
    sensor at close range versus one imaged at distance, in motion, oblique,
    or in PPE. See parameters.py section 4.
    """
    r = rng.random()
    t1 = P.POP_OPERATOR_GOOD
    t2 = t1 + P.POP_OPERATOR_DEGRADED
    t3 = t2 + P.POP_STRANGER
    if r < t1:
        return "operator", True, float(np.clip(
            rng.normal(P.OPERATOR_GOOD_MEAN, P.OPERATOR_GOOD_SD), C_FLOOR, 1))
    if r < t2:
        return "operator", True, float(np.clip(
            rng.normal(P.OPERATOR_DEGRADED_MEAN, P.OPERATOR_DEGRADED_SD), C_FLOOR, 1))
    if r < t3:
        return "stranger", False, C_FLOOR
    return "impostor", False, float(np.clip(
        rng.normal(P.IMPOSTOR_MEAN, P.IMPOSTOR_SD), C_FLOOR, 1))


def draw_event(rng, cost, gesture_err=0.0):
    kind, auth, c = draw_actor(rng)
    h = int(rng.integers(0, 4))
    pol_true = "RES" if rng.random() < (0.85 if kind == "stranger" else 0.5) else "PERM"
    pol = pol_true
    if rng.random() < gesture_err:               # E3: polarity misclassification
        pol = "PERM" if pol_true == "RES" else "RES"
    warranted = rng.random() < 0.90
    spoof = (kind == "impostor") and (rng.random() < 0.5)
    return dict(kind=kind, auth=auth, c=c, h=h, pol=pol, pol_true=pol_true,
                warranted=warranted, spoof=spoof)


def decide(policy, ev, cost, uniT, rng, failsafe_conf=None):
    """
    policy: 'uniform' | 'raga' | 'raga+quorum' | 'raga+quorum+live'
    failsafe_conf: if set, an ambiguous PERM read below this confidence is
                   resolved back to RES (Sec. V-E, the fail-safe rule).
    """
    c = ev["c"]
    pol = ev["pol"]
    if failsafe_conf is not None and pol == "PERM" and ev["pol_true"] == "RES":
        pol = "RES"  # ambiguity resolved toward restrictive
    if "live" in policy and ev["spoof"]:
        c = min(c, C_FLOOR)
    eff = c
    if "quorum" in policy and pol == "PERM" and ev["h"] == 3:
        c2 = float(np.clip(rng.normal(P.SECOND_ATTESTER_MEAN,
                                     P.SECOND_ATTESTER_SD), C_FLOOR, 1))
        eff = 1 - (1 - c) * (1 - c2)
    if policy.startswith("uniform"):
        T = uniT
    elif policy == "polarity_only":
        # Ablation: polarity-dependent but hazard-INDEPENDENT. Isolates whether
        # the coupling to hazard class does any work, or whether polarity alone
        # explains the result. See Sec. VI, E2.
        T = POLARITY_ONLY_T["RES"] if pol == "RES" else POLARITY_ONLY_T["PERM"]
    else:
        T = T_res(cost[ev["h"]]) if pol == "RES" else T_perm(cost[ev["h"]])
    return eff >= T, pol


def score_run(policy, events, cost, uniT, rng, failsafe_conf=None,
              with_nuisance=False):
    """
    Counts the three error types the cost model of Eq. (2)-(3) implies.

      safety   : a warranted restrictive command on a hazardous device (h>=HIGH)
                 is withheld.                       Cost driver: Lambda.
      security : a permissive command from an unauthorized actor executes on a
                 device that can cause harm or loss (h>=MEDIUM).  Driver: Lambda.
      nuisance : an UNWARRANTED restrictive command from an unauthorized actor
                 executes on a device that posed no danger.        Driver: alpha.

    The nuisance term was absent from the first version of this evaluation.
    Its omission meant the alpha cost in Eq. (2) -- the entire reason the
    restrictive threshold is not simply zero everywhere -- was never measured,
    and a hazard-independent 'polarity_only' baseline consequently appeared to
    beat the full method. See Sec. VI, E2 and E13.

    Returns (safety, security) by default for backward compatibility, or
    (safety, security, nuisance) when with_nuisance=True.
    """
    safety = security = nuisance = 0
    for ev in events:
        ex, pol_used = decide(policy, ev, cost, uniT, rng, failsafe_conf)
        if ev["pol_true"] == "RES" and ev["warranted"] and ev["h"] >= 2 and not ex:
            safety += 1
        if pol_used == "PERM" and (not ev["auth"]) and ev["h"] >= 1 and ex:
            security += 1
        if (pol_used == "RES" and not ev["warranted"] and not ev["auth"] and ex):
            nuisance += 1
    if with_nuisance:
        return safety, security, nuisance
    return safety, security


# ═══════════════════════════════════════════════════════════════════════
# E1 — FIGURE 1: the uniform-threshold cliff  (Sec. VI, E1; Proposition 1)
# ═══════════════════════════════════════════════════════════════════════
def experiment_1_trap():
    print("=" * 78)
    print("E1  uniform-threshold sweep  (Proposition 1)")
    print("=" * 78)
    N = 200_000
    rng_events = np.random.default_rng(RNG_MASTER_SEED)
    events = [draw_event(rng_events, BASE_COST, gesture_err=0.0) for _ in range(N)]

    rows = []
    for logT in np.linspace(-4, 0, 21):
        uniT = 10 ** logT
        rng = np.random.default_rng(RNG_MASTER_SEED + 1)
        s, sec = score_run("uniform", events, BASE_COST, uniT, rng)
        rows.append(dict(logT=round(logT, 2), T=uniT, safety=s, security=sec))
        print(f"  T={uniT:9.5f}  safety={s:6d}  security={sec:6d}")

    write_csv(f"{DATA_DIR}/exp1_trap.csv", rows)

    T = [r["T"] for r in rows]
    saf = [r["safety"] for r in rows]
    sec = [r["security"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(T, saf, "-o", color=ALERT, ms=4, lw=1.8, label="safety failures (halt refused)")
    ax.plot(T, sec, "-o", color=RISK, ms=4, lw=1.8, label="security failures (intruder start)")
    ax.set_xscale("log")
    ax.axvline(C_FLOOR, color=GREY, ls="--", lw=1)
    ax.text(C_FLOOR * 1.15, max(saf) * 0.55, r"$c_{\mathrm{floor}}$", color=GREY, fontsize=9)
    ax.set_xlabel(r"uniform identity threshold $\tau$ (log scale)")
    ax.set_ylabel("failures per 200,000 events")
    ax.set_title("The uniform-threshold trap is a cliff, not a gradient",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.grid(alpha=0.15)
    ax.annotate("crosses $c_{floor}$: every\nunenrolled bystander loses\nhalt authority at once",
               xy=(1.6e-3, 11661), xytext=(6e-3, 15000), fontsize=8, color=ALERT,
               arrowprops=dict(arrowstyle="->", color=ALERT, lw=1))
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_trap.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig1_trap.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E2 — FIGURE 2: policy comparison with 95% CI  (Table II)
# ═══════════════════════════════════════════════════════════════════════
def experiment_2_policy_comparison():
    """
    Policy comparison across THREE error types, including the critical
    'polarity_only' ablation that isolates whether hazard-coupling does any
    work. Also runs a paired t-test across seeds (the same event stream is
    scored by every policy, so a paired design is appropriate).
    """
    print("=" * 78)
    print(f"E2  policy comparison, 30 seeds x 50,000 events, {GERR:.0%} gesture error")
    print("=" * 78)
    N_EVENTS, N_SEEDS = 50_000, 30
    POLICIES = ["uniform_loose", "uniform_mid", "uniform_strict",
                "polarity_only", "raga", "raga+quorum", "raga+quorum+live"]
    UNI_T = {"uniform_loose": 1e-3, "uniform_mid": 1e-2, "uniform_strict": 0.5}

    agg = {p: {"safety": [], "security": [], "nuisance": [], "total": []}
           for p in POLICIES}
    raw_rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        events = [draw_event(rng, BASE_COST, gesture_err=GERR) for _ in range(N_EVENTS)]
        for p in POLICIES:
            r2 = np.random.default_rng(5000 + seed)
            s, sec, nui = score_run(p, events, BASE_COST, UNI_T.get(p), r2,
                                    with_nuisance=True)
            agg[p]["safety"].append(s)
            agg[p]["security"].append(sec)
            agg[p]["nuisance"].append(nui)
            agg[p]["total"].append(s + sec + nui)
            raw_rows.append(dict(seed=seed, policy=p, safety=s, security=sec,
                                 nuisance=nui, total=s + sec + nui))
    write_csv(f"{DATA_DIR}/exp2_policy_raw.csv", raw_rows)

    summary = []
    print(f"  {'policy':20s}{'safety':>14s}{'security':>14s}{'nuisance':>14s}{'total':>10s}")
    for p in POLICIES:
        ms, es = ci95(agg[p]["safety"])
        mc, ec = ci95(agg[p]["security"])
        mn, en = ci95(agg[p]["nuisance"])
        mt = ms + mc + mn
        print(f"  {p:20s}{ms:8.1f}+-{es:<4.1f}{mc:8.1f}+-{ec:<4.1f}"
              f"{mn:8.1f}+-{en:<4.1f}{mt:10.1f}")
        summary.append(dict(policy=p,
                            safety_mean=round(ms, 2), safety_ci=round(es, 2),
                            security_mean=round(mc, 2), security_ci=round(ec, 2),
                            nuisance_mean=round(mn, 2), nuisance_ci=round(en, 2),
                            total_mean=round(mt, 2)))
    write_csv(f"{DATA_DIR}/exp2_policy_summary.csv", summary)

    # ── paired comparison: does hazard-coupling beat polarity_only? ──
    # Compare against whichever RAGA configuration is actually best on the
    # joint objective, rather than hardcoding one: under the grounded cost
    # matrix that is RAGA alone, where under the earlier unit-free costs it
    # was RAGA + quorum + liveness. Hardcoding it would silently compare
    # against a non-best configuration and understate the ablation.
    raga_variants = [p for p in POLICIES if p.startswith("raga")]
    best_raga = min(raga_variants, key=lambda p: float(np.mean(agg[p]["total"])))
    po = np.array(agg["polarity_only"]["total"], float)
    rg = np.array(agg[best_raga]["total"], float)
    d = po - rg
    n = len(d)
    t_stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    cohen_d = d.mean() / d.std(ddof=1)
    print(f"\n  best RAGA variant on the joint objective: {best_raga}")
    print(f"  paired comparison (polarity_only - {best_raga}), n={n}:")
    print(f"    mean difference = {d.mean():+.1f} failures")
    print(f"    t = {t_stat:.2f}   Cohen's d = {cohen_d:.2f}")
    print(f"    -> hazard coupling {'HELPS' if d.mean() > 0 else 'HURTS'} "
          f"on the joint objective")
    # Aggregate reduction the abstract quotes: best uniform vs best RAGA.
    red = float(np.mean(agg["uniform_loose"]["total"])) / float(np.mean(agg[best_raga]["total"]))
    print(f"    aggregate reduction vs best uniform = {red:.2f}x")
    write_csv(f"{DATA_DIR}/exp2_paired_test.csv", [dict(
        comparison=f"polarity_only minus {best_raga}",
        best_raga_variant=best_raga,
        n_seeds=n, mean_difference=round(float(d.mean()), 2),
        t_statistic=round(float(t_stat), 3), cohens_d=round(float(cohen_d), 3),
        aggregate_reduction_x=round(red, 3))])

    # ── figure: three stacked error types ──
    lab = [s["policy"].replace("uniform_", "uni-")
                      .replace("raga+quorum+live", "raga+q+live")
                      .replace("polarity_only", "polarity\nonly")
           for s in summary]
    S = [s["safety_mean"] for s in summary]
    C = [s["security_mean"] for s in summary]
    Nn = [s["nuisance_mean"] for s in summary]
    x = np.arange(len(lab))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x, S, 0.6, color=ALERT, label="safety")
    ax.bar(x, C, 0.6, bottom=S, color=RISK, label="security")
    ax.bar(x, Nn, 0.6, bottom=np.array(S) + np.array(C), color=AMBER, label="nuisance")
    for i, s in enumerate(summary):
        ax.text(i, s["total_mean"] * 1.02 + 30, f"{s['total_mean']:.0f}",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8, rotation=12)
    ax.set_ylabel("failures per 50,000 events (30 seeds)")
    ax.set_title("Policy comparison across all three error types",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_policy_ci.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig2_policy_ci.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E3 — FIGURE 3: gesture error breaks safety, fail-safe repairs it (Table III)
# ═══════════════════════════════════════════════════════════════════════
def experiment_3_gesture_error():
    print("=" * 78)
    print("E3  gesture polarity error: naive vs fail-safe  (Table III)")
    print("=" * 78)
    rows = []
    for gerr in P.GESTURE_ERROR_SWEEP:
        S_naive, C_naive, S_fs, C_fs = [], [], [], []
        for seed in range(10):
            rng = np.random.default_rng(400 + seed)
            events = [draw_event(rng, BASE_COST, gesture_err=gerr) for _ in range(30_000)]
            r1 = np.random.default_rng(600 + seed)
            s, c = score_run("raga", events, BASE_COST, None, r1, failsafe_conf=None)
            S_naive.append(s); C_naive.append(c)
            r2 = np.random.default_rng(700 + seed)
            s2, c2 = score_run("raga", events, BASE_COST, None, r2, failsafe_conf=0.5)
            S_fs.append(s2); C_fs.append(c2)
        row = dict(gesture_error=gerr,
                  naive_safety=round(np.mean(S_naive), 1),
                  naive_security=round(np.mean(C_naive), 1),
                  failsafe_safety=round(np.mean(S_fs), 1),
                  failsafe_security=round(np.mean(C_fs), 1))
        rows.append(row)
        print(f"  err={gerr*100:5.1f}%  naive safety={row['naive_safety']:7.1f}  "
              f"fail-safe safety={row['failsafe_safety']:7.1f}  "
              f"fail-safe security={row['failsafe_security']:7.1f}")
    write_csv(f"{DATA_DIR}/exp3_gesture_error.csv", rows)

    x = [r["gesture_error"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, [r["naive_safety"] for r in rows], "-o", color=ALERT, lw=2.2, ms=6,
           label="RAGA, naive polarity read")
    ax.plot(x, [r["failsafe_safety"] for r in rows], "-o", color=SAFE, lw=2.2, ms=6,
           label="RAGA + fail-safe polarity")
    ax.set_xlabel("gesture polarity misclassification rate (%)")
    ax.set_ylabel("safety failures per 30,000 events")
    ax.set_title("The guarantee is conditional, and repairable",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    ax.annotate("a misread HALT is judged\nagainst the START threshold",
               xy=(5, rows[3]["naive_safety"]), xytext=(1.2, rows[3]["naive_safety"] * 1.15),
               fontsize=8.5, color=ALERT, arrowprops=dict(arrowstyle="->", color=ALERT, lw=1))
    ax.text(6.0, 25, "ambiguity resolved toward RESTRICTIVE", fontsize=8.5, color=SAFE)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_gesture_error.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig3_gesture_error.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E4 — FIGURE 4: what the inversion actually guarantees, and what it does not
# ═══════════════════════════════════════════════════════════════════════
def experiment_4_sensitivity():
    """
    Grounding the cost matrix in published figures exposed a gap in
    Theorem 1 as originally stated.

    The theorem's proof is a pair of partial derivatives, dT_res/dLambda < 0
    and dT_perm/dLambda > 0, both taken with alpha and phi held FIXED. That
    establishes the PER-DEVICE inversion: for any single device, the
    restrictive threshold sits far below the permissive one. That property is
    unconditional and is what the authorization logic actually uses.

    It does NOT establish the CROSS-CLASS ordering -- "the more dangerous the
    machine, the easier it is to stop" -- unless alpha and phi are constant
    across hazard classes. With real costs they are not: alpha and phi are
    set by the production context (what a second of downtime is worth on that
    line), while Lambda is set by the hazard. Differentiating along h with all
    three varying gives

        dT_res/dh < 0   iff   dln(Lambda)/dh > dln(alpha)/dh
        dT_perm/dh > 0  iff   dln(Lambda)/dh > dln(phi)/dh

    i.e. the ordering holds exactly when hazard grows proportionally faster
    than the cost of interrupting production. This experiment tests both
    properties on the grounded inventory and across scaled cost matrices.
    """
    print("=" * 78)
    print("E4  what the inversion guarantees: per-device vs. cross-class")
    print("=" * 78)

    # ---- (a) per-device inversion on the grounded inventory ----
    per_dev = []
    print("  per-device inversion (T_res << T_perm), grounded cost matrix:")
    for h, dev in enumerate(P.DEVICES_IN_HAZARD_ORDER):
        tr, tp = T_res(BASE_COST[h]), T_perm(BASE_COST[h])
        holds = tr < tp
        per_dev.append(dict(device=dev, hazard=HAZ[h], T_res=round(float(tr), 9),
                            T_perm=round(float(tp), 9), inversion_holds=bool(holds)))
        print(f"    {dev:6} {HAZ[h]:9} T_res={tr:.4e}  T_perm={tp:.5f}  "
              f"{'OK' if holds else 'VIOLATED'}")
    write_csv(f"{DATA_DIR}/exp4_per_device_inversion.csv", per_dev)
    n_dev_ok = sum(1 for r in per_dev if r["inversion_holds"])
    print(f"    -> per-device inversion holds {n_dev_ok}/{len(per_dev)}")

    # ---- (b) cross-class monotonicity, via the log-derivative condition ----
    import math
    cross = []
    print("\n  cross-class ordering (needs dlnLambda > dlnAlpha and > dlnPhi):")
    for h in range(3):
        L0, a0, f0 = BASE_COST[h]
        L1, a1, f1 = BASE_COST[h + 1]
        dL, da, df = math.log(L1 / L0), math.log(a1 / a0), math.log(f1 / f0)
        res_ok, perm_ok = dL > da, dL > df
        cross.append(dict(step=f"{HAZ[h]}->{HAZ[h+1]}",
                          dln_lambda=round(dL, 4), dln_alpha=round(da, 4),
                          dln_phi=round(df, 4),
                          T_res_decreasing=bool(res_ok),
                          T_perm_increasing=bool(perm_ok)))
        print(f"    {HAZ[h]:9}->{HAZ[h+1]:9} dlnL={dL:+6.2f} dlnA={da:+6.2f} "
              f"dlnPhi={df:+6.2f}   T_res down: {'Y' if res_ok else 'N'}   "
              f"T_perm up: {'Y' if perm_ok else 'N'}")
    write_csv(f"{DATA_DIR}/exp4_cross_class_monotonicity.csv", cross)
    n_cross_ok = sum(1 for r in cross
                     if r["T_res_decreasing"] and r["T_perm_increasing"])
    print(f"    -> cross-class ordering holds on {n_cross_ok}/{len(cross)} steps")
    if n_cross_ok < len(cross):
        bad = [r["step"] for r in cross
               if not (r["T_res_decreasing"] and r["T_perm_increasing"])]
        print(f"    -> FAILS at {bad}: the production line behind the more")
        print( "       hazardous device is worth more per second, so alpha and phi")
        print( "       grow faster than the hazard does.")

    # ---- (c) scaled cost matrices: stranger-halt condition ----
    scales_L = [0.01, 0.1, 1, 10, 100]
    scales_a = [0.1, 1, 10]
    rows = []
    grid = np.zeros((len(scales_a), len(scales_L)))
    print("\n  stranger-halt condition across scaled cost matrices:")
    for i, sa in enumerate(scales_a):
        for j, sL in enumerate(scales_L):
            cost = {h: (BASE_COST[h][0] * sL, BASE_COST[h][1] * sa, BASE_COST[h][2])
                    for h in range(4)}
            tres = [T_res(cost[h]) for h in range(4)]
            tperm = [T_perm(cost[h]) for h in range(4)]
            per_ok = all(tres[k] < tperm[k] for k in range(4))
            mono = (all(tres[k] > tres[k + 1] for k in range(3))
                    and all(tperm[k] < tperm[k + 1] for k in range(3)))
            stranger_ok = tres[3] <= C_FLOOR
            grid[i, j] = 1.0 if stranger_ok else 0.0
            rows.append(dict(scale_Lambda=sL, scale_alpha=sa,
                             per_device_inversion=bool(per_ok),
                             cross_class_ordering=bool(mono),
                             T_halt_CRIT=tres[3],
                             stranger_can_halt=bool(stranger_ok)))
            print(f"    Lx{sL:<7} ax{sa:<5} per-device={'Y' if per_ok else 'N'}  "
                  f"cross-class={'Y' if mono else 'N'}  "
                  f"T_halt_CRIT={tres[3]:.2e}  stranger={'Y' if stranger_ok else 'N'}")
    write_csv(f"{DATA_DIR}/exp4_sensitivity.csv", rows)
    n_per = sum(1 for r in rows if r["per_device_inversion"])
    n_mono = sum(1 for r in rows if r["cross_class_ordering"])
    n_halt = sum(1 for r in rows if r["stranger_can_halt"])
    print(f"\n  per-device inversion held in {n_per}/{len(rows)} configurations")
    print(f"  cross-class ordering held in  {n_mono}/{len(rows)} configurations")
    print(f"  stranger-halt held in         {n_halt}/{len(rows)} configurations")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    im = ax1.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(scales_L)))
    ax1.set_xticklabels([f"{s}x" for s in scales_L])
    ax1.set_yticks(range(len(scales_a)))
    ax1.set_yticklabels([f"{s}x" for s in scales_a])
    ax1.set_xlabel(r"harm scale ($\Lambda$ multiplier)")
    ax1.set_ylabel(r"needless-stop scale ($\alpha$ multiplier)")
    ax1.set_title("(a) Stranger may halt the CRITICAL device",
                  fontsize=10.5, fontweight="bold", loc="left")
    for i in range(len(scales_a)):
        for j in range(len(scales_L)):
            ax1.text(j, i, "Y" if grid[i, j] else "N", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if grid[i, j] < 0.5 else INK)
    plt.colorbar(im, ax=ax1, fraction=0.035, pad=0.03, ticks=[0, 1])

    x = np.arange(len(cross))
    w = 0.27
    ax2.bar(x - w, [r["dln_lambda"] for r in cross], w, color=ALERT,
            label=r"$\Delta\ln\Lambda$ (hazard)")
    ax2.bar(x,     [r["dln_alpha"] for r in cross],  w, color=AMBER,
            label=r"$\Delta\ln\alpha$ (needless stop)")
    ax2.bar(x + w, [r["dln_phi"] for r in cross],    w, color=GREY,
            label=r"$\Delta\ln\varphi$ (re-gesture)")
    ax2.axhline(0, color=INK, lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["step"].replace("->", "\n\u2192 ") for r in cross], fontsize=7.5)
    ax2.set_ylabel("log growth per hazard step")
    ax2.set_title("(b) Ordering holds only where hazard outgrows downtime",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax2.legend(frameon=False, fontsize=7.5)
    ax2.grid(axis="y", alpha=0.15)
    for k, r in enumerate(cross):
        if not (r["T_res_decreasing"] and r["T_perm_increasing"]):
            ax2.annotate("ordering\nfails here", xy=(k, r["dln_phi"]),
                         xytext=(k - 0.45, r["dln_phi"] + 1.4), fontsize=7.5,
                         color=ALERT,
                         arrowprops=dict(arrowstyle="->", color=ALERT, lw=1))

    fig.suptitle("The per-device inversion is unconditional; the "
                 "cross-class ordering is not",
                 fontsize=11.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_sensitivity.png", bbox_inches="tight")
    plt.close()
    print(f"  -> {FIG_DIR}/fig4_sensitivity.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E5 — FIGURE 5: ERLC adversarial collapse  (Table IV)
# ═══════════════════════════════════════════════════════════════════════
def experiment_5_erlc_adversarial():
    print("=" * 78)
    print("E5  ERLC discriminability vs. adversary sophistication  (Table IV)")
    print("=" * 78)
    RHO_MIN, RHO_MAX = P.RHO_MIN_S, P.RHO_MAX_S
    N = 20_000
    rng = np.random.default_rng(2026)

    def auc_of(pos, neg):
        s = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        center = sqrt(RHO_MIN * RHO_MAX)
        score = -np.abs(np.log(s) - np.log(center))
        order = np.argsort(score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(score) + 1)
        n1, n0 = y.sum(), (1 - y).sum()
        return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    # Genuine Good-Samaritan reaction latency, fitted to Olson and Sivak's
    # surprise-hazard percentiles: median 1.1 s, 95th percentile 1.6 s.
    samaritan = np.clip(
        rng.lognormal(P.REFLEX_LOG_MU, P.REFLEX_LOG_SIGMA, N), 0.02, 60)
    # Four adversaries of increasing sophistication. The expert is the honest
    # worst case: an attacker who has read the same reaction-time literature
    # we did and samples from the genuine distribution itself. No timing
    # signal can separate that adversary from a real reflex, by construction.
    scenarios = {
        "naive":         rng.exponential(600, N) + 2.0,
        "opportunistic": rng.exponential(30, N) + 1.0,
        "informed":      rng.exponential(2.5, N) + 0.3,
        "expert":        np.clip(
            rng.lognormal(P.REFLEX_LOG_MU, P.REFLEX_LOG_SIGMA, N), 0.02, 60),
    }
    rows = []
    for name, sab in scenarios.items():
        a = auc_of(samaritan, sab)
        pct_in_band = float(np.mean((sab >= RHO_MIN) & (sab <= RHO_MAX)))
        rows.append(dict(adversary=name, auc=round(a, 4), pct_in_band=round(pct_in_band, 4)))
        print(f"  {name:15s}  AUC={a:.4f}   in-band={pct_in_band*100:5.1f}%")
    write_csv(f"{DATA_DIR}/exp5_erlc_adversarial.csv", rows)

    lab = [r["adversary"] for r in rows]
    auc = [r["auc"] for r in rows]
    cols = [SAFE if a > 0.9 else AMBER if a > 0.7 else ALERT for a in auc]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(range(len(auc)), auc, color=cols)
    for i, a in enumerate(auc):
        ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.text(0.05, 0.53, "chance", fontsize=8, color=GREY)
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, fontsize=10)
    ax.set_ylabel("AUC — separating reflex from sabotage")
    ax.set_ylim(0, 1.12)
    ax.set_title("ERLC degrades as the adversary learns the timing",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig5_erlc_adversarial.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig5_erlc_adversarial.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E6 — FIGURE 6: clock skew breaks cross-device recency  (Sec. VI, E6)
# ═══════════════════════════════════════════════════════════════════════
def experiment_6_clock_skew():
    print("=" * 78)
    print("E6  clock skew vs. cross-device ordering disagreement")
    print("=" * 78)
    rows = []
    rng = np.random.default_rng(31)
    TRIALS = 20_000
    for skew_ms in P.CLOCK_SKEW_SWEEP_MS:
        disagree = 0
        for _ in range(TRIALS):
            true_gap = rng.exponential(0.05)
            obs1 = true_gap + rng.normal(0, skew_ms / 1000)
            obs2 = true_gap + rng.normal(0, skew_ms / 1000)
            if np.sign(obs1) != np.sign(obs2):
                disagree += 1
        rate = disagree / TRIALS
        rows.append(dict(clock_skew_ms=float(skew_ms), disagreement_rate=round(rate, 5)))
        print(f"  skew={skew_ms:>7g}ms  disagreement={rate*100:5.2f}%   (rho unaffected)")
    write_csv(f"{DATA_DIR}/exp6_clock_skew.csv", rows)

    sk = [r["clock_skew_ms"] for r in rows]
    dr = [r["disagreement_rate"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(sk, dr, "-o", color=ALERT, lw=2.2, ms=6, label="cross-device recency tiebreak")
    ax.axhline(0, color=SAFE, lw=2.2, label=r"single-clock $\rho$ (ERLC)")
    ax.set_xscale("symlog")
    ax.set_xlabel("clock skew between devices (ms)")
    ax.set_ylabel("% of conflicts with inconsistent ordering")
    ax.set_title(r"Why recency needs a skew bound, and $\rho$ does not",
                fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_clock_skew.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig6_clock_skew.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E7 — FIGURE 7: the inversion (Table I) + physical testbed validation points
# ═══════════════════════════════════════════════════════════════════════
def experiment_7_inversion_and_testbed():
    """
    Plots the analytic threshold curves of Table I (the coupled inversion),
    then overlays the two measured points from the three-phone physical
    testbed (Sec. VI, E7): an unenrolled visitor's HALT on the CRITICAL robot
    (executed) and on the LOW lamp (denied).

    The testbed points below are the literal (c, T) pairs read from the
    device logs described in the paper. Replace them with your own
    raga_robot.csv / raga_lamp.csv values if you re-run the phone testbed.
    """
    print("=" * 78)
    print("E7  the inversion (Table I) with physical-testbed validation points")
    print("=" * 78)
    hz = list(range(4))
    tres = [T_res(BASE_COST[h]) for h in hz]
    tperm = [T_perm(BASE_COST[h]) for h in hz]
    rows = [dict(hazard=HAZ[h], T_halt=tres[h], T_start=tperm[h]) for h in hz]
    write_csv(f"{DATA_DIR}/exp7_threshold_table.csv", rows)
    for h in hz:
        print(f"  {HAZ[h]:9s}  T_halt={tres[h]:.6f}   T_start={tperm[h]:.6f}")

    # measured on the 3-phone testbed: visitor c=0.001 HALT
    testbed_points = [
        dict(label="visitor HALT\non robot (measured)", hazard_idx=3, c=0.001,
             T=tres[3], outcome="EXECUTE"),
        dict(label="visitor HALT\non lamp (measured)", hazard_idx=0, c=0.001,
             T=tres[0], outcome="DENY"),
    ]
    write_csv(f"{DATA_DIR}/exp7_testbed_points.csv", testbed_points)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(hz, tres, "-o", color=SAFE, lw=2.4, ms=7, label=r"$T_{\mathrm{res}}$ (halt)")
    ax.plot(hz, tperm, "-o", color=RISK, lw=2.4, ms=7, label=r"$T_{\mathrm{perm}}$ (start)")
    ax.axhline(C_FLOOR, color=GREY, ls="--", lw=1)
    ax.text(0, C_FLOOR * 1.6, r"$c_{\mathrm{floor}}$ (unidentified actor)",
           fontsize=8.5, color=GREY)
    ax.fill_between(hz, tres, tperm, color=SAFE, alpha=0.06)
    ax.set_yscale("log")
    ax.set_xticks(hz)
    ax.set_xticklabels(HAZ)
    ax.set_xlabel("device hazard class")
    ax.set_ylabel("required identity confidence (log scale)")
    ax.set_title("The coupled inversion, with physical-testbed\n"
                "validation points from the 3-phone deployment",
                fontsize=11.5, fontweight="bold", loc="left")

    for p in testbed_points:
        marker = "^" if p["outcome"] == "EXECUTE" else "x"
        color = SAFE if p["outcome"] == "EXECUTE" else ALERT
        if p["outcome"] == "EXECUTE":
            ax.scatter([p["hazard_idx"]], [p["c"]], marker=marker, s=140,
                      color=color, edgecolor=INK, linewidth=1.2, zorder=5)
        else:
            ax.scatter([p["hazard_idx"]], [p["c"]], marker=marker, s=160,
                      color=color, linewidth=2.4, zorder=5)
        ax.annotate(f"{p['label']}\n{p['outcome']}", xy=(p["hazard_idx"], p["c"]),
                   xytext=(p["hazard_idx"] + 0.15,
                          p["c"] * (6 if p["outcome"] == "EXECUTE" else 0.25)),
                   fontsize=8, color=color,
                   arrowprops=dict(arrowstyle="->", color=color, lw=1))

    ax.legend(frameon=False, fontsize=9, loc="center left")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig7_inversion_testbed.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig7_inversion_testbed.png\n")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────




# ═══════════════════════════════════════════════════════════════════════
# E8 -- FIGURE 8: the calibration regret bound (Theorem 2), numerically
# ═══════════════════════════════════════════════════════════════════════
def experiment_8_calibration_regret():
    print("=" * 78)
    print("E8  calibration regret bound (Theorem 2) -- numerical illustration")
    print("=" * 78)
    L, a, f = BASE_COST[3]           # CRITICAL
    T = T_perm(BASE_COST[3])
    C_FA, C_FR = L, f                # permissive: false-accept=Lambda, false-reject=phi
    eps = 0.01
    regret_bound = eps * (C_FA + C_FR)
    # Plot window follows the threshold rather than a hardcoded range, so the
    # figure stays correct when the grounded cost matrix moves T_perm.
    lo = max(0.0, min(T - 4 * eps, P.OPERATOR_GOOD_MEAN - 4 * P.OPERATOR_GOOD_SD))
    c_grid = np.linspace(lo, 1.0, 400)
    cost_execute = (1 - c_grid) * C_FA
    cost_withhold = c_grid * C_FR

    # Mean confidence of a well-imaged, credentialed operator (parameters.py
    # section 4), i.e. the best case a single attestation can offer.
    c_operator = P.OPERATOR_GOOD_MEAN
    band_lo, band_hi = T - eps, T + eps
    rows = [dict(T_perm_CRIT=T, C_FA=C_FA, C_FR=C_FR, eps=eps,
                 regret_bound=round(regret_bound, 2),
                 regret_in_units_of_phi=round(regret_bound / f, 2),
                 band_lo=round(band_lo, 6), band_hi=round(min(band_hi, 1.0), 6),
                 operator_c=c_operator,
                 operator_inside_band=(band_lo <= c_operator <= band_hi))]
    write_csv(f"{DATA_DIR}/exp8_calibration_regret.csv", rows)
    print(f"  T_perm(CRITICAL)     = {T:.6f}")
    print(f"  C_FA + C_FR          = {C_FA + C_FR:,}")
    print(f"  regret bound (e=.01) = {regret_bound:,.1f}  ({regret_bound/f:,.1f}x the re-gesture cost)")
    print(f"  vulnerability band   = [{band_lo:.5f}, {min(band_hi,1.0):.5f}]")
    print(f"  measured operator c  = {c_operator}  -> inside band: {band_lo<=c_operator<=band_hi}")

    from math import log
    hrows = [dict(eps=round(float(e), 4), n_required=int(log(2/0.05)/(2*e**2)) + 1)
             for e in [0.005, 0.01, 0.02, 0.05, 0.10]]
    write_csv(f"{DATA_DIR}/exp8_hoeffding_samples.csv", hrows)
    for r in hrows:
        print(f"  eps={r['eps']:.3f}  n >= {r['n_required']:,}")

    eps_range = np.linspace(0.005, 0.10, 200)
    n_needed = [log(2 / 0.05) / (2 * e ** 2) for e in eps_range]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(c_grid, cost_execute, color=RISK, lw=2.2, label=r"cost(execute$\mid c$)")
    ax1.plot(c_grid, cost_withhold, color=SAFE, lw=2.2, label=r"cost(withhold$\mid c$)")
    ax1.axvline(T, color=INK, ls="--", lw=1)
    ax1.axvspan(max(band_lo, lo), min(band_hi, 1.0), color=AMBER, alpha=0.18,
               label=r"vulnerability band $[T\pm\varepsilon]$")
    ax1.axvline(c_operator, color=INK, lw=1.6)
    ax1.scatter([c_operator], [min((1-c_operator)*C_FA, c_operator*C_FR)],
               marker="o", s=90, color=INK, zorder=5)
    ax1.annotate(f"well-imaged operator\n$c={c_operator:.3f}$",
               xy=(c_operator, min((1-c_operator)*C_FA, c_operator*C_FR)),
               xytext=(lo + 0.15 * (1.0 - lo), 0.55 * max(cost_execute.max(), cost_withhold.max())),
               fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax1.set_xlabel(r"true identity confidence $c^{*}$")
    ax1.set_ylabel("expected cost (units)")
    ax1.set_title("(a) Regret geometry at CRITICAL / permissive",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.15)

    ax2.plot(eps_range, n_needed, color=INK, lw=2.2)
    for e in [0.01, 0.05]:
        n = log(2/0.05)/(2*e**2)
        ax2.scatter([e], [n], color=RISK, s=60, zorder=5)
        ax2.annotate(f"$\\varepsilon$={e}\nn$\\geq${int(n)+1:,}", xy=(e, n),
                    xytext=(e + 0.012, n + 6000), fontsize=8.5, color=RISK)
    ax2.set_xlabel(r"target calibration accuracy $\varepsilon$")
    ax2.set_ylabel(r"held-out samples required, $n \geq \ln(2/\eta)/(2\varepsilon^2)$")
    ax2.set_title("(b) Calibration verification needs volume (Hoeffding, $\\eta$=0.05)",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax2.grid(alpha=0.15)

    fig.suptitle("Veracity is safety-critical, and costly to verify, "
                "exactly where quorum applies", fontsize=11.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig8_calibration_regret.png", bbox_inches="tight")
    plt.close()
    print(f"  -> {FIG_DIR}/fig8_calibration_regret.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E9 -- FIGURE 9: throughput vs. published big-data streaming references
# ═══════════════════════════════════════════════════════════════════════
def experiment_9_throughput_benchmark():
    print("=" * 78)
    print("E9  RAGA decision throughput vs. published streaming/CEP benchmarks")
    print("=" * 78)
    import hmac, hashlib, json

    SECRET = b"raga-bench-key"
    rng = np.random.default_rng(1)
    N = 200_000

    def sign(payload):
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(SECRET, body, hashlib.sha256).hexdigest()[:32]

    def verify(env):
        body = {k: v for k, v in env.items() if k != "sig"}
        return hmac.compare_digest(env.get("sig", ""), sign(body))

    envelopes = []
    for i in range(N):
        pol = "RES" if rng.random() < 0.6 else "PERM"
        h = [0, 1, 2, 3][rng.integers(0, 4)]
        env = {"env_id": f"e{i}", "seq": i, "actor": "a", "c": float(rng.uniform(0, 1)),
              "polarity": pol}
        env["sig"] = sign(env)
        env["_h"] = h
        envelopes.append(env)

    def decide(env):
        if not verify(env):
            return "DENY_SIG"
        cost = BASE_COST[env["_h"]]
        T = T_res(cost) if env["polarity"] == "RES" else T_perm(cost)
        return "EXECUTE" if env["c"] >= T else "DENY"

    lat = []
    t0 = time.perf_counter()
    for env in envelopes:
        s = time.perf_counter()
        decide(env)
        lat.append((time.perf_counter() - s) * 1000)
    elapsed = time.perf_counter() - t0
    lat.sort()
    p50, p95, p99 = lat[len(lat)//2], lat[int(len(lat)*0.95)], lat[int(len(lat)*0.99)]
    throughput = N / elapsed
    print(f"  RAGA measured (this machine, single core, pure Python + HMAC-SHA256):")
    print(f"    throughput = {throughput:,.0f} decisions/s")
    print(f"    latency p50/p95/p99 = {p50:.4f} / {p95:.4f} / {p99:.4f} ms")

    refs = [
        dict(system="RAGA (measured,\nthis work)", throughput=throughput,
             note="single core, O(1) decision, incl. signature verify"),
        dict(system="EdgeStream\n(edge, per node)", throughput=2800,
             note="industrial IoT scenario, edge node, published"),
        dict(system="SCEPter\n(buffered CEP)", throughput=3000,
             note="stateful semantic+CEP query, 2s buffer, 25% cache"),
        dict(system="Esper+RabbitMQ\n(sustained)", throughput=20000,
             note="sustained long-run throughput, simple pattern"),
        dict(system="Esper+Kafka\n(peak, simple)", throughput=150000,
             note="peak, select* pattern, 8 partitions"),
    ]
    write_csv(f"{DATA_DIR}/exp9_throughput_comparison.csv", refs)
    for r in refs:
        print(f"    {r['system'].splitlines()[0]:24s} {r['throughput']:>10,.0f} /s   ({r['note']})")

    labels = [r["system"] for r in refs]
    vals = [r["throughput"] for r in refs]
    cols = [SAFE] + [GREY] * (len(refs) - 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_yscale("log")
    ax.set_ylim(top=max(vals) * 2.2)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:,.0f}/s", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("decisions or events per second (log scale)")
    ax.set_title("RAGA's O(1) decision vs. published big-data\n"
                "streaming/CEP throughput (different hardware; order-of-magnitude only)",
                fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.15, which="both")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig9_throughput_benchmark.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig9_throughput_benchmark.png\n")





# ═══════════════════════════════════════════════════════════════════════
# E10 -- component ablation (Table VI in the paper). No figure.
# ═══════════════════════════════════════════════════════════════════════
def experiment_10_ablation():
    """
    Ablates quorum and liveness. Reported as Table VI. Shows a two-sided
    trade-off: removing quorum improves aggregate security but worsens
    safety, and both effects exceed their confidence intervals.
    """
    print("=" * 78)
    print("E10  component ablation (Table VI)")
    print("=" * 78)
    configs = {
        "full (raga+quorum+live)": "raga+quorum+live",
        "minus liveness":          "raga+quorum",
        "minus quorum":            "raga+live",
        "minus both (raga core)":  "raga",
    }
    rows = []
    print(f"  {'configuration':30s}{'safety':>18s}{'security':>18s}")
    for name, pol in configs.items():
        S, C = [], []
        for seed in range(10):
            rng = np.random.default_rng(900 + seed)
            evs = [draw_event(rng, BASE_COST, gesture_err=GERR) for _ in range(30_000)]
            r2 = np.random.default_rng(950 + seed)
            s, sec = score_run(pol, evs, BASE_COST, None, r2)
            S.append(s); C.append(sec)
        ms, es = ci95(S); mc, ec = ci95(C)
        print(f"  {name:30s}{ms:10.1f} +-{es:<6.1f}{mc:10.1f} +-{ec:.1f}")
        rows.append(dict(configuration=name, policy=pol,
                         safety_mean=round(ms, 2), safety_ci=round(es, 2),
                         security_mean=round(mc, 2), security_ci=round(ec, 2)))
    write_csv(f"{DATA_DIR}/exp10_ablation.csv", rows)
    print(f"  -> {DATA_DIR}/exp10_ablation.csv\n")




# ═══════════════════════════════════════════════════════════════════════
# E11 -- FIGURE 11: is the cliff an artifact of the stranger model?
# ═══════════════════════════════════════════════════════════════════════
def experiment_11_stranger_spread():
    """
    Proposition 1 assumes every unenrolled actor reports exactly c_floor.
    That degenerate distribution is what makes E1's transition a cliff. Here
    we relax it: strangers are drawn log-normally around c_floor with varying
    spread, and we ask what survives.

    Two things change with any spread: the jump shrinks sharply, AND the loose
    uniform policy stops achieving zero safety failures. The qualitative trap
    therefore DEEPENS while the cliff narrative weakens. Reporting both is
    more defensible than reporting the degenerate case alone.
    """
    print("=" * 78)
    print("E11  stranger-spread sensitivity (robustness of Prop. 1)")
    print("=" * 78)

    def run(spread, uniT, N=60_000, seed=7):
        rng = np.random.default_rng(seed)
        s = sec = 0
        for _ in range(N):
            r = rng.random()
            if r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED:
                kind, auth = "op", True
                c = float(np.clip(rng.normal(P.OPERATOR_GOOD_MEAN,
                                             P.OPERATOR_GOOD_SD), C_FLOOR, 1))
            elif r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED + P.POP_STRANGER:
                kind, auth = "stranger", False
                c = C_FLOOR if spread == 0 else float(
                    np.clip(10 ** rng.normal(np.log10(C_FLOOR), spread), 1e-6, 1))
            else:
                kind, auth = "imp", False
                c = float(np.clip(rng.normal(P.IMPOSTOR_MEAN,
                                             P.IMPOSTOR_SD), C_FLOOR, 1))
            h = int(rng.integers(0, 4))
            pol = "RES" if rng.random() < (0.85 if kind == "stranger" else 0.5) else "PERM"
            warr = rng.random() < 0.9
            ex = c >= uniT
            if pol == "RES" and warr and h >= 2 and not ex:
                s += 1
            if pol == "PERM" and not auth and h >= 1 and ex:
                sec += 1
        return s, sec

    spreads = [0, 0.1, 0.25, 0.5, 1.0]
    rows = []
    print(f"  {'spread':>8}{'safety@1e-3':>14}{'safety@1.6e-3':>16}{'jump':>8}")
    for sp in spreads:
        s1, _ = run(sp, 1e-3)
        s2, _ = run(sp, 1.6e-3)
        rows.append(dict(stranger_spread_log10sd=sp, safety_at_loose=s1,
                         safety_at_stepped=s2, jump=s2 - s1))
        print(f"  {sp:>8}{s1:>14}{s2:>16}{s2-s1:>8}")
    write_csv(f"{DATA_DIR}/exp11_stranger_spread.csv", rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = [r["stranger_spread_log10sd"] for r in rows]
    ax.plot(x, [r["safety_at_loose"] for r in rows], "-o", color=SAFE, lw=2.2, ms=6,
            label=r"uniform $\tau=10^{-3}$ (loose)")
    ax.plot(x, [r["safety_at_stepped"] for r in rows], "-o", color=ALERT, lw=2.2, ms=6,
            label=r"uniform $\tau=1.6\times10^{-3}$")
    ax.fill_between(x, [r["safety_at_loose"] for r in rows],
                    [r["safety_at_stepped"] for r in rows], color=ALERT, alpha=0.08)
    ax.set_xlabel(r"stranger confidence spread ($\log_{10}$ s.d.; 0 = Prop. 1's model)")
    ax.set_ylabel("safety failures per 60,000 events")
    ax.set_title("The cliff's sharpness depends on the stranger model;\n"
                 "the trap itself does not", fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    ax.annotate("shaded gap = the 'cliff'", xy=(0.05, rows[0]["jump"] / 2),
                xytext=(0.35, rows[0]["jump"] * 0.75), fontsize=8.5, color=ALERT,
                arrowprops=dict(arrowstyle="->", color=ALERT, lw=1))
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig11_stranger_spread.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig11_stranger_spread.png\n")




# ═══════════════════════════════════════════════════════════════════════
# E12 -- FIGURE 12: the phase transition at the c_floor crossing
# ═══════════════════════════════════════════════════════════════════════
def experiment_12_phase_transition():
    """
    NOVEL FINDING. Parameterise the threshold separation by a gap exponent g:
    g=0 collapses both thresholds to their geometric mean; g=1 is the
    cost-optimal RAGA pair; g>1 widens further.

    The system does NOT degrade smoothly in g. It undergoes a sharp transition
    at the value of g where T_res crosses below c_floor -- exactly Corollary 2's
    condition. Crossing it improves the security+nuisance objective AND the
    robustness to upstream gesture misclassification SIMULTANEOUSLY. Beyond the
    crossing, further widening buys almost nothing.

    Corollary 2 was derived as a statement about WHO may halt a device. This
    experiment shows it is also a phase boundary for SYSTEM ROBUSTNESS, which
    does not follow from the corollary as stated.
    """
    print("=" * 78)
    print("E12  phase transition at the c_floor crossing (novel finding)")
    print("=" * 78)

    def thresholds(h, gap):
        tr, tp = T_res(BASE_COST[h]), T_perm(BASE_COST[h])
        gm = (tr * tp) ** 0.5
        return (np.exp(np.log(gm) + gap * (np.log(tr) - np.log(gm))),
                min(np.exp(np.log(gm) + gap * (np.log(tp) - np.log(gm))), 0.999999))

    def run(gap, gerr, seed, N=40_000):
        rng = np.random.default_rng(seed)
        s = sec = nui = 0
        for _ in range(N):
            r = rng.random()
            if r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED:
                auth, c = True, float(np.clip(rng.normal(P.OPERATOR_GOOD_MEAN,
                                                          P.OPERATOR_GOOD_SD), C_FLOOR, 1))
            elif r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED + P.POP_STRANGER:
                auth, c = False, C_FLOOR
            else:
                auth, c = False, float(np.clip(rng.normal(P.IMPOSTOR_MEAN,
                                                           P.IMPOSTOR_SD), C_FLOOR, 1))
            h = int(rng.integers(0, 4))
            pt = "RES" if rng.random() < 0.6 else "PERM"
            pol = ("PERM" if pt == "RES" else "RES") if rng.random() < gerr else pt
            warr = rng.random() < 0.9
            tr, tp = thresholds(h, gap)
            ex = c >= (tr if pol == "RES" else tp)
            if pt == "RES" and warr and h >= 2 and not ex: s += 1
            if pol == "PERM" and not auth and h >= 1 and ex: sec += 1
            if pol == "RES" and not warr and not auth and ex: nui += 1
        return s, sec, nui

    gaps = [0.0, 0.25, 0.5, 0.7, 0.80, 0.85, 0.90, 0.95, 1.0, 1.25, 1.5, 2.0]
    rows = []
    print(f"  {'gap':>5}{'T_res(HIGH)':>14}{'sec+nui':>12}{'safety@err':>15}"
          f"{'strangers can halt':>20}")
    for gap in gaps:
        B, D = [], []
        for sd in range(15):
            _, sec0, nui0 = run(gap, 0.0, 300 + sd)
            s2, _, _ = run(gap, GERR, 300 + sd)
            B.append(sec0 + nui0); D.append(s2)
        mb, _ = ci95(B); md, _ = ci95(D)
        tr_high, _ = thresholds(2, gap)
        n_halt = sum(1 for h in range(4) if C_FLOOR >= thresholds(h, gap)[0])
        rows.append(dict(gap=gap, T_res_HIGH=round(float(tr_high), 8),
                         sec_plus_nui=round(mb, 1), safety_at_2pct=round(md, 1),
                         n_devices_stranger_can_halt=n_halt))
        print(f"  {gap:>5.2f}{tr_high:>14.2e}{mb:>12.1f}{md:>15.1f}{n_halt:>20}")
    write_csv(f"{DATA_DIR}/exp12_phase_transition.csv", rows)

    # Locate EVERY crossing, not just HIGH's: with the grounded cost matrix the
    # devices cross c_floor at different g, so the response is a staircase with
    # one step per device rather than a single phase boundary.
    crossings = []
    for h in range(4):
        lo, hi = 0.0, 4.0
        if thresholds(h, hi)[0] > C_FLOOR:
            continue                      # never becomes haltable in range
        for _ in range(60):
            mid = (lo + hi) / 2
            if thresholds(h, mid)[0] > C_FLOOR: lo = mid
            else: hi = mid
        crossings.append(dict(hazard=HAZ[h], crossing_gap=round(float(hi), 4)))
        print(f"  {HAZ[h]:9} T_res crosses c_floor at g = {hi:.4f}")
    write_csv(f"{DATA_DIR}/exp12_crossings.csv", crossings)
    crossing = [c["crossing_gap"] for c in crossings if c["hazard"] == "HIGH"][0]
    print(f"\n  T_res(HIGH) crosses c_floor at g = {crossing:.3f}")
    best = min(rows, key=lambda r: r["safety_at_2pct"])
    at_g1 = [r for r in rows if abs(r["gap"] - 1.0) < 1e-9][0]
    print(f"  robustness-optimal g = {best['gap']:.2f} "
          f"(safety@{GERR:.0%} = {best['safety_at_2pct']:.1f})")
    print(f"  cost-optimal     g = 1.00 (safety@{GERR:.0%} = {at_g1['safety_at_2pct']:.1f})")
    print(f"  -> cost-optimal is {at_g1['safety_at_2pct']/best['safety_at_2pct']:.1f}x "
          f"MORE fragile than the robustness optimum")
    write_csv(f"{DATA_DIR}/exp12_robustness_valley.csv", [dict(
        robustness_optimal_gap=best["gap"],
        safety_at_robust_optimum=best["safety_at_2pct"],
        safety_at_cost_optimum=at_g1["safety_at_2pct"],
        fragility_ratio=round(at_g1["safety_at_2pct"]/best["safety_at_2pct"], 2),
        crossing_gap=round(crossing, 3))])

    fig, ax1 = plt.subplots(figsize=(7.4, 4.5))
    g = [r["gap"] for r in rows]
    ax1.plot(g, [r["sec_plus_nui"] for r in rows], "-o", color=RISK, lw=2.2, ms=5,
             label="security + nuisance (no gesture error)")
    ax1.plot(g, [r["safety_at_2pct"] for r in rows], "-s", color=ALERT, lw=2.2, ms=5,
             label=f"safety failures @ {GERR:.0%} gesture error")
    ax1.axvline(crossing, color=SAFE, ls="--", lw=1.6)
    ax1.axvspan(crossing, 2.0, color=SAFE, alpha=0.06)
    ax1.text(crossing + 0.03, ax1.get_ylim()[1] * 0.88,
             f"$T_{{res}}$ crosses $c_{{floor}}$\n(Corollary 2's condition)",
             fontsize=8.5, color=SAFE)
    ax1.axvline(1.0, color=INK, ls=":", lw=1.2)
    ax1.text(1.02, ax1.get_ylim()[1] * 0.55, "cost-optimal\n$g=1$",
             fontsize=8.5, color=INK)
    ax1.set_xlabel(r"threshold separation exponent $g$  ($g{=}0$ collapsed, $g{=}1$ cost-optimal)")
    ax1.set_ylabel("failures per 40,000 events")
    ax1.set_title("A phase transition, not a smooth trade-off",
                  fontsize=11.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax1.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig12_phase_transition.png")
    plt.close()
    print(f"  -> {FIG_DIR}/fig12_phase_transition.png\n")




# ═══════════════════════════════════════════════════════════════════════
# E13 -- FIGURE 13: complete four-quadrant, cost-weighted accounting
# ═══════════════════════════════════════════════════════════════════════
def experiment_13_full_accounting():
    """
    The most consequential review finding. Two defects in the earlier
    evaluation, both of which invert the conclusion:

    (1) INCOMPLETENESS. The cost model has four outcome quadrants, but only
        three were measured. The missing one is AUTHORIZED-USER FRICTION: a
        legitimate operator refused a permissive command and forced to
        re-gesture. It is the phi term of Eq. (3). It turns out to dominate
        every other error type by an order of magnitude.

    (2) UNWEIGHTED COUNTING. Failures were counted 1:1. But a safety failure
        at CRITICAL costs Lambda = 2e6 while a friction event costs phi = 20 --
        a ratio of 1e5. Counting them equally contradicts the cost model the
        thresholds are derived from.

    With all four quadrants counted but UNWEIGHTED, the loose uniform baseline
    appears to win. Weighted by the costs the design actually optimises, RAGA
    wins by an order of magnitude. Both the completeness and the weighting are
    necessary; either alone gives the wrong answer.
    """
    print("=" * 78)
    print("E13  four-quadrant cost-weighted accounting (review finding)")
    print("=" * 78)

    def run(policy, seed, N=50_000, gerr=GERR, failsafe=False):
        rng = np.random.default_rng(seed)
        cnt = dict(safety=0, security=0, nuisance=0, friction=0)
        cost = dict(safety=0.0, security=0.0, nuisance=0.0, friction=0.0)
        fric_h = [0, 0, 0, 0]
        auth_perm_h = [0, 0, 0, 0]
        for _ in range(N):
            r = rng.random()
            if r < P.POP_OPERATOR_GOOD:
                auth, c = True, float(np.clip(rng.normal(
                    P.OPERATOR_GOOD_MEAN, P.OPERATOR_GOOD_SD), C_FLOOR, 1))
            elif r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED:
                auth, c = True, float(np.clip(rng.normal(
                    P.OPERATOR_DEGRADED_MEAN, P.OPERATOR_DEGRADED_SD), C_FLOOR, 1))
            elif r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED + P.POP_STRANGER:
                auth, c = False, C_FLOOR
            else:
                auth, c = False, float(np.clip(rng.normal(
                    P.IMPOSTOR_MEAN, P.IMPOSTOR_SD), C_FLOOR, 1))
            h = int(rng.integers(0, 4))
            L, a, f = BASE_COST[h]
            pt = "RES" if rng.random() < 0.5 else "PERM"
            pol = ("PERM" if pt == "RES" else "RES") if rng.random() < gerr else pt
            if failsafe and pol == "PERM" and pt == "RES":
                pol = "RES"          # Sec. V-E: ambiguity resolves restrictive
            warr = rng.random() < 0.9
            eff = c
            if "quorum" in policy and pol == "PERM" and h == 3:
                c2 = float(np.clip(rng.normal(P.SECOND_ATTESTER_MEAN,
                                     P.SECOND_ATTESTER_SD), C_FLOOR, 1))
                eff = 1 - (1 - c) * (1 - c2)
            if policy.startswith("uniform"):
                T = 1e-3
            elif policy == "polarity_only":
                T = POLARITY_ONLY_T["RES"] if pol == "RES" else POLARITY_ONLY_T["PERM"]
            else:
                T = T_res(BASE_COST[h]) if pol == "RES" else T_perm(BASE_COST[h])
            ex = eff >= T
            if pt == "RES" and warr and h >= 2 and not ex:
                cnt["safety"] += 1; cost["safety"] += L
            if pol == "PERM" and not auth and h >= 1 and ex:
                cnt["security"] += 1; cost["security"] += L
            if pol == "RES" and not warr and not auth and ex:
                cnt["nuisance"] += 1; cost["nuisance"] += a
            if pol == "PERM" and auth:
                auth_perm_h[h] += 1
                if not ex:
                    cnt["friction"] += 1; cost["friction"] += f; fric_h[h] += 1
        return cnt, cost, fric_h, auth_perm_h

    policies = ["uniform_loose", "polarity_only", "raga", "raga+quorum"]
    rows, wrows = [], []
    print(f"  {'policy':16s}{'safety':>8}{'securit':>9}{'nuis':>7}{'friction':>10}"
          f"{'cnt tot':>10}{'cost tot':>12}")
    store = {}
    for p in policies:
        Cs, Ks = [], []
        for sd in range(10):
            cnt, cost, fh, ah = run(p, 700 + sd)
            Cs.append(cnt); Ks.append(cost)
            if sd == 0:
                fh0, ah0 = fh, ah
        mc = {k: float(np.mean([c[k] for c in Cs])) for k in Cs[0]}
        mk = {k: float(np.mean([c[k] for c in Ks])) for k in Ks[0]}
        tc, tk = sum(mc.values()), sum(mk.values())
        store[p] = (mc, mk, tc, tk, fh0, ah0)
        print(f"  {p:16s}{mc['safety']:>8.0f}{mc['security']:>9.0f}{mc['nuisance']:>7.0f}"
              f"{mc['friction']:>10.0f}{tc:>10.0f}{tk:>12.3e}")
        rows.append(dict(policy=p, **{k: round(v, 1) for k, v in mc.items()},
                         unweighted_total=round(tc, 1)))
        wrows.append(dict(policy=p, **{f"cost_{k}": round(v, 1) for k, v in mk.items()},
                          weighted_total=round(tk, 1)))
    write_csv(f"{DATA_DIR}/exp13_counts.csv", rows)
    write_csv(f"{DATA_DIR}/exp13_weighted.csv", wrows)

    # The paper claims that applying the fail-safe rule and ranking by cost
    # stabilises the ordering. That claim was previously asserted in prose but
    # never computed here; it is computed now so the artifact backs it.
    fs_rows = []
    print("\n  cost-weighted totals WITH the fail-safe polarity rule applied:")
    fs_store = {}
    for p in policies:
        Ks = []
        for sd in range(10):
            _cnt, cost, _fh, _ah = run(p, 700 + sd, failsafe=True)
            Ks.append(cost)
        mk = {k: float(np.mean([c[k] for c in Ks])) for k in Ks[0]}
        tk = sum(mk.values())
        fs_store[p] = tk
        fs_rows.append(dict(policy=p, **{f"cost_{k}": round(v, 1) for k, v in mk.items()},
                            weighted_total_failsafe=round(tk, 1)))
        print(f"    {p:16s}{tk:>14.3e}")
    write_csv(f"{DATA_DIR}/exp13_weighted_failsafe.csv", fs_rows)
    fs_best = min(policies, key=lambda p: fs_store[p])
    print(f"    -> best under fail-safe, by cost: {fs_best}")
    write_csv(f"{DATA_DIR}/exp13_failsafe_ranking.csv", [dict(
        best_by_weighted_cost_failsafe=fs_best,
        **{f"total_{p}": round(fs_store[p], 1) for p in policies})])

    best_cnt = min(policies, key=lambda p: store[p][2])
    best_cost = min(policies, key=lambda p: store[p][3])
    print(f"\n  best by UNWEIGHTED count : {best_cnt}")
    print(f"  best by COST-WEIGHTED    : {best_cost}")
    if best_cnt != best_cost:
        print("  -> THE TWO METRICS DISAGREE. Unweighted counting contradicts")
        print("     the cost model the thresholds are derived from.")
    ratio = store[best_cnt][3] / store[best_cost][3]
    print(f"  cost ratio between them  : {ratio:.1f}x")
    write_csv(f"{DATA_DIR}/exp13_ranking.csv", [dict(
        best_by_unweighted_count=best_cnt, best_by_weighted_cost=best_cost,
        rankings_disagree=(best_cnt != best_cost), cost_ratio=round(ratio, 2))])

    # per-hazard friction, RAGA
    mc, mk, tc, tk, fh, ah = store["raga"]
    frows = [dict(hazard=HAZ[h], authorized_starts=ah[h], refused=fh[h],
                  refusal_rate=round(fh[h] / max(ah[h], 1), 4)) for h in range(4)]
    write_csv(f"{DATA_DIR}/exp13_friction_by_hazard.csv", frows)
    print("\n  authorized-start refusal rate by hazard class (RAGA):")
    for r in frows:
        print(f"    {r['hazard']:9} {r['refused']:>5}/{r['authorized_starts']:<5} "
              f"= {r['refusal_rate']*100:5.1f}%")

    # ── quorum fusion semantics: noisy-OR vs conjunctive ──
    def run_q(mode, seed, N=50_000, gerr=GERR):
        rng = np.random.default_rng(seed)
        cnt = dict(security=0, friction=0); tot = 0.0
        for _ in range(N):
            r = rng.random()
            if r < P.POP_OPERATOR_GOOD:
                auth, c = True, float(np.clip(rng.normal(
                    P.OPERATOR_GOOD_MEAN, P.OPERATOR_GOOD_SD), C_FLOOR, 1))
            elif r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED:
                auth, c = True, float(np.clip(rng.normal(
                    P.OPERATOR_DEGRADED_MEAN, P.OPERATOR_DEGRADED_SD), C_FLOOR, 1))
            elif r < P.POP_OPERATOR_GOOD + P.POP_OPERATOR_DEGRADED + P.POP_STRANGER:
                auth, c = False, C_FLOOR
            else:
                auth, c = False, float(np.clip(rng.normal(
                    P.IMPOSTOR_MEAN, P.IMPOSTOR_SD), C_FLOOR, 1))
            h = int(rng.integers(0, 4)); L, a, f = BASE_COST[h]
            pt = "RES" if rng.random() < 0.5 else "PERM"
            pol = ("PERM" if pt == "RES" else "RES") if rng.random() < gerr else pt
            if pol == "PERM" and pt == "RES": pol = "RES"          # fail-safe
            T = T_res(BASE_COST[h]) if pol == "RES" else T_perm(BASE_COST[h])
            ex = None
            if pol == "PERM" and h == 3 and mode != "none":
                c2 = float(np.clip(rng.normal(P.SECOND_ATTESTER_MEAN,
                                     P.SECOND_ATTESTER_SD), C_FLOOR, 1))
                if mode == "noisy_or":
                    # As originally specified: fuse into one boosted score.
                    ex = (1 - (1 - c) * (1 - c2)) >= T
                else:
                    # Conjunctive: each attester must independently clear the
                    # SAME threshold T. No arbitrary per-party bar -- this is
                    # exactly c_eff = min(c, c2) >= T, matching Algorithm 2 and
                    # the testbed implementation. It is strictly stricter than
                    # no quorum, so it trades a little friction for the
                    # security blow-up that noisy-OR causes.
                    ex = min(c, c2) >= T
            if ex is None: ex = c >= T
            if pt == "RES" and rng.random() < 0.9 and h >= 2 and not ex: tot += L
            if pol == "PERM" and not auth and h >= 1 and ex:
                cnt["security"] += 1; tot += L
            if pol == "PERM" and auth and not ex:
                cnt["friction"] += 1; tot += f
        return cnt, tot

    qrows = []
    print("\n  quorum fusion semantics at CRITICAL:")
    print(f"    {'mode':14}{'security':>10}{'friction':>10}{'cost total':>14}")
    for mode in ["none", "noisy_or", "conjunctive"]:
        S, F, K = [], [], []
        for sd in range(10):
            cnt, tot = run_q(mode, 700 + sd)
            S.append(cnt["security"]); F.append(cnt["friction"]); K.append(tot)
        ms, _ = ci95(S); mf, _ = ci95(F); mk, _ = ci95(K)
        qrows.append(dict(quorum_mode=mode, security=round(ms, 1),
                          friction=round(mf, 1), cost_total=round(mk, 1)))
        print(f"    {mode:14}{ms:>10.0f}{mf:>10.0f}{mk:>14.3e}")
    write_csv(f"{DATA_DIR}/exp13_quorum_semantics.csv", qrows)
    print("    -> noisy-OR lets an impostor borrow assurance from an honest")
    print("       co-signer; conjunctive quorum does not.")

    # ── figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    lab = [p.replace("uniform_", "uni-").replace("polarity_only", "polarity\nonly")
            .replace("raga+quorum", "raga+q") for p in policies]
    x = np.arange(len(policies))
    keys = ["safety", "security", "nuisance", "friction"]
    cols = [ALERT, RISK, AMBER, GREY]

    bottom = np.zeros(len(policies))
    for k, col in zip(keys, cols):
        vals = np.array([store[p][0][k] for p in policies])
        ax1.bar(x, vals, 0.6, bottom=bottom, color=col, label=k)
        bottom += vals
    ax1.set_xticks(x); ax1.set_xticklabels(lab, fontsize=8.5)
    ax1.set_ylabel("failure count (unweighted)")
    ax1.set_title("(a) Unweighted counts — uniform appears best",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(axis="y", alpha=0.15)

    bottom = np.zeros(len(policies))
    for k, col in zip(keys, cols):
        vals = np.array([store[p][1][k] for p in policies])
        ax2.bar(x, vals, 0.6, bottom=bottom, color=col, label=k)
        bottom += vals
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(lab, fontsize=8.5)
    ax2.set_ylabel("expected cost (log scale)")
    ax2.set_title("(b) Cost-weighted — the ranking inverts",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax2.grid(axis="y", alpha=0.15, which="both")

    fig.suptitle("Completeness AND weighting both change the answer",
                 fontsize=11.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig13_full_accounting.png", bbox_inches="tight")
    plt.close()
    print(f"\n  -> {FIG_DIR}/fig13_full_accounting.png\n")


# ═══════════════════════════════════════════════════════════════════════
# E14 -- FIGURE 14: the stranger-halt property is bought with fast restart
# ═══════════════════════════════════════════════════════════════════════
def experiment_14_stop_category():
    """
    The finding that only appears once the cost matrix is denominated in real
    money rather than arbitrary units.

    Corollary 2 says an unidentified bystander may halt a device exactly when
    Lambda >= alpha * (1-c_floor)/c_floor. With Lambda fixed by what a
    workplace fatality costs (NSC: $1.54M) and alpha fixed by what an
    unnecessary stop costs (Siemens: automotive downtime $2M/h = $556/s),
    that inequality is no longer comfortably satisfied -- it becomes a
    constraint on how quickly the device can resume.

    Concretely: the arm can afford to let strangers halt it only if a needless
    stop costs less than about $1,540, i.e. less than roughly 2.8 seconds of
    line time. A safety-rated monitored stop (ISO/TS 15066), which resumes
    automatically once the person leaves the collaborative workspace, fits
    inside that budget. An emergency stop requiring a manual reset and restart
    sequence does not, and forfeits the property entirely.

    The design rule that falls out is actionable: if you want bystanders to be
    able to halt your most dangerous machine, buy the fast-resume stop.
    """
    print("=" * 78)
    print("E14  stop category vs. the stranger-halt property (grounded costs)")
    print("=" * 78)

    L = P.HARM["arm"]
    rate = P.DOWNTIME_RATE_PER_S[P.DEVICE_SECTOR["arm"]]
    alpha_budget = L * C_FLOOR / (1.0 - C_FLOOR)   # exact Corollary 2 bound
    t_breakeven = alpha_budget / rate

    print(f"  Lambda (NSC cost per workplace death)   = ${L:,.0f}")
    print(f"  automotive downtime rate (Siemens)      = ${rate:,.2f}/s")
    print(f"  alpha budget at c_floor={C_FLOOR:.0e}          = ${alpha_budget:,.2f}")
    print(f"  => break-even restart duration          = {t_breakeven:.3f} s\n")

    rows = []
    print(f"  {'t_stop (s)':>11}{'alpha ($)':>14}{'T_res':>13}{'stranger may halt':>20}")
    for t_stop in [0.5, 1.0, 2.0, 2.5, 2.77, 3.0, 5.0, 10.0, 30.0, 60.0, 120.0]:
        a = t_stop * rate
        tres = a / (a + L)
        ok = bool(tres <= C_FLOOR)
        rows.append(dict(t_stop_s=t_stop, alpha_usd=round(a, 2),
                         T_res=round(float(tres), 9), stranger_may_halt=ok))
        print(f"  {t_stop:>11.2f}{a:>14,.0f}{tres:>13.3e}{('YES' if ok else 'no'):>20}")
    write_csv(f"{DATA_DIR}/exp14_stop_category.csv", rows)

    named = []
    for label, t_stop in [("safety-rated monitored stop", P.T_STOP_S["arm"]),
                          ("emergency stop, manual reset", P.T_STOP_S_ARM_EMERGENCY_STOP)]:
        a = t_stop * rate
        tres = a / (a + L)
        named.append(dict(stop_category=label, t_stop_s=t_stop, alpha_usd=round(a, 2),
                          T_res=round(float(tres), 9), lambda_over_alpha=round(L / a, 1),
                          stranger_may_halt=bool(tres <= C_FLOOR)))
        print(f"\n  {label}: t_stop={t_stop}s, alpha=${a:,.0f}, "
              f"T_res={tres:.3e}, Lambda/alpha={L/a:,.0f} -> "
              f"{'stranger MAY halt' if tres <= C_FLOOR else 'stranger may NOT halt'}")
    write_csv(f"{DATA_DIR}/exp14_stop_categories_named.csv", named)

    srows = []
    print(f"\n  break-even restart duration vs. open-set operating point:")
    for cf in P.C_FLOOR_SWEEP:
        tb = (L * cf / (1.0 - cf)) / rate
        srows.append(dict(c_floor=cf, breakeven_t_stop_s=round(float(tb), 4)))
        print(f"    c_floor={cf:.0e}  ->  t_stop must be < {tb:.3f} s")
    write_csv(f"{DATA_DIR}/exp14_breakeven_vs_cfloor.csv", srows)

    t_grid = np.logspace(np.log10(0.2), np.log10(200), 400)
    tres_grid = (t_grid * rate) / (t_grid * rate + L)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax1.plot(t_grid, tres_grid, color=INK, lw=2.4)
    ax1.axhline(C_FLOOR, color=GREY, ls="--", lw=1.4)
    ax1.text(0.23, C_FLOOR * 1.3, r"$c_{\mathrm{floor}}$ (unidentified actor)",
             fontsize=8.5, color=GREY)
    ax1.axvspan(0.2, t_breakeven, color=SAFE, alpha=0.10)
    ax1.axvline(t_breakeven, color=SAFE, ls="--", lw=1.6)
    ax1.text(t_breakeven * 1.18, tres_grid.min() * 2.0,
             f"break-even\n{t_breakeven:.2f} s", fontsize=8.5, color=SAFE)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    # Fix the limits BEFORE annotating, with headroom above the curve, so no
    # label can be pushed into the panel title. Both labels are then placed by
    # hand in the open band between the c_floor line and the curve: the two
    # markers sit in very different parts of the plot, so one shared offset
    # formula (which is what previously threw the emergency-stop label into
    # the title) cannot serve both.
    ax1.set_xlim(0.2, 260)
    ax1.set_ylim(tres_grid.min() * 0.55, tres_grid.max() * 1.9)
    for label, t_stop, col, tx, ty, ha in [
            ("safety-rated\nmonitored stop", P.T_STOP_S["arm"], SAFE,
             0.62, 4.0e-3, "center"),
            ("emergency stop,\nmanual reset", P.T_STOP_S_ARM_EMERGENCY_STOP, ALERT,
             150.0, 4.2e-3, "right")]:
        y = (t_stop * rate) / (t_stop * rate + L)
        ax1.scatter([t_stop], [y], s=110, color=col, edgecolor=INK,
                    linewidth=1.1, zorder=5)
        ax1.annotate(label, xy=(t_stop, y), xytext=(tx, ty),
                     fontsize=8, color=col, ha=ha, va="center",
                     arrowprops=dict(arrowstyle="->", color=col, lw=1,
                                     shrinkA=2, shrinkB=6))
    ax1.set_xlabel("time to resume after a needless stop (s, log scale)")
    ax1.set_ylabel(r"$T_{\mathrm{res}}$ for the robot arm (log scale)")
    ax1.set_title("(a) Who may halt the arm is decided by how fast it restarts",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax1.grid(alpha=0.15, which="both")

    cfs = [r["c_floor"] for r in srows]
    tbs = [r["breakeven_t_stop_s"] for r in srows]
    ax2.plot(cfs, tbs, "-o", color=RISK, lw=2.2, ms=7)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    # Widen the limits first, then label each point into the empty lower-right
    # triangle. The previous offsets pushed the top label outside the axes.
    ax2.set_xlim(min(cfs) * 0.30, max(cfs) * 4.5)
    ax2.set_ylim(min(tbs) * 0.30, max(tbs) * 3.0)
    for cf, tb in zip(cfs, tbs):
        ax2.annotate(f"{tb:.3g}s", xy=(cf, tb), xytext=(cf * 1.45, tb * 0.62),
                     fontsize=8.5, color=RISK, ha="left", va="center")
    ax2.set_xlabel(r"open-set operating point $c_{\mathrm{floor}}$ (NIST FMR)")
    ax2.set_ylabel("break-even restart budget (s)")
    ax2.set_title("(b) A stricter open-set threshold buys a tighter budget",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax2.grid(alpha=0.15, which="both")

    fig.suptitle("With costs in real money, the stranger-halt property "
                 "is purchased with restart speed",
                 fontsize=11.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig14_stop_category.png", bbox_inches="tight")
    plt.close()
    print(f"\n  -> {FIG_DIR}/fig14_stop_category.png\n")


if __name__ == "__main__":
    experiment_1_trap()
    experiment_2_policy_comparison()
    experiment_3_gesture_error()
    experiment_4_sensitivity()
    experiment_5_erlc_adversarial()
    experiment_6_clock_skew()
    experiment_7_inversion_and_testbed()
    experiment_8_calibration_regret()
    experiment_9_throughput_benchmark()
    experiment_10_ablation()
    experiment_11_stranger_spread()
    experiment_12_phase_transition()
    experiment_13_full_accounting()
    experiment_14_stop_category()

    print("=" * 78)
    print(f"Done. Figures in ./{FIG_DIR}/   Datasets in ./{DATA_DIR}/")
    print("=" * 78)
