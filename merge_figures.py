#!/usr/bin/env python3
"""
merge_figures.py
=================
Combines pairs of the single-panel experimentation figures produced by
raga_experiments.py into two-panel figures, to save space under the IEEE
conference page limit. Reads ONLY the already-computed, already-verified
CSVs in dataset/ -- no simulation is re-run, so every number matches the
tables and prose exactly as before.

Produces (not all of these are necessarily used in the current manuscript;
check GestureRobot.tex for which merged figures are actually \includegraphics'd):
  fig/merged_trap_spread.png       (a) E1 trap        (b) E11 stranger spread
  fig/merged_negresults.png        (a) E3 gesture err  (b) E5 ERLC adversarial
  fig/merged_sensitivity_phase.png (a) E4 heatmap      (b) E12 phase transition
  fig/merged_sysprops.png          (a) E6 clock skew   (b) E9 throughput
  fig/merged_policy_throughput.png (a) E2 policies     (b) E9 throughput
  fig/merged_testbed_phase.png     (a) E7 thresholds   (b) E14 stop category
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

DATA_DIR = "dataset"
FIG_DIR = "fig"
C_FLOOR = 1e-3

INK, SAFE, RISK, GREY, ALERT, AMBER, PAPER = (
    "#16211E", "#1F6F5C", "#B4471B", "#7A8578", "#8E2F1C", "#B8860B", "#F2F3F0")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def read_csv(name):
    with open(f"{DATA_DIR}/{name}") as f:
        return list(csv.DictReader(f))


def panel_label(ax, text):
    ax.text(-0.10, 1.10, text, transform=ax.transAxes, fontsize=13,
             fontweight="bold", va="top", ha="left")


# ═══════════════════════════════════════════════════════════════════════
# Merge 1: E1 trap + E11 stranger spread  -> "the trap, and its robustness"
# ═══════════════════════════════════════════════════════════════════════
def merge_trap_spread():
    rows1 = read_csv("exp1_trap.csv")
    T = [float(r["T"]) for r in rows1]
    saf = [int(r["safety"]) for r in rows1]
    sec = [int(r["security"]) for r in rows1]

    rows11 = read_csv("exp11_stranger_spread.csv")
    x = [float(r["stranger_spread_log10sd"]) for r in rows11]
    loose = [int(r["safety_at_loose"]) for r in rows11]
    stepped = [int(r["safety_at_stepped"]) for r in rows11]
    jump0 = int(rows11[0]["jump"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    ax1.plot(T, saf, "-o", color=ALERT, ms=4, lw=1.8, label="safety failures (halt refused)")
    ax1.plot(T, sec, "-o", color=RISK, ms=4, lw=1.8, label="security failures (intruder start)")
    ax1.set_xscale("log")
    ax1.axvline(C_FLOOR, color=GREY, ls="--", lw=1)
    ax1.text(C_FLOOR * 1.15, max(saf) * 0.55, r"$c_{\mathrm{floor}}$", color=GREY, fontsize=9)
    ax1.set_xlabel(r"uniform identity threshold $\tau$ (log scale)")
    ax1.set_ylabel("failures per 200,000 events")
    ax1.set_title("The uniform-threshold trap is a cliff, not a gradient",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper center")
    ax1.grid(alpha=0.15)
    panel_label(ax1, "(a)")

    ax2.plot(x, loose, "-o", color=SAFE, lw=2.2, ms=6, label=r"uniform $\tau=10^{-3}$ (loose)")
    ax2.plot(x, stepped, "-o", color=ALERT, lw=2.2, ms=6, label=r"uniform $\tau=1.6\times10^{-3}$")
    ax2.fill_between(x, loose, stepped, color=ALERT, alpha=0.08)
    ax2.set_xlabel(r"stranger confidence spread ($\log_{10}$ s.d.; 0 = degenerate)")
    ax2.set_ylabel("safety failures per 60,000 events")
    ax2.set_title("Cliff sharpness depends on the stranger model;\nthe trap itself does not",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.15)
    ax2.annotate("shaded gap = the 'cliff'", xy=(0.05, jump0 / 2),
                 xytext=(0.35, jump0 * 0.75), fontsize=8, color=ALERT,
                 arrowprops=dict(arrowstyle="->", color=ALERT, lw=1))
    panel_label(ax2, "(b)")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/merged_trap_spread.png")
    plt.close()
    print(f"-> {FIG_DIR}/merged_trap_spread.png")


# ═══════════════════════════════════════════════════════════════════════
# Merge 2: E3 gesture error + E5 ERLC adversarial -> "two negative results"
# ═══════════════════════════════════════════════════════════════════════
def merge_negresults():
    rows3 = read_csv("exp3_gesture_error.csv")
    x3 = [float(r["gesture_error"]) * 100 for r in rows3]
    naive = [float(r["naive_safety"]) for r in rows3]
    fs = [float(r["failsafe_safety"]) for r in rows3]

    rows5 = read_csv("exp5_erlc_adversarial.csv")
    lab5 = [r["adversary"] for r in rows5]
    auc = [float(r["auc"]) for r in rows5]
    cols5 = [SAFE if a > 0.9 else AMBER if a > 0.7 else ALERT for a in auc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    ax1.plot(x3, naive, "-o", color=ALERT, lw=2.2, ms=6, label="RAGA, naive polarity read")
    ax1.plot(x3, fs, "-o", color=SAFE, lw=2.2, ms=6, label="RAGA + fail-safe polarity")
    ax1.set_xlabel("gesture polarity misclassification rate (%)")
    ax1.set_ylabel("safety failures per 30,000 events")
    ax1.set_title("The guarantee is conditional, and repairable",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.15)
    ax1.text(4.0, 40, "ambiguity resolved toward RESTRICTIVE", fontsize=7.8, color=SAFE)
    panel_label(ax1, "(a)")

    ax2.bar(range(len(auc)), auc, color=cols5)
    for i, a in enumerate(auc):
        ax2.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax2.axhline(0.5, color=GREY, ls="--", lw=1)
    ax2.text(0.05, 0.53, "chance", fontsize=8, color=GREY)
    ax2.set_xticks(range(len(lab5)))
    ax2.set_xticklabels(lab5, fontsize=9)
    ax2.set_ylabel("AUC, separating reflex from sabotage")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("ERLC degrades as the adversary learns the timing",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax2.grid(axis="y", alpha=0.15)
    panel_label(ax2, "(b)")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/merged_negresults.png")
    plt.close()
    print(f"-> {FIG_DIR}/merged_negresults.png")


# ═══════════════════════════════════════════════════════════════════════
# Merge 3: E4 heatmap + E12 phase transition -> "Corollary 2's boundary"
# ═══════════════════════════════════════════════════════════════════════
def merge_sensitivity_phase():
    rows4 = read_csv("exp4_sensitivity.csv")
    scales_L = sorted(set(float(r["scale_Lambda"]) for r in rows4))
    scales_a = sorted(set(float(r["scale_alpha"]) for r in rows4))
    grid = np.zeros((len(scales_a), len(scales_L)))
    for r in rows4:
        i = scales_a.index(float(r["scale_alpha"]))
        j = scales_L.index(float(r["scale_Lambda"]))
        grid[i, j] = 1.0 if r["stranger_can_halt"].strip().lower() == "true" else 0.0

    rows12 = read_csv("exp12_phase_transition.csv")
    g = [float(r["gap"]) for r in rows12]
    secnui = [float(r["sec_plus_nui"]) for r in rows12]
    safety2 = [float(r["safety_at_2pct"]) for r in rows12]
    valley = read_csv("exp12_robustness_valley.csv")[0]
    crossing = float(valley["crossing_gap"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 3.9))

    im = ax1.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(scales_L)))
    ax1.set_xticklabels([f"{s:g}x" for s in scales_L], fontsize=8.5)
    ax1.set_yticks(range(len(scales_a)))
    ax1.set_yticklabels([f"{s:g}x" for s in scales_a], fontsize=8.5)
    ax1.set_xlabel(r"harm scale ($\Lambda$ multiplier)", fontsize=9)
    ax1.set_ylabel(r"stop-cost scale ($\alpha$ mult.)", fontsize=9)
    ax1.set_title("Stranger-halt condition across 15 cost\nconfigs (green = holds; inversion held 15/15)",
                   fontsize=10, fontweight="bold", loc="left")
    for i in range(len(scales_a)):
        for j in range(len(scales_L)):
            ax1.text(j, i, "Y" if grid[i, j] else "N", ha="center", va="center",
                      fontsize=10, fontweight="bold", color="white" if grid[i, j] < 0.5 else INK)
    plt.colorbar(im, ax=ax1, fraction=0.035, pad=0.03, ticks=[0, 1])
    panel_label(ax1, "(a)")

    ax2.plot(g, secnui, "-o", color=RISK, lw=2, ms=4.5, label="security+nuisance (no err)")
    ax2.plot(g, safety2, "-s", color=ALERT, lw=2, ms=4.5, label="safety @ 2% gesture err")
    ax2.axvline(crossing, color=SAFE, ls="--", lw=1.4)
    ax2.axvspan(crossing, 2.0, color=SAFE, alpha=0.06)
    ax2.text(crossing + 0.03, ax2.get_ylim()[1] * 0.85,
              "$T_{res}$ crosses $c_{floor}$\n(Cor. 2)", fontsize=7.5, color=SAFE)
    ax2.set_xlabel(r"threshold separation exponent $g$", fontsize=9)
    ax2.set_ylabel("failures per 40,000 events", fontsize=9)
    ax2.set_title("Objective and robustness step at the\nsame $c_{floor}$ crossing (Cor. 2)",
                   fontsize=10, fontweight="bold", loc="left")
    ax2.legend(frameon=False, fontsize=7.8)
    ax2.grid(alpha=0.15)
    panel_label(ax2, "(b)")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/merged_sensitivity_phase.png")
    plt.close()
    print(f"-> {FIG_DIR}/merged_sensitivity_phase.png")


# ═══════════════════════════════════════════════════════════════════════
# Merge 4: E6 clock skew + E9 throughput -> "system properties"
# ═══════════════════════════════════════════════════════════════════════
def merge_sysprops():
    rows6 = read_csv("exp6_clock_skew.csv")
    sk = [float(r["clock_skew_ms"]) for r in rows6]
    dr = [float(r["disagreement_rate"]) * 100 for r in rows6]

    rows9 = read_csv("exp9_throughput_comparison.csv")
    labels9 = [r["system"].split("\n")[0] for r in rows9]
    vals9 = [float(r["throughput"]) for r in rows9]
    cols9 = [SAFE] + [GREY] * (len(rows9) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.2))

    ax1.plot(sk, dr, "-o", color=ALERT, lw=2.2, ms=6, label="cross-device recency tiebreak")
    ax1.axhline(0, color=SAFE, lw=2.2, label=r"single-clock $\rho$ (ERLC)")
    ax1.set_xscale("symlog")
    ax1.set_xlabel("clock skew between devices (ms)")
    ax1.set_ylabel("% conflicts, inconsistent ordering")
    ax1.set_title(r"Recency needs a skew bound; $\rho$ does not",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.15)
    panel_label(ax1, "(a)")

    ax2.bar(range(len(vals9)), vals9, color=cols9)
    ax2.set_yscale("log")
    ax2.set_ylim(top=max(vals9) * 2.4)
    for i, v in enumerate(vals9):
        ax2.text(i, v * 1.15, f"{v:,.0f}/s", ha="center", fontsize=7.5, fontweight="bold")
    ax2.set_xticks(range(len(labels9)))
    ax2.set_xticklabels(labels9, fontsize=7.3, rotation=12)
    ax2.set_ylabel("decisions or events/s (log scale)")
    ax2.set_title("RAGA's $O(1)$ decision vs. published\nbig-data streaming/CEP throughput",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax2.grid(axis="y", alpha=0.15, which="both")
    panel_label(ax2, "(b)")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/merged_sysprops.png")
    plt.close()
    print(f"-> {FIG_DIR}/merged_sysprops.png")



# ═══════════════════════════════════════════════════════════════════════
# Merge 5: E2 policy comparison + E9 throughput
# ═══════════════════════════════════════════════════════════════════════
def merge_policy_throughput():
    rows2 = read_csv("exp2_policy_summary.csv")
    lab = [r["policy"].replace("uniform_", "uni-")
                      .replace("raga+quorum+live", "raga+q\n+live")
                      .replace("polarity_only", "polarity\nonly")
           for r in rows2]
    S = [float(r["safety_mean"]) for r in rows2]
    C = [float(r["security_mean"]) for r in rows2]
    Nn = [float(r["nuisance_mean"]) for r in rows2]
    tot = [float(r["total_mean"]) for r in rows2]
    x = np.arange(len(lab))

    rows9 = read_csv("exp9_throughput_comparison.csv")
    labels9 = [r["system"] for r in rows9]
    vals9 = [float(r["throughput"]) for r in rows9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax1.bar(x, S, 0.6, color=ALERT, label="safety")
    ax1.bar(x, C, 0.6, bottom=S, color=RISK, label="security")
    ax1.bar(x, Nn, 0.6, bottom=np.array(S) + np.array(C), color=AMBER, label="nuisance")
    for i, t in enumerate(tot):
        ax1.text(i, t * 1.02 + 30, f"{t:.0f}", ha="center", fontsize=8,
                 fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lab, fontsize=7.5, rotation=12)
    ax1.set_ylabel("failures per 50,000 events (30 seeds)")
    ax1.set_title("(a) Policy comparison, all three error types",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(axis="y", alpha=0.15)

    cols = [SAFE] + [GREY] * (len(vals9) - 1)
    ax2.bar(range(len(vals9)), vals9, color=cols)
    ax2.set_yscale("log")
    ax2.set_ylim(top=max(vals9) * 2.4)
    for i, v in enumerate(vals9):
        ax2.text(i, v * 1.18, f"{v:,.0f}/s", ha="center", fontsize=8,
                 fontweight="bold")
    ax2.set_xticks(range(len(labels9)))
    ax2.set_xticklabels(labels9, fontsize=7.5)
    ax2.set_ylabel("events per second (log scale)")
    ax2.set_title("(b) Decision throughput vs. published CEP figures",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax2.grid(axis="y", alpha=0.15, which="both")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/merged_policy_throughput.png", bbox_inches="tight")
    plt.close()
    print(f"-> {FIG_DIR}/merged_policy_throughput.png")


# ═══════════════════════════════════════════════════════════════════════
# Merge 6: E7 inversion/testbed + E14 stop category
# ═══════════════════════════════════════════════════════════════════════
def merge_testbed_phase():
    rows7 = read_csv("exp7_threshold_table.csv")
    haz = [r["hazard"] for r in rows7]
    tres = [float(r["T_halt"]) for r in rows7]
    tperm = [float(r["T_start"]) for r in rows7]
    hz = list(range(len(haz)))

    pts = read_csv("exp7_testbed_points.csv")
    rows14 = read_csv("exp14_stop_category.csv")
    t_stop = [float(r["t_stop_s"]) for r in rows14]
    tr14 = [float(r["T_res"]) for r in rows14]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax1.plot(hz, tres, "-o", color=SAFE, lw=2.4, ms=7,
             label=r"$T_{\mathrm{res}}$ (halt)")
    ax1.plot(hz, tperm, "-o", color=RISK, lw=2.4, ms=7,
             label=r"$T_{\mathrm{perm}}$ (start)")
    ax1.axhline(C_FLOOR, color=GREY, ls="--", lw=1)
    ax1.text(0, C_FLOOR * 1.7, r"$c_{\mathrm{floor}}$", fontsize=8.5, color=GREY)
    ax1.set_yscale("log")
    ax1.set_xticks(hz)
    ax1.set_xticklabels(haz, fontsize=8.5)
    ax1.set_xlabel("device hazard class")
    ax1.set_ylabel("required identity confidence (log)")
    ax1.set_title("(a) The grounded threshold table, with testbed points",
                  fontsize=10.5, fontweight="bold", loc="left")
    for p in pts:
        idx = int(p["hazard_idx"])
        cval = float(p["c"])
        ex = p["outcome"] == "EXECUTE"
        ax1.scatter([idx], [cval], marker="^" if ex else "x", s=150,
                    color=SAFE if ex else ALERT,
                    edgecolor=INK if ex else None,
                    linewidth=1.2 if ex else 2.4, zorder=5)
    ax1.legend(frameon=False, fontsize=8, loc="center left")
    ax1.grid(alpha=0.15)

    ax2.plot(t_stop, tr14, "-o", color=INK, lw=2.2, ms=5)
    ax2.axhline(C_FLOOR, color=GREY, ls="--", lw=1.4)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ok = [t for t, v in zip(t_stop, tr14) if v <= C_FLOOR]
    if ok:
        ax2.axvspan(min(t_stop), max(ok), color=SAFE, alpha=0.10)
        ax2.text(max(ok) * 1.1, min(tr14) * 1.8,
                 "stranger may halt\n(fast-resume stop)", fontsize=8, color=SAFE)
    ax2.set_xlabel("time to resume after a needless stop (s, log)")
    ax2.set_ylabel(r"$T_{\mathrm{res}}$ for the robot arm (log)")
    ax2.set_title("(b) Bystander halt authority is bought with restart speed",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax2.grid(alpha=0.15, which="both")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/merged_testbed_phase.png", bbox_inches="tight")
    plt.close()
    print(f"-> {FIG_DIR}/merged_testbed_phase.png")


if __name__ == "__main__":
    merge_trap_spread()
    merge_negresults()
    merge_sensitivity_phase()
    merge_sysprops()
    merge_policy_throughput()
    merge_testbed_phase()
    print("\nAll merged figures generated from existing CSVs (no simulation re-run).")
