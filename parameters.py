"""
parameters.py
=============
Every numeric parameter used by the RAGA evaluation, with the published
source it is derived from. This module exists so that no simulation
constant is a free-floating assumption: each value below is either

  (a) [SOURCED]  taken or arithmetically derived from a cited public source,
  (b) [MODEL]    a modelling choice we make explicitly and sweep in
                 sensitivity analysis, because no public source fixes it.

Nothing else is permitted in this file. If a value cannot be labelled (a) or
(b) with a real citation or an explicit sweep, it does not belong here.

SOURCES (full citations in the paper's bibliography):

  [NSC24]   National Safety Council, "Work Injury Costs," Injury Facts, 2024.
            Cost per death $1,540,000; cost per medically consulted injury
            $48,000; total work injury cost $181.4B.
            https://injuryfacts.nsc.org/work/costs/work-injury-costs/

  [SIE23]   Siemens AG / Senseye, "The True Cost of Downtime 2022," 2023.
            Hourly cost of unplanned downtime: automotive $2,000,000/h;
            oil & gas ~$500,000/h; FMCG/CPG $39,000/h. n=56 interviews with
            large industrial organisations, Jan 2021 - Aug 2022.

  [BLS26]   U.S. Bureau of Labor Statistics, average hourly earnings of all
            employees in manufacturing (series CES3000000003), $36.87/h,
            July 2026, seasonally adjusted.

  [OLS86]   P. L. Olson and M. Sivak, "Perception-response time to unexpected
            roadway hazards," Human Factors, vol. 28, no. 1, pp. 91-96, 1986.
            Surprise on-road hazard: 50th percentile total perception-response
            time ~1.1 s, 95th percentile ~1.6 s (n=49 younger, 16 older).

  [BC96]    Broen and Chiang, brake response to a pedestrian stepping hazard:
            observed total brake response range 0.81-2.44 s.

  [LER93]   N. D. Lerner, "Brake perception-reaction times of older and
            younger drivers," Proc. Human Factors and Ergonomics Society,
            1993. Rolling-barrel hazard: mean PRT 1.5 s, 85th percentile
            1.9 s (n=56 braking subjects).

  [WA09]    IAAF/World Athletics Sprint Start Research Project. The 100 ms
            false-start threshold encodes an assumed minimum human reaction
            time; the commissioned study measured simple auditory reactions
            as fast as 80 ms and recommended lowering the limit to 80-85 ms.

  [IPN20]   G. Benitez-Garcia et al., "IPN Hand: A video dataset and benchmark
            for real-time continuous hand gesture recognition," Proc. ICPR,
            2020. Best isolated-recognition accuracy 86.32% (ResNeXt-101,
            RGB-Flow) over 13 classes on a deliberately realistic dataset;
            best continuous (Levenshtein) accuracy 42.47%.

  [FRTE]    NIST Face Recognition Technology Evaluation (FRTE/FRVT) 1:1 and
            1:N. Reporting operating points are FMR = 1e-4, 1e-5, 1e-6;
            leading 1:N algorithms reach FNMR 0.15% at FMR 1e-3 on galleries
            exceeding 10^7 identities.

  [FR25]    Survey of face recognition accuracy across imaging conditions:
            saturated (>99%) on constrained benchmarks (LFW, CFP-FP);
            IJB-C TAR@FAR=1e-4 approx. 96-98%; IJB-S rank-1 50-73%;
            TinyFace rank-1 64-75% under low resolution.

  [MEIN]    Meinberg, "Time synchronization accuracy with NTP." Well-configured
            NTP against a local stratum-1 source on a LAN holds offsets below
            0.01 ms; offsets remain below 1 ms over a WAN path with 146 ms
            packet delay. PTP (IEEE 1588) with hardware timestamping reaches
            sub-microsecond.
"""

# =========================================================================
# 1. MONETARY BASIS
# =========================================================================
# All costs are in US dollars, so that every entry of the cost matrix is
# auditable against a published figure rather than expressed in arbitrary
# "cost units". This is the single biggest change from the first version of
# this evaluation, which used unit-free values.

# [SOURCED] [NSC24]
COST_PER_WORKPLACE_DEATH = 1_540_000.0
COST_PER_MEDICALLY_CONSULTED_INJURY = 48_000.0

# [SOURCED] [SIE23] hourly cost of unplanned downtime, converted to $/second
DOWNTIME_RATE_PER_S = {
    "automotive": 2_000_000.0 / 3600.0,   # $555.56/s
    "oil_and_gas":  500_000.0 / 3600.0,   # $138.89/s
    "fmcg":          39_000.0 / 3600.0,   # $10.83/s
    "domestic":           0.0,            # a home device has no line behind it
}

# [SOURCED] [BLS26] manufacturing average hourly earnings -> $/second
LABOUR_RATE_PER_S = 36.87 / 3600.0        # $0.010242/s


# =========================================================================
# 2. RECOVERY DURATIONS  (the bridge from downtime rate to cost)
# =========================================================================
# alpha (needless-stop cost) and phi (re-gesture cost) are both downtime
# costs; they differ only in how long the device is unavailable. Expressing
# them this way means the cost matrix has exactly two free durations per
# device rather than two free dollar amounts, and both durations are
# physically interpretable.
#
# [MODEL] These durations are engineering estimates, not published
# measurements, and they are the parameters swept in the E4 sensitivity
# analysis. The distinction that matters most -- fast-resume protective stop
# versus emergency stop requiring manual reset -- is treated as a first-class
# experimental variable in E14 rather than fixed here, because the paper's
# stranger-halt property turns out to depend on it.

T_STOP_S = {
    "lamp":  100.0,   # [MODEL] someone left in the dark until they re-switch
    "oven":  600.0,   # [MODEL] a reheat cycle after a needless shutdown
    "agv":     5.0,   # [MODEL] AGVs stop for obstacles routinely and resume
    "arm":     2.0,   # [MODEL] safety-rated monitored stop, automatic resume
}

# The same robot arm behind an emergency stop that needs a manual reset and
# restart sequence, rather than a safety-rated monitored stop. E14 contrasts
# the two; this is the single parameter that decides whether an unidentified
# bystander may halt the most dangerous machine in the room.
T_STOP_S_ARM_EMERGENCY_STOP = 60.0    # [MODEL]

T_RETRY_S = {
    "lamp":    3.0,   # [MODEL] repeat the gesture
    "oven":   30.0,   # [MODEL] repeat, then fall back to the panel
    "agv":    30.0,   # [MODEL]
    "arm":    30.0,   # [MODEL]
}

DEVICE_SECTOR = {
    "lamp": "domestic",
    "oven": "fmcg",
    "agv":  "fmcg",
    "arm":  "automotive",
}


# =========================================================================
# 3. HARM VALUES (Lambda)
# =========================================================================
# Lambda(h) is the expected harm from failing to arrest a hazardous device.
# For devices whose worst credible outcome is a fatality we take the NSC
# cost per workplace death; for those whose worst credible outcome is a
# medically consulted injury we take that figure. The AGV sits between the
# two and is stated as an explicit severity mixture so the assumption is
# visible rather than buried in a round number.

# [MODEL] probability that an un-halted AGV collision proves fatal rather
# than injurious. Swept in E4.
P_FATAL_AGV = 0.30

HARM = {
    # [MODEL] a lamp cannot injure anyone; a nominal non-zero value keeps the
    # threshold algebra well defined and represents residual nuisance only.
    "lamp": 2.0,
    # [SOURCED] [NSC24] burn requiring medical consultation
    "oven": COST_PER_MEDICALLY_CONSULTED_INJURY,
    # [SOURCED+MODEL] [NSC24] severity mixture, see P_FATAL_AGV
    "agv":  P_FATAL_AGV * COST_PER_WORKPLACE_DEATH
            + (1.0 - P_FATAL_AGV) * COST_PER_MEDICALLY_CONSULTED_INJURY,
    # [SOURCED] [NSC24] a robot arm in a shared workspace can kill
    "arm":  COST_PER_WORKPLACE_DEATH,
}

HAZARD_OF_DEVICE = {"lamp": "LOW", "oven": "MEDIUM", "agv": "HIGH", "arm": "CRITICAL"}
DEVICES_IN_HAZARD_ORDER = ["lamp", "oven", "agv", "arm"]
HAZARD_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def alpha_of(device, t_stop_override=None):
    """Needless-stop cost: recovery duration x that sector's downtime rate.
    The lamp has no production line behind it, so its needless-stop cost is
    the disrupted person's time rather than lost output."""
    t = T_STOP_S[device] if t_stop_override is None else t_stop_override
    rate = DOWNTIME_RATE_PER_S[DEVICE_SECTOR[device]]
    if rate == 0.0:
        return t * LABOUR_RATE_PER_S
    return t * rate


def phi_of(device):
    """Re-gesture cost: the delay a wrongly refused legitimate command costs,
    valued at the same rate as any other unavailability of that device."""
    t = T_RETRY_S[device]
    rate = DOWNTIME_RATE_PER_S[DEVICE_SECTOR[device]]
    if rate == 0.0:
        return t * LABOUR_RATE_PER_S
    return t * rate


def cost_triple(device, t_stop_override=None):
    """(Lambda, alpha, phi) in dollars for one device."""
    return HARM[device], alpha_of(device, t_stop_override), phi_of(device)


def build_cost_table(arm_emergency_stop=False):
    """Cost matrix indexed 0..3 by hazard class, as the experiments expect."""
    out = {}
    for i, dev in enumerate(DEVICES_IN_HAZARD_ORDER):
        override = (T_STOP_S_ARM_EMERGENCY_STOP
                    if (arm_emergency_stop and dev == "arm") else None)
        out[i] = cost_triple(dev, override)
    return out


# =========================================================================
# 4. IDENTITY CONFIDENCE
# =========================================================================
# c_floor is the confidence assigned on open-set rejection. We set it at the
# NIST 1:N reporting operating point of FMR = 1e-3, where leading algorithms
# achieve FNMR 0.15% on galleries above 10^7 identities [FRTE]. Tighter
# operating points (1e-4 ... 1e-6) are also standard and are swept in E4,
# because Corollary 2's stranger-halt condition scales as 1/c_floor.
C_FLOOR = 1e-3                                   # [SOURCED] [FRTE]
C_FLOOR_SWEEP = [1e-3, 1e-4, 1e-5, 1e-6]         # [SOURCED] [FRTE]

# Operator confidence is modelled as two sub-populations, reflecting the gap
# between constrained and unconstrained face recognition [FR25]: a
# well-imaged operator facing the sensor, and one imaged at distance, in
# motion, oblique, or wearing PPE. The industrial floor produces both.
#
# [SOURCED-derived] [FR25] good conditions: IJB-C TAR@FAR=1e-4 ~ 96-98%,
# lifted by fusion with a worn IMU and a credential (Tier 1 requires two
# modalities that fail differently), giving a posterior concentrated just
# below 1.
OPERATOR_GOOD_MEAN, OPERATOR_GOOD_SD = 0.995, 0.004
# [SOURCED-derived] [FR25] degraded conditions: IJB-S rank-1 50-73%,
# TinyFace rank-1 64-75%. Wide and substantially lower.
OPERATOR_DEGRADED_MEAN, OPERATOR_DEGRADED_SD = 0.88, 0.08
# [MODEL] impostor who partially defeats one modality.
IMPOSTOR_MEAN, IMPOSTOR_SD = 0.60, 0.25

# [MODEL] population mix. No public source fixes the composition of gesture
# traffic on a factory floor; we state it and sweep the stranger fraction in
# E4. Most events come from enrolled staff; visitors and contractors are a
# minority; a small impostor population is included so security failures are
# measurable at all.
POP_OPERATOR_GOOD = 0.55
POP_OPERATOR_DEGRADED = 0.25
POP_STRANGER = 0.15
POP_IMPOSTOR = 0.05

# [MODEL] second attester's confidence for the quorum experiments.
SECOND_ATTESTER_MEAN, SECOND_ATTESTER_SD = 0.98, 0.03


# =========================================================================
# 5. GESTURE POLARITY MISCLASSIFICATION
# =========================================================================
# The paper's headline sensitivity axis. Anchored, not guessed:
#   - constrained laboratory datasets are saturated above 99% [FR25-analogue]
#   - the best model on IPN Hand, a deliberately realistic 13-class dataset,
#     reaches 86.32% isolated accuracy, i.e. 13.7% overall error [IPN20]
#   - the same benchmark's continuous (Levenshtein) accuracy is 42.47% [IPN20]
# A binary restrictive/permissive decision over a well-separated vocabulary
# is easier than 13-way classification, so the polarity confusion rate sits
# below the 13.7% overall figure but well above the 2% the first version of
# this evaluation assumed. We therefore sweep the whole interval and mark the
# realistic region rather than committing to a single value.
GESTURE_ERROR_SWEEP = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]   # [SOURCED] [IPN20]
# [SOURCED] [IPN20] upper anchor: 100% - 86.32%
IPN_HAND_OVERALL_ERROR = 0.1368
# [MODEL] primary operating point for tables that need a single value. Chosen
# at 5%: above the clean-laboratory regime, below IPN Hand's 13.7% 13-way
# error, on the argument that a two-class polarity decision is easier.
GESTURE_ERROR_PRIMARY = 0.05


# =========================================================================
# 6. EVENT-REACTION LATENCY COUPLING (ERLC)
# =========================================================================
# rho_min: below this, a "reaction" is faster than human physiology allows
# and is therefore an injection, not heroism. World Athletics encodes exactly
# this judgement in the 100 ms false-start threshold; the commissioned study
# measured simple auditory reactions as fast as 80 ms [WA09]. We take the
# codified 100 ms rather than the 80 ms outlier, which is the conservative
# choice: it withholds leniency from fewer genuine reactions.
RHO_MIN_S = 0.100                                  # [SOURCED] [WA09]

# rho_max: above this a gesture is no longer plausibly a reaction to the
# event. Broen and Chiang observed total brake responses out to 2.44 s [BC96];
# Lerner reports an 85th percentile of 1.9 s [LER93]. We take 2.5 s.
RHO_MAX_S = 2.5                                    # [SOURCED] [BC96, LER93]

# The genuine-reflex distribution is fitted to Olson and Sivak's surprise-
# hazard percentiles [OLS86]: median 1.1 s and 95th percentile 1.6 s. For a
# log-normal, mu = ln(1.1) and sigma = (ln 1.6 - ln 1.1)/1.645 = 0.2278.
REFLEX_MEDIAN_S = 1.1                              # [SOURCED] [OLS86]
REFLEX_P95_S = 1.6                                 # [SOURCED] [OLS86]
import math as _math
REFLEX_LOG_MU = _math.log(REFLEX_MEDIAN_S)
REFLEX_LOG_SIGMA = (_math.log(REFLEX_P95_S) - _math.log(REFLEX_MEDIAN_S)) / 1.645

# [MODEL] reflex-band leniency multiplier, "epsilon" in Eq. (4). Set to
# c_floor so that inside the reaction band the restrictive threshold falls
# exactly to the open-set floor: in the band, an unidentified person is as
# able to halt the device as the confidence floor permits, and no more.
G_REFLEX = C_FLOOR


# =========================================================================
# 7. CLOCK SKEW
# =========================================================================
# Sweep anchored on measured synchronisation regimes [MEIN]:
#   PTP hardware timestamping   sub-microsecond
#   NTP, LAN, local stratum-1   < 0.01 ms
#   NTP over a WAN path         < 1 ms
#   heterogeneous consumer IoT  tens of ms and upward, no dedicated sync
CLOCK_SKEW_SWEEP_MS = [0, 0.001, 0.01, 1, 10, 50, 100, 500]   # [SOURCED] [MEIN]
