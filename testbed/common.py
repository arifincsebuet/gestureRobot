"""
common.py -- shared config, threshold math, and envelope crypto for the
physical RAGA testbed described in the paper (Sec. VI, E6-E7):

    "We implemented the decision logic as a stdlib-only Python node on three
    Android phones over UDP with HMAC-SHA256-signed envelopes."

The three nodes are one generator and two device agents (CRITICAL robot,
LOW lamp).

This module is imported by generator.py and device_agent.py. It is pure
standard library (socket, hmac, hashlib, json, time) so it runs unmodified
under Termux's stock python3 on an Android phone, matching the paper's
"stdlib-only Python node" description -- no pip install needed on the
phones themselves.

Implements exactly the two algorithms from the paper (Sec. V-F):
    T_res, T_perm            -- Eqs. (2)-(3), the coupled inversion
    g_res, g_perm            -- Eq. (4), the ERLC threshold modulation
    sign_envelope / verify_envelope -- Algorithm 1 line 16 / Algorithm 2 line 1
"""

import hashlib
import hmac
import json
import time

# ---------------------------------------------------------------------------
# Cost parameters and hazard classes -- Table I of the paper (c_floor = 1e-3).
#
# These mirror ../parameters.py, which carries the citation for every value:
# Lambda from NSC work-injury costs, alpha and phi from recovery durations
# priced at Siemens sector downtime rates and the BLS manufacturing wage.
# They are restated literally here, rather than imported, so that this
# directory stays runnable on a phone with nothing but the standard library
# and no dependency on the simulation package. If parameters.py changes,
# change these too -- test_matches_parameters.py checks that they agree.
# ---------------------------------------------------------------------------
C_FLOOR = 1e-3
HAZARD_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# (Lambda: harm, alpha: needless-stop cost, phi: re-gesture cost), all in USD
COST_TABLE = {
    "LOW":      (2.0,       1.024166,   0.030725),   # lamp
    "MEDIUM":   (48000.0,   6500.0,     325.0),      # oven
    "HIGH":     (495600.0,  54.166667,  325.0),      # AGV
    "CRITICAL": (1540000.0, 1111.111111, 16666.666667),  # robot arm, SRMS
}

# The same robot arm behind an emergency stop needing a manual reset (60 s of
# automotive line time) rather than a safety-rated monitored stop (2 s). E14
# shows this single choice decides whether an unidentified bystander may halt
# it at all: at 60 s the restrictive threshold rises above c_floor.
COST_CRITICAL_EMERGENCY_STOP = (1540000.0, 33333.333333, 16666.666667)

# ERLC reaction band, seconds, fitted to published human data rather than
# assumed (see ../parameters.py section 6 for the citations):
#   rho_min = 0.100 s -- the World Athletics false-start threshold, which
#     codifies the fastest credible human reaction; below it a "reaction" is
#     an injection, not heroism.
#   rho_max = 2.5 s -- the upper end of observed surprise-hazard brake
#     responses (Broen and Chiang observed out to 2.44 s).
# Genuine reflexes are log-normal with median 1.1 s and 95th percentile 1.6 s
# (Olson and Sivak), so 99.98% of them fall inside this band.
RHO_MIN = 0.100
RHO_MAX = 2.5
# "epsilon" in Eq. (4): the reflex-band leniency multiplier. Back-computed
# from the paper's own reported testbed measurement (Sec. VI, E6-E7):
# the lamp's halt threshold drops from T_res=0.33333 (no event) to
# the reported T=0.00033 once rho=0.400s falls in the reflex band, i.e.
# epsilon = 0.00033 / 0.33333 = 0.00099 = c_floor to two significant figures.
# Using c_floor itself is also the natural reading: in the reflex band, the
# restrictive threshold falls all the way to the open-set confidence floor,
# i.e. to the same floor an unidentified stranger already reports.
G_REFLEX = C_FLOOR

# Illustrative staleness bounds (TemporalWindow(pi,h) in Algorithm 2), seconds.
# The paper states restrictive commands get a longer window than permissive
# at the same hazard class (Sec. V-F) but does not fix numeric values -- it
# explicitly says of this component "we neither evaluate it separately nor
# claim it as a contribution". These defaults just need to be longer than
# one UDP round
# trip and are safe to override with --window on the command line.
DEFAULT_WINDOW = {"RES": 5.0, "PERM": 2.0}


def _costs(hazard_class, emergency_stop=False):
    if emergency_stop and hazard_class == "CRITICAL":
        return COST_CRITICAL_EMERGENCY_STOP
    return COST_TABLE[hazard_class]


def t_res(hazard_class, emergency_stop=False):
    """Eq. (2): restrictive (halt) threshold."""
    L, a, _phi = _costs(hazard_class, emergency_stop)
    return a / (a + L)


def t_perm(hazard_class, emergency_stop=False):
    """Eq. (3): permissive (start) threshold."""
    L, _a, phi = _costs(hazard_class, emergency_stop)
    return L / (L + phi)


def g_mod(polarity, rho):
    """
    Eq. (4): Event-Reaction Latency Coupling threshold modulation.
    rho = t_arrival - t_event, both read from the device's own clock.
    rho == None means "no hazard event on this device's timeline" -> treated
    as out-of-band (no leniency), matching Algorithm 2's rho computed from
    max(H_d) with an empty timeline.
    """
    if rho is None:
        return 1.0
    if polarity == "RES":
        if rho < RHO_MIN:
            return 1.0          # faster than any human: injection, not heroism
        if rho <= RHO_MAX:
            return G_REFLEX      # reflex band: lenient
        return 1.0               # no longer reactive
    else:  # PERM: symmetric, elevated (i.e. MORE strict) for small rho
        if rho < RHO_MIN:
            return 1.0 / G_REFLEX
        if rho <= RHO_MAX:
            return 1.0 / G_REFLEX
        return 1.0


def effective_threshold(polarity, hazard_class, rho, emergency_stop=False):
    T = (t_res(hazard_class, emergency_stop) if polarity == "RES"
         else t_perm(hazard_class, emergency_stop))
    g = g_mod(polarity, rho)
    return min(T * g, 1.0)


# ---------------------------------------------------------------------------
# Authority envelope: sign / verify (HMAC-SHA256, per Algorithm 1 line 16 /
# Algorithm 2 line 1). The shared secret plays the role of the generator's
# signing key sk_generator; every device agent holds the same secret as its
# trust anchor (Tier 3 "trust anchors" in Sec. IV-A). For a real deployment
# this would be per-generator asymmetric signing; a shared HMAC secret is
# what the paper's testbed actually used and is sufficient to validate the
# *decision logic*, which is what the testbed is for (Sec. VI-B: "the
# testbed validates distributed decision behaviour, not sensing").
# ---------------------------------------------------------------------------

ENVELOPE_FIELDS = ["actor_id", "role", "c", "c_second", "polarity", "gesture",
                   "scope", "t_g", "envelope_id"]


def _canonical(payload):
    body = {k: payload[k] for k in ENVELOPE_FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign_envelope(payload, secret):
    """payload must contain all of ENVELOPE_FIELDS. Returns payload + 'sig'."""
    mac = hmac.new(secret, _canonical(payload), hashlib.sha256).hexdigest()
    out = dict(payload)
    out["sig"] = mac
    return out


def verify_envelope(envelope, secret):
    """Returns True iff the HMAC signature is valid (Algorithm 2, DENY(sig))."""
    if not all(k in envelope for k in ENVELOPE_FIELDS) or "sig" not in envelope:
        return False
    expected = hmac.new(secret, _canonical(envelope), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, envelope["sig"])


def now():
    return time.time()


def load_secret(secret_arg):
    """--secret takes either a literal string or @/path/to/file."""
    if secret_arg.startswith("@"):
        with open(secret_arg[1:], "rb") as f:
            return f.read().strip()
    return secret_arg.encode()
