"""
common.py -- shared config, threshold math, and envelope crypto for the
physical RAGA testbed described in the paper (Sec. VI, "Physical Testbed"):

    "a physical three-node testbed over UDP with HMAC-signed envelopes...
    one generator, two device agents: CRITICAL robot, LOW lamp."

This module is imported by generator.py and device_agent.py. It is pure
standard library (socket, hmac, hashlib, json, time) so it runs unmodified
under Termux's stock python3 on an Android phone, matching the paper's
"single stdlib-only Python node" description -- no pip install needed on
the phones themselves.

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
# Cost parameters and hazard classes -- Table I of the paper (c_floor = 1e-3)
# ---------------------------------------------------------------------------
C_FLOOR = 1e-3
HAZARD_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# (Lambda: harm, alpha: needless-stop cost, phi: re-gesture cost)
COST_TABLE = {
    "LOW":      (2,       1,   20),   # lamp
    "MEDIUM":   (200,     5,   20),   # oven
    "HIGH":     (100000,  50,  20),   # AGV
    "CRITICAL": (2000000, 200, 20),   # robot arm
}

# ERLC reaction band, seconds. Not independently measured on hardware in this
# paper (Sec. VII, Future Work: "estimating the reaction band [rho_min,
# rho_max]..." is future work); these are the same illustrative values used
# by the E5 Monte Carlo experiment in raga_experiments.py, reused here so the
# testbed and simulation agree. The paper's one physical measurement,
# rho = 0.400s "in band" (Sec. VI, Physical Testbed), is consistent with them.
RHO_MIN = 0.15
RHO_MAX = 1.0
# "epsilon" in Eq. (4): the reflex-band leniency multiplier. Back-computed
# from the paper's own reported testbed measurement (Sec. VI, Physical
# Testbed): the lamp's halt threshold drops from T_res=0.33333 (no event) to
# the reported T=0.00033 once rho=0.400s falls in the reflex band, i.e.
# epsilon = 0.00033 / 0.33333 = 0.00099 = c_floor to two significant figures.
# Using c_floor itself is also the natural reading: in the reflex band, the
# restrictive threshold falls all the way to the open-set confidence floor,
# i.e. to the same floor an unidentified stranger already reports.
G_REFLEX = C_FLOOR

# Illustrative staleness bounds (TemporalWindow(pi,h) in Algorithm 2), seconds.
# The paper states restrictive commands get a longer window than permissive
# at the same hazard class (Sec. V-F) but does not fix numeric values -- it
# explicitly says this component "is included so the procedure is stated
# completely" and is neither evaluated separately nor claimed as a
# contribution. These defaults just need to be longer than one UDP round
# trip and are safe to override with --window on the command line.
DEFAULT_WINDOW = {"RES": 5.0, "PERM": 2.0}


def t_res(hazard_class):
    """Eq. (2): restrictive (halt) threshold."""
    L, a, _phi = COST_TABLE[hazard_class]
    return a / (a + L)


def t_perm(hazard_class):
    """Eq. (3): permissive (start) threshold."""
    L, _a, phi = COST_TABLE[hazard_class]
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


def effective_threshold(polarity, hazard_class, rho):
    T = t_res(hazard_class) if polarity == "RES" else t_perm(hazard_class)
    g = g_mod(polarity, rho)
    return min(T * g, 1.0)


# ---------------------------------------------------------------------------
# Authority envelope: sign / verify (HMAC-SHA256, per Algorithm 1 line 16 /
# Algorithm 2 line 1). The shared secret plays the role of the generator's
# signing key sk_generator; every device agent holds the same secret as its
# trust anchor (Tier 3 "trust anchors" in Sec. IV-A). For a real deployment
# this would be per-generator asymmetric signing; a shared HMAC secret is
# what the paper's testbed actually used and is sufficient to validate the
# *decision logic*, which is what the testbed is for (Sec. VI, Threats to
# Validity: "The testbed validates distributed decision behaviour, not
# sensing").
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
