#!/usr/bin/env python3
"""
device_agent.py -- Tier 3 "device agent" node (Algorithm 2 in the paper).

Runs independently at one device (e.g. the LOW lamp or the CRITICAL robot
arm in the paper's testbed trial). Listens on a UDP port for signed
authority envelopes broadcast by generator.py, and on a second UDP port for
local hazard-event triggers (see trigger_hazard_event.py), which stamp this
device's own event timeline for Event-Reaction Latency Coupling (ERLC).
No inter-agent messaging is used or required (Algorithm 2's per-device
pending set P_d is maintained locally; convergence follows from every agent
running identical logic over the identical broadcast envelope, per Sec. V-C).

Every decision is appended to --log-file as one JSON line with the same
fields Algorithm 2 says to log: EXECUTE/WITHHOLD, T, rho, and rationale.

Usage (reproduce the paper's testbed trial -- run this once per device):

    # Terminal 1: the robot arm (CRITICAL)
    python3 device_agent.py --device-id arm --hazard-class CRITICAL \\
        --bind-port 9101 --control-port 9201 \\
        --secret raga-testbed-demo-secret --log-file arm_decisions.jsonl

    # Terminal 2: the lamp (LOW)
    python3 device_agent.py --device-id lamp --hazard-class LOW \\
        --bind-port 9102 --control-port 9202 \\
        --secret raga-testbed-demo-secret --log-file lamp_decisions.jsonl

Then run generator.py (see its --help) to send envelopes to both, and
trigger_hazard_event.py to test ERLC. See run_local_demo.sh for a scripted
end-to-end run of all three of the paper's testbed observations.

Pure standard library -- runs unmodified under Termux's stock python3.
"""

import argparse
import json
import socket
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common  # noqa: E402


class DeviceAgent:
    def __init__(self, device_id, hazard_class, secret, window, log_file):
        self.device_id = device_id
        self.hazard_class = hazard_class
        self.secret = secret
        self.window = window
        self.log_file = log_file
        self.hazard_timeline = []          # H_d: own-clock hazard-event timestamps
        self.pending = []                  # P_d: recent unexpired envelopes (for R1 arbitration)
        self.lock = threading.Lock()

    # -- Algorithm 2, adapted for a single-device, message-free testbed -----
    def decide(self, envelope):
        t_recv = common.now()

        if not common.verify_envelope(envelope, self.secret):
            return self._finish(envelope, "WITHHOLD", None, None, "DENY(sig)")

        scope = envelope["scope"]
        if scope != "*" and self.device_id not in scope:
            return None  # outside spatial scope: ignore silently, per Algorithm 2 line 5

        age = t_recv - envelope["t_g"]
        window = self.window.get(envelope["polarity"], self.window["RES"])
        if age > window:
            return self._finish(envelope, "WITHHOLD", None, age,
                                 f"discard: stale envelope (age={age:.3f}s > window={window}s)")

        with self.lock:
            rho = (t_recv - max(self.hazard_timeline)) if self.hazard_timeline else None

            # polarity-first arbitration (R1): a RES envelope in-flight dominates
            # any concurrently pending PERM envelope at this device, and vice
            # versa is never true -- restrictive always wins.
            self._expire_pending(t_recv, window)
            for other in self.pending:
                if other["polarity"] != envelope["polarity"] and envelope["polarity"] == "PERM":
                    return self._finish(envelope, "WITHHOLD", None, rho,
                                        "dominated: concurrent RES envelope takes precedence (R1)")
            self.pending.append(envelope)

        T = common.effective_threshold(envelope["polarity"], self.hazard_class, rho)

        c_eff = envelope["c"]
        if envelope.get("c_second") is not None:
            # Conjunctive quorum (paper's corrected design, Sec. VI, E13):
            # each attester must independently clear T -- NOT noisy-OR fusion.
            # NOTE: the paper's own three-phone trial tested only the
            # inversion and ERLC, not quorum; this branch is an extension for
            # anyone who wants to exercise quorum on physical hardware too.
            c_eff = min(envelope["c"], envelope["c_second"])

        outcome = "EXECUTE" if c_eff >= T else "WITHHOLD"
        rationale = (f"c_eff={c_eff:.6f} {'>=' if outcome=='EXECUTE' else '<'} "
                     f"T_eff={T:.6f} (hazard={self.hazard_class}, polarity={envelope['polarity']})")
        return self._finish(envelope, outcome, T, rho, rationale)

    def _expire_pending(self, t_now, window):
        self.pending = [e for e in self.pending if (t_now - e["t_g"]) <= window]

    def _finish(self, envelope, outcome, T, rho, rationale):
        record = {
            "ts": common.now(),
            "device_id": self.device_id,
            "hazard_class": self.hazard_class,
            "envelope_id": envelope.get("envelope_id"),
            "actor_id": envelope.get("actor_id"),
            "polarity": envelope.get("polarity"),
            "c": envelope.get("c"),
            "T": T,
            "rho": rho,
            "outcome": outcome,
            "rationale": rationale,
        }
        line = json.dumps(record)
        print(f"[{self.device_id}] {outcome}  {rationale}")
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")
        return record

    def register_hazard_event(self):
        with self.lock:
            self.hazard_timeline.append(common.now())
        print(f"[{self.device_id}] hazard event registered at t={self.hazard_timeline[-1]:.3f}")


def envelope_listener(agent, bind_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", bind_port))
    print(f"[{agent.device_id}] listening for envelopes on UDP :{bind_port}")
    while True:
        data, _addr = sock.recvfrom(65536)
        try:
            envelope = json.loads(data.decode())
        except (ValueError, UnicodeDecodeError):
            continue
        agent.decide(envelope)


def control_listener(agent, control_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", control_port))
    print(f"[{agent.device_id}] listening for hazard-event triggers on UDP :{control_port}")
    while True:
        data, _addr = sock.recvfrom(4096)
        if data.strip() == b"HAZARD_EVENT":
            agent.register_hazard_event()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--hazard-class", required=True, choices=common.HAZARD_CLASSES)
    ap.add_argument("--bind-port", type=int, required=True,
                     help="UDP port to receive authority envelopes on")
    ap.add_argument("--control-port", type=int, required=True,
                     help="UDP port to receive local hazard-event triggers on (ERLC)")
    ap.add_argument("--secret", required=True)
    ap.add_argument("--res-window", type=float, default=common.DEFAULT_WINDOW["RES"])
    ap.add_argument("--perm-window", type=float, default=common.DEFAULT_WINDOW["PERM"])
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    secret = common.load_secret(args.secret)
    window = {"RES": args.res_window, "PERM": args.perm_window}
    agent = DeviceAgent(args.device_id, args.hazard_class, secret, window, args.log_file)

    t1 = threading.Thread(target=envelope_listener, args=(agent, args.bind_port), daemon=True)
    t2 = threading.Thread(target=control_listener, args=(agent, args.control_port), daemon=True)
    t1.start()
    t2.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{args.device_id}] shutting down")


if __name__ == "__main__":
    main()
