#!/usr/bin/env python3
"""
generator.py -- Tier 2 "envelope generator" node (Algorithm 1 in the paper).

Runs once per gesture: builds a signed authority envelope
    <actor_id, role, c, polarity, gesture, scope, t_g, sig>
and broadcasts it via UDP to every configured device agent. This is the node
the paper describes running on one of the three Android/Termux phones.

Identity confidence c is INJECTED on the command line rather than computed
from camera/IMU fusion -- the paper is explicit that the testbed validates
distributed decision behaviour, not sensing (Sec. VI, Threats to Validity):
"identity confidence is injected, with no gesture recognition in the loop."
That is also why there is no FuseIdentity/LivenessCorrelate step here:
Algorithm 1's sensing-fusion lines (3, 10-13) are the sensing pipeline this
testbed intentionally does not implement.

Usage (reproduce the paper's testbed trial: unenrolled visitor halts the
robot arm but not the lamp):

    python3 generator.py --agents 127.0.0.1:9101,127.0.0.1:9102 \\
        --secret raga-testbed-demo-secret \\
        --actor-id visitor-1 --role stranger --c 0.001 \\
        --polarity RES --gesture halt --scope '*'

Run with --help for all options. Pure standard library (socket, hmac, json,
argparse) -- no pip install required, matching the paper's "single
stdlib-only Python node" description.
"""

import argparse
import socket
import sys
import uuid

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common  # noqa: E402


def parse_agents(s):
    out = []
    for part in s.split(","):
        host, port = part.rsplit(":", 1)
        out.append((host, int(port)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", required=True,
                     help="comma-separated host:port list of device agents to broadcast to")
    ap.add_argument("--secret", required=True,
                     help="shared HMAC secret; literal string, or @path/to/file")
    ap.add_argument("--actor-id", required=True)
    ap.add_argument("--role", default="stranger",
                     choices=["operator", "stranger", "impostor"])
    ap.add_argument("--c", type=float, required=True,
                     help="identity confidence in [c_floor, 1] (injected, per Threats to Validity)")
    ap.add_argument("--c-second", type=float, default=None,
                     help="optional second attester's confidence, for conjunctive-quorum "
                          "testing (Sec. VI, E13). Not part of the paper's own three-phone "
                          "trial, which tested only the inversion and ERLC; included here as "
                          "an extension for anyone extending the physical testbed to quorum.")
    ap.add_argument("--polarity", required=True, choices=["RES", "PERM"])
    ap.add_argument("--gesture", default="halt",
                     help="descriptive gesture label only; polarity (above) drives the decision")
    ap.add_argument("--scope", default="*",
                     help="comma-separated device ids in the spatial scope cone, or '*' for all")
    ap.add_argument("--envelope-id", default=None)
    args = ap.parse_args()

    secret = common.load_secret(args.secret)
    agents = parse_agents(args.agents)
    scope = "*" if args.scope == "*" else args.scope.split(",")

    payload = {
        "actor_id": args.actor_id,
        "role": args.role,
        "c": max(common.C_FLOOR, min(1.0, args.c)),
        "c_second": args.c_second,
        "polarity": args.polarity,
        "gesture": args.gesture,
        "scope": scope,
        "t_g": common.now(),
        "envelope_id": args.envelope_id or str(uuid.uuid4()),
    }
    envelope = common.sign_envelope(payload, secret)
    body = __import__("json").dumps(envelope).encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    for host, port in agents:
        sock.sendto(body, (host, port))
        sent += 1
    sock.close()

    print(f"[generator] broadcast envelope {payload['envelope_id']} "
          f"(actor={args.actor_id} role={args.role} c={payload['c']:.6f} "
          f"polarity={args.polarity} scope={args.scope}) to {sent} agent(s)")


if __name__ == "__main__":
    main()
