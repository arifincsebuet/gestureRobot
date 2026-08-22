#!/usr/bin/env python3
"""
trigger_hazard_event.py -- stamp a device agent's own hazard-event timeline.

In the real deployment this fires automatically when a device agent's own
sensors detect a hazard-relevant transition (Sec. V-D: "each device
maintains an event timeline of hazard-relevant transitions it detected,
stamped on its own clock"). On the physical testbed, where sensing is out
of scope (Threats to Validity), this script stands in for that detection:
it sends a one-word UDP datagram to a device agent's --control-port, which
appends a timestamp (on THAT agent's own clock) to its hazard timeline.

Usage:
    python3 trigger_hazard_event.py --host 127.0.0.1 --port 9202
"""

import argparse
import socket


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"HAZARD_EVENT", (args.host, args.port))
    sock.close()
    print(f"sent HAZARD_EVENT to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
