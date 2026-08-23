#!/usr/bin/env bash
# run_local_demo.sh -- reproduce the paper's three-node testbed observations
# (Sec. VI, E6-E7) on a single machine over localhost UDP,
# for reviewers without three physical phones. On real hardware, replace
# 127.0.0.1 with each phone's LAN address and run generator.py / device_agent.py
# on separate devices exactly as documented in their --help text.
#
# Reproduces:
#   (1) an unenrolled visitor (c=0.001) HALTs the CRITICAL robot arm -> EXECUTE
#   (2) the same visitor cannot HALT the LOW lamp -> WITHHOLD
#   (3) ERLC: the same HALT on the same lamp, 0.4s after a hazard event on
#       that lamp's own timeline, falls in the reflex band -> EXECUTE
#   (4) E14: the same HALT on a CRITICAL arm whose needless-stop cost is that
#       of an emergency stop with manual reset -> WITHHOLD. The stranger-halt
#       property is purchased with restart speed, not granted by the policy.
#
# Usage: bash run_local_demo.sh

set -euo pipefail
cd "$(dirname "$0")"

SECRET="raga-testbed-demo-secret"
rm -f arm_decisions.jsonl lamp_decisions.jsonl arm_estop_decisions.jsonl

echo "=== starting device agents ==="
python3 device_agent.py --device-id arm  --hazard-class CRITICAL \
    --bind-port 9101 --control-port 9201 \
    --secret "$SECRET" --log-file arm_decisions.jsonl &
ARM_PID=$!
python3 device_agent.py --device-id lamp --hazard-class LOW \
    --bind-port 9102 --control-port 9202 \
    --secret "$SECRET" --log-file lamp_decisions.jsonl &
LAMP_PID=$!
trap 'kill $ARM_PID $LAMP_PID 2>/dev/null || true' EXIT
sleep 0.5

echo
echo "=== (1) unenrolled visitor HALTs the robot arm (expect EXECUTE) ==="
python3 generator.py --agents 127.0.0.1:9101 --secret "$SECRET" \
    --actor-id visitor-1 --role stranger --c 0.001 --polarity RES --gesture halt --scope '*'
sleep 0.3

echo
echo "=== (2) same visitor HALTs the lamp, no prior hazard event (expect WITHHOLD) ==="
python3 generator.py --agents 127.0.0.1:9102 --secret "$SECRET" \
    --actor-id visitor-1 --role stranger --c 0.001 --polarity RES --gesture halt --scope '*'
sleep 0.3

echo
echo "=== (3a) register a hazard event on the lamp's own timeline ==="
python3 trigger_hazard_event.py --host 127.0.0.1 --port 9202
echo "    (waiting 0.4s, matching the paper's measured rho=0.400s)"
sleep 0.4

echo "=== (3b) same visitor HALTs the lamp again, now in the reflex band (expect EXECUTE) ==="
python3 generator.py --agents 127.0.0.1:9102 --secret "$SECRET" \
    --actor-id visitor-1 --role stranger --c 0.001 --polarity RES --gesture halt --scope '*'
sleep 0.3

echo
echo "=== (4) E14: the SAME arm behind an emergency stop needing a manual reset ==="
echo "    (a needless stop now costs 60s of line time instead of 2s)"
python3 device_agent.py --device-id arm-estop --hazard-class CRITICAL \
    --emergency-stop --bind-port 9103 --control-port 9203 \
    --secret "$SECRET" --log-file arm_estop_decisions.jsonl &
ESTOP_PID=$!
trap 'kill $ARM_PID $LAMP_PID $ESTOP_PID 2>/dev/null || true' EXIT
sleep 0.5
python3 generator.py --agents 127.0.0.1:9103 --secret "$SECRET" \
    --actor-id visitor-1 --role stranger --c 0.001 --polarity RES --gesture halt --scope '*'
sleep 0.3
echo "    ^ the identical gesture on the identical hazard class is now REFUSED:"
echo "      bystander halt authority was bought with restart speed, not with policy."

echo
echo "=== decision logs ==="
echo "--- arm_decisions.jsonl ---"
cat arm_decisions.jsonl
echo "--- lamp_decisions.jsonl ---"
cat lamp_decisions.jsonl
echo "--- arm_estop_decisions.jsonl ---"
cat arm_estop_decisions.jsonl
