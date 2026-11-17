#!/bin/bash
start=$(date +%s)
for seed in 0 1 2; do for arm in full no_memory no_ledger no_calibration single_return torque_policy; do
  [ -f ../records/m2_${arm}_${seed}.json ] && continue
  now=$(date +%s); [ $((now - start)) -gt 225 ] && { echo "stop for time"; exit 0; }
  python3 m2_stage_a.py $arm $seed 300 '{"NP": 16, "NT": 8, "lrb": 0.02, "sb": 0.05}' m2 2>&1 | grep -v Warning | grep done
done; done
echo ALLDONE
