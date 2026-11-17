#!/bin/sh
# Interim step of attempt 4 (records/l2_attempt4_plan.md) at iteration IT, both arms: waits for the evaluation line of IT in both train.logs, then on the
# CPU at low priority: deterministic phase success of both checkpoints (records/phase_success_curve.json), the group summary (summary.png) and median-cost
# videos (MP4 with GIF preview) from the original start of both arms at the top of the group folder; prints the phase success table.
# usage: sh script/interim_a4.sh IT      (run from the repository root)
IT=$1; P=$(printf %04d $IT)
for A in N I; do
  until RD=$(python -c "from pairl import paths; print(paths.run_dir('l2_a4_$A'))" 2>/dev/null) && grep -q "EVAL it $IT:" "$RD/train.log" 2>/dev/null; do sleep 60; done
done
sleep 20; RN=$(python -c "from pairl import paths; print(paths.run_dir('l2_a4_N'))"); RI=$(python -c "from pairl import paths; print(paths.run_dir('l2_a4_I'))"); GD=$(dirname "$RN"); TS=$(basename "$GD")
CPU="env JAX_PLATFORMS=cpu sh script/safe_run.sh mem=10G nice -n 15 python script/parallel_run.py workers=8"
$CPU eval/diagnostics/phase_success.py records/phase_success_curve.json "$RN:$IT" "$RI:$IT" modes=deterministic > "$GD/interim_phase_it$P.log" 2>&1
JAX_PLATFORMS=cpu sh script/safe_run.sh mem=4G nice -n 15 python eval/figures/group_summary.py "$TS" > /dev/null 2>&1
for R in "$RN" "$RI"; do $CPU eval/figures/episode_video.py "$GD" "$R:$IT" starts=0 only=median_cost >> "$GD/interim_videos.log" 2>&1; done
python - "$IT" <<'PY'
import sys, json
from pairl import paths
it = sys.argv[1]; C = json.load(open(paths.RECORDS / 'phase_success_curve.json')); L1 = json.load(open(paths.RECORDS / 'phase_success.json'))['L1_reference']['deterministic']
rows = [('P1', 'alone', 'P1'), ('P2', 'alone', 'P2'), ('P3', 'alone', 'P3'), ("P3'", 'alone', 'P3c'), ('P4', 'alone', 'P4'), ('full chain', 'chain', 'P1234')]
print(f'| it {it} | L1 reference | N | I |'); print('|---|---|---|---|')
for n, g, q in rows: print(f"| {n} | {L1[g][q]['rate']:.3f} | {C['l2_a4_N:' + it]['deterministic'][g][q]['rate']:.3f} | {C['l2_a4_I:' + it]['deterministic'][g][q]['rate']:.3f} |")
PY
echo INTERIM_A4 "$IT"
