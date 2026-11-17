#!/bin/sh
# Checkpoint report of an L2 run: waits for the evaluation line of iteration IT, then dissection, learning-curve figure, animations from the three
# starts; with signal=1 also restart behavior and the learning-signal checks. Run from the repository root.
# usage: sh script/report_checkpoint.sh RUN_NAME IT PRE_RUN_NAME FIG_SCRIPT [signal=1]
#   example (attempt 3): sh script/report_checkpoint.sh l2_attempt3 300 l2_pilotT_T fig_l2_attempt3.py signal=1
RUN=$1; IT=$2; PRE=$3; FIG=$4; SIG=$5; R=records; F=docs/figures; G=docs/figures/rollouts; P=$(printf "%04d" $IT)
rd() { python -c "from pairl import paths; print(paths.run_dir('$1'))"; }      # run directory by name (data/runs/<group>/<run>)
until [ -n "$RD" ]; do RD=$(rd $RUN 2>/dev/null) || RD=; [ -n "$RD" ] || sleep 20; done
PD=$(rd $PRE)
mkdir -p $G
until grep -q "EVAL it $IT:" $RD/train.log; do sleep 20; done
sleep 20
python eval/diagnostics/l2_dissect.py $RD $IT cost=4 > $RD/dissect_it$P.log 2>&1
python eval/figures/$FIG $RD $PD $F/$RUN.png $R/l1_summary.json
for st in 0 3 2; do python eval/figures/anim_rollout.py $RD $IT $G/${RUN}_start${st}_it$P.gif cost=4 start=$st > $RD/anim_start${st}_it$P.log 2>&1; done
if [ "$SIG" = "signal=1" ]; then
  python eval/diagnostics/l2_restart_behavior.py $R/l2_restart_behavior.json $RD:$IT cost=4 >> $R/l2_restart_behavior.log 2>&1
  for st in 2 3; do python eval/diagnostics/l2_signal_check.py $R/l2_signal_check.json $RD:$IT n=512 start=$st cost=4 >> $R/l2_signal_check.log 2>&1; done
fi
echo REPORTDONE $IT
