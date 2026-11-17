#!/bin/sh
# Interim step of a long run: waits for the built-in evaluation line of iteration IT in the run's train.log, then (CPU, low priority) GIFs of that checkpoint
# and the group's visual summary; with eval=1 also the decision evaluation of that checkpoint (l2_pilotO_eval.py) on the GPU next to the training.
# usage: sh script/interim_report.sh RUN_NAME IT [eval=1]      (run from the repository root)
RUN=$1; IT=$2; EV=$3
until RD=$(python -c "from pairl import paths; print(paths.run_dir('$RUN'))" 2>/dev/null) && grep -q "EVAL it $IT:" "$RD/train.log" 2>/dev/null; do sleep 30; done
sleep 10; GD=$(dirname "$RD"); TS=$(basename "$GD")
if [ "$EV" = "eval=1" ]; then
  sh script/safe_run.sh mem=12G gpu_frac=0.3 python eval/diagnostics/l2_pilotO_eval.py records/l2_pilotO_eval.json "$RD:$IT" > "$RD/pilotO_eval_it$(printf %04d $IT).log" 2>&1
fi
sh script/group_gifs.sh "$RUN" "$IT"
JAX_PLATFORMS=cpu sh script/safe_run.sh mem=4G nice -n 15 python eval/figures/group_summary.py "$TS" > /dev/null 2>&1
echo INTERIM "$RUN" "$IT"
