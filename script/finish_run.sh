#!/bin/sh
# After a run of a group finishes: decision evaluation of its final checkpoint (on the GPU, same platform as the training); then on the CPU at low priority:
# phase success (both modes) of the final checkpoint and the success curve over its checkpoints (every 10 iterations, deterministic), rollout GIFs, the
# group summary and success report (summary.png, eval_*.png, decision.png, success_table.md, funnel.png, success_curve.png, summary.md, INDEX.md), and the
# videos (representative episodes per start, comparison with the previous best, 3 x 3 grid).
# usage: sh script/finish_run.sh RUN_NAME [eval=1]      (run from the repository root, after the training process has ended)
RUN=$1; EV=${2:-eval=1}
RD=$(python -c "from pairl import paths; print(paths.run_dir('$RUN'))") || exit 1; GD=$(dirname "$RD"); TS=$(basename "$GD")
IT=$(python -c "import json; print(json.load(open('$RD/hist.json'))[-1]['it'])"); P=$(printf %04d $IT)
if [ "$EV" = "eval=1" ]; then
  sh script/safe_run.sh mem=16G python eval/diagnostics/l2_pilotO_eval.py records/l2_pilotO_eval.json "$RD:$IT" > "$RD/pilotO_eval_it$P.log" 2>&1
fi
CPU="env JAX_PLATFORMS=cpu sh script/safe_run.sh mem=12G nice -n 15 python script/parallel_run.py workers=6"
$CPU eval/diagnostics/phase_success.py records/phase_success.json "$RD:$IT" > "$RD/phase_success_it$P.log" 2>&1
SPECS=$(python -c "import glob, re; print(' '.join(f'$RD:{int(re.findall(r\"it(\\d+)\", p)[-1])}' for p in sorted(glob.glob('$RD/params_it*.pkl'))))")
$CPU eval/diagnostics/phase_success.py records/phase_success_curve.json $SPECS modes=deterministic > "$RD/phase_success_curve.log" 2>&1
sh script/group_gifs.sh "$RUN" "$IT"
JAX_PLATFORMS=cpu sh script/safe_run.sh mem=4G nice -n 15 python eval/figures/group_summary.py "$TS" > /dev/null 2>&1
$CPU eval/figures/episode_video.py "$GD" "$RD:$IT" > "$RD/videos_it$P.log" 2>&1
PB=$(python eval/figures/prev_best.py "$TS")
[ -n "$PB" ] && $CPU eval/figures/episode_video.py "$GD" "$RD:$IT" compare="$PB" >> "$RD/videos_it$P.log" 2>&1
$CPU eval/figures/episode_video.py "$GD" "$RD:$IT" grid=1 >> "$RD/videos_it$P.log" 2>&1
JAX_PLATFORMS=cpu sh script/safe_run.sh mem=4G nice -n 15 python eval/figures/success_report.py "$TS" > "$RD/success_report.log" 2>&1
echo FINISHED "$RUN" "$IT"
