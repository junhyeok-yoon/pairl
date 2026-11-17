#!/bin/sh
# Rollout GIFs of a run's checkpoint from the three starts, written at the top level of its group folder as <run>_start{0,2,3}_it<NNNN>.gif
# (eval/figures/anim_rollout.py, deterministic policy, four evaluation episodes). Runs on the CPU at low priority, one start after another; frame stride
# every=3 keeps a GIF at about 3-4 MB; a GIF above 5 MB is redone with every=5. The platform is noted in <group>/gifs.md.
# usage: sh script/group_gifs.sh RUN_NAME IT [cost=4]      (run from the repository root)
RUN=$1; IT=$2; COST=${3:-cost=4}; P=$(printf "%04d" $IT)
RD=$(python -c "from pairl import paths; print(paths.run_dir('$RUN'))") || exit 1; GD=$(dirname "$RD")
for st in 0 2 3; do
  OUT=$GD/${RUN}_start${st}_it$P.gif
  [ -s "$OUT" ] && continue
  JAX_PLATFORMS=cpu sh script/safe_run.sh mem=6G nice -n 15 python eval/figures/anim_rollout.py "$RD" $IT "$OUT" $COST start=$st every=3 > /dev/null 2>&1
  if [ "$(stat -c %s "$OUT" 2>/dev/null || echo 0)" -gt 5000000 ]; then
    rm -f "$OUT"; JAX_PLATFORMS=cpu sh script/safe_run.sh mem=6G nice -n 15 python eval/figures/anim_rollout.py "$RD" $IT "$OUT" $COST start=$st every=5 > /dev/null 2>&1; touch "$OUT.stride5"
  fi
  S=3; [ -f "$OUT.stride5" ] && S=5 && rm -f "$OUT.stride5"
  [ -s "$OUT" ] && echo "- \`$(basename "$OUT")\`: made $(date +%F) on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride $S), $(( $(stat -c %s "$OUT") / 1000 )) kB" >> "$GD/gifs.md"
done
