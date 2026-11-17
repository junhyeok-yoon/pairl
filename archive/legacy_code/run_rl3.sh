#!/bin/sh
for s in 0 1 2; do for a in B F A C; do RL_ROUND=2 RL_UBAR=80,35 python3 rl_compare.py $a $s 70 > ../records/rl2_${a}_s${s}.out 2>&1; done; done
echo done > ../records/rl2_all.done
