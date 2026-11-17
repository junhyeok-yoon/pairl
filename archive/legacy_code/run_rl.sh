#!/bin/sh
# 3 seeds x 3 architectures, 70 iterations each, interleaved by seed
for s in 0 1 2; do for a in A B C; do python3 rl_compare.py $a $s 70 > ../records/rl_${a}_s${s}.out 2>&1; done; done
echo done > ../records/rl_all.done
