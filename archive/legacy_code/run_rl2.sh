#!/bin/sh
# proposed architecture with the learned reflex of the spec, with and without reports
for s in 0 1 2; do for a in D E; do python3 rl_compare.py $a $s 70 > ../records/rl_${a}_s${s}.out 2>&1; done; done
echo done > ../records/rl_all2.done
