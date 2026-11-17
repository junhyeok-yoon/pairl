#!/bin/sh
for s in 0 1 2; do for F in psi feat gain target psi,gain,feat,target; do RL_FIX=$F python3 rl_compare.py A $s 70 > ../records/rlfix_$(echo $F | tr , -)_A_s$s.out 2>&1; done; done
echo done > ../records/rlfix_all.done
