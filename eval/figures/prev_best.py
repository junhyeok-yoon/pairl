"""Prints RUN_DIR:IT of the best reported checkpoint outside a group (full chain, then P1-P3, P1&P2, P1; records/phase_success.json), for comparison videos.
Usage: python eval/figures/prev_best.py <timestamp>"""
import sys, json
from pairl import paths
M = json.load(open(paths.RUNS / sys.argv[1] / 'manifest.json')); PS = json.load(open(paths.RECORDS / 'phase_success.json'))
r = lambda k, g, m: PS[k]['deterministic'][g][m]['rate']
ks = [k for k in PS if not k.startswith('_') and k != 'L1_reference' and k.split(':')[0] not in M['runs'] and paths.run_dir(k.split(':')[0]).parent.name < sys.argv[1]]
if ks:
    b = max(ks, key=lambda k: (r(k, 'chain', 'P1234'), r(k, 'chain', 'P123'), r(k, 'chain', 'P12'), r(k, 'alone', 'P1'))); print(f"{paths.run_dir(b.split(':')[0])}:{b.split(':')[1]}")
