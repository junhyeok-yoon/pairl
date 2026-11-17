"""Paired differences of the phase success metrics (records/phase_success.json) between two checkpoints on the same episodes: A minus B for P1, P2, P3, P3'
(episodes with top-face contact in both), P4 alone and the chain (P1&P2, P1-P3, full), with bootstrap 95% CIs (10000 resamples of episodes, numpy
default_rng(0), percentile). Usage: python eval/diagnostics/phase_diff.py KEY_A KEY_B [mode=deterministic|gripper_sampled]  (also importable: diff(R, a, b, mode))"""
import sys, json, numpy as np
from pairl import paths


def _ci(d):
    d = np.asarray(d, float); bs = np.random.default_rng(0).choice(d, (10000, d.size)).mean(1)
    return dict(n=int(d.size), diff=float(d.mean()), ci95=[float(v) for v in np.percentile(bs, [2.5, 97.5])])


def diff(R, a, b, mode='deterministic'):
    A, B = R[a][mode], R[b][mode]; ep = lambda X, g, k: np.asarray(X[g][k]['per_episode'], float)
    out = {k: _ci(ep(A, 'alone', k) - ep(B, 'alone', k)) for k in ('P1', 'P2', 'P3', 'P4')}
    ca, cb = np.asarray(A['P3c_contact_ep'], bool), np.asarray(B['P3c_contact_ep'], bool)
    va, vb = np.full(ca.size, np.nan), np.full(cb.size, np.nan); va[ca] = ep(A, 'alone', 'P3c'); vb[cb] = ep(B, 'alone', 'P3c')
    both = ca & cb; out['P3c'] = _ci(va[both] - vb[both])
    for k in ('P12', 'P123', 'P1234'): out['chain_' + k] = _ci(ep(A, 'chain', k) - ep(B, 'chain', k))
    return out


if __name__ == '__main__':
    R = json.load(open(paths.RECORDS / 'phase_success.json')); mode = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a).get('mode', 'deterministic')
    for k, v in diff(R, sys.argv[1], sys.argv[2], mode).items(): print(f"{k:12s} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n {v['n']}{'  *' if v['ci95'][0] > 0 or v['ci95'][1] < 0 else ''}")
