import json
from pairl import paths

# Table rows of the note generated from the records (single source for the documents and the consistency gate).
def panda_rows(path=str(paths.RECORDS / 'authority_panda.json')):
    rows = []
    for r in json.load(open(path)):
        k = '정지' if r['vbar'] == 0 else f"{r['k1']:g}"
        rows.append(f"{k} & {r['vbar']:g} & {r['rho0_pos']:.3f} & {r['theta_pos']:.3f} & {r['mu_pos']:.3f} & {r['mu_min']:.3f} & {r['mu_med']:.3f} & {r['rho_dir_certified_min']:.3f} \\\\")
    return rows

def mech_rows(path=str(paths.RECORDS / 'mech_arm.jsonl')):
    rec = {r['law']: r for r in (json.loads(l) for l in open(path))}
    ar = lambda x, d=3: f"{x[0]:.{d}f} $\\to$ {x[-1]:.{d}f}"
    rows = []
    for law, lab in (('none', 'Feedback만'), ('adapt', '적응, \\eqref{eq:chat}')):
        r = rec[law]
        V = '--' if law == 'none' else f"{r['V0']:.3f} $\\to$ {r['V_end']:.3f}"
        rows.append(f"{lab} & {ar(r['yield_err'])} & {ar(r['w_mean'])} & {ar(r['effort'], 2)} & {ar(r['pos_err'])} & {V} \\\\")
    return rows, rec

# Learning-comparison tables (rows generated from the records)
NAMES = {'A': 'Proposed', 'B': 'Input policy with funnel filter', 'C': 'Proposed, no reports', 'D': 'Proposed, learned reflex',
         'E': 'Proposed, learned reflex, no reports', 'F': 'Input policy with funnel filter and error integral'}

def rl1_rows(path=str(paths.RECORDS / 'rl_summary.json')):
    import numpy as np
    S = json.load(open(path)); rows = []
    for k in 'BACDE':
        d = S[k]
        rows.append(f"{NAMES[k]} & {min(d['err']):.3f}--{max(d['err']):.3f} & {np.mean(d['ret']):.3f} & {sum(d['train_viol_eps'])} / {sum(d['train_eps'])} & {sum(d['test_viol_eps'])} / {64 * d['seeds']} & {max(d['wmax']):.3f} \\\\")
    return rows

def scripted_rows(path=str(paths.RECORDS / 'rl_scripted.json')):
    return [f"{r['vbar']:g} & {r['kc']:g} & {r['ga']:g} & {r['psip']:g} & {r['err']:.3f} & {r['yield_mean']:.3f} & {r['wmax']:.3f} \\\\" for r in json.load(open(path))]

def rl2_rows(path=str(paths.RECORDS / 'rl2_summary.json')):
    import numpy as np
    R = json.load(open(path))['round2']; rows = []
    for k in 'BFAC':
        per = R[k]
        rows.append(f"{NAMES[k]} & {', '.join(f'{r[chr(115) + chr(115)]:.3f}' for r in per)} & {np.mean([r['ss_interior'] for r in per]):.3f} & {np.mean([r['ss_near'] for r in per]):.3f} & {np.mean([r['err'] for r in per]):.3f} & {sum(r['train_viol_eps'] for r in per)} / {sum(r['train_eps'] for r in per)} \\\\")
    return rows

def cause_rows(path=str(paths.RECORDS / 'rlfix_summary.json')):
    import numpy as np
    S = json.load(open(path)); rows = []
    for key in ['psi', 'gain', 'target', 'feat', 'psi-gain-feat-target']:
        d = S[key]; per = d['per']
        rows.append(f"{d['name']} & {', '.join(f'{p[chr(101) + chr(114) + chr(114)]:.3f}' for p in per)} & {', '.join(f'${100 * p[chr(114) + chr(101) + chr(108)]:+.1f}$' for p in per)} & {'지지' if d['supported'] else '기각'} & {np.mean([p['y1'] for p in per]):.2f} & {sum(p['viol'] for p in per)} \\\\")
    return rows
