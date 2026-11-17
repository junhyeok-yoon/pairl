import json, re, sys
from pairl import paths
R_, N_ = str(paths.RECORDS) + '/', str(paths.NOTES) + '/'
U_ = lambda f: str(paths.run_dir(f.split('/')[0])) + '/' + f.split('/', 1)[1]   # run files by run name (data/runs/<group>/<run>)

# Consistency gate: numbers cited in the documents must match the records; retired strings must be absent.
recs = [json.loads(l) for l in open(R_ + 'closed_loop.jsonl')]
recs = {r['run']: r for r in recs}
assert sorted(recs) == [0, 1, 2, 3, 4], 'closed-loop record incomplete'
note = open(N_ + 'note_core.tex', encoding='utf-8').read()
spec = open(N_ + 'spec_v2.tex', encoding='utf-8').read()      # archived v2 specification holds the v2 status rows
spec3 = open(N_ + 'spec.tex', encoding='utf-8').read()
log = open(R_ + 'log_check_decrease.txt').read()
bad = []

r0, r1, r2, r3, r4 = (recs[i] for i in range(5))
table = [
    (f"제안한 법칙 & 2.0 & 0/64 & {r0['wmax_med']:.3f} & {r0['wmax_max']:.3f} & {1 - r0['m_star']:.3f} & {r0['real_ratio']:.3f}", r0['left'] == 0),
    (f"오차 여유에 대한 양보 제거 & 2.0 & {r1['left']}/64", r1['left'] == 64),
    (f"제안한 법칙 & 9.5 & 0/64 & {r2['wmax_med']:.3f} & {r2['wmax_max']:.3f} & {1 - r2['m_star']:.3f} & {r2['real_ratio']:.3f}", r2['left'] == 0),
    (f"성분별 saturation & 9.5 & {r3['left']}/64 & {r3['wmax_med']:.3f} & {r3['wmax_max']:.3f}", r3['left'] == 3),
    (f"제안한 법칙, 제안 속도 0.3 & 2.0 & 0/64 & {r4['wmax_med']:.3f} & {r4['wmax_max']:.3f} & {1 - r4['m_star']:.3f} & {r4['real_ratio']:.3f}", r4['left'] == 0),
]
for row, cond in table:
    if row not in note: bad.append(f'table row not found: {row}')
    if not cond: bad.append(f'record condition failed for: {row}')
for txt, val in [('0.9998', r4['yield_err']), ('0.050', r0['eta_clear_min']), ('0.054', r2['eta_clear_min']), ('0.817', r4['w0_max'])]:
    if f'{val:.{len(txt.split(".")[1])}f}' != txt or txt not in note: bad.append(f'value {txt} vs record {val}')
for r in (r0, r2, r4):
    if r['above_bound'] != 0 or abs(r['gauge_max'] - 1.0) > 1e-9 or r['tube_clear_min'] < 0: bad.append('bound, input or tube violated in a record')
for m in re.finditer(r'm=(\d) gamma1=([\d.]+): rho0=[\d.]+ m_star=([\d.]+) \| max d\|\|w\|\|/dt \* psi = ([-+\d.]+)', log):
    if float(m.group(4)) >= 0: bad.append('decrease inequality violated in the log')
    if f"{m.group(3)} & ${m.group(4).replace('+', '')}$" not in note: bad.append(f'point-evaluation row not found: {m.group(0)[:40]}')
for s_, doc, name, limit in [(r'\hat\psi_{\min}', spec, 'spec', 1), (r'u_{\mathrm{in}}', spec, 'spec', 0), ('Capability report', spec, 'spec', 0)]:
    if doc.count(s_) > limit: bad.append(f'retired string {s_} occurs {doc.count(s_)} times in {name}')
if '1.2\\times10^7' not in spec or '64개 모두 이탈' not in spec: bad.append('spec status row out of sync')
# adaptation table of the note: rows are generated from the record and must appear verbatim
ad = {r['run']: r for r in (json.loads(l) for l in open(R_ + 'adapt_yield.jsonl'))}
assert sorted(ad) == [0, 1, 2, 3, 4, 5], 'adaptation record incomplete'
def arrow(x, fmt=lambda v: f'{v:.3f}'): return f"{fmt(x[0])} $\\to$ {fmt(x[-1])}"
fs = lambda v: '0' if v < 5e-4 else f'{v:.3f}'
fV = lambda v: f'{v:.3f}' if v < 10 else (f'{v:.1f}' if v < 1000 else f'{v:.0f}')
labels = {0: 'Feedback만', 1: '적응, \\eqref{eq:What}', 3: 'Feedback만', 4: '적응, \\eqref{eq:What}', 5: '적응, $\\lambda\\lambda_{\\mathrm{ff}}$ 없이'}
adapt_rows = []
for i in (0, 1, 3, 4, 5):
    r = ad[i]
    V = '--' if r['law'] == 'none' else f"{r['V0']:.3f} $\\to$ {fV(r['V_end'])}"
    adapt_rows.append(f"{labels[i]} & {r['vs']} & {arrow(r['yield_err'])} & {arrow(r['w_mean'])} & {arrow(r['effort'])} & {arrow(r['sat_frac'], fs)} & {V} \\\\")
for row in adapt_rows:
    if row not in note: bad.append('adaptation row differs from record; expected: ' + row)
for i in (1, 4, 5):
    if ad[i]['v_increases'] != 0: bad.append(f'storage increase on unsaturated steps in run {i}')
if abs(ad[1]['V_end'] - ad[2]['V_end']) > 1e-3 or any(abs(x - y) > 5e-5 for x, y in zip(ad[1]['effort'], ad[2]['effort'])): bad.append('scaled and unscaled laws differ at speed 0.3')
idn = float(open(R_ + 'log_identity.txt').read().split()[0])
if idn > 1e-10 or '2.1\\times10^{-14}' not in note or not (2.05e-14 <= idn < 2.15e-14): bad.append(f'identity check value {idn}')
if '0.944 대 1.000' not in spec or f"{ad[0]['yield_err'][-1]:.3f}" != '0.944' or f"{ad[1]['yield_err'][-1]:.3f}" != '1.000': bad.append('spec adaptation row out of sync')
# authority and mechanical-arm tables: rows generated from the records must appear verbatim in the note
from pairl.tools.lib_tables import panda_rows, mech_rows
mrows, mrec = mech_rows()
for row in panda_rows() + mrows:
    if row not in note: bad.append('row differs from record; expected: ' + row)
pj = json.load(open(R_ + 'authority_panda.json'))
if any(r['rho0_pos'] != 0.0 or r['mu_pos'] != 1.0 for r in pj): bad.append('authority statement of the documents no longer matches the record')
if f"{min(r['mu_min'] for r in pj):.2f}" != '1.09' or '최소 1.09' not in spec: bad.append('spec authority row out of sync')
if mrec['adapt']['v_increases'] != 0 or f"{mrec['none']['yield_err'][-1]:.3f}" != '0.128' or f"{mrec['adapt']['yield_err'][-1]:.3f}" != '1.000' or '0.128 대 1.000' not in spec: bad.append('mechanical-arm statement out of sync')
mpt = json.load(open(R_ + 'mech_arm_points.json'))
if mpt['identity'] > 1e-10 or mpt['authority']['mu_pos'] != 1.0: bad.append('mechanical-arm point checks')
# learning-comparison note: rows generated from the records must appear verbatim; outcome and spec rows must agree
from pairl.tools.lib_tables import rl1_rows, scripted_rows, rl2_rows
learn = open(N_ + 'note_learning.tex', encoding='utf-8').read()
for row in rl1_rows() + scripted_rows() + rl2_rows():
    if row not in learn: bad.append('learning row differs from record: ' + row[:60])
r2 = json.load(open(R_ + 'rl2_summary.json'))
if r2['predictions'] != {'P1': False, 'P2': True, 'P3': True} or '세 seed 모두에서 기각' not in learn: bad.append('round-2 outcome out of sync')
for txt in ('0.289--0.307', '0.517--0.597', '0.422--0.448', '0.119--0.175'):
    if txt not in learn or txt not in spec: bad.append('learning numbers out of sync: ' + txt)
if open(R_ + 'preregistration_round2.sha256').read().strip()[:8] not in learn: bad.append('pre-registration hash')
from pairl.tools.lib_tables import cause_rows
for row in cause_rows():
    if row not in learn: bad.append('cause row differs from record: ' + row[:60])
cs = json.load(open(R_ + 'rlfix_summary.json'))
if cs['criteria']['combined_closes_half'] or any(cs[k]['supported'] for k in ('psi', 'gain', 'target', 'feat')) or '네 진단은 모두 기각되었다' not in learn: bad.append('cause-check outcome out of sync')
if f"{json.load(open(R_ + 'bound_reference.json'))['vbar4_psi0.4']['mean']:.3f}" not in learn or '0.455--0.471' not in spec: bad.append('cause-check numbers out of sync')
for txt in ('0.517--0.597', '0.289--0.307', '0.140--0.342', '30{,}240'):
    if txt not in spec3: bad.append('v3 history numbers out of sync: ' + txt)
r7 = json.load(open(R_ + 'r7_check.json'))
badc = [r for r in r7 if not r['authority_holds']][0]
for txt in (f"{badc['cumulative_debt']:.1f}", f"{badc['min_T']:.2f}", f"{badc['q_min']:.3f}"):
    if txt not in spec3: bad.append('R7 numbers out of sync: ' + txt)
for txt in ('4.800--10.218', '0.579--0.865', '5.902--11.454', '1.091--1.150', '12.314--12.449'):
    if txt not in spec3: bad.append('R8 numbers out of sync: ' + txt)
if '0.605--4.150' not in spec3: bad.append('sensed freeze numbers out of sync')
if '11.5030' not in spec3 or '-6.1243' not in spec3: bad.append('R1 numbers out of sync')
if '5.971--6.002' not in spec3 or '1.0467' not in spec3 or '8.4084' not in spec3: bad.append('M1 numbers out of sync')
if '0.046, 0.055, 0.061' not in spec3 or '2.943' not in spec3: bad.append('M2 numbers out of sync')
if '1.963' not in spec3 or '95.8' not in spec3 or '0.0017' not in spec3 or '-0.055' not in spec3: bad.append('M2 audit numbers out of sync')
from pairl.tools import write_envcheck_tex as _w
_vv = _w.vals()
if any(_vv[k] not in spec3 for k in ('b3_in', 'b3_ood', 'gj', 'r_ood', 'ws_n', 'gr_ood', 'v16f')): bad.append('Stage B environment numbers out of sync')
import json as _j
import numpy as np
if not _j.load(open(R_ + 'stageb_env_check.json'))['all_pass'] or not all(_j.load(open(R_ + 'stageb_dyn_check.json'))['pass'].values()): bad.append('Stage B environment verification not passing')
if '1.0002' not in spec3 or '0.377' not in spec3 or '0.111' not in spec3 or '0.64' not in spec3: bad.append('L0 numbers out of sync')
L1 = _j.load(open(R_ + 'l1_pilot.json'))
def _l1(k, m, f=np.mean): return float(f(np.asarray(L1[k][m])))
_hard = np.asarray(L1['E_hard']) > 0.5; _mo = np.asarray(L1['E_mo']); _w = np.asarray(L1['A_ref']['w_hat']) / 9.81
_er, _es = np.asarray(L1['A_ref']['E_task']), np.asarray(L1['A_slow']['E_task']); _f6 = np.asarray(L1['D_ref_off0.6']['grasp']) < 0.5
_pct = lambda x: f"{100 * x:.1f}"
_cost = lambda k: f"{_l1(k, 'cost'):.2f}"
_band = np.abs(_mo - 0.85) >= 0.2; _mis = ((_w > 0.85) != (_mo > 0.85)) & _band
_txt = [_cost('A_ref'), _cost('A_ref_nomem'), _cost('A_stiff_nomem'), _cost('A_intr_nomem'), _cost('B_intr_nomem'), _cost('B_ref'), _cost('A_ref_qd'), _cost('A_low'),
        f"{_l1('A_ref', 'pos2', np.median):.4f}", f"{np.percentile(L1['A_ref']['pos2'], 90):.4f}", _pct(_l1('A_ref', 'frel', np.median)) + '\\%', _pct(np.percentile(L1['A_ref']['frel'], 90)) + '\\%',
        _pct(np.mean(np.asarray(L1['A_ref']['cons'])[_hard])) + '\\%', _pct(np.mean(np.asarray(L1['A_ref']['cons'])[~_hard])) + '\\%', f"{100 * _l1('A_ref', 'sat'):.2f}" + '\\%',
        _pct(_l1('A_ref_nomem', 'frel', np.median)) + '\\%', f"{np.percentile(L1['A_intr_nomem']['pos2'], 90):.4f}", _pct(_l1('A_intr_nomem', 'frel', np.median)) + '\\%',
        '%.2f, %.2f, %.2f' % tuple(np.percentile(_er, [10, 50, 90])), _pct(_l1('D_ref_prop1.2', 'gate')) + '\\%와 ' + _pct(_l1('D_ref_prop2.0', 'gate')),
        _cost('D_ref_prop1.2') + '와 ' + _cost('D_ref_prop2.0'), ', '.join(_cost(f'D_ref_off{r}') for r in (0.8, 0.6, 0.4)),
        ', '.join(f"{100 * _l1(f'D_ref_off{r}', 'grasp'):.0f}" for r in (0.8, 0.6, 0.4)), ', '.join(f"{100 * _l1(f'D_ref_off{r}', 'gate'):.0f}" for r in (0.8, 0.6, 0.4)),
        ', '.join(_cost(f'D_ledJ_off{r}') for r in (0.8, 0.6, 0.4)), f"{int(_f6.sum())}개의", f"{np.median(_er[_f6]):.2f}", f"{np.median(_es[_f6]):.2f}",
        f"{np.median(np.asarray(L1['E_pe_save']['target'])[~_hard]):.2f}", f"{np.median(L1['E_pe_save']['grasp']):.2f}", f"{np.percentile(np.abs(_w - _mo), 90):.3f}",
        f"{100 * _l1('C_ref', 'grasp'):.0f}" + '\\%만', f"{100 * _l1('C_ref_kc2', 'grasp'):.0f}" + '\\%', f"{_l1('C_ref_kc2', 'pos2', np.median):.3f}", f"({100 * _l1('C_intr_mem', 'grasp'):.0f}" + '\\%)',
        f"{int(_band.sum())}개 가운데 {int(_mis.sum())}개", f"{100 * np.max(L1['A_ref']['guard']):.2f}" + '\\%']
if int(np.sum(_f6 & (_es <= 0.6 * _er))) != 0 or _l1('A_ref', 'viol', np.sum) != 0 or int(np.sum(np.asarray(L1['A_ref']['guard']) > 0)) != 1: bad.append('L1 pilot statements out of sync with the record')
for t_ in _txt:
    if t_ not in spec3: bad.append('L1 pilot number not in the specification: ' + t_)
import hashlib as _hl, re as _re
_S1 = _j.load(open(R_ + 'l1_summary.json')); _R1 = _j.load(open(R_ + 'l1_results.json'))
_A1 = lambda k: {m: np.asarray(v) for m, v in _R1[k].items()}
_st = _S1['stats']; _ref1, _nol1, _bud1 = _A1('ID_ref'), _A1('ID_noled'), _A1('BUD_ref')
_h1 = open(R_ + 'preregistration_l1.hash').read().strip()
if _hl.sha256(open(R_ + 'preregistration_l1.md', 'rb').read()).hexdigest()[:8] != _h1 or _h1 not in spec3: bad.append('L1 pre-registration hash')
if not _S1['all_pass']: bad.append('L1 summary: not all criteria pass but the specification says they do')
_g2 = (_ref1['grasp'] > 0.5) & (_nol1['grasp'] > 0.5); _w1 = _ref1['cost_ph'][:, 2] > _nol1['cost_ph'][:, 2]
_eps_fs = None
_mis1 = (_ref1['class_ok'] < 0.5) & (_ref1['grasp'] > 0.5); _mo1 = np.asarray(_R1['ID_mo'])
_pv = {float(m.group(1)): m.group(2) for m in _re.finditer(r'ID k_c=([0-9.]+)\s+\S+ cost ([0-9.]+)', open(R_ + 'l1_pilot_v2.log').read())}
_t1 = [f"{_st['a1_grasp']:.3f}", f"{_st['a2_class']:.3f}", f"{_st['a3_frel_med']:.3f}, {_st['a3_frel_p90']:.3f}", f"{_st['a4_pos2_med']:.4f}, {_st['a4_pos2_p90']:.4f} m",
       f"{_st['b1_cost_nomem']:.2f}, {_st['b1_cost_nomem_khi']:.2f}", f"v3 {_st['b1_cost_ref']:.2f}의", f"{_st['b2_frel_nomem']:.3f}, {_st['b2_frel_nomem_khi']:.3f}",
       f"{_st['c1_ratio']:.2f} ({_st['c1_frel_noled']:.3f}/{_st['a3_frel_med']:.3f})", f"{_st['c2_press_better']:.3f}", f"{_st['c3_class_noled']:.3f}", f"{_st['d1_cons_hard']:.3f}",
       f"{_st['d2_gate_bud']:.3f}, {_st['d2_gate_any']:.3f}", '%.1f\\cdot10^{-5}' % (_st['d4_tank_bound_min'] / 1e-5), f"{np.mean(_bud1['cost']):.2f}로", f"{np.mean(_bud1['grasp']):.3f}로",
       f"{int(np.sum(_ref1['guard'] > 0))}개 episode", f"{100 * np.max(_ref1['guard']):.2f}" + '\\%', f"{100 * _st['d6_sat']:.2f}" + '\\%',
       f"{np.mean(_A1('HEAVY_ref')['cost']):.2f}와 {np.mean(_A1('BODY_MASS_ref')['cost']):.2f}", f"{np.median(_A1('HEAVY_ref')['frel']):.3f}",
       f"{np.mean(_A1('BODY_LEN_ref')['grasp']):.3f}", f"{np.median(_A1('STIFF_ref')['peak_first']):.1f} N", f"{np.median(_ref1['peak_first']):.1f} N",
       f"{int(_w1.sum())}개", f"{np.median(_ref1['cost_ph'][_w1, 2] - _nol1['cost_ph'][_w1, 2]):.3f}", f"{_mo1[_mis1].min():.3f}--{_mo1[_mis1].max():.3f}",
       ', '.join(_pv[k] for k in (0.0, 1.0, 2.0))]
if abs(np.mean(_A1('BODY_ref')['grasp']) - np.mean(_A1('BODY_LEN_ref')['grasp'])) > 0 or int(_mis1.sum()) != 3 or _st['d4_n_sat_below'] != 0: bad.append('L1 statements out of sync with the record')
for t_ in _t1:
    if t_ not in spec3: bad.append('L1 number not in the specification: ' + t_)
# L2 attempt 1, pilots and the revision-14 decisions (section sec:l2a1)
_J = lambda f: _j.load(open(U_(f) if '/' in f else R_ + f))   # run directories live in data/runs, other records in records/
_H1 = [h for h in _J('l2_attempt1/hist.json') if h['it'] <= 70] + _J('l2_attempt1b/hist.json')
_E1 = [e for e in _J('l2_attempt1/evals.json') if e['it'] <= 70] + [e for e in _J('l2_attempt1b/evals.json') if e['it'] > 70]
_HP = {d: _J(d + '/hist.json') for d in ('l2_pilot_cost2', 'l2_pilot_cost2_grip', 'l2_pilot_v3')}
_tb = [h['tank_bound_min'] for h in _H1 + sum(_HP.values(), []) if h['tank_bound_min'] is not None]
if sum(h['viol'] + h['nan'] for h in _H1 + sum(_HP.values(), [])) != 0 or min(_tb) < 0 or len(_H1) + sum(len(v) for v in _HP.values()) != 190: bad.append('L2 guarantee statement out of sync')
_g1 = max((np.mean(e['nom']['grasp']), e['it']) for e in _E1); _e100 = _E1[-1]['hand']; _md = lambda k: float(np.median(_e100[k]))
_D80 = _J('l2_attempt1b/dissect_it0080.json'); _ch, _l1h = _D80['channel_reference_brain']['learned_hand'], _D80['channel_reference_brain']['l1_hand']
_M = {r['it']: r for r in _J('l2_release_mechanism.json')}; _CA = _J('cost_audit.json'); _FC = _J('l2_face_check.json'); _CC = _J('l2_carry_check.json'); _HS = _J('l2_hand_share.json')
_DP2, _DP3 = _J('l2_pilot_cost2_grip/dissect_it0130.json'), _J('l2_pilot_v3/dissect_it0130.json')
_po = lambda d: [h['p_open_att'] for h in _HP[d]]; _a3 = lambda d: [h['att3'] for h in _HP[d]]
_ev = lambda d: _J(d + '/evals.json')[-1]['hand']
_rng = lambda xs, f='%.2f': (f % min(xs)) + '--' + (f % max(xs))
_t2 = ['%.1f\\cdot10^{-6}' % (min(h['tank_bound_min'] for h in _H1 if h['tank_bound_min'] is not None) / 1e-6), f"{_md('prior_zero'):.2f} N", f"{_md('prior_learned'):.2f} N",
       f"{_md('prior_nominal'):.2f} N", f"{_md('res_learned'):.2f} N", f"{_md('res_class'):.2f} N", f"{_ch['class_given_contact']:.3f}, 상대 힘 오차 {_ch['frel']:.4f}, 유지 오차 {_ch['hold']:.5f} m, 평균 비용 {_ch['cost']:.3f}",
       f"{_l1h['class_given_contact']:.3f}, {_l1h['frel']:.4f}, {_l1h['hold']:.5f} m, {_l1h['cost']:.3f}", ', '.join(f'{x:.3f}' for x in _D80['explore']['critic_explained_variance_by_phase']),
       f"{_D80['reach']['min_dist_median']:.3f} m", f"{100 * _D80['reach']['frac_within_2cm']:.0f}\\%", f"최대 {_g1[0]:.3f}(반복 {_g1[1]})",
       f"{_M[100]['phase2']['p_open']:.3f}(단계 2)와 {_M[100]['phase3']['p_open']:.3f}(단계 3)", f"$-{abs(_M[0]['phase2']['dV_median']):.3f}$(단계 2)과 $-{abs(_M[0]['phase3']['dV_median']):.3f}$(단계 3)",
       f"$+{_M[10]['phase2']['dV_median']:.3f}$과 $+{_M[10]['phase3']['dV_median']:.3f}$", f"반복 10에 {_M[10]['phase2']['p_open']:.3f}와 {_M[10]['phase3']['p_open']:.3f}, 40에 {_M[40]['phase2']['p_open']:.3f}와 {_M[40]['phase3']['p_open']:.3f}, 100에 {_M[100]['phase2']['p_open']:.3f}와 {_M[100]['phase3']['p_open']:.3f}",
       f"단계 2에서 {_rng([_M[i]['phase2']['held_rate_median'] for i in (70, 80, 90, 100)])}, 단계 3에서 {_rng([_M[i]['phase3']['held_rate_median'] for i in (70, 80, 90, 100)])}",
       f"단계 2의 {100 * _CA['1']['phase2']['share_negative']:.1f}\\%, 단계 3의 {100 * _CA['1']['phase3']['share_negative']:.0f}\\%", f"$-{abs(_CA['1']['phase2']['min']):.2f}$과 $-{abs(_CA['1']['phase3']['min']):.2f}$",
       f"기울기 {_CA['1']['side']['slope']:.2f}/m", f"{_po('l2_pilot_cost2')[0]:.3f}에서 {_po('l2_pilot_cost2')[-1]:.3f}", _rng(_po('l2_pilot_cost2_grip'), '%.3f'), _rng(_a3('l2_pilot_cost2_grip')),
       f"{100 * _FC['l2_pilot_cost2_grip:130']['side']:.1f}\\%", f"윗면 접촉은 {100 * _FC['l2_pilot_cost2_grip:130']['top']:.0f}\\%", f"{_FC['l2_pilot_cost2_grip:130']['side_depth_median']:.3f} m",
       f"{np.median(_ev('l2_pilot_cost2_grip')['res_learned']):.2f} N", f"등급 결정이 {_DP2['channel_reference_brain']['learned_hand']['class_given_contact']:.3f}",
       f"{100 * _HS['l2_pilot_cost2_grip:130']['uncalibrated_time_share']:.1f}\\%", f"{100 * _HS['l2_pilot_cost2_grip:130']['uncalibrated_loss_share']:.1f}\\%",
       f"{_HS['l2_pilot_cost2_grip:130']['mse_uncalibrated']:.1f} 대 {_HS['l2_pilot_cost2_grip:130']['mse_calibrated']:.2f}", f"{np.median(_ev('l2_pilot_v3')['res_learned']):.2f} N",
       f"등급 결정은 {_DP3['channel_reference_brain']['learned_hand']['class_given_contact']:.3f}", f"{_HS['l2_pilot_v3:130']['mse_calibrated']:.3f}로 P2({_HS['l2_pilot_cost2_grip:130']['mse_calibrated']:.3f})",
       f"{_HS['l2_pilot_v3:130']['mse_uncalibrated']:.1f}으로 P2({_HS['l2_pilot_cost2_grip:130']['mse_uncalibrated']:.1f})", f"옆면 접촉은 {_FC['l2_pilot_v3:130']['side']:.3f}",
       f"{_CC['l2_pilot_v3:130']['goal_tip_to_target_late']:.3f} m"]
for t_ in _t2:
    if t_ not in spec3: bad.append('L2 number not in the specification: ' + t_)
for f_ in ('l2_attempt1_plan', 'l2_attempt2_plan'):
    if _hl.sha256(open(f'{R_}{f_}.md', 'rb').read()).hexdigest()[:8] != open(f'{R_}{f_}.hash').read().split()[-1]: bad.append('hash of ' + f_)
# attempt 2: decisive experiments (table tab:l2diag)
_G = _J('l2_goal_check.json'); _AC = _J('l2_adv_check.json'); _VC = _J('l2_value_check.json')
_gg = lambda k: f"{_G[k]['goal_to_grasp_point']:.3f} m"
_rows = [('첫 시도', 'l2_attempt1'), ('E4', 'l2_diag_init1_rev13'), ('E3', 'l2_diag_init1_rev14'), ('E5', 'l2_diag_init1_rev14_vnorm'), ('E2', 'l2_diag_cost13'), ('E1', 'l2_diag_relzero')]
_t3 = [f"{_gg(r + ':30')} & {_gg(r + ':40')} \\\\" for n, r in _rows] + [f"0.{int(round(1000 * _G['l2_attempt2b:50']['goal_to_grasp_point'])):03d} m", f"0.{int(round(1000 * _G['l2_attempt2a:50']['goal_to_grasp_point'])):03d} m"]
_e3, _e4 = _VC['l2_diag_init1_rev14:30:cost3'], _VC['l2_diag_init1_rev13:30:cost1']
_t3 += [f"{_AC['l2_diag_init1_rev14:30:cost3']['phase1']['ev']:.2f}으로 개정 13의 {_AC['l2_diag_init1_rev13:30:cost1']['phase1']['ev']:.2f}",
        f"평균 비용은 {_e3['cost3']['episode_cost_mean']:.2f} 대 {_e3['cost1']['episode_cost_mean']:.2f}, 표준편차는 {_e3['cost3']['episode_cost_sd']:.2f} 대 {_e3['cost1']['episode_cost_sd']:.2f}",
        f"(표준편차 {_e3['first_decision_residual_sd']:.2f})", f"뒤 단계의 길이가 {100 * _e3['share_durations']:.0f}\\%, 잡기 결과가 {100 * _e3['share_grasp']:.0f}\\%",
        f"합쳐 {100 * _e3['share_geometry_grasp']:.0f}\\%", f"({_e3['clip_factor_mean']:.3f}와 {_e4['clip_factor_mean']:.3f})",
        f"(평균 비용 {np.mean(_J('l2_diag_init1_rev13/evals.json')[0]['nom']['cost']):.3f})"]
if abs(np.mean(_J('l2_diag_init1_rev13/evals.json')[0]['nom']['cost']) - np.mean(_J('l2_attempt1/evals.json')[0]['nom']['cost'])) > 1e-9: bad.append('E4 does not reproduce attempt 1 at iteration 0')
for t_ in _t3:
    if t_ not in spec3: bad.append('L2 attempt-2 number not in the specification: ' + t_)
# attempt 2: outcome at iteration 300 and its dissection (section sec:l2a2)
_D3 = _J('l2_attempt2/dissect_it0300.json'); _E2 = [e for e in _J('l2_attempt2b/evals.json') if e['it'] < 50] + _J('l2_attempt2/evals.json')
_HL = [h for h in _J('l2_attempt2b/hist.json') if h['it'] <= 50] + _J('l2_attempt2/hist.json'); _SG = _J('l2_signal_check.json'); _HC = _J('l2_hand_check.json')
_S256 = _J('l2_signal_check_n256.json')['l2_attempt2:200']
if [h['it'] for h in _HL] != list(range(1, 301)) or max(h['viol'] for h in _HL) > 0 or max(h['nan'] for h in _HL) > 0 or min(h['tank_bound_min'] for h in _HL) < -0.01: bad.append('attempt 2: (C1) records')
_res = [(e['it'], float(np.median(e['hand']['res_learned']))) for e in _E2]; _r60 = [v for i, v in _res if i >= 60]; _r300 = dict(_res)[300]
_fm = lambda v: f'{v:.3f}' if v >= 0 else f'$-${abs(v):.3f}'
_cc = lambda it, c: f"{_fm(_SG['l2_attempt2:%d' % it][c]['corr_lambda0.95'])} [{_fm(_SG['l2_attempt2:%d' % it][c]['ci95_lambda0.95'][0])}, {_fm(_SG['l2_attempt2:%d' % it][c]['ci95_lambda0.95'][1])}]"
_C1, _C2, _C3 = 'phase1_free_to_grasp_point', 'phase2_held_to_p_star', 'phase3_held_to_p_c'; _sg = lambda it, c, k: _SG['l2_attempt2:%d' % it][c][k]
_n = lambda v: f'{round(v):,}'.replace(',', '{,}')
_ex, _ch, _tbm = _D3['explore'], _D3['channel_reference_brain'], min(h['tank_bound_min'] for h in _HL)
_t4 = [f"최솟값은 ${_tbm / 1e-7:.1f}\\cdot10^{{-7}}$ J였다", f"등급 결정은 {_ch['learned_hand']['class_given_contact']:.3f}였다", f"중앙값은 {_r300:.3f} N\\,m로 기준 1.768",
       f"평가 {len(_r60)}번 가운데 {sum(v > 1.768 for v in _r60)}번이 기준을 넘었고(범위 {min(_r60):.3f}--{max(_r60):.3f} N\\,m)", f"잡기는 {_D3['reach']['grasp']:.3f}",
       f"최근접 거리의 중앙값은 {_D3['reach']['min_dist_median']:.3f} m, 2 cm 안에 든 episode는 {100 * _D3['reach']['frac_within_2cm']:.1f}\\%",
       f"단계 2를 든 채 마친 {_D3['carry']['n_held_phase2']}개 episode에서 단계 2 끝의 goal 손끝은 $p^\\star$에서 {_D3['carry']['goal_tip_to_target_late']:.3f} m",
       f"단계 3에서 든 {_D3['press_detail']['n_held_phase3']}개 episode의 손끝은 윗면보다 {-_D3['press_detail']['tip_height_median_late']:.3f} m 아래, $p_c$에서 옆으로 {_D3['press_detail']['lateral_offset_median_late']:.3f} m",
       f"결정마다 {_ex['p_open_while_attached_by_phase'][1]:.4f}과 {_ex['p_open_while_attached_by_phase'][2]:.4f}였고", f"단계 4 전에 놓은 episode는 {100 * _ex['frac_release_before_phase4']:.1f}\\%",
       "explained variance는 " + ', '.join(f'{v:.3f}' for v in _ex['critic_explained_variance_by_phase']) + "였다",
       f"평가 episode가 {100 * _D3['contact_face']['share_episodes_held_top_force_window']:.0f}\\%로 25\\%에 못 미쳐",
       f"{_cc(200, _C1)}, {_cc(200, _C2)}, {_cc(200, _C3)}이었고", f"{_n(_sg(200, _C1, 'decisions_per_128_episodes'))}, {_n(_sg(200, _C2, 'decisions_per_128_episodes'))}, {_n(_sg(200, _C3, 'decisions_per_128_episodes'))}이었다",
       f"든 episode가 {_S256[_C2]['episodes']}개와 {_S256[_C3]['episodes']}개뿐", f"단계 3은 {_cc(300, _C3)}로 맞았고, 단계 2는 {_cc(300, _C2)}로 미결", f"단계 1도 {_cc(300, _C1)}로 떨어졌다",
       f"{_sg(200, _C2, 'deviation_projection_sd'):.3f} m와 {_sg(200, _C3, 'deviation_projection_sd'):.3f} m(반복 200, 단계 2와 3)에서 {_sg(300, _C2, 'deviation_projection_sd'):.3f} m와 {_sg(300, _C3, 'deviation_projection_sd'):.3f} m(300)",
       "$p^\\star$까지 " + ', '.join(f"{_sg(i, _C2, 'goal_tip_to_target_median'):.3f}" for i in (200, 250, 300)) + " m, $p_c$까지 " + ', '.join(f"{_sg(i, _C3, 'goal_tip_to_target_median'):.3f}" for i in (200, 250, 300)) + " m",
       f"각각 단계 1 결정의 {_sg(300, _C2, 'decisions_per_128_episodes') / _sg(300, _C1, 'decisions_per_128_episodes'):.2f}배와 {_sg(300, _C3, 'decisions_per_128_episodes') / _sg(300, _C1, 'decisions_per_128_episodes'):.2f}배",
       f"$p^\\star$ 쪽으로 {_sg(200, _C2, 'goal_tip_to_target_median') - _sg(300, _C2, 'goal_tip_to_target_median'):.3f} m 다가갔다",
       f"오차는 {min(_HC['per_checkpoint']):.3f}--{max(_HC['per_checkpoint']):.3f} N\\,m(25 백분위 {_HC['per_checkpoint_q25']:.3f} N\\,m)였고, 그 weight를 평균한 손은 {_HC['averaged']:.3f} N\\,m",
       f"128 episode당 {_SG['l2_attempt2:300']['held_near_p_star_per_128']:.2f}개였다(시작 잡는 점 근처는 {_SG['l2_attempt2:300']['held_near_start_grasp_point_per_128']:.1f}개)",
       f"등급 결정 {_ch['learned_hand']['class_given_contact']:.3f}, 상대 힘 오차 {_ch['learned_hand']['frel']:.4f}, 유지 오차 {_ch['learned_hand']['hold']:.4f} m이고, L1의 손으로는 {_ch['l1_hand']['class_given_contact']:.3f}, {_ch['l1_hand']['frel']:.4f}, {_ch['l1_hand']['hold']:.4f} m",
       f"goal이 {_D3['goal_shift_from_learned_load_by_phase'][2]:.3f} rad 움직임", f"(H1) 미충족({_r300:.3f} N\\,m)"]
_t4 += [f"중앙값은 {_sg(200, _C2, 'sd_u_goal_median'):.3f}와 {_sg(200, _C3, 'sd_u_goal_median'):.3f}에서 {_sg(300, _C2, 'sd_u_goal_median'):.3f}과 {_sg(300, _C3, 'sd_u_goal_median'):.3f}으로",
        f"중앙값은 {_sg(200, _C2, 'tanh_factor_median'):.3f}과 {_sg(200, _C3, 'tanh_factor_median'):.3f}에서 {_sg(300, _C2, 'tanh_factor_median'):.3f}과 {_sg(300, _C3, 'tanh_factor_median'):.3f}로"]
if sum(v > 1.768 for i, v in _res if i >= 260) != 4 or len([i for i, v in _res if i >= 260]) != 5: bad.append('attempt 2: last five H1 evaluations')
if any(_SG['l2_attempt2:300'][c]['ci95_lambda1'][0] > 0 or _SG['l2_attempt2:300'][c]['ci95_lambda1'][1] < 0 for c in (_C1, _C2, _C3)): bad.append('attempt 2: lambda-1 intervals')
if _HC['averaged_below_q25'] or _HC['iterations'] != list(range(210, 301, 10)): bad.append('attempt 2: hand check')
for t_ in _t4:
    if t_ not in spec3: bad.append('L2 attempt-2 outcome not in the specification: ' + t_)
# revision 15 and the preparation of pilot R (sections sec:l2pr, sec:l1)
_OP = _J('l2_overpress_check.json')['l2_attempt2:300']; _S14 = _J('l1_summary_rev14.json')['stats']
_sm1 = _re.search(r'it 301: .*?kl ([0-9.]+) clip ([0-9.]+) vl ([0-9.]+)', open(R_ + 'l2_pilotR_smoke_B_nowarm.log').read())
_smw = open(R_ + 'l2_pilotR_smoke_B_warm.log').read(); _evw = [float(x) for x in _re.findall(r'EV orig [0-9.]+ r2 [0-9.]+ r3 ([0-9.]+) \| critic warm-up', _smw)]
_klw = [float(x) for x in _re.findall(r'it 30[67]: .*?kl ([0-9.]+) clip', _smw)]
_a51 = _re.search(r'it 51: .*?kl ([0-9.]+) clip ([0-9.]+) vl ([0-9.]+)', open(U_('l2_attempt2/train.log')).read())
_rd = open(R_ + 'l2_role_defect_check.log').read(); _rdl = [(float(a), float(b)) for a, b in _re.findall(r'w_mode 1 \[\d+\]: new ([0-9.]+) old ([0-9.]+) diff -', _rd)]
_sg36 = _re.search(r'initial tip within GRASP_DIST of the grasp point: (\d+) of (\d+) \(([0-9.]+)%\)', _rd)
_t5 = [f"개정 14에서 {_CA['3']['top']['share_negative']:.3f}, 개정 15에서 {_CA['4']['top']['share_negative']:.0f}이다",
       f"든 단계-3 결정의 {_OP['rev14_release_cheaper_share_of_held']:.3f}({_n(_OP['held_phase3_decisions'])}개 가운데 {_n(_OP['rev14_release_cheaper_decisions'])}개)",
       f"중앙값 ${_OP['force_ratio_median']:.2f}f^\\star$, 90 백분위 ${_OP['force_ratio_p90']:.2f}f^\\star$",
       f"여는 확률은 {_OP['open_probability_rev14_cheaper']:.4f}(다른 곳 {_OP['open_probability_other_held3']:.4f})", f"episode의 {100 * _OP['episodes_releasing_phase3']:.0f}\\%가 단계 3에서 놓았고",
       f"KL은 {float(_sm1.group(1)):.2f}, clip 비율은 {float(_sm1.group(2)):.3f}, value loss는 {float(_sm1.group(3)):.1f}이었다(두 번째 시도의 재개 직후 {float(_a51.group(1)):.3f}, {float(_a51.group(2)):.3f}, {float(_a51.group(3)):.2f})",
       f"explained variance는 {_evw[0]:.3f}에서 {_evw[-1]:.3f}로 올랐고, 그 뒤 첫 두 update의 KL은 {_klw[0]:.3f}과 {_klw[1]:.3f}이었다",
       f"6{{,}}144개 가운데 {_sg36.group(1)}개({float(_sg36.group(3)):.2f}\\%)", f"{_rdl[0][1]:.3f}에서 {_rdl[0][0]:.3f}로, {_rdl[1][1]:.3f}에서 {_rdl[1][0]:.3f}으로",
       f"기준 뇌 {_S14['b1_cost_ref']:.3f}에서 {_st['b1_cost_ref']:.3f}, 기억을 끈 두 손 {_S14['b1_cost_nomem']:.3f}와 {_S14['b1_cost_nomem_khi']:.3f}에서 {_st['b1_cost_nomem']:.3f}과 {_st['b1_cost_nomem_khi']:.3f}"]
if len(_evw) != 5 or len(_klw) != 2 or len(_rdl) != 2 or int(_sg36.group(2)) != 6144: bad.append('pilot R preparation records incomplete')
if _hl.sha256(open(R_ + 'l2_pilotR_plan.md', 'rb').read()).hexdigest()[:8] != open(R_ + 'l2_pilotR_plan.hash').read().split()[-1]: bad.append('pilot R plan hash')
if _hl.sha256(open(R_ + 'l2_pilotT_plan.md', 'rb').read()).hexdigest()[:8] != open(R_ + 'l2_pilotT_plan.hash').read().split()[-1]: bad.append('pilot T plan hash')
if not _j.load(open(R_ + 'stageb_env_check.json'))['pass'].get('V18'): bad.append('V18 missing or failing')
for t_ in _t5:
    if t_ not in spec3: bad.append('revision-15 / pilot R statement not in the specification: ' + t_)
# pilot R outcome (section sec:l2pr)
_EA, _EB = _J('l2_pilotR_A/evals.json'), _J('l2_pilotR_B/evals.json'); _HA, _HB = _J('l2_pilotR_A/hist.json'), _J('l2_pilotR_B/hist.json')
_DA, _DB = _J('l2_pilotR_A/dissect_it0355.json'), _J('l2_pilotR_B/dissect_it0355.json'); _RB = _J('l2_restart_behavior.json'); _OPa = _J('l2_overpress_check.json')
_acc = lambda e: float(np.mean((np.asarray(e['r3']['frel']) < 0.5) & (np.asarray(e['r3']['held_start4']) > 0.5)))
_tw = lambda e: np.asarray(e['r3']['top_window']) > 0.5; _frc = lambda e: float(np.median(np.asarray(e['r3']['frel'])[_tw(e)]))
_h2 = lambda e: np.asarray(e['r2']['held_start3']) > 0.5; _p2 = lambda e: float(np.median(np.asarray(e['r2']['pos2'])[_h2(e)]))
_rh = lambda E: float(np.mean([np.median(e['hand']['res_learned']) for e in E if e['it'] >= 310])); _rha = [np.median(e['hand']['res_learned']) for e in _EA + _EB if e['it'] >= 310]
_tbr = lambda H: min(h['tank_bound_min'] for h in H if h['tank_bound_min'] is not None)
_sgr = lambda k, c: _SG[k][c]
_cs = lambda k, c: f"${_sgr(k, c)['corr_lambda0.95']:.3f}$ $[{_sgr(k, c)['ci95_lambda0.95'][0]:.3f},\\ {_sgr(k, c)['ci95_lambda0.95'][1]:.3f}]$"
_ea, _eb, _e0 = _EA[-1], _EB[-1], _EA[0]
_t6 = [f"A {_acc(_ea):.3f}(64개 가운데 {round(64 * _acc(_ea))}개), B {_acc(_eb):.0f}이었다", f"잡기는 B {np.mean(_eb['nom']['grasp']):.3f}, A {np.mean(_ea['nom']['grasp']):.3f}이고",
       f"최근접 거리의 중앙값은 B {_DB['reach']['min_dist_median']:.4f} m, A {_DA['reach']['min_dist_median']:.4f} m였다",
       f"윗면을 누른 몫은 {np.mean(_tw(_e0)):.3f}에서 A {np.mean(_tw(_ea)):.3f}, B {np.mean(_tw(_eb)):.3f}로", f"중앙값은 {_frc(_e0):.3f}에서 A {_frc(_ea):.3f}, B {_frc(_eb):.3f}이 되었다",
       f"유지 오차의 중앙값은 A {_p2(_ea):.3f} m, B {_p2(_eb):.3f} m였다(반복 300에 {_p2(_e0):.3f} m)", f"평균은 A {_rh(_EA):.3f} N\\,m, B {_rh(_EB):.3f} N\\,m였다",
       f"범위가 {min(_rha):.3f}--{max(_rha):.3f} N\\,m로", f"최솟값은 A ${_tbr(_HA) / 1e-3:.1f}\\cdot10^{{-3}}$ J, B ${_tbr(_HB) / 1e-3:.1f}\\cdot10^{{-3}}$ J였다",
       f"닿지 않은 episode는 {_RB['l2_attempt2:300']['phase3_start']['hover_episodes']}개(반복 300), {_RB['l2_pilotR_A:355']['phase3_start']['hover_episodes']}개(A), {_RB['l2_pilotR_B:355']['phase3_start']['hover_episodes']}개(B)였다",
       "기울기는 " + ', '.join(f"${_RB[k]['phase3_start']['goal_height_on_force_slope']:.3f}$".replace('$-', '$-') for k in ('l2_attempt2:300', 'l2_pilotR_A:355', 'l2_pilotR_B:355')).replace('$-0.', '$-0.') ,
       f"놓은 episode는 {_OPa['l2_attempt2:300']['episodes_releasing_phase3']:.3f}(반복 300), {_OPa['l2_pilotR_A:355']['episodes_releasing_phase3']:.3f}(A), {_OPa['l2_pilotR_B:355']['episodes_releasing_phase3']:.3f}(B)였고",
       "여는 확률은 " + ', '.join(f"{_OPa[k]['open_probability_held_phase3']:.4f}" for k in ('l2_attempt2:300', 'l2_pilotR_A:355', 'l2_pilotR_B:355')) + "이었다",
       f"{_cs('l2_attempt2:300:start2', _C2)}(반복 300), {_cs('l2_pilotR_A:355:start2', _C2)}(A), {_cs('l2_pilotR_B:355:start2', _C2)}(B)",
       f"{_cs('l2_attempt2:300:start3', _C3)}, {_cs('l2_pilotR_A:355:start3', _C3)}, {_cs('l2_pilotR_B:355:start3', _C3)}였다",
       f"2 cm 안에 든 episode는 A {_DA['reach']['frac_within_2cm']:.3f}, B {_DB['reach']['frac_within_2cm']:.3f}인데 가장 가까운 순간에 잡을 만큼 느린 episode는 {_DA['reach']['frac_slow_when_nearest']:.3f}, {_DB['reach']['frac_slow_when_nearest']:.3f}이고"]
if _OPa['l2_pilotR_A:355']['force_ratio_at_release_median'] != 0 or _OPa['l2_pilotR_B:355']['force_ratio_at_release_median'] != 0 or len(_HA) != 55 or len(_HB) != 55 or sum(h['viol'] + h['nan'] for h in _HA + _HB) != 0: bad.append('pilot R statements out of sync')
for t_ in _t6:
    if t_ not in spec3: bad.append('pilot R outcome not in the specification: ' + t_)
# pilot T outcome (section sec:l2pt)
_EJ, _ET = _J('l2_pilotT_J/evals.json'), _J('l2_pilotT_T/evals.json'); _HJ, _HT = _J('l2_pilotT_J/hist.json'), _J('l2_pilotT_T/hist.json')
_DJ, _DT = _J('l2_pilotT_J/dissect_it0100.json'), _J('l2_pilotT_T/dissect_it0100.json'); _rj, _rt = _RB['l2_pilotT_J:100']['phase3_start'], _RB['l2_pilotT_T:100']['phase3_start']
_ej, _et = _EJ[-1], _ET[-1]; _th = lambda E: float(np.mean([np.median(e['hand']['res_learned']) for e in E if e['it'] >= 10]))
_ser = lambda E: [_p2(e) for e in E if e['it'] >= 10]
_t7 = [f"T {_p2(_et):.4f} m로 J의 절반({0.5 * _p2(_ej):.4f} m) 아래였고", f"반복 100에 T {_p2(_et):.3f} m, J {_p2(_ej):.3f} m", f"잡기 T {np.mean(_et['nom']['grasp']):.3f}, J {np.mean(_ej['nom']['grasp']):.3f}",
       f"정확히 누른 몫 T {_acc(_et):.3f}, J {_acc(_ej):.3f}", f"최근접 거리의 중앙값은 T {_DT['reach']['min_dist_median']:.4f} m, J {_DJ['reach']['min_dist_median']:.4f} m이고, 2 cm 안에 든 episode는 64개 가운데 T {round(64 * _DT['reach']['frac_within_2cm'])}개, J {round(64 * _DJ['reach']['frac_within_2cm'])}개였다",
       f"잡는 점에서 T {_DT['aim']['goal_tip_to_grasp_point_median']:.3f} m, J {_DJ['aim']['goal_tip_to_grasp_point_median']:.3f} m였다", f"느린 episode는 T {_DT['reach']['frac_slow_when_nearest']:.3f}, J {_DJ['reach']['frac_slow_when_nearest']:.3f}이었다",
       f"T {_cs('l2_pilotT_T:100:start2', _C2)}, J {_cs('l2_pilotT_J:100:start2', _C2)}이고", f"T {_cs('l2_pilotT_T:100:start3', _C3)}, J {_cs('l2_pilotT_J:100:start3', _C3)}였다",
       f"단계 2에서 T {_sgr('l2_pilotT_T:100:start2', _C2)['deviation_projection_sd']:.4f} m, J {_sgr('l2_pilotT_J:100:start2', _C2)['deviation_projection_sd']:.4f} m, 단계 3에서 T {_sgr('l2_pilotT_T:100:start3', _C3)['deviation_projection_sd']:.4f} m, J {_sgr('l2_pilotT_J:100:start3', _C3)['deviation_projection_sd']:.4f} m였다",
       f"평균은 T {_th(_ET):.3f} N\\,m, J {_th(_EJ):.3f} N\\,m이고, 기준 뇌의 등급 결정은 T {_DT['channel_reference_brain']['learned_hand']['class_given_contact']:.3f}, J {_DJ['channel_reference_brain']['learned_hand']['class_given_contact']:.3f}이었다",
       f"최솟값은 J ${_tbr(_HJ) / 1e-9:.1f}\\cdot10^{{-9}}$ J, T ${_tbr(_HT) / 1e-4:.1f}\\cdot10^{{-4}}$ J였다",
       f"64개 가운데 {_rt['hover_episodes']}개에서 물체를 $p_c$ 위(옆으로 {_rt['lateral_offset_median_held_window']:.3f} m, 윗면 위 {_rt['hover_tip_height_median']:.3f} m)", f"힘은 중앙값 ${_rt['force_ratio_median_contact']:.2f}f^\\star$였다",
       f"J는 {_rj['contact_episodes']}개에서 닿았고(힘 중앙값 ${_rj['force_ratio_median_contact']:.2f}f^\\star$), {_rj['hover_episodes']}개는 윗면 위 {_rj['hover_tip_height_median']:.3f} m, 옆으로 {_rj['lateral_offset_median_held_window']:.3f} m",
       f"(0.0082 대 0.0145 m)"]
if not (_p2(_et) <= 0.5 * _p2(_ej) and np.mean(_et['nom']['grasp']) >= np.mean(_ej['nom']['grasp']) - 0.05) or any(t >= j for t, j in zip(_ser(_ET), _ser(_EJ))): bad.append('pilot T decision out of sync')
if len(_HJ) != 100 or len(_HT) != 100 or sum(h['viol'] + h['nan'] for h in _HJ + _HT) != 0: bad.append('pilot T guarantee records')
if _hl.sha256(open(R_ + 'l2_pilotT_plan.md', 'rb').read()).hexdigest()[:8] != open(R_ + 'l2_pilotT_plan.hash').read().split()[-1]: bad.append('pilot T plan hash (outcome)')
if _hl.sha256(open(R_ + 'l2_attempt3_plan.md', 'rb').read()).hexdigest()[:8] != open(R_ + 'l2_attempt3_plan.hash').read().split()[-1]: bad.append('attempt 3 plan hash')
for t_ in _t7:
    if t_ not in spec3: bad.append('pilot T outcome not in the specification: ' + t_)
# attempt 3 outcome (section sec:l2a3)
_E3 = {e['it']: e for e in _ET if e['it'] < 100}; _E3.update({e['it']: e for e in _J('l2_attempt3/evals.json')}); _H3 = [h for h in _HT if h['it'] <= 100] + _J('l2_attempt3/hist.json')
_D3s = {100: _DT}; _D3s.update({i: _J(f'l2_attempt3/dissect_it{i:04d}.json') for i in (150, 200, 250, 300)}); _D300 = _D3s[300]; _r300 = _RB['l2_attempt3:300']['phase3_start']
_res = lambda i: float(np.median(_E3[i]['hand']['res_learned'])); _h1 = [_res(i) for i in (260, 270, 280, 290, 300)]; _LG = _J('l2_ledger_check.json')
_ccl = lambda e, t: (np.asarray(e[t]['class_ok'])[(np.asarray(e[t]['fbar']) > 0) & ((np.asarray(e[t]['grasp']) > 0.5) if t == 'nom' else True)])
_stp = [_D3s[i]['contact_face']['share_episodes_held_top_force_window'] for i in (100, 150, 200, 250, 300)]
_L1R = _J('l1_results.json'); _c_ref, _c_nol = float(np.mean(_L1R['ID_ref']['cost'])), float(np.mean(_L1R['ID_noled']['cost']))
_me3 = [abs(_LG[k]['phase3']['mass_error_median']) for k in _LG]
_t8 = [f"최솟값은 ${_tbr(_H3) / 1e-4:.1f}\\cdot10^{{-4}}$ J였다", ', '.join(f'{v:.3f}' for v in _h1) + f" N\\,m이고 평균은 {np.mean(_h1):.3f} N\\,m였다",
       f"평가 110--150의 평균은 {np.mean([_res(i) for i in (110, 120, 130, 140, 150)]):.3f}이었다",
       f"등급 결정은 {_D300['channel_reference_brain']['learned_hand']['class_given_contact']:.3f}(64개 가운데 {round(64 * _D300['channel_reference_brain']['learned_hand']['class_given_contact'])}개)였다",
       ', '.join(f"{_D3s[i]['channel_reference_brain']['learned_hand']['class_given_contact']:.3f}" for i in (100, 150, 200, 250)) + '였다',
       f"64개 가운데 {round(64 * _stp[-1])}개({100 * _stp[-1]:.1f}\\%)였다", ', '.join(f'{100 * v:.1f}' if v else '0' for v in _stp[:4]) + '\\%였다',
       f"유지 오차는 {_p2(_E3[300]):.4f} m였고(반복 100, 150, 200, 250에 " + ', '.join(f'{_p2(_E3[i]):.4f}' for i in (100, 150, 200, 250)) + ' m)',
       f"정확히 누른 몫은 {_acc(_E3[300]):.3f}이었고(" + ', '.join(f'{_acc(_E3[i]):.3f}' for i in (100, 150, 200, 250)) + ')',
       f"64개 가운데 {_r300['contact_episodes']}개가 닿았으며 닿은 경우의 힘은 중앙값 ${_r300['force_ratio_median_contact']:.2f}f^\\star$(90 백분위 ${_r300['force_ratio_p90_contact']:.2f}f^\\star$)",
       f"잡기는 {_D300['reach']['grasp']:.3f}, 최근접 거리의 중앙값은 {_D300['reach']['min_dist_median']:.4f} m, 2 cm 안에 든 몫은 {_D300['reach']['frac_within_2cm']:.3f}, 가장 가까운 순간에 잡을 만큼 느린 몫은 {_D300['reach']['frac_slow_when_nearest']:.3f}였다",
       f"단계 3에 물체를 든 episode는 {_D300['press_detail']['n_held_phase3']}개였다",
       f"반복 200에 {_cs('l2_attempt3:200:start2', _C2)}, 300에 {_cs('l2_attempt3:300:start2', _C2)}이고, 단계 3의 $p_c$ 방향은 {_cs('l2_attempt3:200:start3', _C3)}, {_cs('l2_attempt3:300:start3', _C3)}이었다",
       f"{min(_sgr(f'l2_attempt3:{i}:start{s_}', c)['deviation_projection_sd'] for i in (200, 300) for s_, c in ((2, _C2), (3, _C3))):.4f}--{max(_sgr(f'l2_attempt3:{i}:start{s_}', c)['deviation_projection_sd'] for i in (200, 300) for s_, c in ((2, _C2), (3, _C3))):.4f} m였다",
       f"$\\mu_2$에서 {_ccl(_E3[300], 'r2').mean():.3f}({_ccl(_E3[300], 'r2').size}개), $\\mu_3$에서 {_ccl(_E3[300], 'r3').mean():.3f}({_ccl(_E3[300], 'r3').size}개), $\\mu_0$에서 {_ccl(_E3[300], 'nom').mean():.3f}({_ccl(_E3[300], 'nom').size}개)",
       f"45개 가운데 {_LG['l2_attempt3:300']['phase3']['class_rule_correct_at_last_held']:.3f}이 맞았다(반복 100에는 {_LG['l2_pilotT_T:100']['phase3']['held_episodes']}개 가운데 {_LG['l2_pilotT_T:100']['phase3']['class_rule_correct_at_last_held']:.3f}",
       f"중앙값으로 {min(_me3):.2f}--{max(_me3):.2f} kg 작았다", f"평균 비용은 {_c_ref:.3f}, 장부를 가린 기준 뇌는 {_c_nol:.3f}이었고(차이 {_c_nol - _c_ref:.3f})",
       f"평가 비용은 {np.mean(_E3[300]['nom']['cost']):.3f}이다"]
if len(_H3) != 300 or sum(h['viol'] + h['nan'] for h in _H3) != 0 or any(h['n_sat_below'] for h in _H3) or np.mean(_h1) < 1.768 or _D300['channel_reference_brain']['learned_hand']['class_given_contact'] >= 0.95 or _stp[-1] < 0.25: bad.append('attempt 3 criteria out of sync')
if _LG['l2_attempt3:300']['phase3']['held_episodes'] != 45: bad.append('attempt 3 ledger count')
_HR, _HF1, _HF3, _HC, _TX = _J('l2_hand_replay.json'), _J('l2_hand_fresh_b140.json'), _J('l2_hand_fresh_b300.json'), _J('l2_hand_contact.json'), _J('l2_transfer_exposure.json')
_H3b = _J('l2_attempt3/hist.json'); _hw = lambda a, b, k: float(np.mean([h[k] for h in _H3b if a <= h['it'] <= b]))
_cv = lambda R, m: [x[1] for x in R['arms'][m]['res_curve'][1:]]
_sh3 = [_HR['start3'][f'hand{i}']['share_contact'] for i in (100, 140, 200)]
_t9 = ["처음 시작에서 " + ', '.join(f"{_HR['start0'][f'hand{i}']['role_loss']:.3f}" for i in (100, 140, 200)) + "이고 단계 3 시작에서 " + ', '.join(f"{_HR['start3'][f'hand{i}']['role_loss']:.3f}" for i in (100, 140, 200)) + "이었다",
       "옮긴 오차는 " + ', '.join(f"{_res(i) if i > 100 else float(np.median(_ET[-1]['hand']['res_learned'])):.3f}" for i in (100, 140, 200)) + " N\\,m이다",
       f"평균 {_hw(121, 140, 'hand_loss'):.3f}에서 201--220의 {_hw(201, 220, 'hand_loss'):.3f}로", f"닿은 몫은 {_hw(121, 140, 'touch3'):.3f}에서 {_hw(201, 220, 'touch3'):.3f}이 되었다",
       f"전체의 {100 * _HR['start3']['contact_share_of_steps']:.0f}\\%이고, 가중 role loss의 {100 * min(_sh3):.0f}--{100 * max(_sh3):.0f}\\%",
       f"옮긴 오차는 시작에 {_HF1['arms']['all']['res_curve'][0][1]:.3f}이었고",
       ', '.join(f'{v:.3f}' for v in _cv(_HF1, 'all')) + f"(평균 {np.mean(_cv(_HF1, 'all')):.3f})", ', '.join(f'{v:.3f}' for v in _cv(_HF3, 'all')) + f"(평균 {np.mean(_cv(_HF3, 'all')):.3f})",
       ', '.join(f'{v:.3f}' for v in _cv(_HF3, 'mask')) + f"(평균 {np.mean(_cv(_HF3, 'mask')):.3f})",
       f"모든 step {_HC['arms']['all']['res_curve'][-1][1]:.3f}, 닿은 step을 뺀 경우 {_HC['arms']['mask']['res_curve'][-1][1]:.3f}, 접촉의 시작과 끝을 짐의 변화로 다룬 경우 {_HC['arms']['reset']['res_curve'][-1][1]:.3f}였다",
       "300의 뇌에서 " + ', '.join(f"{_TX[k]['held']['0.2-0.5 rad'] + _TX[k]['held']['0.5-9 rad']:.3f}" for k in ('l2_pilotT_T:100', 'l2_attempt3:140', 'l2_attempt3:200', 'l2_attempt3:300')) + "였다",
       "몫은 " + ', '.join(f"{_TX[k]['weight_moving_share']:.3f}" for k in ('l2_pilotT_T:100', 'l2_attempt3:140', 'l2_attempt3:200', 'l2_attempt3:300')) + "이었다"]
if _HF1['hand_it'] != 140 or _HF3['brain_it'] != 300 or _HF1['brain_it'] != 140 or _HC['brain_it'] != 300 or _HR['brain_it'] != 200: bad.append('hand dissection records do not match the description')
for t_ in _t9:
    if t_ not in spec3: bad.append('attempt 3 hand dissection not in the specification: ' + t_)
for t_ in _t8:
    if t_ not in spec3: bad.append('attempt 3 outcome not in the specification: ' + t_)
# status at iteration 300 (section sec:l2status)
_g3 = lambda k: np.array([np.nan if h.get(k) is None else h[k] for h in _H3], float); _it3 = _g3('it')
_w3 = lambda a, b, k: float(np.nanmean(_g3(k)[(_it3 >= a) & (_it3 <= b)]))
_sl3 = lambda a, b: float(np.polyfit(_it3[(_it3 >= a) & (_it3 <= b)], _g3('cost')[(_it3 >= a) & (_it3 <= b)], 1)[0])
_ec3 = lambda i: float(np.mean(_E3[i]['nom']['cost'])); _kl3 = np.array([h['kl'] for h in _H3b]); _cf3 = np.array([h['clipfrac'] for h in _H3b])
_TR = _J('l2_traj_check.json'); _NR = _J('l2_never_rest.json')['l2_attempt3:300']; _t0d, _t0s = _TR['start0']['deterministic'], _TR['start0']['sampled']
_oc, _rh, _pr = _t0d['outcome'], _t0d['reach'], _t0d['press']; _co = _pr['regression_const_light_heavy_classgap']
_mt = lambda v: f'${v:.3f}$' if v < 0 else f'{v:.3f}'
_t10 = [f"반복 0, 100, 300에 {_ec3(0):.2f}, {_ec3(100):.2f}, {_ec3(300):.2f}이었고", f"표준오차는 {np.asarray(_E3[300]['nom']['cost']).std() / 8:.2f}이다",
        f"101--200에 {_mt(_sl3(101, 200))}, 201--300에 {_mt(_sl3(201, 300))}, 251--300에 {_mt(_sl3(251, 300))}였다",
        f"중앙값은 {np.median(_kl3):.4f}이고, clip된 표본의 몫은 중앙값 {np.median(_cf3):.2f}이었다. 반복 1--10의 평균은 {_w3(1, 10, 'kl'):.3f}와 {_w3(1, 10, 'clipfrac'):.3f}이었다",
        f"0.211에서 261--300의 {_w3(261, 300, 'sd_goal'):.4f}로, 강성 행동이 {_w3(1, 10, 'sd_K'):.3f}에서 {_w3(261, 300, 'sd_K'):.3f}로",
        f"처음 시작 {_w3(261, 300, 'ev_orig'):.3f}, 단계 2 시작 {_w3(261, 300, 'ev_r2'):.3f}, 단계 3 시작 {_w3(261, 300, 'ev_r3'):.3f}이었고, 단계 4의 결정에서는 {_D300['explore']['critic_explained_variance_by_phase'][3]:.3f}이었다",
        f"단계 1에 잡은 {_oc['grasp_in_phase1']['episodes']}개가 {_oc['grasp_in_phase1']['cost']:.3f}, 단계 2에 잡은 {_oc['grasp_in_phase2']['episodes']}개가 {_oc['grasp_in_phase2']['cost']:.3f}(단계 2 시작 뒤 지연의 중앙값 {_oc['grasp_in_phase2']['grasp_delay_into_phase2_median_s']:g} s), 잡지 못한 {_oc['never']['episodes']}개가 {_oc['never']['cost']:.3f}이었다",
        f"든 몫이 {_NR['never']['share_ever_inside_2cm']:.3f}이고 최근접 거리의 중앙값이 {_NR['never']['min_dist_median']:.4f} m였다. 잡는 점은 반경 {_NR['never']['grasp_point_radius_median']:.3f} m, 높이 {_NR['never']['grasp_point_height_median']:.3f} m(중앙값)로, 단계 1에 잡은 episode의 {_NR['phase1']['grasp_point_radius_median']:.3f} m, {_NR['phase1']['grasp_point_height_median']:.3f} m보다",
        f"중앙값 {_rh['speed_inside_2cm_median']:.3f} rad/s였고", f"결정의 몫은 {_rh['share_inside_2cm_slow']:.3f}였다", f"중앙값은 {_NR['dp_norm_inside_ball_median_m']:.4f} m였다",
        f"힘을 낸 {_pr['episodes_with_window_force']}개의", f"\\bar f\\approx{_co[0]:.3f}+{_co[1]:.3f}\\,f_{{\\mathrm{{light}}}}+{_co[2]:.3f}\\,f_{{\\mathrm{{heavy}}}}{_co[3]:.3f}\\,h\\,(f",
        f"중앙값은 {_pr['relative_error_median']:.3f}이고, 두 목표의 중간 힘으로 눌렀다면 {_pr['relative_error_if_mid_force']:.3f}이다",
        f"든 {_t0d['release']['held_at_phase4']}개 가운데 결정론적 평가(logit $>0$)에서 놓은 몫은 {_t0d['release']['released_share']:.3f}였다",
        f"표본으로 뽑으면 {_t0s['release']['released_share']:.3f}이 놓았고, 지연의 중앙값은 {_t0s['release']['delay_median_s']:.2f} s였다",
        f"여는 확률은 {_D300['explore']['p_open_while_attached_by_phase'][3]:.3f}이므로", f"결정론적 평가에서 {_t0d['cost_by_phase'][3]:.3f}, 표본에서 {_t0s['cost_by_phase'][3]:.3f}였다"]
if abs(_t0d['cost'] - _ec3(300)) > 1e-9: bad.append('trajectory check does not reproduce the evaluation at 300')
if _w3(1, 10, 'sd_goal') < 0.2105 or _w3(1, 10, 'sd_goal') >= 0.2115: bad.append('status: sd_goal 1-10')
for t_ in _t10:
    if t_ not in spec3: bad.append('status at 300 not in the specification: ' + t_)
# handoff document (docs/20261115_handoff.md): key numbers against the records
_HO = open(paths.HANDOFF, encoding='utf-8').read(); _PC = open(R_ + 'port_check_cloud.log', encoding='utf-8').read()
_t11 = [f"평가 260–300의 손 전이 잔차 평균 {np.mean(_h1):.3f} N·m", f"등급 결정 {_D300['channel_reference_brain']['learned_hand']['class_given_contact']:.3f}(기준 0.95)",
        f"윗면을 누른 몫 {_stp[-1]:.3f}", f"정확히 누른 몫 {_acc(_E3[300]):.3f}", f"유지 오차 {_p2(_E3[300]):.4f} m", f"평가 비용 {_ec3(300):.2f}(L1 기준 뇌 {_c_ref:.3f})",
        f"평균 비용 {_oc['never']['cost']:.3f}", f"{_t0d['release']['held_at_phase4']}개 중 {_t0d['release']['released_share']:.3f}만 놓지만 표본 gripper로는 {_t0s['release']['released_share']:.3f}",
        f"0.44에서 300에 {_res(300):.2f}(중앙값)", f"L1 규칙으로 {_LG['l2_attempt3:300']['phase3']['class_rule_correct_at_last_held']:.3f}",
        f"KL 중앙값 {np.median(_kl3):.4f}, clip 몫 {np.median(_cf3):.2f}, goal 표준편차 {_w3(1, 10, 'sd_goal'):.3f} → {_w3(261, 300, 'sd_goal'):.4f}, 훈련 비용 기울기 ${_sl3(251, 300):.3f}$/반복(251–300)",
        f"최소 tank-bound margin ${_tbr(_H3) / 1e-4:.1f}\\cdot10^{{-4}}$ J", f"반복 101–300은 {sum(h['t_it'] for h in _H3b) / 3600:.2f} h"]
if f"{float(np.median(_E3[30]['hand']['res_learned'])):.2f}" != '0.44': bad.append('handoff: transfer residual at 30')
for k in ('17.370263', '6.421145', '4.681267', 'REPRODUCTION EXACT', '7.79 s per batch, 1498 decisions/s', '26.99 s per batch, 1716 decisions/s'):
    if k not in _PC: bad.append('port check log: ' + k)
for k in ('17.370263, 6.421145, 4.681267', '7.79 s/batch(1498 결정/s)', '26.99 s/batch(1716 결정/s)'):
    if k not in _HO: bad.append('handoff: port check value ' + k)
for t_ in _t11:
    if t_ not in _HO: bad.append('handoff number not found: ' + t_)
# platform reproduction: per-start means of the PC baseline (records/port_check_pc_baseline.json) against verified_values.md and the handoff
_PB = _j.load(open(R_ + 'port_check_pc_baseline.json'))['starts']; _VV = open(R_ + 'verified_values.md', encoding='utf-8').read()
_pbm = []
for _tg in ('nom', 'r2', 'r3'):
    _cs = [float.fromhex(x) for x in _PB[_tg]['cost_hex']]
    if len(_cs) != 64 or _cs != _PB[_tg]['cost'] or f"{np.mean(_cs):.6f}" != _PB[_tg]['mean_cost']: bad.append('pc baseline record inconsistent: ' + _tg)
    _pbm.append(f"{np.mean(_cs):.6f}")
if _pbm != ['17.373252', '6.421073', '4.681267']: bad.append('pc baseline means changed: ' + ', '.join(_pbm))
if 'mean cost ' + ', '.join(_pbm) not in _VV: bad.append('verified_values: pc baseline means')
if ', '.join(_pbm) not in _HO: bad.append('handoff: pc baseline means')
if 'mean cost ' + ', '.join(f"{float(np.mean(_J('l2_attempt3/evals.json')[-1][t]['cost'])):.6f}" for t in ('nom', 'r2', 'r3')) not in _VV: bad.append('verified_values: cloud means')
div = open(R_ + 'verified_values.md', encoding='utf-8').read().split('## Divergence (open)')[1].split('##')[0].strip()
if div != '(none)': bad.append('divergence list not empty')
# contact-permitting constraint (specification revision 17): its own gate, script/check_sync_contact.py
sys.path.insert(0, str(paths.ROOT / 'script')); import check_sync_contact
bad += ['contact: ' + b for b in check_sync_contact.bad]
print('sync findings:', len(bad))
for b in bad: print(' -', b)
sys.exit(1 if bad else 0)
