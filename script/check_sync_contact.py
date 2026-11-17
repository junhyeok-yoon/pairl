"""Consistency gate of the contact-permitting constraint in the specification: every number cited in the sections "접촉을 허용하는 제약" and
"접촉력 상한의 상수와 수치 확인" of spec.tex and in records/verified_values.md (contact sections) is recomputed from the records and compared as a string;
the plan hash chains, the freshness of the figures and the divergence ledger are checked. Prints 'sync findings: N'.
Imported by script/check_sync.py (findings prefixed 'contact:'); alone: python script/check_sync_contact.py [PATH/TO/spec.tex]"""
import json, hashlib, os, sys
from pathlib import Path
from pairl import paths
R, FIG = paths.RECORDS, paths.FIGURES
TEX = Path(sys.argv[1]) if __name__ == '__main__' and len(sys.argv) > 1 else paths.NOTES / 'spec.tex'; bad = []
tex = open(TEX, encoding='utf-8').read(); vv = open(R / 'verified_values.md', encoding='utf-8').read()
K, C, Pk, D, A = (json.load(open(R / f)) for f in ('contact_bound_constants.json', 'contact_bound_check.json', 'contact_peak_attempt3.json', 'contact_class_design.json', 'contact_authority_limit.json'))
def need(s, where=('tex', 'vv'), what=''):
    for w in where:
        if s not in (tex if w == 'tex' else vv): bad.append(f'{what or s!r} missing in {w}')
neg = lambda x, nd: f'$-{abs(x):.{nd}f}$' if x < 0 else f'{x:.{nd}f}'
row = lambda vals: ' & '.join(vals) + r' \\'
sci = lambda x: (lambda m, e: f'{m}\\times10^{int(e)}')(*f'{x:.3e}'.split('e'))
# sampled constants
cs = [K['cases'][k] for k in ('in_xi0', 'in_xi1', 'ood_xi0', 'ood_xi1')]
need(row([f"{c['n_region']:,}".replace(',', '{,}') for c in cs]), ('tex',), 'table constants: region counts')
for key, f in (('lam_nn_min', lambda c: f"{c['lam_nn_min']:.4f}"), ('theta 0.5', lambda c: f"{c['mu_0.5']['min']:.4f}"), ('theta 0.8', lambda c: neg(c['mu_0.8']['min'], 4)),
               ('share', lambda c: f"{c['mu_0.8']['share_nonpositive']:.5f}" if c['mu_0.8']['share_nonpositive'] > 0 else '0'), ('speed', lambda c: f"{c['vc_max']:.3f}"),
               ('a_in', lambda c: f"{c['a_box_max']:.1f}"), ('relief', lambda c: neg(c['a_relief_max_slow'], 1))):
    need(row([f(c) for c in cs]), ('tex',), f'table constants: {key}')
for c in cs:
    for s_ in (f"{c['lam_nn_min']:.4f}", f"{c['mu_0.5']['min']:.4f}", f"{abs(c['mu_0.8']['min']):.4f}", f"{c['vc_max']:.3f}", f"{c['a_box_max']:.1f}", f"{abs(c['a_relief_max_slow']):.1f}"): need(s_, ('vv',))
ci = K['class_in']
for s_ in (f"\\underline\\theta={ci['mu_low']:.4f}", f"\\bar v_c={ci['vc']:.3f}", f"\\bar a_{{\\mathrm{{in}}}}={ci['a_box']:.1f}", f"a_r={ci['a_r']:.2f}"): need(s_, ('tex',))
for s_ in (f"{ci['mu_low']:.4f}", f"{ci['vc']:.3f}", f"{ci['a_box']:.1f}", f"{ci['a_r']:.2f}", f"{K['present_parts']['F_a']:.1f}", f"{K['present_parts']['impact_k20000']:.1f}"): need(s_)
for k in ('k20000', 'k200000'): need(sci(K['present'][k]), ('tex',)); need(f"{K['present'][k]:.1f}", ('vv',))
if tex.count(sci(K['present']['k20000'])) != 2: bad.append('present bound not cited twice (section and status table)')
if 2 * ci['a_box'] / ci['mu_low'] <= 60: bad.append('2 a_in / theta does not exceed 60 N; the text says no speed limit closes the condition')
# authority condition
b3 = D['b3']
for s_ in (f"{b3['60']['in']:.2f}", f"{b3['60']['ood']:.2f}", f"{abs(b3['100']['in']):.2f}", f"{abs(b3['100']['ood']):.2f}"): need(s_)
if not (b3['60']['in'] > 0 and b3['60']['ood'] > 0 and b3['100']['in'] < 0 and b3['100']['ood'] < 0): bad.append('signs of the (B3) margins differ from the text')
for ub in ((120, 60, 25), (150, 75, 40), (180, 90, 50), (240, 120, 60)):
    a, b = (A['-'.join(map(str, ub)) + s_] for s_ in ('_in', '_ood'))
    need(f"({ub[0]}, {ub[1]}, {ub[2]}) & {a['margin_free']:.2f} & {b['margin_free']:.2f} & {a['fstar_all']:.2f} & {b['fstar_all']:.2f} \\\\", ('tex',), f'table authority {ub}')
    for x in (a['margin_free'], b['margin_free'], a['fstar_all'], b['fstar_all'], a['fstar_region'], b['fstar_region']): need(f'{x:.2f}', ('vv',))
a, b = A['120-60-25_in'], A['120-60-25_ood']
for s_ in (f"{a['fstar_all']:.2f} N", f"{b['fstar_all']:.2f} N", f"{a['fstar_region']:.2f} N과 {b['fstar_region']:.2f} N"): need(s_, ('tex',))
if abs(A['120-60-25_in']['margin_60'] - b3['60']['in']) > 1e-9 or abs(A['120-60-25_ood']['margin_60'] - b3['60']['ood']) > 1e-9: bad.append('E0: reimplemented margin differs from D1')
if round(b['fstar_all']) != 72 or round(A['240-120-60_ood']['fstar_all']) != 157: bad.append('rounded force limits 72 N / 157 N differ from the text')
if not (70.0 < min(a['fstar_all'], b['fstar_all'])) or '\\bar f_c=70$ N' not in tex: bad.append('decided force limit 70 N is not below F* or not stated')
need(f"최소 {D['f_stat']['min']:.1f} N, 중앙값 {D['f_stat']['median']:.1f} N", ('tex',)); need(f"{D['f_stat']['min']:.1f}", ('vv',)); need(f"{D['f_stat']['median']:.1f}", ('vv',))
# speed limits
for k, ks in ((20000, r'$2\times10^4$'), (200000, r'$2\times10^5$'), (1000000, r'$10^6$')):
    v = [D['speed'][f'class_nu05_k{k}_f{f}_td{td}'] for f in (60, 100, 200) for td in (4, 2)]
    need(ks + ' & ' + ' & '.join(f'{x:.3f}' for x in v) + r' \\', ('tex',), f'table speed {k}')
    for x in v: need(f'{x:.3f}', ('vv',))
sp = lambda name, k, f, td=4: f"{D['speed'][f'{name}_k{k}_f{f}_td{td}']:.3f}"
for s_ in (f"{sp('class_nu05', 200000, 60)} m/s이고", f"{sp('class_nu05', 1000000, 60)} m/s로", f"{sp('class_nu05', 200000, 200)} m/s와 {sp('class_nu05', 1000000, 200)} m/s", f"60 N에서 {sp('class_nu05', 20000, 60)} m/s",
           ', '.join(sp('q05_nu08', k, 60) for k in (20000, 200000, 1000000)) + ' m/s'): need(s_, ('tex',))
if abs(D['sets']['class_nu05']['theta'] - ci['mu_low']) > 0 or abs(D['sets']['class_nu05']['a_r'] - ci['a_r']) > 0: bad.append('class constants of the speed table differ from contact_bound_constants.json')
# friction 0.8 and energy
c0, c1_ = D['cases']['in_xi0'], D['cases']['in_xi1']
need(f"{100 * c0['share_nonpositive']:.3f}\\%", ('tex',)); need(f"{100 * c1_['share_nonpositive']:.3f}\\%", ('tex',))
need(', '.join(f"{c0[k]:.4f}" for k in ('theta_q01', 'theta_q05', 'theta_median')) + ' kg$^{-1}$', ('tex',)); need(', '.join(f"{c1_[k]:.4f}" for k in ('theta_q01', 'theta_q05', 'theta_median')) + ' kg$^{-1}$', ('tex',))
need(f"{c0['kin_max']:.3f} J(물체를 들면 {c1_['kin_max']:.3f} J)", ('tex',)); need(f"{c0['kin_max_slow']:.3f} J과 {c1_['kin_max_slow']:.3f} J", ('tex',))
need(', '.join(f"{D['energy'][f'k{k}_f60']:g}" for k in (20000, 200000, 1000000)) + ' J', ('tex',))
for c in (c0, c1_):
    for k in ('theta_q01', 'theta_q05', 'theta_median'): need(f'{c[k]:.4f}', ('vv',))
    for k in ('kin_max', 'kin_max_slow'): need(f'{c[k]:.3f}', ('vv',))
# registered checks C1, C2, C4
c1, c2 = C['c1'], C['c2']
if not (c1['pass_bound'] and c1['pass_tight'] and c2['pass_bound'] and c2['pass_count']): bad.append('a registered check did not pass')
if f"{c1['constant_undamped']['r_l'][0]:.6f}" != '1.000000' or f"{c1['constant_undamped']['r_l'][-1]:.6f}" != '1.000000': bad.append('C1 tightness string')
for s_ in ('1.000000', f"{c1['constant']['r_l'][-1]:.6f}", f"{c1['constant']['r_f'][-1]:.6f}", f"{c1['random']['r_l'][-1]:.6f}", f"{c1['random']['r_f'][-1]:.6f}",
           f"{c2['r_l'][-1]:.6f}", f"{c2['r_f'][-1]:.6f}", f"{c2['max_fn_all']:.1f}"): need(s_)
if (c1['bangbang']['r_l'][-1], c1['bangbang']['r_f'][-1]) != (c1['constant']['r_l'][-1], c1['constant']['r_f'][-1]): bad.append('C1 bang-bang and constant maxima differ; the text reports them together')
need(f"{c2['intervals_valid']:,}".replace(',', '{,}'), ('tex',)); need(str(c2['intervals_valid']), ('vv',))
if c2['intervals_mu_nonpositive'] != 0 or c2['n'] != 3000: bad.append('C2 counts differ from the text')
st = Pk['starts']
for key, lab in (('0', 'original'), ('2', 'phase 2'), ('3', 'phase 3')):
    v = st[key]; sh = '0' if v['share_above_cert'] == 0 else f"{v['share_above_cert']:.3f}"
    need(f"{lab} & {v['n_contact']} & {v['peak_median']:.2f} & {v['peak_p90']:.2f} & {v['peak_max']:.2f} & {sh} \\\\", ('tex',), f'table peak {lab}')
    for x in (v['peak_median'], v['peak_p90'], v['peak_max']): need(f'{x:.2f}', ('vv',))
if round(st['3']['share_above_cert'] * st['3']['n_contact']) != 5 or st['3']['n_contact'] != 63: bad.append('share above 60 N is not 5 of 63')
if st['3']['peak_max'] <= A['120-60-25_in']['fstar_all']: bad.append('max first-contact peak does not exceed F*; the text says it does')
# plan hash chains
for name in ('contact_bound_check_plan', 'contact_class_plan'):
    hl = [l for l in open(R / f'{name}.hash').read().split('\n') if l.strip()]
    if hashlib.sha256((R / f'{name}.md').read_bytes()).hexdigest()[:8] != hl[-1].split()[-1]: bad.append(f'{name}: last hash line does not match the plan file')
    need(hl[0].split()[-1], ('tex',), f'{name}: registered plan hash')
# divergence ledger and freshness
if vv.split('## Divergence (open)')[1].split('##')[0].strip() != '(none)': bad.append('divergence ledger not empty')
mt = os.path.getmtime
for fig, src in (('contact_bound_check.png', 'contact_bound_check_c2.npz'), ('force_speed_design.png', 'contact_class_design.json')):
    if mt(FIG / fig) < mt(R / src): bad.append(f'{fig} older than {src}')
if __name__ == '__main__':
    for b_ in bad: print('  -', b_)
    print(f'sync findings: {len(bad)}'); sys.exit(1 if bad else 0)
