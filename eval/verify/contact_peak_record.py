"""First-contact peak force of a learned policy (check C4 of records/contact_bound_check_plan.md): deterministic rollouts of the 64 evaluation
episodes from the three starts, as in port_check.py. Run from code/stageb of the bundle (flat imports) or from the repository root (package imports).
Generalized 2026-11-17 to any run directory: optional OUT_JSON (default records/contact_peak_<run>_it<IT>.json; C4 wrote records/contact_peak_attempt3.json),
n=N evaluation episodes per start (default 64 as in C4; the first n of the 256-episode evaluation set), and the share above 70 N (decided force limit).
Usage: python contact_peak_record.py RUN_DIR IT [OUT_JSON] [n=64]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, numpy as np, jax.numpy as jnp
from pathlib import Path
try:
    from pairl.learn import l1, l2
    from pairl.stageb import env, controller as C, params as P
    from pairl import paths; RECORDS = paths.RECORDS
except ImportError:
    import l1, l2, env, controller as C, params as P
    RECORDS = Path(__file__).resolve().parents[1] / 'records'

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
RUN, IT = POS[0], int(POS[1]); F_CERT, F_LIM = 60.0, 70.0     # normal force of the sampled certificate of (B3); decided force limit
NE = int(OPT.get('n', 64)); OUTF = Path(POS[2]) if len(POS) > 2 else RECORDS / f'contact_peak_{Path(RUN).name}_it{IT:04d}.json'
env.COST_V = 4; cfg = C.default_cfg()
ck = pickle.load(open(f'{RUN}/params_it{IT:04d}.pkl', 'rb')); l2.set_mode(ck)
bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
ev = l1.with_E0(l1.episodes(6000, NE, lim=0.7), P.E_NOM); batch = l2.make_batch_fn(cfg); REC = dict(run=Path(RUN).name, it=IT, f_cert=F_CERT, f_limit=F_LIM, n=NE, starts={})
for start in (0, 2, 3):
    eps = ev if start == 0 else [env.restart(p, t, start) for p, t in ev]; pb, tb = env.stack(eps)
    st, m, dec, rr = batch(pb, tb, jax.random.split(jax.random.PRNGKey(555), NE), bp, hp, 1.0, 0.0)
    s = l2.summarize(st, m); pk = np.asarray(s['peak_first']); hit = pk > 0; ke = np.asarray(pb['ke'])
    REC['starts'][str(start)] = dict(mean_cost=float(np.mean(s['cost'])), n_contact=int(hit.sum()), peak_median=float(np.median(pk[hit])), peak_p90=float(np.percentile(pk[hit], 90)),
                                     peak_max=float(pk[hit].max()), share_above_cert=float(np.mean(pk[hit] > F_CERT)), share_above_limit=float(np.mean(pk[hit] > F_LIM)), ke_at_max=float(ke[hit][np.argmax(pk[hit])]))
    print(f'start {start}: mean cost {np.mean(s["cost"]):.6f} | contact in {hit.sum()} of {NE} | peak_first median {np.median(pk[hit]):.2f} N, p90 {np.percentile(pk[hit], 90):.2f} N, '
          f'max {pk[hit].max():.2f} N | share above {F_CERT:.0f} N: {np.mean(pk[hit] > F_CERT):.3f}', flush=True)
json.dump(REC, open(OUTF, 'w'), indent=1)
