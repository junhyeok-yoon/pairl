"""Hand diagnostic of attempt 2 (plan, amendment 7 (a)): median transfer residual on the 64 evaluation episodes for the hand of each listed checkpoint and for the
hand whose weights are the average of those checkpoints. Usage: python3 l2_hand_check.py OUT_JSON RUN IT1 IT2 ..."""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import params as P

OUT, RUN, ITS = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
ev_nom = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM)
hps = [jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f'{RUN}/params_it{i:04d}.pkl', 'rb'))['hp']) for i in ITS]
med = lambda hp: float(np.median(l2.hand_diagnostics(hp, ev_nom)['res_learned']))
per = [med(hp) for hp in hps]
avg = jax.tree_util.tree_map(lambda *a: sum(a) / len(a), *hps); ra = med(avg)
R = dict(run=RUN.rstrip('/').split('/')[-1], iterations=ITS, per_checkpoint=per, per_checkpoint_q25=float(np.percentile(per, 25)), per_checkpoint_median=float(np.median(per)),
         averaged=ra, averaged_below_q25=bool(ra < np.percentile(per, 25)))
json.dump(R, open(OUT, 'w'), indent=1); print(json.dumps(R))
