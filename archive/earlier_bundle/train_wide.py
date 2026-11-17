import pickle
from jax import random
import arm3 as P
import train_arm3 as TR

_, g, k, ke = pickle.load(open("arm3_classical.pkl", "rb"))["c0"]
k0, k1, k2, k3 = random.split(random.PRNGKey(0), 4)
TR.train(P.ph_wide_op, P.NZ, P.init_ph_wide(k1, gamma0=g, k0=k, ke0=ke), k2, 250, 3e-3, "wide", bs=16)
