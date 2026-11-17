import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panda as P


def chain(q):
    Rs, ps = P.fk(jnp.asarray(q))
    pts = [np.zeros(3)] + [np.asarray(p) for p in ps]
    flange = np.asarray(ps[-1] + Rs[-1] @ jnp.array([0.0, 0.0, P.D_FLANGE]))
    return np.array(pts + [flange]), np.asarray(Rs[-1])


fig = plt.figure(figsize=(10, 4.2))
ax = fig.add_subplot(1, 2, 1, projection="3d")
a, eta, rp = P.sample_tasks(random.PRNGKey(3), 1)
ts = np.linspace(0, 6, 7)
for i, t in enumerate(ts):
    qd = np.asarray(P.ref(t, rp[0])[0])
    pts, R7 = chain(qd)
    al = 0.25 + 0.75 * (i == 0)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="tab:blue" if i == 0 else "0.5", lw=2.5 if i == 0 else 1, ms=3, alpha=al)
pts, R7 = chain(np.asarray(P.Q0))
ax.scatter(*pts[-1], s=90, color="tab:orange", label="payload (0-3 kg, unknown)")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_xlim(-0.3, 0.7); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 1.0)
ax.set_title("7-DOF arm (Panda kinematics): nominal pose\nand reference snapshots over 6 s", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
ax2 = fig.add_subplot(1, 2, 2)
t = np.linspace(0, 6, 601)
qd = np.array([np.asarray(P.ref(tt, rp[0])[0]) for tt in t])
for j in range(P.N):
    ax2.plot(t, qd[:, j], lw=1.2, label=f"joint {j+1}")
ax2.set_xlabel("time [s]"); ax2.set_ylabel("reference joint angle [rad]"); ax2.grid(alpha=0.3)
ax2.legend(fontsize=7, ncol=2); ax2.set_title("one sampled reference task", fontsize=9)
plt.tight_layout(); plt.savefig("note/fig_panda_arm.png", dpi=150)
print("ok")
