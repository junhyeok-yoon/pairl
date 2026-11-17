import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = pickle.load(open("struct_panda.pkl", "rb"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for i, r in enumerate(S):
    t = np.arange(len(r["V"])) * 0.02
    col = plt.cm.viridis(i / len(S))
    v = r["V"] - r["V"][-1]
    ax[0].semilogy(t[:-1], v[:-1] / v[0] + 1e-6, color=col, lw=1.1)
    ax[1].semilogy(t, r["en"] + 1e-4, color=col, lw=0.9)
ax[0].set_title("7-DOF, random weights: (V(t) - V(8 s)) / (V(0) - V(8 s))", fontsize=9)
ax[1].set_title("7-DOF, random weights: |q - q_d| [deg]", fontsize=9)
for a in ax:
    a.set_xlabel("time [s]"); a.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig("note/fig_panda_struct.png", dpi=150); plt.close()

try:
    E = pickle.load(open("eval_panda.pkl", "rb"))
except FileNotFoundError:
    raise SystemExit("eval missing")
res, traj = E["res"], E["traj"]
import os
have2 = os.path.exists("eval_panda2.pkl")
if have2:
    E2 = pickle.load(open("eval_panda2.pkl", "rb"))
    for s_ in res:
        res[s_]["pH2"] = E2["res"][s_]["pH"]
    traj[("C", "pH2")] = E2["traj"][("C", "pH")]
COL = {"nominal": "0.85", "C0": "0.6", "C1": "0.15", "pH": "lightskyblue", "pH2": "tab:blue"}
LAB = {"nominal": "fixed nominal model", "C0": "C0 Slotine-Li", "C1": "C1 oracle basis", "pH": "pH learned, 30 it.", "pH2": "pH learned, 140 it."}
names = ["nominal", "C0", "C1", "pH"] + (["pH2"] if have2 else [])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.1, 1.4]})
sets = [s for s, _ in E["sets"]]
xl = {"A": "in-dist.", "B": "payload 6 kg,\nfriction x2", "C": "pick-up", "D": "15 s horizon"}
x, w = np.arange(len(sets)), 0.8 / len(names)
for i, cn in enumerate(names):
    ax[0].bar(x + (i - (len(names) - 1) / 2) * w, [res[s][cn]["rms"] for s in sets], w, color=COL[cn], label=LAB[cn])
ax[0].set_xticks(x); ax[0].set_xticklabels([xl[s] for s in sets], fontsize=8); ax[0].set_ylabel("tracking error RMS [deg]")
ax[0].set_yscale("log"); ax[0].grid(axis="y", alpha=0.3, which="both"); ax[0].legend(fontsize=7)
for cn in ["C0", "C1"] + (["pH2"] if have2 else ["pH"]):
    o = traj[("C", cn)][0]
    t = np.arange(o.shape[0]) * 0.004
    ax[1].plot(t, np.degrees(np.linalg.norm(o[:, :7], axis=1)), color=COL[cn], lw=1.0, label=LAB[cn])
ax[1].axvline(3.0, color="k", ls=":", lw=0.8)
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("|q - q_d| [deg]"); ax[1].grid(alpha=0.3)
ax[1].set_title("payload pick-up at t = 3 s (one task)", fontsize=9); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("note/fig_panda_perf.png", dpi=150); plt.close()
print("figs done")
