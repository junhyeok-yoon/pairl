"""Constraint-layer storage inequality: for tau_b = -H^{-1} grad Phi(w), Phi(w) = Psi(rho(w)) convex, U_b(q) = Phi(H^{-1}(q - c)):
   qd^T tau_b + dU_b/dt <= 0. Compared with the normalized-direction form tau_b = -s g/|g| (earlier revision)."""
import numpy as np, json
from pairl import paths
PG, K1, KB, RON, RFULL = 8, 2.0, 20.0, 0.8, 0.9
QC = np.array([0.9, -1.3, -0.5]); QH = np.array([0.9, 1.1, 1.2])
rho = lambda w: np.sum(np.abs(w) ** PG, -1) ** (1.0 / PG)
grad = lambda w: np.abs(w) ** (PG - 1) * np.sign(w) / rho(w)[..., None] ** (PG - 1)
st = lambda r: np.clip((r - RON) / (RFULL - RON), 0, 1) * KB * r / np.maximum(1 - r ** 2, 1e-12)      # Psi'(rho)
rng = np.random.default_rng(0); n = 2_000_000
w = rng.uniform(-1, 1, (n, 3)); wq = rng.uniform(-1, 1, (n, 3)); keep = (rho(w) < 0.999) & (rho(wq) < 0.999); w, wq = w[keep], wq[keep]
dq = K1 * QH * (w - wq)                                             # joint velocity implied by (w, w_q)
tau_new = -st(rho(w))[:, None] * grad(w) / QH
g = grad(w) / QH; tau_old = -st(rho(w))[:, None] * g / np.linalg.norm(g, axis=-1, keepdims=True)
dU = st(rho(wq)) * np.sum(grad(wq) * (dq / QH), -1)                 # dU_b/dt = Psi'(rho(w_q)) grad rho(w_q)^T H^{-1} qd
dU_old = dU / np.linalg.norm(grad(wq) / QH, axis=-1)                # candidate storage rate for the normalized form (any scaling shown below)
new = np.sum(dq * tau_new, -1) + dU
# normalized form: does some storage U = c * Phi(w_q) work? test c on a grid
p_old = np.sum(dq * tau_old, -1)
best = min((float(np.max(p_old + c * dU)), c) for c in np.linspace(0.2, 3.0, 57))
R = dict(n=int(keep.sum()), new_form_max=float(new.max()), new_form_fraction_active=float(np.mean(st(rho(w)) + st(rho(wq)) > 0)),
         old_form_best_max_over_scalings=best[0], old_form_best_scaling=best[1])
print(json.dumps(R)); json.dump(R, open(str(paths.RECORDS / 'barrier_storage_check.json'), 'w'), indent=1)
