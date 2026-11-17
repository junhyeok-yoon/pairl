"""Single source of the Stage B numbers. Values fixed by the design study (records/stageb_design.json)."""
import numpy as np

L_NOM = np.array([0.40, 0.35, 0.25]); M_NOM = np.array([3.0, 2.0, 1.0]); IR = 0.02
UB = np.array([120.0, 60.0, 25.0])                       # input limits [N m]
QC = np.array([0.9, -1.3, -0.5]); QH = np.array([0.9, 1.1, 1.2])   # joint box center and half-widths [rad]
PG = 8                                                   # exponent of the gauge
BOX = ((0.35, 0.75), (-0.15, 0.35))                      # workspace of grasp points and targets [m]
WE = 0.15                                                # half-width of the block [m]
TS, TH = 0.002, 0.05                                     # control and decision periods [s]
NB = int(round(TH / TS))
GRASP_DIST, GRASP_SPEED = 0.02, 0.1                      # [m], [rad/s]
CLEAR = 0.03                                             # grasp point and initial tip at least this far above the block top [m]
LOBJ_MAX = 0.08                                          # largest distance between the object's center of mass and grasp point [m]
MARG_NORMAL, MARG_HARD = 0.25, (0.08, 0.20)              # gauge margins of normal and limit-adjacent targets
P_HARD, P_BUDGET = 0.3, 0.3
E_NOM = 60.0                                             # nominal energy budget [J]
M_TH = 0.85                                              # payload class threshold [kg]; heavy class: m_obj > M_TH
MO_ID = ((0.2, 0.65), (1.05, 1.5))                       # in-distribution payload: uniform on the union [kg]
F_RANGE, F_SEP = (3.0, 15.0), 4.0                        # force targets of the two classes [N] and their least separation [N]
RHO_BUDGET = (0.6, 1.0)                                  # budget episodes: E0 = Delta T + rho E_task, rho uniform here
SIG_TAU, SIG_P, SIG_DQ, Q_QUANT, BIAS_TAU = 0.05, 0.002, 0.01, 1e-4, 0.05
NOISE_CLIP = 3.0                                         # torque noise truncated at 3 sd

def _two_sided(rng, lo1, hi1, lo2, hi2, n):
    return np.where(rng.random(n) < 0.5, rng.uniform(lo1, hi1, n), rng.uniform(lo2, hi2, n))

def make_par(rng, ood=False, payload=None, v=2):
    """Plant parameters of one episode (no block: placed far away). v=1: task version of L0 and the L1 pilot."""
    if ood:
        L = L_NOM * _two_sided(rng, 0.85, 0.95, 1.05, 1.15, 3); m = M_NOM * _two_sided(rng, 0.7, 0.9, 1.1, 1.3, 3); mo = rng.uniform(1.5, 2.5)
    else:
        L = L_NOM.copy(); m = M_NOM * rng.uniform(0.9, 1.1, 3)
        if v == 1: mo = rng.uniform(0.2, 1.5)
        else:                                                 # one uniform draw, as in v=1, mapped onto the two class intervals
            u = rng.uniform(0.0, 1.0); (a1, b1), (a2, b2) = MO_ID
            mo = a1 + 2 * u * (b1 - a1) if u < 0.5 else a2 + 2 * (u - 0.5) * (b2 - a2)
    if payload is not None: mo = payload
    Lobj = rng.uniform(0.0, LOBJ_MAX)                          # distance between the object's center of mass and its grasp point
    rr, ra = rng.uniform(0.0, LOBJ_MAX + GRASP_DIST), rng.uniform(-np.pi, np.pi)   # attached offset (direct-attachment tests); env sets it at each grasp
    return dict(L=L, m=m, lc=L / 2, Icm=m * L ** 2 / 12, Ir=np.float64(IR), Fv=rng.uniform(0.05, 0.3, 3), d=rng.uniform(-0.5, 0.5, 3),
                mo=np.float64(mo), Lobj=np.float64(Lobj), rx=np.float64(rr * np.cos(ra)), ry=np.float64(rr * np.sin(ra)),
                xe=np.float64(10.0), ye=np.float64(-10.0), we=np.float64(WE), ke=np.float64(0.0), de=np.float64(0.0), nu=np.float64(0.0))
