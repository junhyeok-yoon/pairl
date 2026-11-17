"""Independent references for the Stage B dynamics: (1) SymPy Lagrangian in joint coordinates, (2) MuJoCo."""
import numpy as np, sympy as sp, mujoco

def sympy_model():
    q = sp.symbols('q1:4'); dq = sp.symbols('dq1:4')
    L = sp.symbols('L1:4'); m = sp.symbols('m1:4'); lc = sp.symbols('lc1:4'); Ic = sp.symbols('Ic1:4')
    Ir, mo, rx, ry, g0 = sp.symbols('Ir mo rx ry g0')
    th = [q[0], q[0] + q[1], q[0] + q[1] + q[2]]; om = [dq[0], dq[0] + dq[1], dq[0] + dq[2] + dq[1]]
    joint = [sp.Matrix([0, 0])]
    for i in range(3):
        joint.append(joint[-1] + L[i] * sp.Matrix([sp.cos(th[i]), sp.sin(th[i])]))
    com = [joint[i] + lc[i] * sp.Matrix([sp.cos(th[i]), sp.sin(th[i])]) for i in range(3)]
    obj = joint[3] + rx * sp.Matrix([sp.cos(th[2]), sp.sin(th[2])]) + ry * sp.Matrix([-sp.sin(th[2]), sp.cos(th[2])])
    qv, dqv = sp.Matrix(q), sp.Matrix(dq)
    vel = lambda r: r.jacobian(qv) * dqv
    KE = sum(sp.Rational(1, 2) * m[i] * (vel(com[i]).T * vel(com[i]))[0] + sp.Rational(1, 2) * Ic[i] * om[i] ** 2 for i in range(3))
    KE += sp.Rational(1, 2) * mo * (vel(obj).T * vel(obj))[0] + sp.Rational(1, 2) * Ir * sum(d ** 2 for d in dq)
    PE = g0 * (sum(m[i] * com[i][1] for i in range(3)) + mo * obj[1])
    M = sp.hessian(KE, dq)
    dKdq = sp.Matrix([KE]).jacobian(qv).T
    bias = (sp.Matrix([KE]).jacobian(dqv).T).jacobian(qv) * dqv - dKdq + sp.Matrix([PE]).jacobian(qv).T      # C dq + g
    Jt = joint[3].jacobian(qv)
    args = list(q) + list(dq) + list(L) + list(m) + list(lc) + list(Ic) + [Ir, mo, rx, ry, g0]
    return sp.lambdify(args, [M, bias, joint[3], Jt, sp.Matrix([PE]).jacobian(qv).T, KE, PE], 'numpy')

def mujoco_model(par, xi):
    L, m, lc, Ic = [np.asarray(par[k], float) for k in ('L', 'm', 'lc', 'Icm')]
    Fv, Ir = np.asarray(par['Fv'], float), float(par['Ir'])
    obj = f'<body pos="{float(L[2] + par["rx"])} {float(par["ry"])} 0"><inertial pos="0 0 0" mass="{float(par["mo"])}" diaginertia="1e-12 1e-12 1e-12"/></body>' if xi else ''
    def body(i, inner):
        pos = "0 0 0" if i == 0 else f"{L[i-1]} 0 0"
        return (f'<body pos="{pos}"><joint type="hinge" axis="0 0 1" damping="{Fv[i]}" armature="{Ir}"/>'
                f'<inertial pos="{lc[i]} 0 0" mass="{m[i]}" diaginertia="{Ic[i]} {Ic[i]} {Ic[i]}"/>{inner}</body>')
    xml = ('<mujoco><compiler angle="radian"/><option gravity="0 -9.81 0" timestep="0.0005" integrator="RK4"><flag contact="disable"/></option><worldbody>'
           + body(0, body(1, body(2, f'<site name="tip" pos="{L[2]} 0 0"/>{obj}'))) + '</worldbody></mujoco>')
    mdl = mujoco.MjModel.from_xml_string(xml)
    return mdl, mujoco.MjData(mdl)
