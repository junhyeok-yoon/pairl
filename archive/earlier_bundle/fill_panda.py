import os
import pickle

E = pickle.load(open("eval_panda.pkl", "rb"))
res = E["res"]
cols = [("nominal", "nominal"), ("C0", "C0"), ("C1", "C1, gain 100")]
if os.path.exists("eval_panda_c1b.pkl"):
    r = pickle.load(open("eval_panda_c1b.pkl", "rb"))["res"]
    for s in res:
        res[s]["C1b"] = r[s]["C1"]
    cols.append(("C1b", "C1, gain 30"))
cols.append(("pH", "pH, 30회"))
if os.path.exists("eval_panda2.pkl"):
    r = pickle.load(open("eval_panda2.pkl", "rb"))["res"]
    for s in res:
        res[s]["pH2"] = r[s]["pH"]
    cols.append(("pH2", "pH, 140회"))


def cell(r, key, d):
    v = f"{r[key]:.{d}f}"
    return v + (f" ({100*r['div']:.0f}\\%)" if r["div"] > 0 else "")


out = []
for key, d, cap, lab in [("rms", 3, "Tracking 오차 RMS [도]", "tab:panda"), ("dtau", 2, "Torque 편차 RMS [N\\,m]", "tab:panda_tau")]:
    out += ["\\begin{table}[H]", "\\centering\\small",
            f"\\caption{{7-DOF: {cap}. 괄호는 발산 비율이며, 값은 발산하지 않은 궤적만으로 계산했다.}}", f"\\label{{{lab}}}",
            "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
            "시험 & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for sid, name in E["sets"]:
        out.append(name + " & " + " & ".join(cell(res[sid][c[0]], key, d) for c in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
block = "%%TAB_PANDA_BEGIN%%\n" + "\n".join(out) + "%%TAB_PANDA_END%%"
p = "note/note.tex"
s = open(p, encoding="utf-8").read()
i0, i1 = s.index("%%TAB_PANDA_BEGIN%%"), s.index("%%TAB_PANDA_END%%") + len("%%TAB_PANDA_END%%")
open(p, "w", encoding="utf-8").write(s[:i0] + block + s[i1:])
for sid, name in E["sets"]:
    print(sid, {c[0]: (round(float(res[sid][c[0]]["rms"]), 3), round(float(res[sid][c[0]]["dtau"]), 2), float(res[sid][c[0]]["div"])) for c in cols})
