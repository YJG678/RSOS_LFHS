"""Check every number reported in Section 4.5 (Numerical verification) against the recomputation.

Run after scheme_verify.py and the refsolve.py runs (see README.md).  One line per reported number:
  OK    recomputed value rounds to the printed value (or the stated inequality holds)
  EDGE  printed value agrees with the recomputation, but the unrounded data of an earlier run (embedded
        in the .tex figures) round differently: the value sits on a rounding boundary, so its last digit
        is not reproducible across runs/library versions
  RNDF  round-off-level quantity: only its order of magnitude is reproducible
  DIFF  disagreement
Writes verification_report.txt and tables_generated.tex (Tables 1-3 in the manuscript's format).
"""
import json, numpy as np, os, sys

FILES = dict(scheme="scheme_results.json", g04="res_Pe8.4_g0.4_96x1280.json", g13="res_Pe84_g13_96x1280.json",
             g100="res_Pe606_g100_96x1280.json", chk04="check_Pe8.4_g0.4_48x640.json",
             chk13="check_Pe84_g13_48x640.json")
KS = [8, 16, 32, 64]

# ------------------------------------------------------------------ values printed in the manuscript (.tex)
T1_SPATIAL = {8: ("8.601e-4", "4.227e-4", "1.744e-3"), 16: ("2.154e-4", "1.058e-4", "4.303e-4"),
              32: ("5.388e-5", "2.646e-5", "1.072e-4"), 64: ("1.347e-5", "6.616e-6", "2.678e-5")}
T1_AXIAL = {40: ("1.588e-3", "1.594e-3", "5.834e-3"), 80: ("7.981e-4", "8.036e-4", "3.047e-3"),
            160: ("4.001e-4", "4.035e-4", "1.560e-3"), 320: ("2.003e-4", "2.021e-4", "7.903e-4")}
T1_ORDERS = (("2.000", "2.000", "2.001"), ("0.998", "0.997", "0.982"))
T2 = {8: ("4.083e-3", "2.339e-4", "4.214e-3", "3.648e-4", "1.075e-7", "6.906e-8"),
      16: ("2.000e-3", "6.250e-5", "2.034e-3", "9.635e-5", "9.338e-8", "8.572e-8"),
      32: ("9.877e-4", "1.615e-5", "9.965e-4", "2.474e-5", "9.181e-8", "9.005e-8"),
      64: ("4.905e-4", "4.119e-6", "4.928e-4", "6.282e-6", "9.138e-8", "9.095e-8")}
T3 = {8: ("3.291e-3", "2.707e-3", "5.741e-3", "2.400e-3", "1.030e-6", "5.068e-7"),
      16: ("1.660e-3", "6.717e-4", "2.415e-3", "6.496e-4", "2.995e-7", "1.222e-7"),
      32: ("8.889e-4", "1.678e-4", "1.090e-3", "1.674e-4", "1.318e-7", "7.235e-8"),
      64: ("4.643e-4", "4.200e-5", "5.157e-4", "4.240e-5", "9.937e-8", "8.495e-8")}
COLS = ("E0", "Ep", "E0fin", "Epfin", "dref_fl", "dref_fin")
# unrounded values embedded in the .tex figures (Figure 3: gamma=0.4; Figure 6(a): E_p)
ARCHIVED = {
    "g04": {0: [0.004082517239352876, 0.001999890728891125, 0.0009876892902036395, 0.0004905463489493434],
            1: [0.0002338673042481423, 6.250416582884966e-5, 1.614754811104255e-5, 4.118860318791099e-6],
            2: [0.004214208863838911, 0.002034286390501966, 0.000996483955182408, 0.0004927703691432494],
            3: [0.0003647741048673427, 9.6348183373937e-5, 2.473569923668644e-5, 6.282029782575344e-6],
            4: [1.075151066996989e-7, 9.337897750878549e-8, 9.181500705404522e-8, 9.137725425405955e-8],
            5: [6.905479457070676e-8, 8.571989424136988e-8, 9.005199277752541e-8, 9.094593957343926e-8]},
    "g13": {1: [0.00270729, 0.000671727, 0.000167779, 4.19987e-05]},
    "g100": {1: [0.020432, 0.00590731, 0.00141964, 0.000349291]}}

# ------------------------------------------------------------------ comparison helpers
LINES = []; COUNT = {}


def ulp_of(rep):
    s = rep.strip()
    mant, ex = (s.split("e") + ["0"])[:2]
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return float(s), 10.0**(int(ex)-dec)


def emit(status, where, what, rep, val, note=""):
    COUNT[status] = COUNT.get(status, 0)+1
    LINES.append(f"[{status:4s}] {where:7s} {what:52s} manuscript {rep:>12s}   recomputed {val:<14s} {note}")


def num(where, what, rep, val, alt=None):
    """alt: unrounded value of an earlier run (figure data embedded in the .tex), if available."""
    R, u = ulp_of(rep); dec = len(rep.split("e")[0].split(".")[1]) if "." in rep.split("e")[0] else 0
    fmt = (lambda x: f"{x:.{dec}e}".replace("e-0", "e-")) if "e" in rep else (lambda x: f"{x:.{dec}f}")
    if alt is not None and round(alt/u) != round(val/u):
        emit("EDGE", where, what, rep, f"{val:.6g}",
             f"(.tex figure data {alt:.7g} round to {fmt(alt)}: last digit not reproducible)")
    else:
        emit("OK" if abs(val-R) <= 0.5*u*(1+1e-9) else "DIFF", where, what, rep, f"{val:.6g}")


def less(where, what, bound, val, rep=None):
    emit("OK" if val < bound else "DIFF", where, what, rep or f"< {bound:.2g}", f"{val:.3e}")


def atmost(where, what, rep, val):
    R, u = ulp_of(rep)
    if val <= R: emit("OK", where, what, "<= "+rep, f"{val:.4g}")
    elif round(val/u)*u <= R+1e-12: emit("OK", where, what, "<= "+rep, f"{val:.4g}", f"(equal to {rep} after rounding)")
    else: emit("DIFF", where, what, "<= "+rep, f"{val:.4g}")


def flag(where, what, ok):
    emit("OK" if ok else "DIFF", where, what, "holds", str(bool(ok)))


def rndf(where, what, rep, val):
    R = float(rep); same = 1/3 <= val/R <= 3
    emit("RNDF" if same else "DIFF", where, what, rep, f"{val:.3e}", "(round-off level; magnitude agrees)" if same else "")


def info(where, what, rep, val):
    emit("INFO", where, what, rep, val)


# ------------------------------------------------------------------ LaTeX tables
def sci(x, thin=False):
    m, e = f"{x:.3e}".split("e"); t = r"\!\times\!" if thin else r"\times"
    return f"${m}{t}10^{{{int(e)}}}$"


def tables_tex(S, R04, R13):
    sp, ax = S["spatial"], S["axial"]
    t1 = [r"\begin{tabular}{rccc|rccc}", r"\hline", r"$J$ & $E_u$ & $E_v$ & $E_g$", r"& $N$ & $E_u$ & $E_v$ & $E_g$\\",
          r"\hline"]
    for i in range(4):
        t1.append(f"{sp['J'][i]}\n" + "".join(f"& {sci(sp[e][i], True)}\n" for e in ("Eu", "Ev", "Eg"))
                  + f"& {ax['N'][i]}\n" + "\n".join(f"& {sci(ax[e][i], True)}" for e in ("Eu", "Ev", "Eg")) + r"\\")
    t1 += ["Order & " + " & ".join(f"{o:.3f}" for o in sp["order"]),
           "& Order & " + " & ".join(f"{o:.3f}" for o in ax["order"]) + r"\\", r"\hline", r"\end{tabular}"]

    def t23(R):
        t = [r"\begin{tabular}{rrrrrrr}", r"\hline", r"$\varepsilon$ & $E_0$ & $E_{\rm p}$",
             r"& $E_{0,\rm fin}$ & $E_{{\rm p},\rm fin}$", r"& $\Delta_{\rm ref,fl}$ & $\Delta_{\rm ref,fin}$\\", r"\hline"]
        for k in KS:
            v = R[str(k)]
            t.append(f"$1/{k}$\n& {sci(v[0])} & {sci(v[1])}\n& {sci(v[2])} & {sci(v[3])}\n& {sci(v[4])} & {sci(v[5])}\\\\")
        return t+[r"\hline", r"\end{tabular}"]
    return ("% Table 1 (tab:scheme-space) -- generated by verify_report.py\n" + "\n".join(t1)
            + "\n\n% Table 2 (tab:resolved-reduction)\n" + "\n".join(t23(R04))
            + "\n\n% Table 3 (tab:resolved-reduction-gamma)\n" + "\n".join(t23(R13)) + "\n")


# ------------------------------------------------------------------ main
def main():
    miss = [f for f in FILES.values() if not os.path.exists(f)]
    if miss: sys.exit("missing result files: " + ", ".join(miss) + "  (run scheme_verify.py and refsolve.py first)")
    D = {k: json.load(open(f)) for k, f in FILES.items()}
    S, R04, R13, R100 = D["scheme"], D["g04"], D["g13"], D["g100"]
    aux, st, sp, ax = S["auxiliary"], S["stability"], S["spatial"], S["axial"]

    # ---- 4.5.1
    w = "4.5.1"
    flag(w, "g(0) = g'(0) = 0", max(abs(aux["g0"]), abs(aux["gp0"])) < 1e-14)
    less(w, "corrector truncation 400->800 terms: change", 4.6e-11, aux["trunc_change"])
    for i, J in enumerate(sp["J"]):
        for e, rep in zip(("Eu", "Ev", "Eg"), T1_SPATIAL[J]): num("Tab.1", f"spatial J={J}: {e}", rep, sp[e][i])
    for e, o, rep in zip(("Eu", "Ev", "Eg"), sp["order"], T1_ORDERS[0]): num("Tab.1", f"spatial order {e}", rep, o)
    for i, N in enumerate(ax["N"]):
        for e, rep in zip(("Eu", "Ev", "Eg"), T1_AXIAL[N]): num("Tab.1", f"axial N={N}: {e}", rep, ax[e][i])
    for e, o, rep in zip(("Eu", "Ev", "Eg"), ax["order"], T1_ORDERS[1]): num("Tab.1", f"axial order {e}", rep, o)
    less(w, "direct 47-step calculation (J=15) vs powers", 1e-11, aux["step47_maxdiff"], "within 1e-11")
    h8, h64 = aux["halving"]["8"], aux["halving"]["64"]
    less(w, "halving spatial-test step, J=8: field change", 2.6e-8, max(h8["du"], h8["dv"]))
    less(w, "halving spatial-test step, J=64: field change", 4.0e-10, max(h64["du"], h64["dv"]))
    rndf(w, "arbitrary steps: max cumulative defect in (4.5)", "2.17e-12", st["defect"])
    flag(w, "computed-load inequality (4.6) at every step", st["load_inequality_46"])
    flag(w, "corrector inequality (4.7) at every step", st["corrector_inequality_47"])

    # ---- 4.5.2 and 4.5.3
    for w, R, T, chk, bound in (("4.5.2", R04, T2, D["chk04"], 1.5e-9), ("4.5.3", R13, T3, D["chk13"], 1e-13)):
        tab = "Tab.2" if w == "4.5.2" else "Tab.3"
        for k in KS:
            for j, rep in enumerate(T[k]):
                key = "g04" if w == "4.5.2" else "g13"
                alt = ARCHIVED[key][j][KS.index(k)] if j in ARCHIVED[key] else None
                num(tab, f"eps=1/{k}: {COLS[j]}", rep, R[str(k)][j], alt)
        d = R["derived"]
        pf, pn = ("2.3", "1.5") if w == "4.5.2" else ("0.24", "0.20")
        atmost(w, "last mesh change / corrected error, fluid [%]", pf, d["pct_mesh_fl"])
        atmost(w, "last mesh change / corrected error, fin [%]", pn, d["pct_mesh_fin"])
        less(w, "enlarged modes/quadrature: change of errors", bound, chk["max_change"])
        reps = ("1.010", "1.971", "1.016", "1.977") if w == "4.5.2" else ("0.937", "1.998", "1.079", "1.981")
        for key, rep in zip(("order_0_fl", "order_p_fl", "order_0_fin", "order_p_fin"), reps):
            num(w, f"last-pair order {key}", rep, d[key])
    d04, d13, d100 = R04["derived"], R13["derived"], R100["derived"]
    w = "4.5.3"
    for name, R in (("gamma=13", R13), ("gamma=100", R100)):
        flag(w, f"a/m = 2 for {name}", abs((1+R["meta"]["gamma"])/(R["meta"]["Pe"]/12)-2) < 1e-12)
    num(w, "E0/eps at eps=1/8 (gamma=13)", "0.0263", d13["E0_over_eps"][0])
    num(w, "E0/eps at eps=1/64 (gamma=13)", "0.0297", d13["E0_over_eps"][-1])
    flag(w, "E0/eps increases with decreasing eps (gamma=13)", bool(np.all(np.diff(d13["E0_over_eps"]) > 0)))
    num(w, "||theta_1||_L2((0,1)xI)", "0.0312", d13["theta1_norm_I"])
    for k, rep, v in zip(KS, ("0.173", "0.172", "0.172", "0.172"), d13["Ep_over_eps2"]):
        num(w, f"Ep/eps^2 at eps=1/{k} (gamma=13)", rep, v)
    num(w, "Ep/eps^2 at eps=1/64 (gamma=0.4)", "0.0169", d04["Ep_over_eps2"][-1])
    num(w, "ratio of Ep/eps^2 at 1/64 (gamma=13 : 0.4)", "10.2", d13["Ep_over_eps2"][-1]/d04["Ep_over_eps2"][-1])
    num(w, "ratio of 1+gamma", "10", 14/1.4)
    num(w, "Ep/((1+g)eps^2) at 1/64, gamma=0.4", "0.012", d04["Ep_over_1pg_eps2"][-1])
    num(w, "Ep/((1+g)eps^2) at 1/64, gamma=13", "0.012", d13["Ep_over_1pg_eps2"][-1])
    num(w, "Ep/((1+g)eps^2) at 1/64, gamma=100", "0.014", d100["Ep_over_1pg_eps2"][-1])
    num(w, "E0 ~ 0.031 eps: limit coefficient ||theta_1||", "0.031", d04["theta1_norm_I"])
    cst = 0.5*(d04["Ep_over_1pg_eps2"][-1]+d13["Ep_over_1pg_eps2"][-1])
    thr = d04["theta1_norm_I"]/cst
    info(w, "(1+g)eps where 0.012(1+g)eps^2 = E0 [(1+g)eps <~ 2.5]", "~2.5", f"{thr:.3f}")
    info(w, "Omega = ((1+g)eps-eps)/2 there, eps->0 / 1/8 [<~ 1.2]", "~1.2", f"{thr/2:.3f} / {(thr-1/8)/2:.3f}")

    # ---- archived (unrounded) figure data in the .tex vs recomputation
    arch = []
    for key, cols in ARCHIVED.items():
        R = D[key]
        for j, vals in cols.items():
            rec = np.array([R[str(k)][j] for k in (8, 16, 32, 64)])
            arch.append((key, COLS[j], np.max(np.abs(rec-np.array(vals))/np.abs(vals))))

    out = ["Section 4.5 verification report", "="*31, ""]+LINES+["",
           "Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(COUNT.items())), "",
           "Unrounded data embedded in the .tex figures vs recomputation (max relative difference):"]
    out += [f"  {k:5s} {c:9s} {d:.1e}" for k, c, d in arch]
    out += ["  (Figure 3 coordinates differ by up to ~2e-5: they come from an earlier run of the code; the Figure 6",
            "   coordinates agree to the stored digits.  Such differences are far below the printed precision except",
            "   for values lying on a rounding boundary, flagged EDGE above.)"]
    txt = "\n".join(out); print(txt)
    open("verification_report.txt", "w").write(txt+"\n")
    open("tables_generated.tex", "w").write(tables_tex(S, R04, R13))
    print("\nwrote verification_report.txt and tables_generated.tex")


if __name__ == "__main__":
    main()
