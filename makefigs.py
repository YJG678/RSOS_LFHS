"""Generate Figures 1-6 of the manuscript as PDF and PNG files, and figdata.json with the PGFPlots
coordinate strings used by the inline figures of the .tex source.

  fig1_geometry          zero-clearance cell and conjugate shroud layer (schematic)
  fig2_velocity_layers   12 W^eps at eps = 1/8 and the X-averaged velocity for eps = 1/8, 1/16, 1/32
  fig3_reduction_errors  fluid and fin reduction errors      (needs res_Pe8.4_g0.4_96x1280.json)
  fig4_fields            T, (T-theta)/eps, (T-theta_p)/eps^2 at Z = 0.1, eps = 1/8
  fig5_scaled_profiles   scaled reference profiles at Z = 0.1, eps = 1/8,...,1/64
  fig6_gamma_dependence  corrected error versus fin conductance (needs the three res_*_96x1280.json)

Usage:  python makefigs.py [--figs 1 2 3 4 5 6] [--outdir figures] [--formats pdf png] [--no-pgf]
Figures 4-5 need the (96,1280) reference at Z = 0.1 for four eps (about a minute); the fields are cached
in fields_Z0.1.npz (delete the file to recompute).  Figures are sized for a 6.5 in text width.
"""
import argparse, json, logging, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter, NullLocator, LogLocator, NullFormatter
from refsolve import Wfield, Ref, Reduced, interp

# ------------------------------------------------------------------ style (figures/figure_style.tex)
FIGBLUE, FIGORANGE, FIGGREEN, FIGINK, FIGGRAY = "#0072B2", "#D55E00", "#00866A", "#203442", "#677480"


def tint(col, pct):
    """xcolor 'col!pct' (pct % colour, rest white)."""
    return tuple(pct/100*np.array(mcolors.to_rgb(col))+(1-pct/100))


SEQBLUE = LinearSegmentedColormap.from_list(
    "seqblue", np.array([(240, 246, 251), (120, 175, 214), (0, 114, 178), (0, 56, 94)])/255)
DIVBO = LinearSegmentedColormap.from_list("divbo", np.array([(0, 114, 178), (232, 232, 232), (213, 94, 0)])/255)
GRIDC = tint("gray", 15)
TEXTW = 6.5                                   # inches (10pt article, 1in margins)
KS = (8, 16, 32, 64)
EPS_TICKS = ([1/64, 1/32, 1/16, 1/8], ["$1/64$", "$1/32$", "$1/16$", "$1/8$"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 7, "axes.linewidth": 0.5, "lines.linewidth": 0.7, "lines.markersize": 4,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.titlepad": 6,
    "lines.markeredgewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
logging.getLogger("fontTools").setLevel(logging.ERROR)            # silence font-subsetting chatter
GFMT = FuncFormatter(lambda x, _: f"${x:g}$")                     # pgfplots-style tick labels: 0, 0.5, 1
BOLD = font_manager.FontProperties(fname=os.path.join(matplotlib.get_data_path(), "fonts/ttf/cmb10.ttf"), size=9)


def gticks(ax, x=True, y=True):
    if x: ax.xaxis.set_major_formatter(GFMT)
    if y: ax.yaxis.set_major_formatter(GFMT)


def colorbar(fig, pc, ax, ticks):
    cb = fig.colorbar(pc, ax=ax, aspect=35, pad=0.04, ticks=ticks)
    cb.ax.yaxis.set_major_formatter(GFMT); cb.ax.tick_params(labelsize=7, width=.4); cb.outline.set_linewidth(.4)
    return cb


def grid(ax):
    ax.grid(True, which="major", color=GRIDC, lw=0.4); ax.set_axisbelow(True)


def eps_axis(ax):
    ax.xaxis.set_major_locator(FixedLocator(EPS_TICKS[0])); ax.xaxis.set_major_formatter(FixedFormatter(EPS_TICKS[1]))
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlim(1/64/1.15, 1/8*1.15)


def save(fig, name, A):
    os.makedirs(A.outdir, exist_ok=True)
    for ext in A.formats:
        fig.savefig(os.path.join(A.outdir, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("wrote", os.path.join(A.outdir, name)+".{"+",".join(A.formats)+"}")


def load_res(fn):
    if not os.path.exists(fn):
        raise SystemExit(f"{fn} not found: run refsolve.py first (see README.md)")
    return json.load(open(fn))


# ------------------------------------------------------------------ Figure 1: schematic
def figure1(A):
    W, H = 16.15, 5.75                                    # TikZ bounding box in cm
    fig = plt.figure(figsize=(W/2.54, H/2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-0.15, 16.0); ax.set_ylim(-0.65, 5.1); ax.axis("off")
    pad = 0.1                                             # TikZ inner sep (cm) for west anchors

    def text(x, y, s, size=9, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center")
        return ax.text(x, y, s, fontsize=size, color=FIGINK, **kw)

    def line(xs, ys, col, lw, **kw): ax.plot(xs, ys, color=col, lw=lw, solid_capstyle="butt", **kw)
    def rect(x0, y0, x1, y1, col): ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc=col, ec="none"))

    def arrow(p, q, col=FIGINK, lw=0.8, style="-|>", ms=9):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=col, lw=lw, mutation_scale=ms,
                                                        shrinkA=0, shrinkB=0))
    dashed = dict(ls=(0, (3, 2)))                          # TikZ 'densely dashed'
    white = dict(fc="white", ec="none", pad=1.5)
    ax.text(0+pad, 4.85, "(a) Zero-clearance cell", fontproperties=BOLD, color=FIGINK, va="center")
    ax.text(8.45+pad, 4.85, "(b) Shroud-layer magnification", fontproperties=BOLD, color=FIGINK, va="center")
    # (a) thin-fin reference cell
    rect(1, .8, 5, 3.9, tint(FIGBLUE, 7)); rect(1, 3.5, 5, 3.9, tint(FIGORANGE, 13)); rect(1, .8, 5, 1.25, tint(FIGGRAY, 10))
    line([1, 1], [.8, 3.9], FIGORANGE, 1.8); line([5, 5], [.8, 3.9], FIGORANGE, 1.8)
    line([.72, 5.28], [3.9, 3.9], FIGGRAY, 2); line([.72, 5.28], [.8, .8], FIGORANGE, 2)
    line([1, 5], [3.5, 3.5], tint(FIGORANGE, 70), 0.4, **dashed); line([1, 5], [1.25, 1.25], tint(FIGGRAY, 70), 0.4, **dashed)
    text(3, 4.25, r"adiabatic shroud: $\partial_yT=0$", 8)
    text(3, 3.71, r"$1-y=O(\varepsilon)$", 7, bbox=dict(fc=tint(FIGORANGE, 13), ec="none", pad=0.3))
    text(3, 1.025, r"$y=O(\varepsilon)$", 7, bbox=dict(fc=tint(FIGGRAY, 10), ec="none", pad=0.3))
    ax.add_patch(Circle((3, 2.8), .14, fc="none", ec=FIGBLUE, lw=.9)); ax.add_patch(Circle((3, 2.8), .035, fc=FIGBLUE, ec="none"))
    text(3, 2.25, r"$w(x,y)\,\boldsymbol{e}_Z$")
    text(3, 1.62, "streamwise flow", 8, linespacing=1.15)
    arrow((.38, .8), (.38, 3.9), FIGGRAY, .55, "<|-|>", 7); text(.38, 2.35, r"$H^*$", bbox=white)
    line([.28, .85], [.8, .8], FIGGRAY, .4); line([.28, .85], [3.9, 3.9], FIGGRAY, .4)
    arrow((1, .22), (5, .22), FIGGRAY, .55, "<|-|>", 7); text(3, .22, r"$S^*=\varepsilon H^*$", bbox=white)
    line([1, 1], [.12, .58], FIGGRAY, .4); line([5, 5], [.12, .58], FIGGRAY, .4)
    text(3, -.3, r"isothermal base: $T=\Theta_f=1$", 8)
    text(5.35+pad, 2.1, "thin fin\n" r"$\Omega=\varepsilon\gamma/2$", 8, ha="left", multialignment="left")
    line([5, 5.28], [2.62, 2.3], FIGGRAY, .55)
    # (b) conjugate half-strip corrector
    rect(9.2, .85, 12.5, 3.9, tint(FIGBLUE, 7))
    line([9.2, 9.2], [.85, 3.9], FIGORANGE, 1.7); line([12.5, 12.5], [.85, 3.9], FIGORANGE, 1.7)
    line([8.95, 12.75], [3.9, 3.9], FIGGRAY, 2); line([9.2, 12.5], [.85, .85], FIGGRAY, .4, **dashed)
    arrow((10.85, 1.25), (10.85, .48)); text(11.05+pad, .54, r"$Y\to{-\infty}$", 8, ha="left")
    text(10.85, 2.58, "conjugate tip cell\n" r"$(\psi(X,Y),\psi_{\mathrm{f}}(Y))$", 9, linespacing=1.7)
    text(9.2, .08, r"$X=0$", 8); text(12.5, .08, r"$X=1$", 8)
    text(13.05+pad, 3.7, "$Y=0$\n" r"$\psi_Y=\psi_{\mathrm{f}}'=0$", 8, ha="left", multialignment="left", linespacing=1.4)
    text(13.05+pad, 2.28, r"$X=x/\varepsilon$" "\n" r"$Y=(y-1)/\varepsilon$", 8, ha="left", multialignment="left",
         linespacing=1.6)
    text(13.05+pad, 1.15, "matching to the\nchannel interior", 8, ha="left", multialignment="left")
    arrow((5.32, 3.7), (8.82, 3.7)); text(7.07, 3.79, r"scale by $\varepsilon^{-1}$", 8, va="bottom")
    text(11.65, -.4, r"corner moment: $\sigma_0=-\dfrac{31\zeta(5)}{4\pi^5}$", 8)
    save(fig, "fig1_geometry", A)


# ------------------------------------------------------------------ Figure 2: velocity layers
def xavg_velocity(y, k, nmodes=2000):
    n = 2*np.arange(nmodes)+1.0; t = n[:, None]*np.pi*k
    q = (np.exp(-t*y)+np.exp(-t*(1-y)))/(1+np.exp(-t))
    return 1-12*((8/(n*np.pi)**4)[:, None]*q).sum(0)


def figure2(A):
    X = np.linspace(0, 1, 201)
    y = np.unique(np.r_[np.linspace(0, .25, 201), np.linspace(.25, .75, 101), np.linspace(.75, 1, 201)])
    W = 12*Wfield(X, y, 1/8); W[[0, -1], :] = 0; W[:, [0, -1]] = 0
    fig = plt.figure(figsize=(TEXTW, 2.75), layout="constrained")
    ga, gb = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.0])
    ax = fig.add_subplot(ga)
    pc = ax.pcolormesh(X, y, W.T, shading="gouraud", cmap=SEQBLUE, vmin=0, vmax=1.5, rasterized=True)
    colorbar(fig, pc, ax, [0, .5, 1, 1.5])
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .25, .5, .75, 1], xlabel="$X$", ylabel="$y$",
           title=r"(a) $12W^\varepsilon(X,y)$, $\varepsilon=1/8$")
    ax.tick_params(which="both", top=False, right=False); gticks(ax)
    ax = fig.add_subplot(gb)
    yl = np.unique(np.r_[np.linspace(0, .3, 601), np.linspace(.3, .7, 81), np.linspace(.7, 1, 601)])
    sty = {8: dict(color=tint(FIGBLUE, 45), ls="-"), 16: dict(color=FIGBLUE, ls=(0, (3, 2))), 32: dict(color=FIGINK, ls=(0, (1, 2)))}
    for k, lab in zip((8, 16, 32), (r"$\varepsilon=1/8$", "$1/16$", "$1/32$")):
        ax.plot(yl, xavg_velocity(yl, k), lw=1, label=lab, **sty[k])
    ax.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="$y$", ylabel=r"$12\int_0^1W^\varepsilon\,dX$", title="(b) $X$-averaged velocity")
    gticks(ax); grid(ax); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3, handlelength=2.6, columnspacing=1.2)
    save(fig, "fig2_velocity_layers", A)


# ------------------------------------------------------------------ Figure 3: reduction errors (gamma = 0.4)
def slope_segment(ax, p, q, label, pos, where="above"):
    ax.plot([p[0], q[0]], [p[1], q[1]], color="gray", ls=(0, (3, 3)), lw=0.6)
    x = np.exp(np.log(p[0])+pos*(np.log(q[0])-np.log(p[0]))); y = np.exp(np.log(p[1])+pos*(np.log(q[1])-np.log(p[1])))
    if where == "above": ax.annotate(label, (x, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    else: ax.annotate(label, (x, y), xytext=(3, -3), textcoords="offset points", ha="left", va="top", fontsize=7)


def figure3(A):
    r = load_res("res_Pe8.4_g0.4_96x1280.json"); e = np.array([1/k for k in KS]); E = np.array([r[str(k)] for k in KS])
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.95), layout="constrained")
    for ax, (i0, ip, im), title in zip(axs, ((0, 1, 4), (2, 3, 5)), ("(a) Fluid temperature", "(b) Fin temperature")):
        ax.loglog(e, E[:, i0], color=FIGBLUE, marker="o", mfc="none", label="Leading")
        ax.loglog(e, E[:, ip], color=FIGORANGE, marker="s", mfc="none", label="Corrected")
        ax.loglog(e, E[:, im], color=tint("black", 65), marker="x", ls=(0, (1, 2)), label="Mesh change")
        slope_segment(ax, (0.019, 4e-5), (0.039, 0.000117626), "$3/2$", 0.6)
        ax.set(ylim=(3e-8, .018), xlabel=r"$\varepsilon$", ylabel="Temperature error norm", title=title)
        eps_axis(ax); ax.yaxis.set_major_locator(LogLocator(10, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(10, subs=np.arange(2, 10), numticks=12)); ax.yaxis.set_minor_formatter(NullFormatter())
        grid(ax); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=2.2)
    save(fig, "fig3_reduction_errors", A)


# ------------------------------------------------------------------ reference fields at Z = 0.1 (Figures 4-5)
PE, GAM, ZF = 8.4, 0.4, 0.1


def reference_fields(cache="fields_Z0.1.npz"):
    if os.path.exists(cache):
        d = np.load(cache); return {k: d[k] for k in d.files}
    out = {}
    for k in KS:
        print(f"  reference field eps=1/{k} on (96,1280) ...", flush=True)
        ref = Ref(96, 1280, 1/k, PE, GAM); out[f"U{k}"] = ref.field(ZF)
    out["X"], out["y"] = ref.X, ref.y
    np.savez_compressed(cache, **out); return out


def scaled_fields(F, k, Xs, ys, red):
    """T, (T-theta)/eps, (T-theta_p)/eps^2 on the grid Xs x ys (indexing [X, y])."""
    eps = 1/k; T = 1+interp(F["X"], Xs)@F[f"U{k}"]@interp(F["y"], ys).T
    th = 1+red.u(ys, ZF); t1 = red.th1(ys, ZF)
    return T, (T-th[None, :])/eps, (T-(th+eps*t1)[None, :])/eps**2


def figure4(A, F, red):
    Xs = np.linspace(0, 1, 97); ys = np.linspace(0, 1, 241)
    T, E0, Ep = scaled_fields(F, 8, Xs, ys, red)
    fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.75), layout="constrained")
    specs = [(T, SEQBLUE, None, None, [.8, .9, 1], r"(a) $T^\varepsilon$"),
             (E0, SEQBLUE, 0, None, [0, .05, .1, .15], r"(b) $(T^\varepsilon-\theta)/\varepsilon$"),
             (Ep, DIVBO, -0.095, 0.095, [-.05, 0, .05], r"(c) $(T^\varepsilon-\theta_{\mathrm{p}}^\varepsilon)/\varepsilon^2$")]
    for i, (ax, (Fv, cm, lo, hi, ticks, title)) in enumerate(zip(axs, specs)):
        pc = ax.pcolormesh(Xs, ys, Fv.T, shading="gouraud", cmap=cm, vmin=lo, vmax=hi, rasterized=True)
        colorbar(fig, pc, ax, ticks)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .25, .5, .75, 1], xlabel="$X$", title=title)
        if i == 0: ax.set_ylabel("$y$")
        ax.tick_params(which="both", top=False, right=False); gticks(ax)
    save(fig, "fig4_fields", A)


def figure5(A, F, red):
    ys = np.linspace(0, 1, 61); yf = np.linspace(0, 1, 201); Xf = np.linspace(0, 1, 161)
    mk = {8: dict(marker="o", ms=3.2), 16: dict(marker="s", ms=3.0), 32: dict(marker="^", ms=3.6), 64: dict(marker="x", ms=4.0)}
    lab = {8: r"$\varepsilon=1/8$", 16: "$1/16$", 32: "$1/32$", 64: "$1/64$"}
    m, a = PE/12, 1+GAM; h = 1e-5
    Phi = lambda X: (2*X**3-X**4-X)/24-X*(X-1)/(24*a)
    thZ = (red.u(np.array([.5]), ZF+h)-red.u(np.array([.5]), ZF-h))[0]/(2*h)
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.75), layout="constrained")
    ax = axs[0]; ax.plot(yf, red.th1(yf, ZF), color=FIGORANGE, lw=1.1, label=r"$\theta_1$")
    th = 1+red.u(ys, ZF)
    for k in KS:
        fin = 1+interp(F["y"], ys)@F[f"U{k}"][0]
        ax.plot(ys[::3], ((fin-th)*k)[::3], ls="none", color=FIGINK, mfc="none", mew=.6, label=lab[k], **mk[k])
    ax.set(xlim=(0, 1), ylim=(0, None), yticks=[0, .05, .1, .15], xlabel="$y$", ylabel=r"$(\Theta_f-\theta)/\varepsilon$",
           title="(a) Scaled leading fin error")
    gticks(ax); grid(ax); ax.legend(loc="upper left")
    ax = axs[1]; ax.plot(Xf, PE*thZ*Phi(Xf), color=FIGORANGE, lw=1.1, label=r"$\widehat{\mathrm{Pe}}\,\theta_Z\Phi(X)$")
    for k in KS:
        Uy = np.asarray(interp(F["y"], np.array([.5]))@F[f"U{k}"].T).ravel()
        ax.plot(F["X"][::8], ((Uy-Uy[0])*k*k)[::8], ls="none", color=FIGINK, mfc="none", mew=.6, label=lab[k], **mk[k])
    ax.set(xlim=(0, 1), yticks=[-.04, -.02, 0], xlabel="$X$", ylabel=r"$(T^\varepsilon-\Theta_f)/\varepsilon^2$",
           title=r"(b) Cross-channel variation at $y=\frac{1}{2}$")
    gticks(ax); grid(ax); ax.legend(loc="upper center")
    save(fig, "fig5_scaled_profiles", A)


# ------------------------------------------------------------------ Figure 6: gamma dependence
RUNS = (("res_Pe8.4_g0.4_96x1280.json", 0.4, FIGBLUE, "o"), ("res_Pe84_g13_96x1280.json", 13, FIGORANGE, "s"),
        ("res_Pe606_g100_96x1280.json", 100, FIGGREEN, "^"))


def figure6(A):
    e = np.array([1/k for k in KS])
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.75), layout="constrained")
    for fn, g, col, mkr in RUNS:
        Ep = np.array([load_res(fn)[str(k)][1] for k in KS])
        axs[0].loglog(e, Ep, color=col, marker=mkr, mfc="none", label=rf"$\gamma={g:g}$")
        axs[1].semilogx(e, Ep/((1+g)*e**2), color=col, marker=mkr, mfc="none")
    slope_segment(axs[0], (0.02, 2e-6), (0.05, 1.25e-5), "$2$", 0.55, where="below")
    axs[0].set(ylim=(1e-6, 4e-2), xlabel=r"$\varepsilon$", ylabel=r"$E_{\mathrm{p}}$", title="(a) Corrected fluid error")
    axs[0].yaxis.set_minor_locator(LogLocator(10, subs=np.arange(2, 10), numticks=12)); axs[0].yaxis.set_minor_formatter(NullFormatter())
    axs[0].legend(loc="lower right")
    axs[1].set(ylim=(0, 0.02), yticks=[0, .005, .01, .015, .02], xlabel=r"$\varepsilon$",
               ylabel=r"$E_{\mathrm{p}}/((1+\gamma)\varepsilon^2)$", title=r"(b) Rescaled by $(1+\gamma)\varepsilon^2$")
    gticks(axs[1], x=False)
    for ax in axs: eps_axis(ax); grid(ax)
    save(fig, "fig6_gamma_dependence", A)


# ------------------------------------------------------------------ PGFPlots coordinates (as in the .tex)
def pgf_data(F, red, have_res):
    coords = lambda xs, ys, fmt="%.5g": " ".join(f"({fmt % x},{fmt % y})" for x, y in zip(xs, ys))
    surf = lambda X, Y, Z: ("\n".join(" ".join(f"({x:.4g},{y:.4g},{Z[i, j]:.4g})" for i, x in enumerate(X))
                                      for j, y in enumerate(Y)), len(Y))
    out = {}
    Xp = np.linspace(0, 1, 21); yp = np.unique(np.r_[np.linspace(0, .25, 21), np.linspace(.25, .75, 11), np.linspace(.75, 1, 21)])
    W = 12*Wfield(Xp, yp, 1/8); W[[0, -1], :] = 0; W[:, [0, -1]] = 0
    out['Vsurf'], out['Vrows'] = surf(Xp, yp, W)
    yl = np.unique(np.r_[np.linspace(0, .3, 61), np.linspace(.3, .7, 9), np.linspace(.7, 1, 61)])
    for k in (8, 16, 32): out[f'Vbar{k}'] = coords(yl, xavg_velocity(yl, k))
    if F is not None:
        ys = np.linspace(0, 1, 61); Xs = np.linspace(0, 1, 25); a = 1+GAM; h = 1e-5
        Phi = lambda X: (2*X**3-X**4-X)/24-X*(X-1)/(24*a)
        T, E0, Ep = scaled_fields(F, 8, Xs, ys, red)
        out['Tsurf'], out['Trows'] = surf(Xs, ys, T); out['E0surf'], _ = surf(Xs, ys, E0); out['Epsurf'], _ = surf(Xs, ys, Ep)
        out['ranges'] = dict(T=(T.min(), T.max()), E0=(E0.min(), E0.max()), Ep=(Ep.min(), Ep.max()))
        th = 1+red.u(ys, ZF)
        for k in KS:
            fin = 1+interp(F["y"], ys)@F[f"U{k}"][0]; out[f'P1_{k}'] = coords(ys[::3], ((fin-th)*k)[::3])
            Uy = np.asarray(interp(F["y"], np.array([.5]))@F[f"U{k}"].T).ravel()
            out[f'P2_{k}'] = coords(F["X"][::8], ((Uy-Uy[0])*k*k)[::8])
        out['P1_th1'] = coords(ys, red.th1(ys, ZF))
        thZ = (red.u(np.array([.5]), ZF+h)-red.u(np.array([.5]), ZF-h))[0]/(2*h)
        Xf = np.linspace(0, 1, 81); out['P2_pred'] = coords(Xf, PE*thZ*Phi(Xf))
    if have_res:
        r = load_res("res_Pe8.4_g0.4_96x1280.json"); e = [1/k for k in KS]
        for j, key in enumerate(("F3_E0", "F3_Ep", "F3_E0fin", "F3_Epfin", "F3_dfl", "F3_dfin")):
            out[key] = coords(e, [r[str(k)][j] for k in KS], "%.16g")
        for fn, g, _, _ in RUNS:
            r = load_res(fn); ks = [64, 32, 16, 8]
            out[f'G0_{g:g}'] = coords([1/k for k in ks], [r[str(k)][0] for k in ks], "%.6g")
            out[f'Gp_{g:g}'] = coords([1/k for k in ks], [r[str(k)][1] for k in ks], "%.6g")
            out[f'Gs_{g:g}'] = coords([1/k for k in ks], [r[str(k)][1]*k*k/(1+g) for k in ks], "%.6g")
    json.dump(out, open("figdata.json", "w")); print("wrote figdata.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--formats", nargs="*", default=["pdf", "png"])
    ap.add_argument("--no-pgf", action="store_true", help="do not write figdata.json")
    A = ap.parse_args()
    need_ref = bool({4, 5} & set(A.figs)) or not A.no_pgf
    F = reference_fields() if need_ref else None
    red = Reduced(PE, GAM)
    for i in A.figs:
        if i == 1: figure1(A)
        elif i == 2: figure2(A)
        elif i == 3: figure3(A)
        elif i == 4: figure4(A, F, red)
        elif i == 5: figure5(A, F, red)
        elif i == 6: figure6(A)
    if not A.no_pgf:
        pgf_data(F, red, all(os.path.exists(fn) for fn, *_ in RUNS))
