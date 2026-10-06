"""Independent resolved reference for the zero-clearance fluid--fin cell (bilinear FE on (X,y)),
compared with the sequential reduction theta + eps*theta_1.  Sections 4.5.2-4.5.3 of the manuscript.

Usage
  python refsolve.py Pe gam NX Ny [I0 I1] [--sequence]
      Reduction errors for eps = 1/8,...,1/64 on the mesh (NX,Ny) and reference-mesh changes against
      (NX/2,Ny/2).  Prints Table 2/3 rows and the derived numbers quoted in the text; writes
      res_Pe{Pe}_g{gam}_{NX}x{Ny}.json   ("8","16","32","64" -> [E0,Ep,E0fin,Epfin,dref_fl,dref_fin]).
      --sequence additionally reports the changes between all four nested meshes.
  python refsolve.py Pe gam NX Ny --check [--modes M --vel V --ng Q --nz NZ --nred NR]
      Refinement check at eps = 1/8 and 1/64: baseline (24 reference modes, 512 velocity modes,
      4 Gauss points per coordinate, 20 axial points, 160 reduced modes) against the enlarged
      parameters on the same mesh; writes check_Pe{Pe}_g{gam}_{NX}x{Ny}.json.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, time, json, argparse
from scipy.special import betainc, zeta
SIG0 = -31*zeta(5)/(4*np.pi**5)
BASE = dict(nmodes=24, vel=512, ng=4, nz=20, nred=160)      # parameters used for Tables 2 and 3


def gauss01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x+1)/2, w/2
GP, GW = gauss01(4)


def p1(g):
    h = np.diff(g)
    M = sp.diags([np.r_[h, 0]/3+np.r_[0, h]/3, h/6, h/6], [0, 1, -1]).tocsr()
    K = sp.diags([np.r_[1/h, 0]+np.r_[0, 1/h], -1/h, -1/h], [0, 1, -1]).tocsr()
    return M, K


def interp(g, pts):
    e = np.clip(np.searchsorted(g, pts, side='right')-1, 0, len(g)-2); t = (pts-g[e])/(g[e+1]-g[e])
    r = np.arange(len(pts))
    return sp.csr_matrix((np.r_[1-t, t], (np.r_[r, r], np.r_[e, e+1])), shape=(len(pts), len(g)))


def quad(g, ng=4):
    gp, gw = gauss01(ng); h = np.diff(g)
    return (g[:-1, None]+h[:, None]*gp[None, :]).ravel(), (h[:, None]*gw[None, :]).ravel()


def Wfield(Xq, yq, eps, nmodes=512):
    n = 2*np.arange(nmodes)+1.0; om = 4/(n*np.pi)**3
    S = np.sin(np.outer(Xq, n*np.pi))*om
    t = n[:, None]*np.pi/eps
    Q = (np.exp(-t*yq[None, :])+np.exp(-t*(1-yq[None, :])))/(1+np.exp(-t))
    return Xq[:, None]*(1-Xq[:, None])/2-S@Q


def theta_in(y): return 1-0.5*betainc(13, 13, np.clip((y-0.2)/0.6, 0, 1))


class Ref:
    """Bilinear FE reference (4.21); modal propagation with nmodes generalised eigenpairs."""
    def __init__(s, NX, Ny, eps, Pe, gam, nmodes=24, vel=512, ng=4):
        s.NX, s.Ny, s.eps = NX, Ny, eps
        s.X = np.arange(NX+1)/NX; s.y = (1-np.cos(np.pi*np.arange(Ny+1)/Ny))/2
        Mx, Kx = p1(s.X); My, Ky = p1(s.y)
        K = sp.kron(Kx, My)/eps**2+sp.kron(Mx, Ky)
        gp, gw = gauss01(ng)
        Xq, _ = quad(s.X, ng); yq, _ = quad(s.y, ng)
        Wq = Wfield(Xq, yq, eps, vel).reshape(NX, ng, Ny, ng)
        hx = np.diff(s.X); hy = np.diff(s.y)
        B = np.stack([1-gp, gp], 1)
        Mel = np.einsum('xpyq,p,q,pa,pc,qb,qd->xyabcd', Wq, gw, gw, B, B, B, B, optimize=True) \
            * hx[:, None, None, None, None, None]*hy[None, :, None, None, None, None]
        ex, ey = np.meshgrid(np.arange(NX), np.arange(Ny), indexing='ij')
        rows = []; cols = []; vals = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        rows.append(((ex+a)*(Ny+1)+ey+b).ravel()); cols.append(((ex+c)*(Ny+1)+ey+d).ravel())
                        vals.append(Mel[:, :, a, b, c, d].ravel())
        nf = (NX+1)*(Ny+1)
        M = Pe*sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nf, nf))
        # constraints: base removed, both sidewalls share the fin dof
        I, J = np.meshgrid(np.arange(NX+1), np.arange(Ny+1), indexing='ij')
        red = np.where((I == 0) | (I == NX), (NX-1)*Ny+J-1, (I-1)*Ny+J-1); keep = (J > 0)
        nr = NX*Ny
        s.P = sp.csr_matrix((np.ones(keep.sum()), ((I*(Ny+1)+J)[keep], red[keep])), shape=(nf, nr))
        Kr = (s.P.T@K@s.P).tolil()
        Kf = Ky[1:, 1:].tocoo(); off = (NX-1)*Ny
        Kr = (Kr.tocsr()+gam*sp.csr_matrix((Kf.data, (Kf.row+off, Kf.col+off)), shape=(nr, nr))).tocsc()
        Mr = (s.P.T@M@s.P).tocsc()
        lu = spla.splu(Kr)
        OP = spla.LinearOperator((nr, nr), matvec=lu.solve)
        lam, V = spla.eigsh(Kr, k=nmodes, M=Mr, sigma=0, which='LM', OPinv=OP, tol=1e-12)
        o = np.argsort(lam); s.lam = lam[o]; V = V[:, o]
        u0 = np.tile(theta_in(s.y)-1, (NX+1, 1)).ravel()
        u0r = np.zeros(nr); u0r[red[keep]] = u0[(I*(Ny+1)+J)[keep]]
        s.alpha = V.T@(Mr@u0r); s.PV = s.P@V

    def field(s, Z): return (s.PV@(s.alpha*np.exp(-s.lam*Z))).reshape(s.NX+1, s.Ny+1)


class Reduced:
    """Series solutions of (3.10) and (3.12) with N sine modes."""
    def __init__(s, Pe, gam, N=160):
        s.m, s.a = Pe/12, 1+gam; s.kap = s.a/s.m; s.b = SIG0*Pe
        s.ln = (np.arange(1, N+1)-.5)*np.pi; s.e1 = np.sqrt(2)*np.sin(s.ln)
        al = np.zeros(N)
        for lo, hi in [(0.2, 0.8), (0.8, 1.0)]:
            x, w = np.polynomial.legendre.leggauss(2000); x = lo+(hi-lo)*(x+1)/2; w = w*(hi-lo)/2
            al += (np.sqrt(2)*np.sin(np.outer(s.ln, x)))@(w*(theta_in(x)-1))
        s.al = al; s.A = s.kap*s.ln**2
        s.G = s.b*s.A*al*s.e1                      # g(z)=sum G_j exp(-A_j z)
        s.pn = -np.sqrt(2)*(-1.0)**np.arange(N)/s.ln**4

    def E(s, y): return np.sqrt(2)*np.sin(np.outer(s.ln, y))
    def u(s, y, Z): return (s.al*np.exp(-s.A*Z))@s.E(y)

    def th1(s, y, Z, nj=60):
        # boundary lift + rapidly convergent remainder; avoids g'' (which amplifies round-off in alpha)
        ex = np.exp(-s.A*Z); G = s.G[:nj]; g = (G*ex[:nj]).sum(); gp = -(G*s.A[:nj]*ex[:nj]).sum()
        Bl = (g*y+gp/s.kap*(y**3/6-y/2))/s.a
        D = s.A[:, None]-s.A[None, :nj]; ii = np.arange(nj); D[ii, ii] = 1
        T = (ex[None, :nj]-ex[:, None])/D; T[ii, ii] = Z*ex[:nj]
        c = s.e1/s.m*(T@G)                                   # modal coefficients of theta_1
        Bn = (g*s.e1/s.ln**2+gp/s.kap*s.pn)/s.a              # modal coefficients of the lift
        return Bl+(c-Bn)@s.E(y)

    def th1_norm(s, I=(0.05, 0.2), nz=40, ny=400):
        """||theta_1||_{L2((0,1) x I)}."""
        y, wy = gauss01(ny); z, wz = gauss01(nz); z = I[0]+(I[1]-I[0])*z; wz = wz*(I[1]-I[0])
        return float(np.sqrt(sum(w*(wy@s.th1(y, Z)**2) for Z, w in zip(z, wz))))


def errors(ref, red, eps, I=(0.05, 0.2), nz=20, coarse=None, ng=4):
    """[E0, Ep, E0fin, Epfin, dref_fl, dref_fin] of (4.22) on the quadrature of `ref`."""
    Xq, wx = quad(ref.X, ng); yq, wy = quad(ref.y, ng); Ix = interp(ref.X, Xq); Iy = interp(ref.y, yq)
    if coarse is not None: Cx = interp(coarse.X, Xq); Cy = interp(coarse.y, yq)
    z, wz = np.polynomial.legendre.leggauss(nz); z = I[0]+(I[1]-I[0])*(z+1)/2; wz = wz*(I[1]-I[0])/2
    out = np.zeros(6)
    for Z, w in zip(z, wz):
        U = ref.field(Z); Uq = Ix@U@Iy.T; Uf = Iy@U[0]
        u = red.u(yq, Z); t1 = red.th1(yq, Z)
        f = lambda D: wx@(D**2)@wy
        out[0] += w*f(Uq-u[None, :]); out[1] += w*f(Uq-(u+eps*t1)[None, :])
        out[2] += w*(wy@(Uf-u)**2); out[3] += w*(wy@(Uf-u-eps*t1)**2)
        if coarse is not None:
            Uc = coarse.field(Z); out[4] += w*f(Uq-Cx@Uc@Cy.T); out[5] += w*(wy@(Uf-Cy@Uc[0])**2)
    return np.sqrt(out)


# ----------------------------------------------------------------------------- reporting helpers
def tex_sci(x, sig=4):
    m, e = f"{x:.{sig-1}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def table_rows(res, ks):
    lines = []
    for k in ks:
        v = res[k]
        lines.append(f"$1/{k}$\n& {tex_sci(v[0])} & {tex_sci(v[1])}\n& {tex_sci(v[2])} & {tex_sci(v[3])}"
                     f"\n& {tex_sci(v[4])} & {tex_sci(v[5])}\\\\")
    return "\n".join(lines)


def derived(res, ks, gam, red, I):
    """Numbers quoted in the text of Sections 4.5.2-4.5.3."""
    E = np.array([res[k] for k in ks]); eps = np.array([1/k for k in ks])
    i, j = ks.index(32), ks.index(64)
    return dict(
        pct_mesh_fl=100*np.max(E[:, 4]/E[:, 1]), pct_mesh_fin=100*np.max(E[:, 5]/E[:, 3]),
        order_0_fl=np.log2(E[i, 0]/E[j, 0]), order_p_fl=np.log2(E[i, 1]/E[j, 1]),
        order_0_fin=np.log2(E[i, 2]/E[j, 2]), order_p_fin=np.log2(E[i, 3]/E[j, 3]),
        E0_over_eps=(E[:, 0]/eps).tolist(), Ep_over_eps2=(E[:, 1]/eps**2).tolist(),
        Ep_over_1pg_eps2=(E[:, 1]/((1+gam)*eps**2)).tolist(),
        theta1_norm_I=red.th1_norm(I))


def run(Pe, gam, NX, Ny, I=(0.05, 0.2), ks=(8, 16, 32, 64), sequence=False):
    red = Reduced(Pe, gam, BASE['nred']); res = {}; seq = {}
    P = (BASE['nmodes'], BASE['vel'], BASE['ng'])
    for k in ks:
        t = time.time(); eps = 1/k
        fine = Ref(NX, Ny, eps, Pe, gam, *P); coarse = Ref(NX//2, Ny//2, eps, Pe, gam, *P)
        e = errors(fine, red, eps, I, BASE['nz'], coarse, BASE['ng']); res[k] = e.tolist()
        print(f"1/{k}: E0={e[0]:.4e} Ep={e[1]:.4e} E0f={e[2]:.4e} Epf={e[3]:.4e} dfl={e[4]:.3e} dfin={e[5]:.3e}"
              f"  lam1={fine.lam[0]:.5f} lam24={fine.lam[-1]:.1f} [{time.time()-t:.0f}s]", flush=True)
        if sequence:      # changes between consecutive nested meshes (NX/8,Ny/8) -> ... -> (NX,Ny)
            meshes = [Ref(NX//q, Ny//q, eps, Pe, gam, *P) for q in (8, 4)]+[coarse, fine]
            seq[k] = [errors(meshes[l], red, eps, I, BASE['nz'], meshes[l-1], BASE['ng'])[4:6].tolist()
                      for l in (1, 2, 3)]
            print("      mesh changes (1->2, 2->3, 3->4): fluid " + " ".join(f"{s[0]:.3e}" for s in seq[k])
                  + " | fin " + " ".join(f"{s[1]:.3e}" for s in seq[k]), flush=True)
    d = derived(res, list(ks), gam, red, I)
    print("\nTable rows (LaTeX):\n"+table_rows(res, ks))
    print(f"\nlast mesh changes: at most {d['pct_mesh_fl']:.3f}% (fluid) and {d['pct_mesh_fin']:.3f}% (fin) "
          "of the corrected errors")
    print(f"last-pair orders: (a0_fl, ap_fl) = ({d['order_0_fl']:.3f}, {d['order_p_fl']:.3f}),  "
          f"(a0_fin, ap_fin) = ({d['order_0_fin']:.3f}, {d['order_p_fin']:.3f})")
    print("E0/eps            :", " ".join(f"{v:.5f}" for v in d['E0_over_eps']))
    print("Ep/eps^2          :", " ".join(f"{v:.5f}" for v in d['Ep_over_eps2']))
    print("Ep/((1+g) eps^2)  :", " ".join(f"{v:.5f}" for v in d['Ep_over_1pg_eps2']))
    print(f"||theta_1||_L2((0,1)x I) = {d['theta1_norm_I']:.5f}")
    out = {str(k): res[k] for k in ks}
    out['meta'] = dict(Pe=Pe, gamma=gam, NX=NX, Ny=Ny, I=list(I), **BASE)
    out['derived'] = d
    if sequence: out['sequence'] = {str(k): v for k, v in seq.items()}
    fn = f"res_Pe{Pe:g}_g{gam:g}_{NX}x{Ny}.json"; json.dump(out, open(fn, "w"), indent=1); print("wrote", fn)


def check(Pe, gam, NX, Ny, I, enh, ks=(8, 64)):
    """Change of the reported errors when the numerical parameters are enlarged (same mesh)."""
    out = dict(meta=dict(Pe=Pe, gamma=gam, NX=NX, Ny=Ny, I=list(I), base=BASE, enlarged=enh))
    worst = 0.0
    for k in ks:
        eps = 1/k; ee = []
        for P in (BASE, enh):
            t = time.time()
            ref = Ref(NX, Ny, eps, Pe, gam, P['nmodes'], P['vel'], P['ng'])
            ee.append(errors(ref, Reduced(Pe, gam, P['nred']), eps, I, P['nz'], None, P['ng'])[:4])
            print(f"  1/{k} {P}: " + " ".join(f"{v:.12e}" for v in ee[-1]) + f" [{time.time()-t:.0f}s]", flush=True)
        dlt = np.abs(ee[1]-ee[0]); worst = max(worst, dlt.max())
        out[str(k)] = dict(base=ee[0].tolist(), enlarged=ee[1].tolist(), change=dlt.tolist())
        print(f"1/{k}: max change of (E0,Ep,E0fin,Epfin) = {dlt.max():.3e}", flush=True)
    out['max_change'] = worst
    print(f"max change over eps = 1/{ks[0]}, 1/{ks[1]}: {worst:.3e}")
    fn = f"check_Pe{Pe:g}_g{gam:g}_{NX}x{Ny}.json"; json.dump(out, open(fn, "w"), indent=1); print("wrote", fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("Pe", type=float); ap.add_argument("gam", type=float)
    ap.add_argument("NX", type=int); ap.add_argument("Ny", type=int)
    ap.add_argument("I", type=float, nargs="*", default=[0.05, 0.2])
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--check", action="store_true")
    for key in BASE: ap.add_argument("--"+("modes" if key == "nmodes" else key), dest=key, type=int, default=None)
    A = ap.parse_args(); I = tuple(A.I)
    if A.check:
        enh = {key: (getattr(A, key) if getattr(A, key) is not None else BASE[key]) for key in BASE}
        check(A.Pe, A.gam, A.NX, A.Ny, I, enh)
    else:
        run(A.Pe, A.gam, A.NX, A.Ny, I, sequence=A.sequence)
