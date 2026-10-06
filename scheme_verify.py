"""Section 4.5.1: verification of the sequential finite-element / backward-Euler scheme (3.18).

Reproduces Table 1 and every auxiliary number quoted in Section 4.5.1:
  * g(0) = g'(0) = 0 for the three-mode data (4.19);
  * truncation check of the analytic reference corrector (400 -> 800 sine terms);
  * spatial refinement (k = h^2/1e5) and axial refinement (J = 80, k = L/N), with last-pair orders;
  * uniform-step propagation by powers of the backward-Euler matrix, evaluated exactly in its
    M-orthonormal eigenbasis, checked against a direct 47-step calculation with J = 15;
  * halving the axial step of the spatial test at J = 8 and J = 64;
  * arbitrary-step stability test (J = 36, 90 steps on [0,0.3], seed 453): cumulative defect of the
    energy identity (4.5) and the inequalities (4.6)-(4.7) at every step.
Writes scheme_results.json (unrounded).

Note on the powers: a literal np.linalg.matrix_power of the 2J x 2J step matrix accumulates about
N*eps_mach relative round-off (N = 8.2e7 steps at J = 64, i.e. ~1e-9), which would swamp the
4e-10 halving change.  The eigenbasis evaluation, rho^N = exp(-N log1p(k mu)), is exact to round-off.
"""
import numpy as np, json
from scipy.linalg import eigh, solve

a, m, b, L = 1.4, 0.7, -0.3, 0.2
kap = a/m
lam3 = (np.arange(1, 4)-0.5)*np.pi
r = kap*lam3**2
c = -0.5*np.array([1, 1/6, 1/50])
G = b*r*c*(-1.0)**np.arange(3)                 # g(Z) = sum_j G_j exp(-r_j Z)
SEED = 453                                     # numpy.random.default_rng(SEED).uniform(-4, 2, 90)


def u_ex(y, Z): return (c*np.exp(-r*Z))@np.sin(np.outer(lam3, y))
def g_ex(Z): return np.exp(-np.outer(np.atleast_1d(Z), r))@G
def gp_ex(Z): return np.exp(-np.outer(np.atleast_1d(Z), r))@(-r*G)


def conv(A, R, Z):
    """int_0^Z exp(-A_i (Z-z)) exp(-R_k z) dz  for all (i,k); exact also when A_i = R_k."""
    D = A[:, None]-R[None, :]; same = (D == 0)
    return np.where(same, Z*np.exp(-R[None, :]*Z),
                    (np.exp(-R[None, :]*Z)-np.exp(-A[:, None]*Z))/np.where(same, 1.0, D))


def theta1(y, Z, nterms=400, lift=1):
    """Reference corrector of (3.12) for the data (4.19): boundary lift + analytic sine convolution.
    lift=1: B = g y/a, remainder driven by g' (series ~ n^-4; used for the reference values);
    lift=2: B = (g y + g' p/kappa)/a of Lemma 5.3, remainder driven by g'' (independent cross-check)."""
    ln = (np.arange(1, nterms+1)-0.5)*np.pi; A = kap*ln**2; A[:3] = r      # identical floats for n = j
    sg = (-1.0)**np.arange(nterms)
    gZ = g_ex(Z)[0]; gpZ = gp_ex(Z)[0]
    if lift == 1:
        coef = -(2/a)*sg/ln**2*(conv(A, r, Z)@(-r*G))
        return gZ*y/a+coef@np.sin(np.outer(ln, y))
    coef = (2/(a*kap))*sg/ln**4*(conv(A, r, Z)@(r**2*G))
    return (gZ*y+gpZ*(y**3/6-y/2)/kap)/a+coef@np.sin(np.outer(ln, y))


XG, WG = np.polynomial.legendre.leggauss(10); XG = (XG+1)/2; WG = WG/2      # 10 points per element


def gauss_points(J):
    h = 1/J
    return ((np.arange(J)*h)[:, None]+h*XG[None, :]).ravel()


def l2err(U, f):
    """|| u_h - f ||_{L2(0,1)} for nodal values U (base value 0), 10 Gauss points per element."""
    J = len(U); h = 1/J; Uf = np.r_[0, U]
    uh = Uf[:-1, None]*(1-XG)+Uf[1:, None]*XG
    return np.sqrt(h*np.sum(WG*(uh-f(gauss_points(J)).reshape(J, -1))**2))


def fe(J):
    """Unit P1 mass and stiffness on y_i = i/J, base node eliminated, exact integration (3.19)."""
    h = 1/J
    M1 = np.diag(np.full(J, 4*h/6)); M1[-1, -1] = 2*h/6
    M1 += np.diag(np.full(J-1, h/6), 1)+np.diag(np.full(J-1, h/6), -1)
    K1 = np.diag(np.full(J, 2/h)); K1[-1, -1] = 1/h
    K1 += np.diag(np.full(J-1, -1/h), 1)+np.diag(np.full(J-1, -1/h), -1)
    return M1, K1


class Scheme:
    def __init__(s, J):
        s.J = J; s.M1, s.K1 = fe(J); s.M = m*s.M1; s.K = a*s.K1
        s.mu, s.Phi = eigh(s.K, s.M)                       # Phi^T M Phi = I
        s.y = np.arange(1, J+1)/J; s.u0 = u_ex(s.y, 0)       # u_h^0 = R_h u_in (nodal interpolant)
        s.al = s.Phi.T@(s.M@s.u0); s.tip = s.Phi[-1]
        s.Gh = b*s.al*s.mu*s.tip                            # g_h^sd(Z) = sum Gh_i exp(-mu_i Z)

    def norm(s, e): return np.sqrt(e@s.M1@e)

    # exact semidiscrete evolution
    def gsd(s, Z): return np.exp(-np.outer(np.atleast_1d(Z), s.mu))@s.Gh
    def usd(s, Z): return s.Phi@(s.al*np.exp(-s.mu*Z))
    def vsd(s, Z): return s.Phi@(s.tip*(conv(s.mu, s.mu, Z)@s.Gh))

    def power(s, k, N):
        """(u_h^N, v_h^N) for N uniform steps k: powers of the BE matrix in its eigenbasis."""
        rho = 1/(1+k*s.mu); rN = np.exp(-N*np.log1p(k*s.mu))
        D = s.mu[None, :]-s.mu[:, None]; same = (D == 0)
        GS = np.where(same, (k*N*rN*rho)[:, None], (rN[:, None]-rN[None, :])/np.where(same, 1.0, D))
        return s.Phi@(s.al*rN), s.Phi@(b*s.tip*(GS@(s.tip*s.mu*s.al)))

    def step(s, ks, u0=None):
        """Direct application of (3.18) on the axial grid with steps ks; returns u^n, v^n, g^n."""
        u = s.u0.copy() if u0 is None else u0.copy(); v = np.zeros(s.J); U = [u]; V = [v]; g = []
        for k in ks:
            S = s.M/k+s.K
            un = solve(S, s.M@u/k, assume_a='pos'); gn = -b*(un[-1]-u[-1])/k
            rhs = s.M@v/k; rhs[-1] += gn; v = solve(S, rhs, assume_a='pos'); u = un
            U.append(u); V.append(v); g.append(gn)
        return np.array(U), np.array(V), np.array(g)

    def blockpower(s, k, N):
        """Literal np.linalg.matrix_power of the 2J x 2J step matrix (diagnostic only)."""
        J = s.J; S = s.M/k+s.K; A = solve(S, s.M/k); w = solve(S, np.eye(J)[:, -1])
        cT = -(b/k)*(A[-1]-np.eye(J)[-1]); T = np.zeros((2*J, 2*J))
        T[:J, :J] = A; T[J:, J:] = A; T[J:, :J] = np.outer(w, cT)
        x = np.linalg.matrix_power(T, N)@np.r_[s.u0, np.zeros(J)]
        return x[:J], x[J:]


def table1():
    zq, wq = np.polynomial.legendre.leggauss(100); zq = L*(zq+1)/2; wq = wq*L/2   # 100 axial points
    sp = dict(J=[8, 16, 32, 64], N=[], Eu=[], Ev=[], Eg=[])
    for J in sp['J']:
        s = Scheme(J); k = (1/J)**2/1e5; N = int(round(L/k)); u, v = s.power(k, N)
        sp['N'].append(N)
        sp['Eu'].append(l2err(u, lambda y: u_ex(y, L)))
        sp['Ev'].append(l2err(v, lambda y: theta1(y, L)))
        sp['Eg'].append(float(np.sqrt(wq@(s.gsd(zq)-g_ex(zq))**2)))
    ax = dict(J=80, N=[40, 80, 160, 320], Eu=[], Ev=[], Eg=[])
    s = Scheme(80)
    for N in ax['N']:
        k = L/N; U, V, g = s.step([k]*N); Zn = k*np.arange(1, N+1)
        ax['Eu'].append(s.norm(U[-1]-s.usd(L))); ax['Ev'].append(s.norm(V[-1]-s.vsd(L)))
        ax['Eg'].append(float(np.sqrt(k*np.sum((g-s.gsd(Zn))**2))))
    for T in (sp, ax):
        T['order'] = [float(np.log2(T[e][-2]/T[e][-1])) for e in ('Eu', 'Ev', 'Eg')]
    return sp, ax


def auxiliary():
    out = dict(g0=float(g_ex(0.0)[0]), gp0=float(gp_ex(0.0)[0]))
    # truncation of the reference corrector: all values used for E_v (Gauss points) and the nodes
    pts = np.unique(np.concatenate([np.r_[gauss_points(J), np.arange(1, J+1)/J] for J in (8, 16, 32, 64)]))
    out['trunc_change'] = float(np.abs(theta1(pts, L, 400)-theta1(pts, L, 800)).max())
    out['lift1_vs_lift2'] = float(np.abs(theta1(pts, L, 800, 1)-theta1(pts, L, 800, 2)).max())
    # powers of the BE matrix vs a direct 47-step calculation, J = 15
    s = Scheme(15); d = []
    for k in (L/47, (1/15)**2/1e5):
        U, V, _ = s.step([k]*47); u, v = s.power(k, 47)
        d.append(max(np.abs(U[-1]-u).max(), np.abs(V[-1]-v).max()))
    out['step47_maxdiff'] = float(max(d))
    # halving the axial step of the spatial test
    out['halving'] = {}
    for J in (8, 64):
        s = Scheme(J); k = (1/J)**2/1e5; N = int(round(L/k))
        u1, v1 = s.power(k, N); u2, v2 = s.power(k/2, 2*N)
        out['halving'][str(J)] = dict(du=s.norm(u1-u2), dv=s.norm(v1-v2))
        if J == 64:
            ub, vb = s.blockpower(k, N)
            out['matrix_power_roundoff_J64'] = float(max(s.norm(ub-u1), s.norm(vb-v1)))
    return out


def stability(J=36, n=90, Lz=0.3):
    xi = np.random.default_rng(SEED).uniform(-4, 2, n)
    ks = Lz*np.exp(xi)/np.exp(xi).sum()
    s = Scheme(J); U, V, g = s.step(ks)
    S = np.r_[[-solve(s.M, s.K@s.u0)], (U[1:]-U[:-1])/ks[:, None]]       # s^0 = -A_h R_h u_in, s^n = D_n u^n
    # identity s^0 = P_h[(a/m) u_in''] of Theorem 4.3
    h = 1/J; yq = gauss_points(J).reshape(J, -1); f = (a/m)*(-(c*lam3**2)@np.sin(np.outer(lam3, yq.ravel()))).reshape(J, -1)
    load = np.zeros(J+1); load[:-1] += h*(f*(1-XG))@WG; load[1:] += h*(f*XG)@WG
    s0_identity = float(np.abs(solve(s.M1, load[1:])-S[0]).max())
    def energy(Z, w):   # m|z^n|^2 + m sum|z^j-z^{j-1}|^2 + w sum k_j |z^j'|^2
        dZ = Z[1:]-Z[:-1]
        return (m*np.einsum('ni,ij,nj->n', Z, s.M1, Z)
                + np.r_[0, np.cumsum(m*np.einsum('ni,ij,nj->n', dZ, s.M1, dZ)
                                     + w*ks*np.einsum('ni,ij,nj->n', Z[1:], s.K1, Z[1:]))])
    defect = 0.0; defect_rel = 0.0
    for Z in (U, S):
        E = energy(Z, 2*a); d = np.abs(E[1:]-E[0]).max(); defect = max(defect, d); defect_rel = max(defect_rel, d/E[0])
    uinpp2 = np.sum(c**2*lam3**4)/2                                          # ||u_in''||^2
    Sg = np.cumsum(ks*g**2); B1 = m*b*b/(2*a)*(S[0]@s.M1@S[0]); B2 = a*b*b/(2*m)*uinpp2; B3 = b*b/(2*m)*uinpp2
    Ev = energy(V, a)[1:]
    ok46 = bool(np.all(Sg <= B1) and B1 <= B2); ok47 = bool(np.all(Ev <= Sg/a) and np.all(Sg/a <= B3))
    return dict(seed=SEED, rng="numpy.random.default_rng(seed).uniform(-4,2,90)", kmin=float(ks.min()),
                kmax=float(ks.max()), defect=float(defect), defect_rel=float(defect_rel),
                load_inequality_46=ok46, corrector_inequality_47=ok47,
                margin46_rel=float(np.min((B1-Sg)/B1)), margin47_rel=float(np.min((Sg/a-Ev)/(Sg/a))),
                s0_identity=s0_identity)


def tex_cell(x):
    mm, e = f"{x:.3e}".split("e")
    return f"${mm}\\!\\times\\!10^{{{int(e)}}}$"


def table1_tex(sp, ax):
    rows = []
    for i in range(4):
        rows.append(f"{sp['J'][i]}\n" + "".join(f"& {tex_cell(sp[e][i])}\n" for e in ('Eu', 'Ev', 'Eg'))
                    + f"& {ax['N'][i]}\n" + "".join(f"& {tex_cell(ax[e][i])}\n" for e in ('Eu', 'Ev', 'Eg'))[:-1] + "\\\\")
    rows.append("Order & " + " & ".join(f"{o:.3f}" for o in sp['order']) + "\n& Order & "
                + " & ".join(f"{o:.3f}" for o in ax['order']) + "\\\\")
    return "\n".join(rows)


if __name__ == "__main__":
    sp, ax = table1(); aux = auxiliary(); st = stability()
    print("Table 1 (spatial: k = h^2/1e5;  axial: J = 80, k = 0.2/N)")
    print("   J        E_u        E_v        E_g  |    N        E_u        E_v        E_g")
    for i in range(4):
        print(f"{sp['J'][i]:4d} " + " ".join(f"{sp[e][i]:.5e}" for e in ('Eu', 'Ev', 'Eg'))
              + f" | {ax['N'][i]:4d} " + " ".join(f"{ax[e][i]:.5e}" for e in ('Eu', 'Ev', 'Eg')))
    print("order " + " ".join(f"{o:11.4f}" for o in sp['order']) + " |      " + " ".join(f"{o:11.4f}" for o in ax['order']))
    print("\nTable 1 rows (LaTeX):\n" + table1_tex(sp, ax))
    print(f"\ng(0) = {aux['g0']:.2e},  g'(0) = {aux['gp0']:.2e}")
    print(f"reference corrector, 400 -> 800 terms: max change {aux['trunc_change']:.3e} "
          f"(one-term lift vs Lemma-5.3 lift at 800 terms: {aux['lift1_vs_lift2']:.1e})")
    print(f"direct 47-step calculation (J=15) vs powers: max difference {aux['step47_maxdiff']:.1e}")
    for J, d in aux['halving'].items():
        print(f"halving the spatial-test step, J={J}: |du| = {d['du']:.3e}, |dv| = {d['dv']:.3e}")
    print(f"(diagnostic) literal matrix_power vs eigenbasis powers at J=64: {aux['matrix_power_roundoff_J64']:.1e}")
    print(f"arbitrary steps (seed {st['seed']}, k in [{st['kmin']:.2e}, {st['kmax']:.2e}]): "
          f"max cumulative defect in (4.5) = {st['defect']:.3e} (relative {st['defect_rel']:.1e})")
    print(f"   (4.6) holds at every step: {st['load_inequality_46']} (min relative margin {st['margin46_rel']:.2e});"
          f"  (4.7) holds at every step: {st['corrector_inequality_47']} (min relative margin {st['margin47_rel']:.2e})")
    print(f"   s^0 = P_h[(a/m) u_in''] identity: max difference {st['s0_identity']:.1e}")
    json.dump(dict(params=dict(a=a, m=m, b=b, L=L), spatial=sp, axial=ax, auxiliary=aux, stability=st),
              open("scheme_results.json", "w"), indent=1)
    print("wrote scheme_results.json")
