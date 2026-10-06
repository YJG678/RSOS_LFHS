# Verification scripts for Section 4.5 and Figures 1–6

```
python run_all.py
```
Runs everything in about 5 minutes on one core (< 1 GB RAM). Requires Python 3 with NumPy, SciPy and matplotlib.
It writes `environment.txt`, the logs in `logs/`, and the outputs listed below.

| Script | Reproduces | Output |
|---|---|---|
| `scheme_verify.py` | Sec. 4.5.1: Table 1 and the auxiliary checks (truncation of the reference corrector, 47-step check, step halving, arbitrary-step stability test) | `scheme_results.json` |
| `refsolve.py 8.4 0.4 96 1280 --sequence` | Sec. 4.5.2: Table 2, mesh-change percentages, last-pair orders | `res_Pe8.4_g0.4_96x1280.json` |
| `refsolve.py 84 13 96 1280 --sequence` | Sec. 4.5.3: Table 3, orders, E0/ε, Ep/ε², ‖θ₁‖ | `res_Pe84_g13_96x1280.json` |
| `refsolve.py 606 100 96 1280` | γ = 100 run (text of 4.5.3, Fig. 6) | `res_Pe606_g100_96x1280.json` |
| `refsolve.py 8.4 0.4 48 640 --check --modes 36 --vel 1024 --ng 6 --nz 32 --nred 320` | refinement check of 4.5.2 | `check_Pe8.4_g0.4_48x640.json` |
| `refsolve.py 84 13 48 640 --check --modes 80 --vel 1024 --nz 32 --nred 320` | refinement check of 4.5.3 | `check_Pe84_g13_48x640.json` |
| `verify_report.py` | compares every number of Sec. 4.5 with the manuscript | `verification_report.txt`, `tables_generated.tex` (Tables 1–3) |
| `makefigs.py` | Figures 1–6 | `figures/fig1_geometry … fig6_gamma_dependence` (.pdf, .png); `figdata.json` (PGFPlots coordinates used in the .tex) |

`refsolve.py` prints the LaTeX rows of Tables 2–3 and the derived numbers; `scheme_verify.py` prints those of Table 1.
`--sequence` also reports the changes between all four nested reference meshes.
`makefigs.py --figs 3 6` draws selected figures only. The Z = 0.1 reference fields for Figures 4–5 are cached in `fields_Z0.1.npz`.

## Report status codes
* **OK**: the recomputed value rounds to the printed value, or the stated inequality holds.
* **EDGE**: the value lies on a rounding boundary. The printed digit agrees with this run, but the unrounded data of an earlier run (embedded in the .tex figures) round the other way. This applies to Δ_ref,fin(1/8) and Δ_ref,fl(1/32).
* **RNDF**: a round-off-level quantity. Only its order of magnitude is reproducible. This applies to the cumulative defect in (4.5).

## Numerical notes
* **Matrix powers.** In the spatial test, uniform steps are propagated by powers of the backward-Euler matrix, evaluated in its M-orthonormal eigenbasis as ρᴺ = exp(−N log1p(kμ)). A literal `numpy.linalg.matrix_power` of the step matrix accumulates about N·ε_mach relative round-off. At J = 64 that is N = 8.2·10⁷ steps, or roughly 10⁻⁹, which would swamp the 4·10⁻¹⁰ step-halving change.
* **Reference corrector.** The reference corrector for Table 1 uses the lift g·y/a with an analytic sine convolution of g′ (400 terms). Its 400→800-term change is 4.5·10⁻¹¹.
* **Random steps.** The arbitrary steps use `numpy.random.default_rng(453).uniform(-4, 2, 90)`.
