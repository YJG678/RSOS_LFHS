"""Reproduce every number of Section 4.5 and Figures 1-6 (about 5 minutes on one core, < 1 GB RAM).

    python run_all.py            # all steps; logs in logs/, environment record in environment.txt
"""
import os, platform, subprocess, sys, time
import numpy, scipy, matplotlib

PY = sys.executable
STEPS = [
    ("scheme",      ["scheme_verify.py"]),                                                    # Sec. 4.5.1, Table 1
    ("ref_g0.4",    ["refsolve.py", "8.4", "0.4", "96", "1280", "--sequence"]),               # Sec. 4.5.2, Table 2
    ("ref_g13",     ["refsolve.py", "84", "13", "96", "1280", "--sequence"]),                 # Sec. 4.5.3, Table 3
    ("ref_g100",    ["refsolve.py", "606", "100", "96", "1280"]),                             # gamma = 100 run
    ("check_g0.4",  ["refsolve.py", "8.4", "0.4", "48", "640", "--check",
                     "--modes", "36", "--vel", "1024", "--ng", "6", "--nz", "32", "--nred", "320"]),
    ("check_g13",   ["refsolve.py", "84", "13", "48", "640", "--check",
                     "--modes", "80", "--vel", "1024", "--nz", "32", "--nred", "320"]),
    ("report",      ["verify_report.py"]),                                                    # compare with manuscript
    ("figures",     ["makefigs.py"]),                                                         # Figures 1-6
]

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    with open("environment.txt", "w") as f:
        f.write(f"python {platform.python_version()}\nnumpy {numpy.__version__}\nscipy {scipy.__version__}\n"
                f"matplotlib {matplotlib.__version__}\nplatform {platform.platform()}\n"
                f"date {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    for name, cmd in STEPS:
        t = time.time(); print(f"[{name}] python {' '.join(cmd)}", flush=True)
        with open(os.path.join("logs", name+".log"), "w") as log:
            rc = subprocess.call([PY]+cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"    -> logs/{name}.log  ({time.time()-t:.0f} s){'  FAILED' if rc else ''}", flush=True)
        if rc: sys.exit(rc)
    print(f"done in {time.time()-t0:.0f} s; see verification_report.txt, tables_generated.tex and figures/")
    print(next((l.strip() for l in open("verification_report.txt") if l.startswith("Summary:")), ""))
