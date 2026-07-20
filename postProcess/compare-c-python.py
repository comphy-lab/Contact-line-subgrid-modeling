# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""
# `compare-c-python.py` — cross-validate the C GLE solver against the Python reference

This script reproduces, end to end, the comparison between the C implementation
of the Generalized Lubrication Equation (GLE) solver (`gle-ode/gle-solve`,
built on the headers in `src-local/`) and the historical Python reference
(`GLE_solver.py`). Both solve the *same* boundary-value problem for a vertical
plate withdrawing from a bath,

$$
\\frac{\\mathrm{d}h}{\\mathrm{d}s} = \\sin\\theta, \\qquad
\\frac{\\mathrm{d}\\theta}{\\mathrm{d}s} = \\omega, \\qquad
\\frac{\\mathrm{d}\\omega}{\\mathrm{d}s}
   = \\frac{3\\,\\mathrm{Ca}\\,M(\\theta,\\mu_r)}{h\\,(h + 3\\lambda)}
     - \\frac{\\cos\\theta}{\\ell_{\\mathrm{cap}}^{2}},
$$

with inner conditions $h(s_0) = h_0$, $\\theta(s_0) = \\theta_0$ and the
far-field condition $\\omega(s_{\\max}) = 0$. The Python code uses SciPy
collocation (`solve_bvp`); the C code uses adaptive Runge–Kutta–Cash–Karp
shooting on the contact-line curvature $\\omega_0$.

## What it does

1. Imports the Python solver from `python/GLE_solver.py` (its final home),
   falling back to the repository root if that path does not yet exist, and
   reads the physical parameters ($\\mathrm{Ca}$, $\\mu_r$, $\\lambda$,
   $\\ell_{\\mathrm{cap}}$, $\\theta_0$) straight from the module so the two
   solvers stay locked to identical physics. Note $g^{*} = 1/\\ell_{\\mathrm{cap}}^{2}$
   is passed to the C solver as `grav`.
2. Runs the Python reference (`run_solver_and_plot`) and the C solver
   (compiled on the fly with `cc -I src-local`, so no stale binary is trusted).
3. Interpolates the Python solution onto the C solver's native adaptive nodes —
   $\\theta$ linearly, $h$ in log-space — and reports the maximum and RMS
   relative errors over $s \\in [2,\\,5\\times10^{5}]$, plus the apparent-angle
   difference at $s = 10^{4}$ (that is $0.01\\,\\ell_{\\mathrm{cap}}$).
4. Writes a two-panel, publication-style figure to `img/c-vs-python.png`: the
   $\\theta(s)$ overlay on a logarithmic $s$-axis, and the relative error on
   log–log axes.

## The far-field caveat

The condition $\\omega(s_{\\max}) = 0$ pins the curvature but leaves the
far-field *angle* soft: the gain $\\mathrm{d}\\theta(s_{\\max})/\\mathrm{d}\\omega_0$
runs to $\\sim 5\\times10^{7}$, while $\\mathrm{d}\\omega(s_{\\max})/\\mathrm{d}\\omega_0
\\approx 1$. A shooting residual tolerance of $5\\times10^{-8}$ therefore admits
a few degrees of slack in $\\theta(s_{\\max})$. Agreement is consequently
excellent where the problem is well conditioned ($s \\lesssim 10^{4}$, relative
error $< 3\\times10^{-4}$) and degrades monotonically towards the cap. This is a
property of the boundary-value problem, not of either implementation — the two
right-hand sides agree to machine precision.

## Usage

```bash
uv run postProcess/compare-c-python.py            # full 10^6-node reference
uv run postProcess/compare-c-python.py --quick    # cap N_grid at 2x10^5 for CI
```

Author: Vatsal Sanjay — CoMPhy Lab, Durham University.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Matplotlib must be told to use a headless backend *before* pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- comparison configuration --------------------------------------------------
WINDOW = (2.0, 0.5e6)     # s-range for the reported error metrics
S_APPARENT = 1.0e4        # apparent-angle probe (0.01 * l_cap)
QUICK_NGRID = 200_000     # capped mesh for --quick (CI-speed) runs


def find_repo_root() -> Path:
    """Locate the repository root.

    Honours the ``GLE_REPO`` environment override; otherwise walks upward from
    this file looking for a directory that carries both ``src-local`` and a
    copy of the Python reference (in ``python/`` or at the root).
    """
    env = os.environ.get("GLE_REPO")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        has_src = (cand / "src-local").is_dir()
        has_py = (cand / "python" / "GLE_solver.py").exists() or \
                 (cand / "GLE_solver.py").exists()
        if has_src and has_py:
            return cand
    # last resort: parent of postProcess/
    return here.parent.parent


def import_python_reference(repo: Path):
    """Import ``GLE_solver`` from ``python/`` (preferred) or the repo root."""
    for sub in ("python", "."):
        cand = (repo / sub / "GLE_solver.py")
        if cand.exists():
            sys.path.insert(0, str(cand.parent))
            import GLE_solver  # noqa: E402  (deferred, path-dependent import)
            return GLE_solver
    raise FileNotFoundError(
        f"GLE_solver.py not found under {repo}/python or {repo}")


def run_python(gle, quick: bool):
    """Run the SciPy reference and return ``(s, h, theta_rad)`` arrays."""
    if quick:
        gle.N_grid = min(gle.N_grid, QUICK_NGRID)
    print(f"[python] N_grid = {gle.N_grid:,}  (quick={quick})")
    with tempfile.TemporaryDirectory() as tmp:
        _, s, h, theta, _ = gle.run_solver_and_plot(GUI=False, output_dir=tmp)
    return np.asarray(s), np.asarray(h), np.asarray(theta)


def run_c(repo: Path, gle):
    """Compile and run the C solver at the matching parameters.

    Returns ``(s, h, theta_rad)``. The physical parameters are taken from the
    imported Python module so the two solvers are guaranteed identical.
    """
    src = repo / "gle-ode" / "gle-solve.c"
    inc = repo / "src-local"
    grav = 1.0 / gle.l_cap ** 2                    # g* = 1 / l_cap^2
    theta_deg = float(np.rad2deg(gle.theta0))
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "gle-solve"
        profile = Path(tmp) / "c-profile.csv"
        cc = os.environ.get("CC", "cc")
        subprocess.run(
            [cc, "-O2", "-std=c99", f"-I{inc}", "-o", str(binary),
             str(src), "-lm"],
            check=True, capture_output=True, text=True)
        args = [
            str(binary),
            f"slip={gle.lambda_slip:g}", "s0=1", "h0=1",
            f"grav={grav:.12e}", f"theta_mic_deg={theta_deg:g}",
            f"Ca={gle.Ca:g}", f"mu_r={gle.mu_r:g}",
            "outer_bc=omega_zero", f"smax_cap={gle.l_cap:g}",
            "rtol=1e-12", "atol=1e-14", f"profile_out={profile}",
        ]
        print("[c] " + " ".join(args[1:]))
        out = subprocess.run(args, check=True, capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if line.startswith(("omega0", "theta_app", "residual")):
                print("    " + line)
        data = np.genfromtxt(profile, delimiter=",", names=True)
    return data["s"], data["h"], np.deg2rad(data["theta_deg"])


def interp_python(s_query, s_py, h_py, th_py):
    """Interpolate the (dense) Python solution onto ``s_query``.

    ``theta`` linearly, ``h`` in log-space (it spans many decades).
    """
    theta = np.interp(s_query, s_py, th_py)
    h = np.exp(np.interp(s_query, s_py, np.log(h_py)))
    return h, theta


def compute_metrics(s_c, h_c, th_c, s_py, h_py, th_py):
    """Relative-error metrics over ``WINDOW`` and the apparent-angle probe."""
    lo, hi = WINDOW
    mask = (s_c >= lo) & (s_c <= hi)
    s_w = s_c[mask]
    h_py_w, th_py_w = interp_python(s_w, s_py, h_py, th_py)
    rel_th = np.abs(th_c[mask] - th_py_w) / np.abs(th_py_w)
    rel_h = np.abs(h_c[mask] - h_py_w) / np.abs(h_py_w)

    h_a_py, th_a_py = interp_python(S_APPARENT, s_py, h_py, th_py)
    th_a_c = np.interp(S_APPARENT, s_c, th_c)

    return {
        "n": s_w.size,
        "theta_max": float(rel_th.max()),
        "theta_rms": float(np.sqrt(np.mean(rel_th ** 2))),
        "h_max": float(rel_h.max()),
        "h_rms": float(np.sqrt(np.mean(rel_h ** 2))),
        "app_c_deg": float(np.rad2deg(th_a_c)),
        "app_py_deg": float(np.rad2deg(th_a_py)),
        "app_diff_deg": float(np.rad2deg(abs(th_a_c - th_a_py))),
        "app_rel": float(abs(th_a_c - th_a_py) / abs(th_a_py)),
    }


def make_figure(out_png, s_c, h_c, th_c, s_py, h_py, th_py, m):
    """Two-panel publication-style comparison figure."""
    # The Python reference calls plt.style.use('seaborn-...') as a side effect;
    # reset to a clean sheet before imposing the publication rcParams.
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.linewidth": 1.1,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
    })
    c_py, c_c = "#1f4e79", "#c0392b"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.0))

    # panel 1: theta(s) overlay, log-x
    ax1.semilogx(s_py, np.rad2deg(th_py), "-", color=c_py, lw=2.0,
                 label="Python (solve_bvp)")
    ax1.semilogx(s_c, np.rad2deg(th_c), "o", color=c_c, ms=3.5, mfc="none",
                 mew=1.0, label="C shooting")
    ax1.axvline(S_APPARENT, color="0.5", ls=":", lw=1.0)
    ax1.set_xlabel(r"$s/\lambda$")
    ax1.set_ylabel(r"$\theta(s)$  [deg]")
    ax1.set_title(r"Interface angle: C shooting vs SciPy collocation")
    ax1.legend(loc="upper left")
    ax1.margins(x=0.02)

    # panel 2: relative error, log-log
    lo, hi = WINDOW
    mask = (s_c >= lo) & (s_c <= hi)
    s_w = s_c[mask]
    h_py_w, th_py_w = interp_python(s_w, s_py, h_py, th_py)
    rel_th = np.abs(th_c[mask] - th_py_w) / np.abs(th_py_w)
    rel_h = np.abs(h_c[mask] - h_py_w) / np.abs(h_py_w)
    ax2.loglog(s_w, rel_th, "-", color=c_c, lw=1.8, label=r"$\theta$ rel. error")
    ax2.loglog(s_w, rel_h, "--", color=c_py, lw=1.6, label=r"$h$ rel. error")
    ax2.axhline(1e-3, color="0.4", ls=":", lw=1.0)
    ax2.text(s_w[-1], 1.3e-3, r"scipy tol $\sim 10^{-3}$", fontsize=9,
             color="0.35", ha="right", va="bottom")
    ax2.set_xlabel(r"$s/\lambda$")
    ax2.set_ylabel(r"relative error")
    ax2.set_title(r"Pointwise relative error (Python onto C nodes)")
    ax2.legend(loc="upper left")

    txt = (f"max rel $\\theta$ = {m['theta_max']:.2e}\n"
           f"rms rel $\\theta$ = {m['theta_rms']:.2e}\n"
           f"$\\Delta\\theta_{{\\rm app}}(10^4)$ = {m['app_diff_deg']:.2e} deg")
    ax2.text(0.97, 0.05, txt, transform=ax2.transAxes, fontsize=9,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help=f"cap Python N_grid at {QUICK_NGRID:,} for CI-speed runs")
    args = ap.parse_args()

    repo = find_repo_root()
    print(f"[repo] {repo}")
    gle = import_python_reference(repo)

    s_py, h_py, th_py = run_python(gle, args.quick)
    s_c, h_c, th_c = run_c(repo, gle)

    m = compute_metrics(s_c, h_c, th_c, s_py, h_py, th_py)

    lo, hi = WINDOW
    print("\n=== relative-error metrics ===")
    print(f"window s in [{lo:g}, {hi:g}]  ({m['n']} C nodes)")
    print(f"  theta : max = {m['theta_max']:.3e}   rms = {m['theta_rms']:.3e}")
    print(f"  h     : max = {m['h_max']:.3e}   rms = {m['h_rms']:.3e}")
    print(f"apparent angle at s = {S_APPARENT:g}:")
    print(f"  C  = {m['app_c_deg']:.6f} deg   Python = {m['app_py_deg']:.6f} deg")
    print(f"  |diff| = {m['app_diff_deg']:.3e} deg  (rel {m['app_rel']:.3e})")

    make_figure(repo / "img" / "c-vs-python.png",
                s_c, h_c, th_c, s_py, h_py, th_py, m)


if __name__ == "__main__":
    main()
