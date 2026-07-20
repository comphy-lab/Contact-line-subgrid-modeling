# Contact-line-subgrid-modeling

A multiscale treatment of the moving-contact-line singularity. The Generalized
Lubrication Equation (GLE) resolves the interface from the Navier-slip scale
($\sim$ nm) up to the DNS grid scale; a Basilisk two-phase DNS resolves it from
the grid scale up to the macroscale. The two are coupled at the grid size: each
DNS timestep, the GLE hands the DNS an *apparent* contact angle, and takes the
DNS interface curvature as its own outer boundary condition. The subgrid wedge
never has to be resolved on the DNS mesh — the GLE carries the viscous bending
of the interface across the decades of length scale below it.

## Headline result

The steady dip-coating branch reproduced through its entrainment fold: the
bifurcation diagram $\Delta(\mathrm{Ca})$ of Fig. 4b of Snoeijer & Andreotti,
*Annu. Rev. Fluid Mech.* **45**, 269–292 (2013).

![Reproduction of Fig. 4b: meniscus rise versus capillary number, GLE
continuation against the digitized theory curve and Delon et al. (2008)
experiments](img/fig4b-reproduction.png)

Continuation of the GLE (in C, this repository; solid black) against the theory
curve vector-digitized from the paper's PDF (thick grey) and the five
experimental series of Delon et al. (2008). The saddle-node fold sits at
$\mathrm{Ca}^{*} = 1.0544\times10^{-2}$ versus $1.054\times10^{-2}$ digitized.
The slip length is calibrated *to the fold*, so recovering $\mathrm{Ca}^{*}$ is
circular and is **not** a test. The genuine out-of-sample checks are the fold
height $\Delta^{*} = 1.440$ versus $1.446$ digitized (not fit to), and the full
branch shape — including the upper-branch approach to the Landau–Levich
asymptote — which carries no further free parameters. Microscopic parameters:
$\theta_e = 53.46^{\circ}$, $\lambda/\ell_\gamma = 7.46\times10^{-6}$
(the digitization provenance is in
[data/fig4b-digitized/CALIBRATION.md](data/fig4b-digitized/CALIBRATION.md)).

## The model

State vector $y = (h, \theta, \omega, \zeta)$ parameterised by arc length $s$
along the liquid–gas interface, measured from the contact line, in the
formulation of Snoeijer (2006):

$$
\frac{\mathrm{d}h}{\mathrm{d}s} = \sin\theta, \qquad
\frac{\mathrm{d}\theta}{\mathrm{d}s} = \omega, \qquad
\frac{\mathrm{d}\omega}{\mathrm{d}s}
   = \frac{3\,\mathrm{Ca}\;M(\theta,\mu_r)}{h\,(h + 3\lambda)} + G(\theta),
\qquad
\frac{\mathrm{d}\zeta}{\mathrm{d}s} = \cos\theta .
$$

Here $h$ is the film thickness, $\theta$ the local interface inclination,
$\omega$ the curvature, $\lambda$ the Navier slip length, and $G(\theta)$ the
gravity term. Lengths are non-dimensionalised by the capillary length
$\ell_\gamma = \sqrt{\gamma/\rho g}$. $\mathrm{Ca} = \eta U/\gamma > 0$ is a
**receding** contact line (dip-coating: plate withdrawn from the bath);
$\mathrm{Ca} < 0$ is advancing.

The mobility $M(\theta,\mu_r)$ is the two-fluid Huh–Scriven wedge factor (Chan,
Snoeijer & Eggers 2013),

$$
M(\theta,\mu_r) = \frac{2\sin^3\theta\,\bigl[\mu_r^2 f_1(\theta)
   + 2\mu_r f_3(\theta) + f_1(\pi-\theta)\bigr]}
  {3\,\bigl[\mu_r f_1(\theta) f_2(\pi-\theta)
   \;\boldsymbol{+}\; f_1(\pi-\theta) f_2(\theta)\bigr]},
$$

with $f_1(\theta) = \theta^2 - \sin^2\theta$, $f_2(\theta) = \theta -
\sin\theta\cos\theta$, $f_3(\theta) = \theta(\pi-\theta) + \sin^2\theta$. The
**plus** sign in the denominator is essential and load-bearing: every earlier C
port in this repository's history carried a minus sign there — flagged in review
on PRs #9, #10 and #13, and never fixed — which flips the whole viscous term at
small $\mu_r$. The corrected mobility lives in
[src-local/gle-model.h](src-local/gle-model.h) and is validated against the
Python reference to machine precision. In the one-fluid limit $M(\theta,0) \to
1$ as $\theta \to 0$, recovering classical lubrication $h''' = 3\,\mathrm{Ca}/
h^2$. The function is not named `f`: in Basilisk that name is the VoF volume
fraction (issue #4).

## Numerics

Two complementary solvers share the same right-hand side; both are header-only
C99 with no external dependencies.

**Shooting** ([src-local/gle-shoot.h](src-local/gle-shoot.h)) — adaptive
embedded Cash–Karp RK5(4) integration
([src-local/gle-integrate.h](src-local/gle-integrate.h)) with a
static-meniscus-manifold outer boundary condition, reducing the BVP to a single
scalar equation in the contact-line curvature $\omega_0$. This is the fast
single-solve path and the engine of the DNS coupling. Away from the fold it is
reliable up to roughly $0.6\,\mathrm{Ca}^{*}$ on the lower branch; near the fold
the residual window in $\omega_0$ becomes exponentially narrow.

**Collocation** ([src-local/gle-collocate.h](src-local/gle-collocate.h)) — a
fixed hybrid-mesh implicit-midpoint discretisation with a bordered banded-LU
Newton solve, marching the meniscus rise $\Delta$, which is monotone along the
entire branch. Parameterising by $\Delta$ renders the fold benign and needs no
special turning-point handling. This is precisely what the mesh-readaptation of
scipy's `solve_bvp` inside a continuation loop could not do — the failure mode
of the historical Python continuation (issues #14 and #16). The older
shooting-based continuation is kept in
[src-local/gle-continuation.h](src-local/gle-continuation.h) for provenance and
lower-branch scans only; it is not safe through the fold.

Verified in the solver stack: the Cash–Karp tableau against its published
coefficients; the banded-LU border solve against a dense reference to
$\sim 10^{-14}$; second-order mesh convergence of the fold location; and
cross-agreement between the shooting and collocation solvers of $\sim 2\times
10^{-6}$ at equal $\mathrm{Ca}$.

## Validation

Cross-validated against the historical Python `solve_bvp` reference
([python/GLE_solver.py](python/GLE_solver.py)), which solves the *same* BVP by
SciPy collocation. Driver:
[postProcess/compare-c-python.py](postProcess/compare-c-python.py); figure:
[img/c-vs-python.png](img/c-vs-python.png).

| Check | Result |
| --- | --- |
| Huh–Scriven mobility, C vs Python | machine precision |
| Interface profile $\theta(s)$, well-conditioned window | rel. error $< 2.3\times10^{-4}$ |
| Apparent-angle difference at $s = 10^4$ (`--quick`, post tolerance fix) | $5\times10^{-4}$ deg |
| Gravity-rescaling invariance ($g^{*}$ rescaled) | $4.5\times10^{-6}$ |
| Static ($\mathrm{Ca}\to 0$) limit | exact |
| Fig. 4b branch overlay | see above |

The far-field angle is a soft condition (its gain runs to $\sim 5\times10^7$),
so agreement is excellent where the problem is well conditioned
($s \lesssim 10^4$) and degrades monotonically towards the cap. That is a
property of the boundary-value problem, not of either implementation: the two
right-hand sides agree to machine precision.

## Quick start

```bash
make                 # build the standalone C solvers (any C99 compiler + libm)
make test            # smoke test; also compiles the Basilisk case if qcc is present
./reproduce-fig4b.sh # trace the branch and regenerate img/fig4b-reproduction.png
```

Single solve at one capillary number:

```bash
cd gle-ode && ./gle-solve fig4b.params Ca=5e-3
```

Trace the bifurcation branch through the fold:

```bash
cd gle-ode && ./gle-continuation fig4b.params branch_out=output/branch.csv
```

The GLE-coupled DNS case (serial Basilisk):

```bash
cd simulationCases
qcc -O2 -disable-dimensions -I../src-local contactline-gle.c -o run -lm
```

Requirements: any C99 compiler and `libm` for the solvers — **no GSL, no
SUNDIALS**. Basilisk `qcc` only for the DNS case. `uv`/Python only for the
plotting and cross-validation scripts (each carries its own inline dependency
metadata).

## Repository structure

```
Contact-line-subgrid-modeling
├── src-local - header-only C99 solver stack (no external dependencies)
│   ├── gle-model.h - GLE parameters and right-hand side; corrected Huh–Scriven mobility
│   ├── gle-integrate.h - adaptive Cash–Karp RK5(4) integrator with thickness-event stopping
│   ├── gle-shoot.h - single-shooting BVP solve on the contact-line curvature
│   ├── gle-collocate.h - fixed-mesh implicit-midpoint collocation; fold-free branch tracer
│   ├── gle-continuation.h - legacy shooting-based arclength continuation (lower branch only)
│   ├── gle-basilisk.h - the GLE ↔ DNS coupling seam
│   └── gle-params.h - key=value runtime parameter loader with CLI overrides
├── gle-ode - standalone drivers and parameter files
│   ├── gle-solve.c - single GLE solve, writes the interface profile
│   ├── gle-continuation.c - traces the dip-coating branch through the fold
│   ├── fig4b.params - calibrated parameters reproducing the Fig. 4b theory curve
│   └── Makefile - builds both drivers with cc -lm
├── simulationCases - Basilisk two-phase DNS cases
│   ├── contactline-gle.c - plate coating with the per-timestep GLE subgrid coupling
│   └── contactline.c - baseline case with a fixed contact angle (no GLE coupling)
├── postProcess - uv-runnable plotting and validation
│   ├── plot-fig4b.py - overlays the traced branch on the digitized reference
│   └── compare-c-python.py - cross-validates the C solver against the Python reference
├── python - historical reference implementation and linearized fixture
│   ├── GLE_solver.py - SciPy solve_bvp reference (shares the C physics exactly)
│   ├── validate-linearized-gle.py - checks the linearized GLE against Kansal et al. (2024)
│   ├── reference-data/kansal-minkush-linearized.csv - digitized linearized-GLE reference
│   └── requirements.txt - Python dependencies for the reference scripts
├── data/fig4b-digitized - vector-digitized reference data (do not modify)
│   ├── CALIBRATION.md - axis calibration and extraction provenance
│   ├── theory_curve.csv - the review's multiscale-lubrication theory curve
│   └── symbols_*.csv - the five experimental series of Delon et al. (2008)
├── img - generated figures committed for the README and docs
│   ├── fig4b-reproduction.png - the headline reproduction figure (also .pdf)
│   └── c-vs-python.png - the C-vs-Python cross-validation figure
├── _Archive - superseded code kept for provenance (do not resurrect)
├── .github - documentation pipeline and website generator
├── Makefile - top-level build delegating to gle-ode
├── reproduce-fig4b.sh - one-shot branch trace + figure regeneration
└── LICENSE
```

## DNS coupling status and roadmap

The coupling seam is implemented and runtime-verified.
[simulationCases/contactline-gle.c](simulationCases/contactline-gle.c) runs a
per-step `event`: it samples the DNS interface curvature at the grid scale,
solves the subgrid GLE with that curvature as its outer condition, and refreshes
the height-function contact angle `theta_gle`, which is observed to respond to
the DNS curvature during a run.

Three items remain before production use, all documented in the source:

- build the signed capillary number from the **local** contact-line speed
  (plate speed minus interface speed), not the fixed plate speed used in the
  demonstration;
- reconcile the sign convention of Basilisk's `curvature()` with the GLE's
  $\mathrm{d}\theta/\mathrm{d}s > 0$ toward-the-bath convention before the
  curvature is handed across;
- carry out a grid-convergence study of the coupled system (the curvature
  sample is currently taken in the interfacial cell nearest the plate).

## References

- Snoeijer, J. H. (2006). Free-surface flows with large slopes: beyond
  lubrication theory. *Phys. Fluids* **18**, 021701.
  [doi:10.1063/1.2171190](https://doi.org/10.1063/1.2171190)
- Snoeijer, J. H. & Andreotti, B. (2013). Moving contact lines: scales,
  regimes, and dynamical transitions. *Annu. Rev. Fluid Mech.* **45**, 269–292.
  [doi:10.1146/annurev-fluid-011212-140734](https://doi.org/10.1146/annurev-fluid-011212-140734)
- Delon, G., Fermigier, M., Snoeijer, J. H. & Andreotti, B. (2008).
  Relaxation of a dewetting contact line. *J. Fluid Mech.* **604** (source of
  the experimental series in Fig. 4b).
- Huh, C. & Scriven, L. E. (1971). Hydrodynamic model of steady movement of a
  solid/liquid/fluid contact line. *J. Colloid Interface Sci.* **35**, 85–101
  (the wedge-flow mobility).
- Chan, T. S., Snoeijer, J. H. & Eggers, J. (2013). Theory of the forced
  wetting transition. *Phys. Fluids* **24**, 072104 (two-fluid mobility).
- Kansal, M. *et al.* (2024). *Eur. Phys. J. Spec. Top.*
  [doi:10.1140/epjs/s11734-024-01443-5](https://doi.org/10.1140/epjs/s11734-024-01443-5)
  (linearized-GLE reference data).
- Luo, K. & Gao, P. (2025). *J. Fluid Mech.* — explicit theory with a
  closed-form flux $Q$ (a target for future validation).
- Afkhami, S., Zaleski, S. & Bussmann, M. (2009). A mesh-dependent model for
  applying dynamic contact angles to VOF simulations. *J. Comput. Phys.*
  **228**, 5370–5389
  [doi:10.1016/j.jcp.2009.04.027](https://doi.org/10.1016/j.jcp.2009.04.027).

---

Vatsal Sanjay · CoMPhy Lab, Department of Physics, Durham University ·
`vatsal.sanjay@comphy-lab.org`
