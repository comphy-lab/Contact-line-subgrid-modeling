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

![Reproduction of Fig. 4b: meniscus rise versus capillary number for the
legacy, reference-matched Chan, and direct Luo–Gao models](img/fig4b-reproduction.png)

Steady contact-line height $z/\ell_\gamma$ versus capillary number for a plate
withdrawn from a silicone-oil bath. The dashed black line retains the legacy
Chan branch with $c_\lambda=3$; blue uses the Scott--Hocking one-phase closure
in the Chan GLE; red uses the direct Luo--Gao GLE. The review's digitised
lubrication theory is thick grey and the steady Delon et al. (2008) data are
coloured symbols. Only the symbols are digitised: the thin coloured traces in
the original review figure are transient elevations mapped through
$\widetilde{\mathrm{Ca}}(t) = \eta(U_p - \mathrm{d}z_{cl}/\mathrm{d}t)/\gamma$.

The legacy effective slip was calibrated to the digitised fold at its reported
resolution. The Chan--Scott--Hocking and direct Luo--Gao slips were then fitted
to that legacy branch's computed fold,
$\mathrm{Ca}^{*}=1.0544249\times10^{-2}$. Fold agreement is therefore circular
and is **not** a validation. At $\theta_e=53.46^\circ$ and $M=0$, the fitted
values are $\lambda/\ell_\gamma=7.46\times10^{-6}$ (legacy),
$8.8078\times10^{-6}$ (Chan--Scott--Hocking), and $8.48805\times10^{-6}$
(direct Luo--Gao). The out-of-sample fold heights are respectively 1.43908,
1.43907 and 1.43919, versus 1.446 digitised; the full branch shape is the other
check. The reproduction command now resamples the external digitised theory
curve and the generated branch onto 421 uniformly spaced heights: the legacy
branch has maximum $|\Delta\mathrm{Ca}|=2.012\times10^{-4}$ and RMS
$1.044\times10^{-4}$, within fixed budgets, and every generated branch must
reach its requested upper-branch height. None of these fitted effective lengths
is an independently measured Navier slip length. Digitisation and calibration
provenance are recorded in
[data/fig4b-digitized/CALIBRATION.md](data/fig4b-digitized/CALIBRATION.md).

![Chan and direct Luo–Gao branches at the same microscopic slip
length](img/fig4b-model-comparison.png)

The separate equal-input comparison fixes
$\lambda/\ell_\gamma=7.46\times10^{-6}$ in both equations and performs no
refit. With the implemented finite start $s_0=\lambda$, the direct Luo--Gao
fold is 0.41690% above the reference-matched Chan fold. This number is not
cutoff-independent: matched sweeps at $s_0/\lambda=0.1$, 0.01 and 0.001 give
0.6289%, 0.6819% and 0.6903%, respectively. The robust conclusion is that the
model-form shift is sub-percent for this case. The comparison includes both
models' matching constants, different local viscous terms and their common
finite-start convention; no finite-$s_0$ inner correction is applied.

The full methods document — GLE derivation, the two solvers, the calibration,
and the component-by-component validation — is
[docs/gle-theory-and-implementation.pdf](docs/gle-theory-and-implementation.pdf)
(source [docs/gle-theory-and-implementation.tex](docs/gle-theory-and-implementation.tex)).

## The model

State vector $y = (h, \theta, \omega, \zeta)$ parameterised by arc length $s$
along the liquid–gas interface, measured from the contact line, in the
formulation of Snoeijer (2006):

$$
\frac{\mathrm{d}h}{\mathrm{d}s} = \sin\theta, \qquad
\frac{\mathrm{d}\theta}{\mathrm{d}s} = \omega, \qquad
\frac{\mathrm{d}\omega}{\mathrm{d}s}
   = \frac{3\,\mathrm{Ca}\;M(\theta,\mu_r)}
           {h\,(h + c_\lambda\lambda)} + G(\theta),
\qquad
\frac{\mathrm{d}\zeta}{\mathrm{d}s} = \cos\theta .
$$

Here $h$ is the film thickness, $\theta$ the local interface inclination,
$\omega$ the curvature, $\lambda$ the Navier slip length,
$c_\lambda=c(\theta_e,\mu_r)$ the finite-angle microscopic cutoff coefficient,
and $G(\theta)$ the gravity term. This is the Chan *et al.* form of the GLE.
For that model $c_\lambda$ is resolved once from the case inputs
$(\theta_e,\mu_r)$ and is constant along the trajectory; $c_\lambda=3$ is
retained only as the explicit legacy small-angle, one-phase convention. A
second runtime model evaluates the slip-resolved Luo--Gao GLE directly and
does not use $c_\lambda$. Lengths are non-dimensionalised by the capillary
length $\ell_\gamma = \sqrt{\gamma/\rho g}$. $\mathrm{Ca} = \eta U/\gamma > 0$ is a
**receding** contact line (dip-coating: plate withdrawn from the bath);
$\mathrm{Ca} < 0$ is advancing.

The mobility $M(\theta,\mu_r)$ is the two-fluid Huh–Scriven wedge factor (Chan,
Snoeijer & Eggers 2012),

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

### Microscopic closure and the alternative GLE

The Chan cutoff is represented by the Cox matching constant

$$
Q = \ln\!\left(\frac{\sin\theta_e}{c_\lambda}\right)+1,
\qquad
c_\lambda=\sin\theta_e\,\exp(1-Q).
$$

The numerical authorities for $Q$ are open-source generators in
[`gle-ode/reference-generator`](gle-ode/reference-generator). One solves the
Scott--Hocking $\mu_r=0$ singular integral problem; the other solves the
two-phase constant-Navier-slip Stokes wedge. Both check convergence and freeze
their results before dependency-free C data are emitted. These Python
dependencies belong to table generation, not to the solver used by Basilisk.

The dependency-free runtime consumes that frozen table and records which
closure supplied each result. Its automatic order is: the Scott--Hocking
$\mu_r=0$ reference branch; the corrected right-angle branch; interpolation
of $Q$ in the convergence-checked table; and Luo--Gao's explicit $Q$ as a
clearly marked approximate fallback. The Scott--Hocking branch consumes 70
direct integral-equation nodes and the analytic Hocking right-angle anchor.
Its 71 independent off-node checks give a maximum interpolation discrepancy of
$4.25\times10^{-5}$ in $Q_i$; arbitrary angles are therefore a numerically
converged reference, not an analytic exact formula. Interpolation is performed
in $Q$, not in $c_\lambda$.
The corrected right-angle values include
$c(\pi/2,0)=1.1229189671$, $c(\pi/2,0.02)=1.20925446$,
$c(\pi/2,1)=2.05206963$ and $c(\pi/2,10)=1.47616934$; the branch obeys
$c(\pi/2,\mu_r)=c(\pi/2,1/\mu_r)$. The finite-ratio values are regression
anchors computed with the published four-significant-figure constant
$h_b=-1.539$; their displayed tail digits are not extra physical precision.

The direct Luo--Gao model is not the Chan equation with an explicit value of
$c_\lambda$. It applies Luo and Gao's local slippery-wedge approximation: the
rational flux coefficients depend on the evolving angle $\theta(s)$. Both
models recover the same no-slip Huh--Scriven mobility for $h\gg\lambda$, but
they differ within the slip-to-mesoscale region. At $\mu_r=0$, Luo--Gao's
matched value
$c_{\mathrm{LG}}=2\sin^3\theta_e/f_2(\theta_e)$ can be used as the approximate
fallback in the Chan equation; the resulting constant-$c$ trajectory is still
not the direct Luo--Gao trajectory because the latter updates this factor with
the local angle. The published two-phase Luo--Gao expression has the wedge
domain $0<\theta<\pi$; only its $\mu_r=0$ reduction has an exact even extension
through the small negative angles of an oscillatory film tail. The Fig. 4b
direct-model comparison is therefore made in that one-phase limit.

Use Chan with Scott--Hocking, the right-angle branch, or the numerical table
when reference inner-Stokes matching covers the case. Use direct Luo--Gao when
an explicit local closure is preferred or no table value is available, while
retaining its approximation status and, at finite $\mu_r$, its
$0<\theta<\pi$ domain restriction.

![Finite-viscosity comparison of Luo--Gao matching against the numerical
inner-Stokes reference](img/finite-m-closure-comparison.png)

At the 270 certified table nodes, $Q_{\mathrm{LG}}-Q_{\mathrm{FEM}}$ ranges
from $-0.95525$ to $-0.00710$ (RMS $0.22665$), corresponding to
$c_{\lambda,\mathrm{LG}}/c_{\lambda,\mathrm{FEM}}$ between $1.00713$ and
$2.59932$. The largest mismatch is at $\theta_e=150^\circ$, $M=10^{-2}$.
Each square is an actual converged FEM node on the deliberately nonuniform
grid; the plot performs no interpolation between them. This comparison is at
the matching-constant level. The integrated Fig. 4b comparison above is the
separate one-phase GLE test.

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
| Inner-Stokes $Q$ generator | 270 converged nodes; max node sensitivity $5.65\times10^{-4}$ |
| Frozen $Q$ interpolation | 234 independent cell centres; max $|\Delta Q|=3.90\times10^{-4}$, propagated budget $1.84\times10^{-3}$ |
| Corrected right-angle cutoff | one-phase analytic anchor and viscosity-inversion symmetry |
| Interface angle $\theta(s)$, well-conditioned window | max rel. error $2.74\times10^{-5}$; limit $2.3\times10^{-4}$ |
| Interface thickness $h(s)$, well-conditioned window | max rel. error $3.96\times10^{-5}$; limit $5\times10^{-4}$ |
| Apparent-angle difference at $s = 10^4$ (`--quick`) | $5.30\times10^{-4}$ deg; limit $10^{-3}$ deg |
| Gravity-rescaling invariance ($g^{*}$ rescaled) | $4.5\times10^{-6}$ |
| Static ($\mathrm{Ca}\to 0$) limit | finite-$s_0$ value $0.626995031426$ versus contact-line value $0.626990589862$; offset $s_0\cos\theta_e=4.44\times10^{-6}$ |
| Fig. 4b external theory curve | max $|\Delta\mathrm{Ca}|=2.012\times10^{-4}$; RMS $1.044\times10^{-4}$ over 421 uniform heights |

The Fig. 4b study separates calibration from comparison. The legacy effective
$\lambda$ is fitted to the digitised fold; each new model is fitted to the
computed legacy fold. Fold agreement is circular, so only the fold height and
branch shape are out-of-sample checks. A second comparison holds the input
$\lambda$, $\theta_e$ and $\mu_r$ fixed between Chan-with-reference-$Q$ and
direct Luo--Gao. That equal-$\lambda$ comparison exposes the full formulation
difference which an independent recalibration would otherwise hide.

The far-field angle is a soft condition (its gain runs to $\sim 5\times10^7$),
so agreement is excellent where the problem is well conditioned
($s \lesssim 10^4$) and degrades monotonically towards the cap. That is a
property of the boundary-value problem, not of either implementation: the two
right-hand sides agree to machine precision.

## Quick start

```bash
make                 # build the standalone C solvers (any C99 compiler + libm)
make test            # regressions; runs the bounded Basilisk seam test when qcc is present
./reproduce-fig4b.sh # trace the branch and regenerate img/fig4b-reproduction.png
```

Single solve at one capillary number:

```bash
cd gle-ode && ./gle-solve fig4b.params Ca=5e-3
```

Select the case-level Chan closure or the direct Luo--Gao model at runtime:

```bash
cd gle-ode
./gle-solve fig4b.params c_method=auto
./gle-solve fig4b.params gle_model=luo_gao
```

Evaluate the Chan matching constant for a prescribed microscopic angle and
viscosity ratio without solving a GLE profile:

```bash
cd gle-ode
./gle-cutoff theta_mic_deg=60 mu_r=0.1 c_method=auto
```

The output names the resolved authority (`scott_hocking`,
`corrected_right_angle`, `reference_table`, or the explicitly approximate
`luo_gao_approx`) together with $Q$, $\log c$ and $c$. The separate
`luo_gao_approximation` flag says only whether that fallback supplied the
value; the named method carries the reference or interpolation provenance.

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
SUNDIALS**. Basilisk `qcc` is needed only for the DNS case. `uv`/Python is
confined to development-time plots, cross-validation, and the open-source inner-Stokes
reference generator; none enters the production C solver.

## Repository structure

```
Contact-line-subgrid-modeling
├── src-local - header-only C99 solver stack (no external dependencies)
│   ├── gle-model.h - common parameters, corrected Huh–Scriven mobility and model dispatcher
│   ├── gle-model-chan.h - Chan constant-cutoff GLE
│   ├── gle-model-luo-gao.h - direct Luo–Gao slippery-wedge GLE
│   ├── gle-slip-closure.h - named case-level Q and c closure policy
│   ├── gle-slip-reference.h - Scott–Hocking and frozen-table runtime data
│   ├── gle-slip-table-data.h - generated dependency-free Q table
│   ├── gle-integrate.h - adaptive Cash–Karp RK5(4) integrator with thickness-event stopping
│   ├── gle-shoot.h - single-shooting BVP solve on the contact-line curvature
│   ├── gle-collocate.h - fixed-mesh implicit-midpoint collocation; fold-free branch tracer
│   ├── gle-continuation.h - legacy shooting-based arclength continuation (lower branch only)
│   ├── gle-basilisk.h - the GLE ↔ DNS coupling seam
│   └── gle-params.h - key=value runtime parameter loader with CLI overrides
├── gle-ode - standalone drivers and parameter files
│   ├── gle-cutoff.c - evaluates c(theta_e,M) without solving a GLE profile
│   ├── gle-solve.c - single GLE solve, writes the interface profile
│   ├── gle-continuation.c - traces the dip-coating branch through the fold
│   ├── reference-generator - open-source two-phase Stokes-wedge Q generator
│   ├── fig4b.params - calibrated parameters reproducing the Fig. 4b theory curve
│   └── Makefile - builds all three drivers with cc -lm
├── simulationCases - Basilisk two-phase DNS cases
│   ├── contactline-gle.c - plate coating with the per-timestep GLE subgrid coupling
│   └── contactline.c - baseline case with a fixed contact angle (no GLE coupling)
├── postProcess - uv-runnable plotting and validation
│   ├── plot-fig4b.py - overlays the traced branch on the digitized reference
│   ├── plot-finite-m-closure-comparison.py - compares finite-M FEM and Luo–Gao matching
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
│   ├── fig4b-model-comparison.png - equal-slip Chan/Luo–Gao comparison
│   ├── finite-m-closure-comparison.png - finite-M FEM/Luo–Gao matching comparison
│   └── c-vs-python.png - the C-vs-Python cross-validation figure
├── docs - LaTeX methods document
│   ├── gle-theory-and-implementation.tex - theory, numerics, and validation write-up
│   └── gle-theory-and-implementation.pdf - the compiled methods document
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
converts its sign to the GLE orientation, solves the subgrid GLE at the actual
local cell size, and refreshes the associated height-function contact angle.
The signed capillary number uses the plate speed relative to the measured
contact-line speed, while a dumped scalar preserves the dynamic angle across
restart. The remaining production qualification is a grid-convergence study
of the coupled curvature and contact-line-position samples.

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
- Chan, T. S., Snoeijer, J. H. & Eggers, J. (2012). Theory of the forced
  wetting transition. *Phys. Fluids* **24**, 072104 (two-fluid mobility).
- Chan, T. S., Kamal, C., Snoeijer, J. H., Sprittles, J. E. & Eggers, J.
  (2020). Cox--Voinov theory with slip. *J. Fluid Mech.* **900**, A8.
  [doi:10.1017/jfm.2020.499](https://doi.org/10.1017/jfm.2020.499)
- Hocking, L. M. (1977). A moving fluid interface. Part 2. The removal of the
  force singularity by a slip flow. *J. Fluid Mech.* **79**, 209--229.
  [doi:10.1017/S0022112077000123](https://doi.org/10.1017/S0022112077000123)
- Scott, J. F. (2020). Calculation of a key function in the asymptotic
  description of moving contact lines. *Q. J. Mech. Appl. Math.* **73**,
  279--291.
  [doi:10.1093/qjmam/hbaa012](https://doi.org/10.1093/qjmam/hbaa012)
- Kansal, M. *et al.* (2024). *Eur. Phys. J. Spec. Top.*
  [doi:10.1140/epjs/s11734-024-01443-5](https://doi.org/10.1140/epjs/s11734-024-01443-5)
  (linearized-GLE reference data).
- Luo, J. & Gao, P. (2025). Explicit theory of moving contact lines.
  *J. Fluid Mech.* **1019**, A52.
  [doi:10.1017/jfm.2025.10587](https://doi.org/10.1017/jfm.2025.10587)
- Afkhami, S., Zaleski, S. & Bussmann, M. (2009). A mesh-dependent model for
  applying dynamic contact angles to VOF simulations. *J. Comput. Phys.*
  **228**, 5370–5389
  [doi:10.1016/j.jcp.2009.04.027](https://doi.org/10.1016/j.jcp.2009.04.027).

---

Vatsal Sanjay · CoMPhy Lab, Department of Physics, Durham University ·
`vatsal.sanjay@comphy-lab.org`
