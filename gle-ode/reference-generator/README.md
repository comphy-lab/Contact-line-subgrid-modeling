# Constant-slip inner-Stokes reference

This directory contains the two numerical authorities for the finite matching
constant in the Chan *et al.* Generalised Lubrication Equation (GLE): the
Scott--Hocking one-phase integral problem and the constant-slip two-phase
Stokes wedge. They are validation tools, not runtime dependencies:
`src-local/` consumes frozen tables of $Q$, while the generators retain NumPy,
SciPy and scikit-fem as explicit development dependencies.

## Problem solved

Lengths, the phase-A viscosity and the contact-line speed are scaled so that
$\lambda=\mu_A=U=1$. Phase A occupies $0<\phi<\theta_e$, and phase B
occupies $\theta_e<\phi<\pi$, with $M=\mu_B/\mu_A$. Each sector solves

$$
-\nabla p_i + \mu_i\nabla^2\boldsymbol{u}_i=0,
\qquad \nabla\mathbin{\cdot}\boldsymbol{u}_i=0.
$$

The discretisation uses separate conforming P2/P1 Taylor--Hood meshes on a
logarithmic radial grid. On both solid walls it imposes no penetration and the
same Navier condition,

$$
\boldsymbol{u}\mathbin{\cdot}\boldsymbol{n}=0,
\qquad
\boldsymbol{t}\mathbin{\cdot}\boldsymbol{\sigma}\boldsymbol{n}
=-(\mu_i/\lambda)(u_t-U_t).
$$

At the flat interface, Lagrange multipliers impose zero normal velocity in each
phase and a common tangential velocity. Tangential-traction continuity then
follows from the weak problem. No normal-traction continuity condition is
imposed: that stress jump belongs to the subsequent $O(\mathrm{Ca})$
free-surface deformation problem. Independent mean-pressure constraints fix
the two pressure gauges.

The outer arc receives the exact two-phase Huh--Scriven wedge velocity. The
inner arc is set to zero, the regular contact-line limit, and is moved towards
the origin during the sensitivity study.

## Extracting $Q$

The code integrates the Navier traction on both walls at every radial knot. In
the overlap region the cumulative force is

$$
\frac{W(r)}{\mu_A U}
=-\frac{3F(\theta_e,M)}{\sin\theta_e}\log(r/\lambda)+H,
\qquad H=h_1+Mh_2,
$$

where $F<0$ is the signed mobility used by Chan *et al.* The analytic slope is
held fixed when extracting $H$; a free-slope fit is retained as a diagnostic.
The output quantity is

$$
Q=1-\frac{\sin\theta_e}{3F}H,
\qquad
\log c=1+\log(\sin\theta_e)-Q.
$$

The frozen runtime table interpolates **$Q$, not $c$**. The C closure then
reconstructs $c=\sin\theta_e\exp(1-Q)$. Interpolating $Q$ preserves the
additive matching constant and avoids a large dynamic range near
$\theta_e\to\pi$.

Runtime interpolation uses a local cubic Lagrange polynomial on four
neighbouring angle nodes and a local quartic Lagrange polynomial on five
neighbouring $\log_{10}M$ nodes: a 20-node tensor-product stencil. It remains
a small, dependency-free C calculation. The higher order in $\log_{10}M$ is
needed because $Q$ bends sharply near the obtuse-angle, low-$M$ edge; both
bilinear and the 16-node cubic--cubic stencil failed the independent
cell-centre audit and are not used. On the final grid, cubic--cubic gave a
maximum discrepancy of $1.091\times10^{-3}$; cubic--quartic reduces this to
$3.892\times10^{-4}$, with a maximum propagated checkpoint budget of
$1.831\times10^{-3}$.

For $M=0$, `scott_hocking.py` solves the public integral equation of
[Scott (2020)](https://hal.science/hal-03227614) and freezes
`data/scott-hocking-m0-nodes.csv`; `data/scott-hocking-m0.csv` remains the
independent transcription of Scott's published Table 1. The convention
conversion is $Q=1+Q_i$. Only the right-angle value
$Q(\pi/2,0)=1+\gamma_E-\log 2$ is analytic exact. The arbitrary-angle branch is
a numerically converged reference. The two-phase right-angle checks use the
corrected expression reported by
[Luo & Gao (2025)](https://doi.org/10.1017/jfm.2025.10587). The governing GLE
and force convention follow
[Chan *et al.* (2020)](https://doi.org/10.1017/jfm.2020.499).

## Scott--Hocking one-phase generator

The one-phase generator solves Scott's equation (2.1),

$$
1-e^{-\rho}k(\rho)=\int_{-\infty}^{\infty}
L(\rho-\rho')k(\rho')\,\mathrm d\rho',
$$

and evaluates $Q_i=k_\infty^{-1}\int\Gamma\,\mathrm d\rho$ from equation
(2.11). Subtracting $k(\rho)$ inside the convolution removes the logarithmic
kernel singularity. The analytic $K_0$ series in Scott's Appendix B supplies
both exterior tails of the truncated interval. Composite Simpson integration
is applied separately on either side of the known jump in $\Gamma$ at
$\rho=0$.

The frozen nonuniform grid has 70 direct nodes. It is refined around the
turning point of the regularised large-angle function and towards both angle
limits. Every node is compared with coarser and wider-domain calculations. The
maximum recorded sensitivity is $6.618\times10^{-5}$ in $Q_i$. Direct solves at
all 69 interior cell midpoints and at the midpoints of both asymptotic endpoint
intervals give 71 independent checks. Their maximum interpolation discrepancy
is $4.240\times10^{-5}$ and the largest discrepancy-plus-solve-sensitivity
budget is $1.793\times10^{-4}$. The generated values reproduce Scott's
published Table 1 to $3.106\times10^{-6}$.

Runtime interpolation is performed on the regular quantities

$$
Q_i-\log(\alpha/3),\qquad
Q_i-\frac{\pi}{\pi-\alpha},
$$

with Scott's endpoint asymptotics. These asymptotic endpoint intervals are
numerically checked but are not labelled analytic exact.

## Numerical convergence checks

For every requested node, `table` compares the reported $Q$ with four
independent sensitivity solves:

- a mesh with 75% of the logarithmic radial resolution;
- an inner radius divided by four, retaining the logarithmic resolution;
- an outer radius multiplied by four, retaining the logarithmic resolution;
- the same fit window shifted by one radial decade.

The maximum of those four changes and the within-window drift is recorded as
`estimated_error_Q`. This is a convergence **sensitivity estimate**, not a
rigorous a posteriori error bound. The command also gates the fitted Cox slope,
the normwise backward error of every KKT solve, and the relative residual of
the interface and pressure-gauge constraints. It refuses to write if any row
fails unless `--allow-unconverged` is deliberately supplied; such a manifest
cannot subsequently be frozen.

The production tolerances are $10^{-3}$ in the $Q$ sensitivity estimate,
$2\times10^{-3}$ in relative slope error, and $10^{-10}$ for both KKT residual
measures. Since $\delta\log c=-\delta Q$, a small error in $Q$ transfers
directly to the relative error in $c$ to leading order.

## Production workflow

`uv` reads the pinned PEP 723 environments directly from the generator
scripts. The pinned NumPy/SciPy stack supports Python 3.9--3.12; the upper
bound prevents `uv` from attempting unsupported source builds on Python 3.13.
The one-phase reference can be regenerated and verified independently of the
finite-$M$ FEM evidence:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  uv run gle-ode/reference-generator/scott_hocking.py generate

uv run gle-ode/reference-generator/scott_hocking.py verify
```

This writes the node CSV, independent checkpoint CSV, manifest and
`src-local/gle-slip-scott-data.h`. For a single integral-equation solve:

```bash
uv run gle-ode/reference-generator/scott_hocking.py solve \
  --theta-rad 1.0 --spacing 0.05 --radius 40
```

The two-phase public anchors and a single-case diagnostic are:

```bash
uv run gle-ode/reference-generator/generate.py anchors

uv run gle-ode/reference-generator/generate.py solve \
  --theta-deg 60 --viscosity-ratio 0.1 --json
```

Generate the stored half-plane table on the adaptively refined 27 by 10 grid:

```bash
uv run gle-ode/reference-generator/generate.py table \
  --theta-deg-grid "30,32.5,35:145:5,147.5,150" \
  --log10-M-grid=-2,-1.875,-1.75,-1.5:0:0.25 \
  --inner-radius 1e-8 --outer-radius 1e8 --radial-cells 480 \
  --fit-lower 1e5 --fit-upper 1e7 \
  --tolerance-Q 1e-3 --tolerance-slope 2e-3 --tolerance-linear 1e-10 \
  --output gle-ode/reference-generator/data/two-phase-q-nodes.csv
```

Generate an independent FEM solve at the centre of every table cell:

```bash
uv run gle-ode/reference-generator/generate.py table \
  --theta-deg-grid "31.25,33.75,37.5:142.5:5,146.25,148.75" \
  --log10-M-grid=-1.9375,-1.8125,-1.625:-0.125:0.25 \
  --inner-radius 1e-8 --outer-radius 1e8 --radial-cells 480 \
  --fit-lower 1e5 --fit-upper 1e7 \
  --tolerance-Q 1e-3 --tolerance-slope 2e-3 --tolerance-linear 1e-10 \
  --output gle-ode/reference-generator/data/two-phase-q-checkpoints.csv
```

Freeze the nodes, check all cell centres, and emit the dependency-free C data:

```bash
python3 gle-ode/reference-generator/freeze_table.py merge \
  --shards gle-ode/reference-generator/data/two-phase-q-nodes.csv \
  --output gle-ode/reference-generator/data/two-phase-q.csv

python3 gle-ode/reference-generator/freeze_table.py check \
  --table gle-ode/reference-generator/data/two-phase-q.csv \
  --checkpoints gle-ode/reference-generator/data/two-phase-q-checkpoints.csv \
  --tolerance-observed-Q 1e-3 --tolerance-error-budget-Q 3e-3 \
  --tolerance-right-angle-Q 1e-3 \
  --tolerance-symmetry-Q 1e-3 \
  --output gle-ode/reference-generator/data/two-phase-q-interpolation-audit.csv

python3 gle-ode/reference-generator/freeze_table.py emit-c \
  --table gle-ode/reference-generator/data/two-phase-q.csv \
  --output src-local/gle-slip-table-data.h
```

The production node grid has 27 angle values and 10 stored viscosity values
(270 nodes). The checkpoint grid has 26 by 9 values (234 cell centres).
Independent node shards may replace the single `--shards` input above, but
each shard must have the same generator source hash, method, mesh, fit window,
tolerances and convergence checks. Their union must be a complete rectangular
grid unless an exact target grid is selected. Merge split cell-centre runs in
the same way with the explicit checkpoint mode, for example:

```bash
python3 gle-ode/reference-generator/freeze_table.py merge \
  --kind checkpoints \
  --shards path/to/checkpoint-shards/*.csv \
  --output gle-ode/reference-generator/data/two-phase-q-checkpoints.csv
```

Adaptive refinements may overlap a larger, previously frozen dataset without
forming a rectangular union. In that case, select the exact Cartesian output
grid after merging the inputs:

```bash
python3 gle-ode/reference-generator/freeze_table.py merge \
  --kind checkpoints \
  --shards path/to/original.csv path/to/refinement-shards/*.csv \
  --theta-grid "31.25,33.75,37.5:142.5:5,146.25,148.75" \
  --log10-M-grid=-1.9375,-1.8125,-1.625:-0.125:0.25 \
  --output gle-ode/reference-generator/data/two-phase-q-checkpoints.csv
```

Comma-separated values and inclusive `start:stop:step` ranges can be mixed.
Every input CSV and manifest is hash-, convergence- and configuration-checked
before any row is discarded. The selected coordinates must all exist, and the
result must be a complete rectangular grid; missing points or duplicate input
nodes stop the merge. The output manifest records the selected grid, row counts
and each validated input hash.

At each checkpoint, the audit records

$$
|Q_{\mathrm{cubic}\times\mathrm{quartic}}-Q_{\mathrm{FEM}}|
+\epsilon_{Q,\mathrm{checkpoint}}
+\sum_{k\in\mathcal S} |w_k|\epsilon_{Q,k},
$$

where $\mathcal S$ is the 20-node tensor stencil. The absolute weights give a
conservative linear propagation of the node sensitivity estimates. This is an
empirical cell-centre interpolation discrepancy augmented by the node and
checkpoint sensitivity estimates. It is **not** a mathematical global bound
on the interpolation error everywhere inside each cell. In particular,
checking the centre does not rule out a larger off-centre discrepancy.

The audit separately checks the numerical $M=1$ column against
$Q(\theta,1)=Q(\pi-\theta,1)$. The emitted C header averages each reflected
pair in that column, imposing the exact identity while changing the raw FEM
values only within their recorded convergence sensitivity. The CSV retains
the unmodified solves.

## Provenance and stale-evidence rejection

Each generated row and manifest records the SHA-256 of the exact
`generate.py` source. Each manifest also records the SHA-256 of its CSV.
`freeze_table.py merge` rejects a non-converged manifest, a source-hash
mismatch, altered CSV contents, or incompatible numerical settings. The
interpolation audit records the hashes of the frozen table, independent
checkpoint CSV and audit CSV. `emit-c` verifies all three immediately before
writing the header, so changing any evidence file makes the validation stale
and blocks emission. Regenerate or rerun the affected stage; do not patch a
manifest by hand.

The Scott--Hocking evidence follows the same rule on a separate hash path:
its manifest binds the exact `scott_hocking.py`, node CSV and independent
checkpoint CSV, and the generated C header embeds all three hashes. This
separation means regenerating one-phase evidence does not change the frozen
two-phase FEM provenance.

## Runtime domain and limits

The frozen table stores
$30^\circ\leq\theta_e\leq150^\circ$ and
$-2\leq\log_{10}M\leq0$. The exact phase-exchange identity

$$
Q(\theta_e,M)=Q(\pi-\theta_e,1/M)
$$

extends the finite-viscosity runtime range to $10^{-2}\leq M\leq10^2$ without
extrapolation. The $M=0$ limit remains the separate Scott--Hocking branch.
Cases outside the frozen table domain use the explicitly labelled
Luo--Gao approximation only when that fallback policy is selected.

The FEM solver itself rejects angles below $3^\circ$ and above $177^\circ$;
the production table is intentionally narrower because the increasingly
slender sectors require a separate asymptotic mesh study.

Run the deterministic freeze tests and the coarse FEM tests with:

```bash
python3 gle-ode/reference-generator/tests/test_scott_reference.py
uv run gle-ode/reference-generator/tests/test_scott_solver.py
python3 gle-ode/reference-generator/tests/test_freeze_table.py
uv run gle-ode/reference-generator/tests/test_generator.py
```
