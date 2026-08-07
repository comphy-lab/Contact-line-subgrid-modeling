# AGENTS.md

Authoritative agent guidance for `comphy-lab/Contact-line-subgrid-modeling`: a
multiscale contact-line solver where a Generalized Lubrication Equation (GLE)
subgrid model resolves the interface from the slip length to the DNS grid scale
and couples to a Basilisk two-phase DNS above it.

> The `CLAUDE.md` shim in this repo is a local, gitignored `@AGENTS.md`
> pointer. **Do not commit `CLAUDE.md`.** Edit this file instead.

## Layout

- `src-local/` — header-only C99 solver stack, no external dependencies. Physics
  in `gle-model.h`; adaptive Cash–Karp integrator in `gle-integrate.h`; shooting
  BVP in `gle-shoot.h`; fold-free collocation branch tracer in
  `gle-collocate.h`; legacy shooting continuation in `gle-continuation.h`; the
  GLE ↔ DNS seam in `gle-basilisk.h`; the `key=value` loader in `gle-params.h`.
- `gle-ode/` — standalone drivers (`gle-cutoff.c`, `gle-solve.c`,
  `gle-continuation.c`), the calibrated `fig4b.params`, and a self-contained
  `Makefile`; `reference-generator/` contains the open-source Scott--Hocking
  one-phase integral solver and two-phase inner-Stokes FEM generator, with
  their convergence evidence.
- `simulationCases/` — Basilisk DNS cases; `contactline-gle.c` is the
  GLE-coupled case, `contactline.c` the fixed-angle baseline.
- `postProcess/` — `uv`-runnable plotting (`plot-fig4b.py`,
  `plot-finite-m-closure-comparison.py`) and cross-validation
  (`compare-c-python.py`); each script carries inline PEP 723 dependencies.
- `python/` — the historical `solve_bvp` reference (`GLE_solver.py`) and the
  linearized-GLE fixture (`validate-linearized-gle.py` + `reference-data/`).
- `data/fig4b-digitized/` — vector-digitized reference data. **Do not modify**;
  the calibration is documented in `CALIBRATION.md`.
- `_Archive/` — superseded code kept for provenance only. **Do not resurrect.**
  The old C mobility (`gle_physics.h`) carries a sign error in the Huh–Scriven
  denominator (flagged on PRs #9/#10/#13, never fixed there).

## Build and test

```bash
make                 # build gle-cutoff, gle-solve and gle-continuation
make test            # smoke solve; compiles the Basilisk case when qcc exists
sh tests/verify-reference-table.sh # independently check frozen table evidence
./reproduce-fig4b.sh # trace the branch and regenerate the Fig. 4b figure
# Basilisk DNS case, serial:
cd simulationCases && qcc -O2 -disable-dimensions -I../src-local \
    contactline-gle.c -o run -lm
```

## Conventions

- Documentation is literate Markdown inside `/** ... */` docstrings, rendered by
  the `.github` docs pipeline; LaTeX via MathJax (`$...$`, `$$...$$`).
- British English, mechanism-first, concise, zero hype.
- No `f`, `p`, or `h` as global identifiers — they collide with Basilisk's VoF
  field, pressure, and height function (issue #4). Solver functions are static
  inline in headers.
- **No external dependencies in `src-local/`.** Adding GSL, SUNDIALS, or CMake is
  off the table — that path was tried and closed (PRs #9/#3). Any C99 compiler
  with `libm`, or Basilisk's `qcc`, must build the whole stack.
- Parameter files are `key = value` lines with `#` comments; a bare CLI argument
  is a parameter file, and `key=value` CLI arguments override, left to right.
  Angles are given as `theta_mic_deg` (degrees) in parameter files.

## Validation gates (any physics change)

Run all of the following before committing a change that touches the model or
either solver:

1. `make test` — smoke solve plus the `qcc` compile check.
2. `sh tests/verify-reference-table.sh` — the frozen one- and two-phase
   reference data, independent checkpoints, interpolation audit, and emitted C
   headers must reproduce from their public generators.
3. `./reproduce-fig4b.sh` — every branch must reach its requested upper height;
   the legacy fold must lie within the digitisation uncertainty of
   $\mathrm{Ca}^{*} = 1.053717\times10^{-2}$ and $\Delta^{*}=1.44617$; and the
   421-height external theory-curve comparison must satisfy maximum
   $|\Delta\mathrm{Ca}|<2.2\times10^{-4}$ and RMS $<1.1\times10^{-4}$.
4. `uv run postProcess/compare-c-python.py --quick` — C vs Python mobility to
   machine precision; maximum angle and thickness errors below
   $2.3\times10^{-4}$ and $5\times10^{-4}$ in the well-conditioned window;
   apparent-angle difference below $10^{-3}$ degrees.
5. Gravity-rescaling invariance:
   ```bash
   cd gle-ode
   ./gle-continuation fig4b.params slip=3.73e-6 grav=4.0 Delta_max=1.8
   ```
   `fold_Ca` must stay invariant to $\sim 10^{-3}$; `fold_Delta` halves to
   $\sim 0.720$ (both lengths rescale with $\sqrt{g^{*}}$).

## Commits

Imperative mood, ~50-character subject, focused body explaining what and why.
No AI signatures, no co-authored-by tags, no tool advertisements.
