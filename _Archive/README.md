# Archive

Superseded code kept for provenance, not for use. The active solver stack
lives in `src-local/gle-*.h`, `gle-ode/`, and `python/`.

## GSL/SUNDIALS-era C solver

`GLE_solver-GSL.c`, `Makefile-GSL` (the original root Makefile), and the
five `gle_*.h` headers (`GLE_solver-GSL.h`, `gle_io.h`, `gle_ode_systems.h`,
`gle_optimization.h`, `gle_physics.h`, `gle_shooting.h`) are the earlier
GSL-dependent shooting-method implementation.

`gle_physics.h` carries a known sign error in the Huh-Scriven mobility
denominator: `f_combined()` computes

```c
double denominator = 3.0 * (mu_r * f1_theta * f2_pi_minus_theta - f1_pi_minus_theta * f2_theta);
```

with a minus sign where a plus is required. This has been flagged on PRs
#9, #10, and #13 and was never fixed in this implementation. The
corrected mobility function lives in `src-local/gle-model.h`.

## Exploratory Python scripts

`GLE_solver_v3.py` (horizontal-plate variant), `GLE_solver_v4.py`
(h-based boundary condition variant), `huh_scriven_velocity.py`
(Huh-Scriven velocity field exploration), and `compare_results.py`
(Python/C output comparison) were exploratory scripts from earlier
iterations of the solver. The maintained Python reference implementation
is `python/GLE_solver.py`.

## Basilisk contact-line header

`contact-fixed.h` is an old patched copy of Basilisk's `contact.h`. Its
fixes — height-preferred normals and a per-step height-field refresh —
are now upstream in Basilisk itself, so this local patched copy is no
longer needed.
