/**
# gle-basilisk.h — the GLE ↔ DNS coupling seam

Connects the subgrid Generalized Lubrication Equation to a Basilisk
two-phase DNS: the GLE resolves the contact-line region **from the slip
length up to the grid size** $\Delta$, and hands the DNS an *apparent*
contact angle at that scale; the DNS resolves everything from $\Delta$ up to
the macroscopic scale and, in return, provides the interface curvature that
the GLE uses as its outer boundary condition. This is the multiscale
strategy of the ARFM 2013 review (Snoeijer & Andreotti) applied as a grid-
scale boundary condition, in the spirit of Afkhami, Zaleski & Bussmann
(J. Comput. Phys. 2009) mesh-dependent angles.

Per DNS time step (an `event (i++)`):

1. measure the interface curvature $\kappa_{\mathrm{DNS}}$ near the contact
   line at the grid scale and convert it to the GLE orientation
   $\omega_{\mathrm{DNS}}$;
2. solve the GLE on $s \in [s_0, s_\Delta]$ with $h(s_\Delta) = \Delta$,
   inner conditions $h(s_0) = h_0$, $\theta(s_0) = \theta_e$, and outer
   condition $\omega(s_\Delta) = \omega_{\mathrm{DNS}}$ (shooting on the
   contact-line curvature $\omega_0$);
3. impose $\theta_{\mathrm{app}} = \theta(s_\Delta)$ through the
   `contact_angle()` height-function boundary condition of
   Basilisk's `contact.h`.

The header is plain C99 — it compiles with `qcc` inside a Basilisk case and
with any host compiler for testing. It deliberately avoids Basilisk
identifiers (`f`, `p`, `h` are not used as global names; see issue #4).

## Usage sketch (inside a Basilisk case)

```c
#include "gle-basilisk.h"

double theta_gle = 90.0*M_PI/180.0;   // updated every step
vector hei[];
hei.t[bottom] = contact_angle (theta_gle);

int main() {
  f.height = hei;                 // required before run()
  run();
}

event gle_boundary (i++) {
  // Prepare gle_case_params once after theta_e, M and the model are known.
  GLEParams gp = gle_case_params;
  gp.Ca = Ca;                    // instantaneous plate/line speed
  double omega = gle_dns_curvature_near_cl (...);   // in GLE orientation
  double th = gle_dns_apparent_angle (&gp, omega, Delta, theta_gle);
  if (isfinite (th))
    theta_gle = th;
}
```

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#ifndef GLE_BASILISK_H
#define GLE_BASILISK_H

#include "gle-shoot.h"
#include "gle-slip-closure.h"

/**
### gle_dns_residual()

Outer-condition residual for the grid-scale problem:
$\mathcal{R}(\omega_0) = \omega\big|_{h=\Delta} - \omega_{\mathrm{DNS}}$,
with the trajectory state at $h = \Delta$ returned through `out` (may be
`NULL`). `omega_dns` must already use the GLE convention: positive
$d\theta/ds$ while marching away from the contact line. Curvature sign
conversion belongs at the DNS call site because it depends on phase and
coordinate orientation.
*/
static inline double gle_dns_residual_prepared (const GLEParams *gp,
						double omega0,
						double Delta_grid,
						double omega_dns,
						double theta_out[1]) {
  double s = gle_s0 (gp);
  double y[4] = { gle_h0 (gp), gp->theta_mic, omega0,
		  gle_s0 (gp)*cos (gp->theta_mic) };
  int st = gle_integrate_prepared (gp, &s, y, Delta_grid, gp->smax_cap,
				   NULL, NULL);
  if (st != GLE_OK)
    return (y[1] < gp->theta_mic ? -1.0e3 : 1.0e3);
  if (theta_out)
    theta_out[0] = y[1];
  return y[2] - omega_dns;
}

static inline double gle_dns_residual (const GLEParams *gp, double omega0,
				       double Delta_grid, double omega_dns,
				       double theta_out[1]) {
  GLEParams prepared;
  GLECutoffResult cutoff;
  if (gle_model_prepare_copy (gp, &prepared, &cutoff) != GLE_CUTOFF_OK)
    return NAN;
  return gle_dns_residual_prepared (&prepared, omega0, Delta_grid, omega_dns,
				    theta_out);
}

/**
### gle_dns_apparent_angle()

Solves the grid-scale GLE boundary-value problem and returns the apparent
contact angle $\theta(h = \Delta)$ to impose on the DNS, or `NAN` on
failure. Newton on $\omega_0$ with bracket/bisection fallback, seeded from
the previous time step's solution through `theta_guess` (the corresponding
$\omega_0$ is re-estimated internally, so any reasonable angle works — pass
the current DNS contact angle on the first call).

#### Parameters
- `gp`: GLE parameters (`Ca`, `slip`, `theta_mic`, model and tolerances;
  `grav` is usually 0 at the grid scale). Calling `gle_model_prepare()` once
  after setting the case inputs exposes its cutoff provenance and catches a
  bad case at startup. This entry point also prepares defensively, so an
  unresolved automatic Chan cutoff can never silently retain the legacy
  default `c_slip`; the caller-owned parameter set is not modified.
- `omega_dns`: interface curvature at the DNS grid scale after conversion to
  the GLE's positive-$d\theta/ds$ orientation (`0` recovers a
  curvature-free inner solution).
- `Delta_grid`: the DNS grid size, in the same length unit as `gp->slip`.
- `theta_guess`: previous apparent angle (seeds the shooting).

#### Returns
The apparent angle in radians, or `NAN` if no converged solution exists
(e.g. beyond the entrainment transition at this `Ca`).
*/
static inline double gle_dns_apparent_angle (const GLEParams *gp,
					     double omega_dns,
					     double Delta_grid,
					     double theta_guess) {
  const double tolR = 1.0e-8;
  if (!gp || !isfinite (omega_dns) || !isfinite (Delta_grid) ||
      !isfinite (theta_guess) || theta_guess <= 0.0 || theta_guess >= M_PI)
    return NAN;
  GLEParams prepared;
  GLECutoffResult cutoff;
  if (gle_model_prepare_copy (gp, &prepared, &cutoff) != GLE_CUTOFF_OK ||
	      Delta_grid <= gle_h0 (&prepared))
    return NAN;
  /* seed omega0: the far-field curvature target plus the wedge estimate
     linking theta_guess to theta_mic across the log region */
  double w = omega_dns
    + 2.0*(theta_guess - prepared.theta_mic)/fmax (Delta_grid, 1.0e-30);
  double th = theta_guess;
  double R = gle_dns_residual_prepared (&prepared, w, Delta_grid, omega_dns,
					&th);
  for (int it = 0; it < 50; it++) {
    if (fabs (R) < tolR)
      return th;
    double dw = fmax (1.0e-8*fabs (w), 1.0e-11);
    double Rp = gle_dns_residual_prepared (&prepared, w + dw, Delta_grid,
					   omega_dns, NULL);
    double Rm = gle_dns_residual_prepared (&prepared, w - dw, Delta_grid,
					   omega_dns, NULL);
    double dRdw = (Rp - Rm)/(2.0*dw);
    if (dRdw == 0.0 || !isfinite (dRdw))
      break;
    double step = -R/dRdw;
    double cap = 0.5*fmax (fabs (w), 1.0);
    if (fabs (step) > cap)
      step = copysign (cap, step);
    w += step;
    R = gle_dns_residual_prepared (&prepared, w, Delta_grid, omega_dns, &th);
  }
  if (fabs (R) < tolR)
    return th;

  /* bracket + bisection fallback */
  double span = fmax (1.0e-3*fabs (w), 1.0), a = w, b = w, Ra = R, Rb = R;
  int bracketed = 0;
  for (int it = 0; it < 120 && !bracketed; it++) {
    a = w - span;
    b = w + span;
    Ra = gle_dns_residual_prepared (&prepared, a, Delta_grid, omega_dns, NULL);
    Rb = gle_dns_residual_prepared (&prepared, b, Delta_grid, omega_dns, NULL);
    if (isfinite (Ra) && isfinite (Rb) && Ra*Rb < 0.0)
      bracketed = 1;
    else
      span *= 1.6;
  }
  if (!bracketed)
    return NAN;
  for (int it = 0; it < 200; it++) {
    double m = 0.5*(a + b);
    double Rm2 = gle_dns_residual_prepared (&prepared, m, Delta_grid,
					    omega_dns, &th);
    if (Ra*Rm2 <= 0.0)
      b = m;
    else {
      a = m; Ra = Rm2;
    }
    if (fabs (Rm2) < tolR || fabs (b - a) < 1.0e-13*fmax (fabs (m), 1.0))
      return (fabs (Rm2) < tolR ? th : NAN);
  }
  return NAN;
}

#endif /* GLE_BASILISK_H */
