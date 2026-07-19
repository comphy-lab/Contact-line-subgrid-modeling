/**
# gle-shoot.h — the GLE boundary-value problem by shooting

Solves the dip-coating boundary-value problem of
[gle-model.h](gle-model.h) by single shooting on the contact-line curvature.

## Formulation

Inner boundary (contact line, $s = s_0 \sim \lambda$):
$$h(s_0) = h_0, \qquad \theta(s_0) = \theta_e, \qquad
  \omega(s_0) = \omega_0 \; \text{(unknown)} .$$

Outer boundary (bath): integrate until $h = H_{\mathrm{match}} \gg 1$, where
the viscous term has decayed like $3\,\mathrm{Ca}\,M/h$, and require the
trajectory to lie on the static-meniscus manifold,
$$\mathcal{R}(\omega_0) \equiv \omega\big|_{h=H} -
  \sqrt{2\,\bigl(1 - \sin\theta\big|_{h=H}\bigr)} = 0 .$$

One unknown, one condition. The meniscus rise follows from the hydrostatic
balance $\omega = z$ that holds on the static manifold:
$$\Delta = \zeta\big|_{h=H} + \omega\big|_{h=H},$$
and on the lower branch the apparent contact angle is
$\theta_{\mathrm{app}} = \arcsin(1 - \Delta^2/2)$.

This single-shooting formulation is the reason the pseudo-arclength
continuation in [gle-continuation.h](gle-continuation.h) is a $2\times 2$
problem: the branch is parameterised by $(\omega_0, \mathrm{Ca})$ alone. The
collocation-based Python attempts (issues #14, #16) failed precisely because
their extended system dragged the whole adaptive `solve_bvp` mesh into the
continuation unknowns.

The far field amplifies perturbations like $e^{s}$ (the flat-bath fixed point
has a growing mode), so `H_match` should stay moderate: $H \approx 5$ gives a
matching error $\mathcal{O}(3\,\mathrm{Ca}\,M/H) \sim 10^{-3}$ on the residual
while keeping the shooting map well-conditioned in double precision. The same
amplification sets the achievable residual floor ($\sim 10^{-8}$ with the
default integrator tolerances), which is why the Newton tolerance sits at
$5\times10^{-8}$ rather than machine precision.

## Robustness

`gle_shoot()` runs a damped Newton iteration with a finite-difference
derivative; on failure it falls back to geometric bracket expansion around the
initial guess followed by bisection. Trajectories that exit the physical
domain before reaching $H$ return signed penalties chosen so that bisection
still brackets: an interface that collapses ($\theta \to 0$, $h \to 0$)
signals $\omega_0$ too small (negative residual); one that folds over
($\theta \to \pi$) signals $\omega_0$ too large (positive residual).

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#ifndef GLE_SHOOT_H
#define GLE_SHOOT_H

#include "gle-integrate.h"

/**
## Solution record
*/
enum gle_shoot_status {
  GLE_SHOOT_CONVERGED = 0,
  GLE_SHOOT_FAIL = 1
};

typedef struct {
  double omega0;      /* converged contact-line curvature                   */
  double Delta;       /* meniscus rise z_cl (units of l_gamma)              */
  double theta_app;   /* apparent angle from Delta (NAN past the fold)      */
  double theta_min;   /* minimum interface angle along the profile          */
  double theta_end;   /* angle at the matching point                        */
  double omega_end;   /* curvature at the matching point                    */
  double s_end;       /* arc length of the matching point                   */
  double residual;    /* final boundary residual                            */
  int iters;          /* Newton + bisection iterations used                 */
  int status;         /* enum gle_shoot_status                              */
} GLESolution;

/**
### gle_shoot_residual()

Evaluates the outer boundary residual $\mathcal{R}(\omega_0)$ for given
parameters, filling `out` (may be `NULL`) with trajectory observables. The
`sampler` observer, when non-`NULL`, sees every accepted integration step —
used by the drivers to dump interface profiles.

Penalty values ($\pm 10^{3}$) are returned when the trajectory leaves the
domain or exhausts its budget before reaching $H_{\mathrm{match}}$; their sign
encodes which side of the solution $\omega_0$ lies on (see header notes).
*/
typedef struct {
  double theta_min;
  GLESampler user_sampler;
  void *user_ctx;
} gle_shoot_obs;

static void gle_shoot_observer (void *ctx, double s, const double y[4]) {
  gle_shoot_obs *ob = (gle_shoot_obs *) ctx;
  if (y[1] < ob->theta_min)
    ob->theta_min = y[1];
  if (ob->user_sampler)
    ob->user_sampler (ob->user_ctx, s, y);
}

static double gle_shoot_residual (const GLEParams *p, double omega0,
				  GLESolution *out, GLESampler sampler,
				  void *sctx) {
  double s = gle_s0 (p);
  double y[4] = { gle_h0 (p), p->theta_mic, omega0,
		  gle_s0 (p)*cos (p->theta_mic) };
  gle_shoot_obs ob = { p->theta_mic, sampler, sctx };

  int st = gle_integrate (p, &s, y, p->H_match, p->smax_cap,
			  gle_shoot_observer, &ob);

  double R;
  if (st == GLE_OK)
    R = y[2] - gle_static_curvature (y[1]);
  else
    /* domain exit or budget: signed penalty for bracketing */
    R = (y[1] < p->theta_mic ? -1.0e3 : 1.0e3);

  if (out) {
    out->omega0 = omega0;
    out->iters = 0;
    out->theta_min = ob.theta_min;
    out->theta_end = y[1];
    out->omega_end = y[2];
    out->s_end = s;
    out->residual = R;
    if (st == GLE_OK) {
      out->Delta = y[3] + y[2];        /* zeta + hydrostatic curvature */
      double sa = 1.0 - 0.5*out->Delta*out->Delta;
      out->theta_app = (fabs (sa) <= 1.0 ? asin (sa) : NAN);
    }
    else {
      out->Delta = NAN;
      out->theta_app = NAN;
    }
    out->status = (st == GLE_OK ? GLE_SHOOT_CONVERGED : GLE_SHOOT_FAIL);
  }
  return R;
}

/**
### gle_shoot()

Solves $\mathcal{R}(\omega_0) = 0$ starting from `omega0_guess`.

Strategy:

1. damped Newton with central finite differences (up to 60 iterations,
   convergence at $|\mathcal{R}| < 10^{-9}$);
2. on stagnation or penalty contamination, geometric bracket expansion
   around the best iterate, then bisection to machine precision, then one
   final residual evaluation to populate the solution record.

#### Parameters
- `p`: model parameters (`p->Ca` is the operating capillary number).
- `omega0_guess`: initial guess; for $\mathrm{Ca} \to 0$ the static value
  $\omega_0 \approx \sqrt{2(1-\sin\theta_e)}$ is a good start.
- `sol`: output record (required).

#### Returns
`0` on convergence, `1` on failure (`sol->status` mirrors this).
*/
static int gle_shoot (const GLEParams *p, double omega0_guess,
		      GLESolution *sol) {
  const double tolR = 5.0e-8;
  double w = omega0_guess;
  int iters = 0;

  /* --- damped Newton --- */
  double R = gle_shoot_residual (p, w, sol, NULL, NULL);
  for (int it = 0; it < 60 && fabs (R) < 1.0e2; it++) {
    iters++;
    if (fabs (R) < tolR) {
      sol->iters = iters;
      sol->status = GLE_SHOOT_CONVERGED;
      return 0;
    }
    double dw = fmax (1.0e-8*fabs (w), 1.0e-11);
    double Rp = gle_shoot_residual (p, w + dw, NULL, NULL, NULL);
    double Rm = gle_shoot_residual (p, w - dw, NULL, NULL, NULL);
    double dRdw = (Rp - Rm)/(2.0*dw);
    if (dRdw == 0.0 || !isfinite (dRdw))
      break;
    double step = -R/dRdw;
    double cap = 0.25*fmax (fabs (w), 1.0);   /* damping */
    if (fabs (step) > cap)
      step = copysign (cap, step);
    double wn = w + step, Rn = gle_shoot_residual (p, wn, sol, NULL, NULL);
    /* simple backtracking */
    for (int bt = 0; bt < 8 && fabs (Rn) > fabs (R) && fabs (R) < 1e2; bt++) {
      step *= 0.5;
      wn = w + step;
      Rn = gle_shoot_residual (p, wn, sol, NULL, NULL);
      iters++;
    }
    w = wn;
    R = Rn;
  }
  if (fabs (R) < tolR) {
    sol->iters = iters;
    sol->status = GLE_SHOOT_CONVERGED;
    return 0;
  }

  /* --- bracket around the best point seen, then bisect --- */
  double a = omega0_guess, b = omega0_guess;
  double Ra = gle_shoot_residual (p, a, NULL, NULL, NULL), Rb = Ra;
  double span = fmax (1.0e-6*fabs (omega0_guess), 1.0e-6);
  int bracketed = 0;
  for (int it = 0; it < 200 && !bracketed; it++) {
    a = omega0_guess - span;
    b = omega0_guess + span;
    Ra = gle_shoot_residual (p, a, NULL, NULL, NULL);
    Rb = gle_shoot_residual (p, b, NULL, NULL, NULL);
    iters += 2;
    if (isfinite (Ra) && isfinite (Rb) && Ra*Rb < 0.0)
      bracketed = 1;
    else
      span *= 1.6;
  }
  if (!bracketed) {
    sol->iters = iters;
    sol->status = GLE_SHOOT_FAIL;
    return 1;
  }
  for (int it = 0; it < 200; it++) {
    iters++;
    double m = 0.5*(a + b);
    double Rm = gle_shoot_residual (p, m, NULL, NULL, NULL);
    if (Ra*Rm <= 0.0) {
      b = m; Rb = Rm;
    }
    else {
      a = m; Ra = Rm;
    }
    if (fabs (b - a) < 1.0e-14*fmax (fabs (a), 1.0) || fabs (Rm) < tolR)
      break;
  }
  w = 0.5*(a + b);
  R = gle_shoot_residual (p, w, sol, NULL, NULL);
  sol->iters = iters;
  sol->status = (fabs (R) < 1.0e-6 ? GLE_SHOOT_CONVERGED : GLE_SHOOT_FAIL);
  return sol->status == GLE_SHOOT_CONVERGED ? 0 : 1;
}

/**
### gle_solve_ca()

The complementary 1-D solve: finds $\mathrm{Ca}$ such that
$\mathcal{R}(\omega_0, \mathrm{Ca}) = 0$ at **fixed** $\omega_0$. This is the
corrector used near the fold of the bifurcation diagram, where
$\partial\mathcal{R}/\partial\omega_0 = 0$ (that is what a saddle-node *is*)
and solving for $\omega_0$ at fixed $\mathrm{Ca}$ becomes singular — while
$\partial\mathcal{R}/\partial\mathrm{Ca}$ stays finite and the roles can be
swapped.

Strategy mirrors `gle_shoot()`: secant iteration seeded at
`Ca_guess`$\,(1 \pm 10^{-6})$ with relative step clamping, then geometric
bracket expansion (relative in $\mathrm{Ca}$) plus bisection as fallback.
Both stages are derivative-free — the residual is too stiff transverse to the
solution branch for finite-difference Newton to be trustworthy at anything
but the smallest steps.

#### Parameters
- `p`: model parameters; `p->Ca` is overwritten during the search.
- `omega0`: the fixed contact-line curvature.
- `Ca_guess`: starting capillary number ($> 0$).
- `sol`: output record (required).

#### Returns
`0` on convergence (final $\mathrm{Ca}$ left in `p->Ca` and reflected in
`sol`), `1` on failure.
*/
static int gle_solve_ca (GLEParams *p, double omega0, double Ca_guess,
			 GLESolution *sol) {
  const double tolR = 5.0e-8;
  int iters = 0;

  /* --- secant --- */
  double ca0 = Ca_guess, ca1 = Ca_guess*(1.0 + 1.0e-6);
  p->Ca = ca0;
  double R0 = gle_shoot_residual (p, omega0, sol, NULL, NULL);
  p->Ca = ca1;
  double R1 = gle_shoot_residual (p, omega0, sol, NULL, NULL);
  for (int it = 0; it < 30; it++) {
    iters++;
    if (fabs (R1) < tolR && fabs (R1) < 1.0e2) {
      p->Ca = ca1;
      sol->iters = iters;
      sol->status = GLE_SHOOT_CONVERGED;
      return 0;
    }
    if (R1 == R0 || fabs (R1) >= 1.0e2 || fabs (R0) >= 1.0e2)
      break;
    double ca2 = ca1 - R1*(ca1 - ca0)/(R1 - R0);
    double cap = 0.2*ca1;                  /* clamp to 20% relative moves */
    if (fabs (ca2 - ca1) > cap)
      ca2 = ca1 + copysign (cap, ca2 - ca1);
    if (ca2 <= 0.0)
      ca2 = 0.5*ca1;
    ca0 = ca1; R0 = R1;
    ca1 = ca2;
    p->Ca = ca1;
    R1 = gle_shoot_residual (p, omega0, sol, NULL, NULL);
  }

  /* --- bracket + bisection --- */
  double span = 1.0e-6*Ca_guess;
  double a = Ca_guess, b = Ca_guess, Ra = 0.0, Rb = 0.0;
  int bracketed = 0;
  for (int it = 0; it < 100 && !bracketed; it++) {
    a = Ca_guess - span;
    b = Ca_guess + span;
    if (a <= 0.0)
      a = 1.0e-3*Ca_guess;
    p->Ca = a;
    Ra = gle_shoot_residual (p, omega0, NULL, NULL, NULL);
    p->Ca = b;
    Rb = gle_shoot_residual (p, omega0, NULL, NULL, NULL);
    iters += 2;
    if (isfinite (Ra) && isfinite (Rb) && Ra*Rb < 0.0)
      bracketed = 1;
    else
      span *= 1.6;
    if (span > 0.5*Ca_guess && !bracketed)
      break;
  }
  if (!bracketed) {
    sol->iters = iters;
    sol->status = GLE_SHOOT_FAIL;
    return 1;
  }
  double m = Ca_guess, Rm = 1.0;
  for (int it = 0; it < 200; it++) {
    iters++;
    m = 0.5*(a + b);
    p->Ca = m;
    Rm = gle_shoot_residual (p, omega0, NULL, NULL, NULL);
    if (Ra*Rm <= 0.0)
      b = m;
    else {
      a = m; Ra = Rm;
    }
    if (fabs (b - a) < 1.0e-15*Ca_guess || fabs (Rm) < tolR)
      break;
  }
  p->Ca = m;
  double R = gle_shoot_residual (p, omega0, sol, NULL, NULL);
  sol->iters = iters;
  sol->status = (fabs (R) < 1.0e-6 ? GLE_SHOOT_CONVERGED : GLE_SHOOT_FAIL);
  return sol->status == GLE_SHOOT_CONVERGED ? 0 : 1;
}

#endif /* GLE_SHOOT_H */
