/**
# gle-integrate.h — adaptive Runge–Kutta integrator for the GLE

A self-contained embedded Runge–Kutta–Cash–Karp 5(4) integrator with
proportional step control, specialised to the 4-component GLE state of
[gle-model.h](gle-model.h). No external dependencies (no GSL, no SUNDIALS):
the whole solver stack compiles with `cc -lm` or with Basilisk's `qcc`.

Why adaptive: the GLE spans the slip scale $\lambda \sim 10^{-6}\,\ell_\gamma$
to the capillary scale $\ell_\gamma$ in a single trajectory — six decades of
arc length. Near the contact line the curvature gradient behaves like
$\mathrm{d}\omega/\mathrm{d}s \sim 1/s$, which a fixed-step method cannot
afford; the embedded 5(4) pair keeps the local error controlled while the step
grows geometrically away from the contact line.

Integration terminates on a **thickness event** $h = h_{\mathrm{target}}$,
localised to machine precision by bisection on the final step size; this is
how the outer matching point $h = H_{\mathrm{match}}$ of the dip-coating
problem is hit exactly.

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#ifndef GLE_INTEGRATE_H
#define GLE_INTEGRATE_H

#include "gle-model.h"

/**
## Return codes
*/
enum gle_integrate_status {
  GLE_OK = 0,               /* reached h = h_target                          */
  GLE_ERR_DOMAIN = 1,       /* state left physical domain (theta -> 0 or pi,
			       or h -> 0): no steady solution this way       */
  GLE_ERR_SMAX = 2,         /* exceeded arc-length cap without the event     */
  GLE_ERR_STEPS = 3,        /* exhausted the step budget                     */
  GLE_ERR_STEPSIZE = 4      /* step size underflow                           */
};

/**
### GLESampler

Optional per-step observer: called once per *accepted* step with the current
arc length and state, e.g. to record interface profiles. Pass `NULL` to
disable.
*/
typedef void (*GLESampler) (void *ctx, double s, const double y[4]);

/**
### gle_rkck_step()

One embedded Cash–Karp step of size `ds` from state `y` (arc position `s`).
Writes the 5th-order solution into `y5[4]` and returns the scalar error
estimate normalised by the mixed tolerance
$\mathrm{tol}_i = \mathrm{atol} + \mathrm{rtol}\,\max(|y_i|, |y^5_i|)$
(RMS over components; a value $\le 1$ means the step is acceptable).

#### Returns
The error norm, or a negative value if the right-hand side reported a domain
violation at any stage point.
*/
static inline double gle_rkck_step (const GLEParams *p, const double y[4],
				    double ds, double y5[4]) {
  static const double
    b21 = 1./5.,
    b31 = 3./40., b32 = 9./40.,
    b41 = 3./10., b42 = -9./10., b43 = 6./5.,
    b51 = -11./54., b52 = 5./2., b53 = -70./27., b54 = 35./27.,
    b61 = 1631./55296., b62 = 175./512., b63 = 575./13824.,
    b64 = 44275./110592., b65 = 253./4096.,
    c1 = 37./378., c3 = 250./621., c4 = 125./594., c6 = 512./1771.,
    dc1 = 37./378. - 2825./27648., dc3 = 250./621. - 18575./48384.,
    dc4 = 125./594. - 13525./55296., dc5 = -277./14336.,
    dc6 = 512./1771. - 1./4.;

  double k1[4], k2[4], k3[4], k4[4], k5[4], k6[4], yt[4];

  if (gle_rhs (p, y, k1)) return -1.0;
  for (int i = 0; i < 4; i++)
    yt[i] = y[i] + ds*b21*k1[i];
  if (gle_rhs (p, yt, k2)) return -1.0;
  for (int i = 0; i < 4; i++)
    yt[i] = y[i] + ds*(b31*k1[i] + b32*k2[i]);
  if (gle_rhs (p, yt, k3)) return -1.0;
  for (int i = 0; i < 4; i++)
    yt[i] = y[i] + ds*(b41*k1[i] + b42*k2[i] + b43*k3[i]);
  if (gle_rhs (p, yt, k4)) return -1.0;
  for (int i = 0; i < 4; i++)
    yt[i] = y[i] + ds*(b51*k1[i] + b52*k2[i] + b53*k3[i] + b54*k4[i]);
  if (gle_rhs (p, yt, k5)) return -1.0;
  for (int i = 0; i < 4; i++)
    yt[i] = y[i] + ds*(b61*k1[i] + b62*k2[i] + b63*k3[i] + b64*k4[i]
		       + b65*k5[i]);
  if (gle_rhs (p, yt, k6)) return -1.0;

  double errsum = 0.0;
  for (int i = 0; i < 4; i++) {
    y5[i] = y[i] + ds*(c1*k1[i] + c3*k3[i] + c4*k4[i] + c6*k6[i]);
    double erri = ds*(dc1*k1[i] + dc3*k3[i] + dc4*k4[i] + dc5*k5[i]
		      + dc6*k6[i]);
    double tol = p->atol + p->rtol*fmax (fabs (y[i]), fabs (y5[i]));
    double r = erri/tol;
    errsum += r*r;
  }
  return sqrt (errsum/4.0);
}

/**
### gle_integrate()

Integrates the GLE from (`*s`, `y`) until the film thickness reaches
`h_target` (rising crossing), the arc length exceeds `smax`, or an error
condition occurs. On success, (`*s`, `y`) hold the state *at* the event to
within a relative tolerance of $10^{-13}$ on $h$.

Step control: accept when the Cash–Karp error norm $E \le 1$; the next step is
$\mathrm{d}s \leftarrow \mathrm{d}s \cdot \min(5,\max(0.2,\,0.9E^{-1/5}))$.
Steps whose stage evaluations leave the physical domain are retried at half
the size until the step underflows, which is reported as `GLE_ERR_DOMAIN`
(the trajectory genuinely exits through $\theta \to 0$, $\theta \to \pi$ or
$h \to 0$ — for the dip-coating problem this is the signature of a shooting
parameter on the wrong side of the solution).

#### Parameters
- `p`: model parameters (integrator tolerances included).
- `s`: in/out arc-length position.
- `y`: in/out state $(h,\theta,\omega,\zeta)$.
- `h_target`: thickness event ($\le 0$ disables; then runs to `smax`).
- `smax`: arc-length cap for this call.
- `sampler`, `sctx`: optional per-accepted-step observer.

#### Returns
An `enum gle_integrate_status` value.
*/
static inline int gle_integrate (const GLEParams *p, double *s, double y[4],
				 double h_target, double smax,
				 GLESampler sampler, void *sctx) {
  double ds = 0.01*fmax (*s, p->slip);   /* gentle initial step */
  long nstep = 0;

  if (sampler)
    sampler (sctx, *s, y);

  while (nstep++ < p->max_steps) {
    if (*s >= smax)
      return GLE_ERR_SMAX;
    if (ds > smax - *s)
      ds = smax - *s;

    double y5[4];
    double err = gle_rkck_step (p, y, ds, y5);

    if (err < 0.0) {                     /* domain violation inside step */
      ds *= 0.5;
      if (ds < 1e-15*fmax (*s, 1.0))
	return GLE_ERR_DOMAIN;
      continue;
    }
    if (!(err <= 1.0)) {                 /* reject: shrink (this also
					     rejects NaN error norms, which
					     fail every ordinary comparison) */
      ds *= fmax (0.2, 0.9*pow (err, -0.2));
      if (ds < 1e-15*fmax (*s, 1.0))
	return GLE_ERR_STEPSIZE;
      continue;
    }

    /* accepted step; check for the thickness event inside it */
    if (h_target > 0.0 && y[0] < h_target && y5[0] >= h_target) {
      double lo = 0.0, hi = ds;          /* bisect on step size */
      for (int it = 0; it < 200; it++) {
	double mid = 0.5*(lo + hi);
	double ym[4];
	if (gle_rkck_step (p, y, mid, ym) < 0.0) {
	  hi = mid;
	  continue;
	}
	if (ym[0] >= h_target)
	  hi = mid;
	else
	  lo = mid;
	if ((hi - lo) < 1e-13*ds)
	  break;
      }
      double yf[4];
      if (gle_rkck_step (p, y, hi, yf) >= 0.0) {
	for (int i = 0; i < 4; i++)
	  y[i] = yf[i];
	*s += hi;
	if (sampler)
	  sampler (sctx, *s, y);
	return GLE_OK;
      }
      return GLE_ERR_DOMAIN;
    }

    for (int i = 0; i < 4; i++)
      y[i] = y5[i];
    *s += ds;
    if (sampler)
      sampler (sctx, *s, y);

    ds *= fmin (5.0, fmax (0.2, 0.9*pow (fmax (err, 1e-16), -0.2)));
  }
  return GLE_ERR_STEPS;
}

#endif /* GLE_INTEGRATE_H */
