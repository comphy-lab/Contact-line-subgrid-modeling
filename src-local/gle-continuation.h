/**
# gle-continuation.h — arclength continuation of GLE solution branches

Traces the steady-state branch $(\mathrm{Ca}, \Delta)$ of the dip-coating
problem *through* its fold bifurcation — the bifurcation diagram of Fig. 4b of
[Snoeijer & Andreotti (2013)](https://doi.org/10.1146/annurev-fluid-011212-140734):
meniscus rise $\Delta = z_{\mathrm{cl}}/\ell_\gamma$ versus capillary number,
with a saddle-node at $\mathrm{Ca}^{*}$ (where $\theta_{\mathrm{app}} \to 0$
and $\Delta \to \sqrt{2}$) and an upper branch approaching the critical
Landau–Levich speed as $\Delta \to \infty$.

> **Superseded for branch tracing.** [gle-collocate.h](gle-collocate.h) is
> now the branch-tracing workhorse (see its header for why). Near the fold,
> the $(\omega_0, \mathrm{Ca})$ chart this routine walks is genuinely
> degenerate, and the shooting-based march below can silently *retrace the
> lower branch* instead of crossing onto the upper one, rather than failing
> loudly. This routine is retained for provenance and for lower-branch-only
> use (fast scans well away from the fold); it is not safe for tracing a
> branch through the fold.

## Method: secant predictor + local-parameterization corrector

Because [gle-shoot.h](gle-shoot.h) reduces the BVP to a **single scalar
equation** $\mathcal{R}(\omega_0, \mathrm{Ca}) = 0$, the branch lives in the
$(\omega_0, \mathrm{Ca})$ plane and continuation needs no discretisation mesh
among its unknowns — the failure mode of `GLE_continuation_v4.5.py` (issues
#14 and #16), where scipy's `solve_bvp` re-adapted its mesh between residual
evaluations and made the extended system inconsistent, cannot occur here by
construction.

Each step is a **secant (arclength) predictor** followed by a **1-D
corrector** in the local-parameterization style of Rheinboldt (1980; the
fold-handling strategy of AUTO's ancestors): freeze whichever of
$\{\mathrm{Ca}, \omega_0\}$ the branch currently varies *fastest* in
(relative terms), and solve the scalar equation for the other:

- lower branch, away from the fold: either choice works; the selector picks
  the variable with the larger relative secant increment;
- at the fold: $\partial\mathcal{R}/\partial\omega_0 = 0$ **by definition of
  a saddle-node**, so solving for $\omega_0$ at fixed $\mathrm{Ca}$ turns
  singular there — while the $\mathrm{Ca}$-secant vanishes. The selector
  therefore freezes $\omega_0$ (whose secant stays finite) and solves for
  $\mathrm{Ca}$ via `gle_solve_ca()`, which is regular through the fold.

A genuinely 2-D Newton corrector on the extended (pseudo-arclength) system
was tried and rejected: the residual is amplified like $e^{s}$ transverse to
the branch, so its linear range in $\omega_0$ ($\sim 10^{-5}$) is microscopic
compared to any useful step, and finite-difference Jacobians of the coupled
system are meaningless at every usable step size. The 1-D correctors dodge
this entirely because bracketing + bisection needs no derivative at all.

## Step control

Steps are measured as a factor $\alpha$ of the previous secant ($\alpha = 1$
repeats the previous step). Corrector failure — or a corrected point landing
suspiciously far from the predictor — halves $\alpha$; fast convergence grows
it by $1.3\times$ up to `alpha_max`. The fold is detected from the sign
change of the $\mathrm{Ca}$-secant and refined by quadratic interpolation in
$\Delta$.

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#ifndef GLE_CONTINUATION_H
#define GLE_CONTINUATION_H

#include <stdio.h>
#include "gle-shoot.h"

/**
## Branch data structures
*/
typedef struct {
  double Ca;
  double omega0;
  double Delta;
  double theta_app;   /* NAN past the fold */
  double theta_min;
  double s_end;
  double residual;
  int iters;
} GLEBranchPoint;

/**
### GLEContOpts

Continuation controls.

#### Fields
- `Ca_start`: first (small) capillary number on the lower branch.
- `alpha0`, `alpha_min`, `alpha_max`: initial / minimal / maximal step
  factor relative to the previous secant.
- `max_points`: branch-point budget.
- `Delta_max`: stop once the meniscus rise exceeds this (upper branch runs to
  $\Delta \to \infty$; Fig. 4b needs $\Delta \lesssim 3.6$).
- `Ca_stop_min`: stop if $\mathrm{Ca}$ falls below this after the fold.
- `verbose`: progress lines to `stderr` (`>= 2` also logs corrector
  failures).
*/
typedef struct {
  double Ca_start;
  double alpha0, alpha_min, alpha_max;
  int max_points;
  double Delta_max;
  double Ca_stop_min;
  int verbose;
} GLEContOpts;

static inline GLEContOpts gle_default_cont_opts (void) {
  GLEContOpts o;
  o.Ca_start = 1.0e-6;
  o.alpha0 = 1.0;
  o.alpha_min = 1.0e-4;
  o.alpha_max = 4.0;
  o.max_points = 2000;
  o.Delta_max = 3.7;
  o.Ca_stop_min = 1.0e-4;
  o.verbose = 0;
  return o;
}

/**
### gle_branch_csv_header(), gle_branch_csv_row()

CSV serialisation of branch points (`theta` columns in degrees).
*/
static inline void gle_branch_csv_header (FILE *fp) {
  fprintf (fp, "index,Ca,Delta,omega0,theta_app_deg,theta_min_deg,"
	   "s_end,residual,corrector_iters\n");
}

static inline void gle_branch_csv_row (FILE *fp, int idx,
				       const GLEBranchPoint *b) {
  fprintf (fp, "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.6e,%.3e,%d\n",
	   idx, b->Ca, b->Delta, b->omega0,
	   b->theta_app*180.0/M_PI, b->theta_min*180.0/M_PI,
	   b->s_end, b->residual, b->iters);
}

/**
### gle_continuation()

Traces the branch from `opts->Ca_start` on the lower branch, through the
fold, up the high-$\Delta$ branch. Points are stored in `branch[]`
(caller-allocated of size `opts->max_points`) and streamed to `csv` when
non-`NULL`.

The first two points use natural continuation in $\mathrm{Ca}$ (robust
shooting on $\omega_0$, seeded by the static meniscus curvature
$\sqrt{2(1-\sin\theta_e)}$); every subsequent point is a secant-predictor /
1-D-corrector step as described in the header notes, with a sanity guard
rejecting corrected points that land further than $3\alpha$ previous-secants
from the base point (branch-jumping protection near the fold, where two
solutions coexist at the same $\mathrm{Ca}$).

The fold is located from the sign change of the $\mathrm{Ca}$-secant and
refined by quadratic interpolation of $\mathrm{Ca}(\Delta)$ through the three
neighbouring points; written to `fold_Ca` / `fold_Delta` when non-`NULL`.

#### Returns
The number of branch points computed (≥ 1 on any success; 0 = total
failure).
*/
static inline int gle_continuation (GLEParams *p, const GLEContOpts *opts,
				    GLEBranchPoint *branch, FILE *csv,
				    double *fold_Ca, double *fold_Delta) {
  if (opts->max_points < 2)
    return 0;                 /* branch[] has no room for the two natural-
				  continuation seed points pushed below;
				  bail out before writing past its end
				  (confirmed heap overflow otherwise) */
  int n = 0;
  GLESolution sol;

  if (csv)
    gle_branch_csv_header (csv);

  /* --- point 0: natural solve at Ca_start --- */
  p->Ca = opts->Ca_start;
  double w_guess = gle_static_curvature (p->theta_mic, p->grav);
  if (gle_shoot (p, w_guess, &sol)) {
    if (opts->verbose)
      fprintf (stderr, "gle_continuation: failed at Ca_start = %g\n",
	       opts->Ca_start);
    return 0;
  }
#define GLE_PUSH_POINT()						\
  do {									\
    branch[n].Ca = p->Ca;						\
    branch[n].omega0 = sol.omega0;					\
    branch[n].Delta = sol.Delta;					\
    branch[n].theta_app = sol.theta_app;				\
    branch[n].theta_min = sol.theta_min;				\
    branch[n].s_end = sol.s_end;					\
    branch[n].residual = sol.residual;					\
    branch[n].iters = sol.iters;					\
    if (csv) {								\
      gle_branch_csv_row (csv, n, &branch[n]);				\
      fflush (csv);							\
    }									\
    n++;								\
  } while (0)

  GLE_PUSH_POINT ();

  /* --- point 1: natural solve at 2 Ca_start --- */
  p->Ca = 2.0*opts->Ca_start;
  if (gle_shoot (p, sol.omega0, &sol)) {
    if (opts->verbose)
      fprintf (stderr, "gle_continuation: failed at second point\n");
    return n;
  }
  GLE_PUSH_POINT ();

  /* --- secant-predictor / local-parameterization-corrector march --- */
  double alpha = opts->alpha0;
  int fold_seen = 0;
  double prev_dCa = branch[1].Ca - branch[0].Ca;
  /* secant base: the last point distinct from the branch head (duplicates
     must never enter the secant, or the predictor dies) */
  double base_w = branch[0].omega0, base_Ca = branch[0].Ca;
  int zero_steps = 0;

  while (n < opts->max_points) {
    double sig_w = branch[n-1].omega0 - base_w;
    double sig_ca = branch[n-1].Ca - base_Ca;

    /* selector: relative secant increments */
    double rel_w = fabs (sig_w)/fmax (fabs (branch[n-1].omega0), 1.0);
    double rel_ca = fabs (sig_ca)/fmax (fabs (branch[n-1].Ca), 1.0e-12);

    double w_new = 0.0, Ca_new = 0.0;
    int ok = 0;
    while (!ok) {
      double w_pred = branch[n-1].omega0 + alpha*sig_w;
      double Ca_pred = branch[n-1].Ca + alpha*sig_ca;
      int fail;
      if (rel_ca >= rel_w && Ca_pred > 0.0) {
	/* freeze Ca, solve omega0 (regular away from the fold) */
	p->Ca = Ca_pred;
	fail = gle_shoot (p, w_pred, &sol);
	w_new = sol.omega0;
	Ca_new = Ca_pred;
	/* branch-jump guard: near the fold two omega0 roots coexist at the
	   same Ca and the fallback bracket expansion in gle_shoot() can
	   land on the other one; reject corrections far from the
	   predictor. (The solve-Ca corrector below needs no guard: Ca is
	   single-valued in omega0 through a Ca-fold.) */
	if (!fail && fabs (sig_w) > 0.0 &&
	    fabs (w_new - w_pred) > 3.0*alpha*fabs (sig_w))
	  fail = 1;
      }
      else {
	/* freeze omega0, solve Ca (regular at and beyond the fold) */
	fail = gle_solve_ca (p, w_pred,
			     (Ca_pred > 0.0 ? Ca_pred : branch[n-1].Ca),
			     &sol);
	w_new = w_pred;
	Ca_new = p->Ca;
      }
      /* orientation guard: never march backward along the branch. Right
	 after the fold the two omega0 roots are still close and a corrector
	 can land on the already-computed branch; the backward step projects
	 negatively on the tangent and is rejected here. */
      if (!fail) {
	double sw = fmax (fabs (branch[n-1].omega0), 1.0);
	double sc = fmax (branch[n-1].Ca, 1.0e-12);
	if (((w_new - branch[n-1].omega0)/sw)*(sig_w/sw)
	    + ((Ca_new - branch[n-1].Ca)/sc)*(sig_ca/sc) < 0.0)
	  fail = 1;
      }
      ok = !fail;
      if (!ok) {
	if (opts->verbose >= 2)
	  fprintf (stderr, "    corrector fail at point %d (Ca = %.6e, "
		   "alpha = %.3g)\n", n, branch[n-1].Ca, alpha);
	alpha *= 0.5;
	if (alpha < opts->alpha_min) {
	  if (opts->verbose)
	    fprintf (stderr,
		     "gle_continuation: step underflow at point %d "
		     "(Ca = %.6e, Delta = %.4f)\n",
		     n, branch[n-1].Ca, branch[n-1].Delta);
	  return n;
	}
      }
    }
    /* zero-step: the corrected point coincides with the branch head
       (predictor displacement below solver resolution, typical right after
       a fold crossing has ground alpha down). Inflate alpha and retry
       without pushing a duplicate — a duplicate would zero the secant and
       freeze the march for good. */
    if (fabs (w_new - branch[n-1].omega0) <=
	1.0e-12*fmax (fabs (branch[n-1].omega0), 1.0) &&
	fabs (Ca_new - branch[n-1].Ca) <=
	1.0e-12*fmax (branch[n-1].Ca, 1.0e-12)) {
      if (opts->verbose >= 2)
	fprintf (stderr, "    zero step at point %d, inflating alpha to "
		 "%.3g\n", n, 2.0*alpha);
      alpha *= 2.0;
      if (++zero_steps > 60) {
	if (opts->verbose)
	  fprintf (stderr, "gle_continuation: stuck at point %d "
		   "(Ca = %.6e, Delta = %.4f)\n",
		   n, branch[n-1].Ca, branch[n-1].Delta);
	return n;
      }
      continue;
    }
    zero_steps = 0;
    base_w = branch[n-1].omega0;
    base_Ca = branch[n-1].Ca;
    p->Ca = Ca_new;
    sol.omega0 = w_new;
    GLE_PUSH_POINT ();

    if (alpha < opts->alpha_max)
      alpha *= 1.3;

    /* fold detection: Ca-secant sign change */
    double dCa = branch[n-1].Ca - branch[n-2].Ca;
    if (!fold_seen && n > 2 && dCa*prev_dCa < 0.0) {
      fold_seen = 1;
      /* quadratic vertex of Ca(Delta) through the last three points
	 (Delta is monotone through the fold) */
      double ca1 = branch[n-3].Ca, ca2 = branch[n-2].Ca, ca3 = branch[n-1].Ca;
      double d1 = branch[n-3].Delta, d2 = branch[n-2].Delta,
	d3 = branch[n-1].Delta;
      double denom = (d1 - d2)*(d1 - d3)*(d2 - d3);
      double fCa = ca2, fD = d2;
      if (fabs (denom) > 0.0) {
	double A = (d3*(ca2 - ca1) + d2*(ca1 - ca3) + d1*(ca3 - ca2))/denom;
	double B = (d3*d3*(ca1 - ca2) + d2*d2*(ca3 - ca1)
		    + d1*d1*(ca2 - ca3))/denom;
	if (A != 0.0) {
	  fD = -B/(2.0*A);
	  double C = ca1 - A*d1*d1 - B*d1;
	  fCa = A*fD*fD + B*fD + C;
	}
      }
      if (fold_Ca) *fold_Ca = fCa;
      if (fold_Delta) *fold_Delta = fD;
      if (opts->verbose)
	fprintf (stderr,
		 "gle_continuation: fold at Ca* = %.6e, Delta* = %.4f\n",
		 fCa, fD);
    }
    prev_dCa = dCa;

    if (opts->verbose && n % 20 == 0)
      fprintf (stderr, "  point %4d: Ca = %.6e  Delta = %.4f  alpha = %.3g\n",
	       n, branch[n-1].Ca, branch[n-1].Delta, alpha);

    if (branch[n-1].Delta > opts->Delta_max)
      break;
    if (fold_seen && branch[n-1].Ca < opts->Ca_stop_min)
      break;
  }
#undef GLE_PUSH_POINT
  return n;
}

#endif /* GLE_CONTINUATION_H */
