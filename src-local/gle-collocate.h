/**
# gle-collocate.h — fixed-mesh collocation solver for GLE branch tracing

A damped-Newton collocation solver for the dip-coating GLE boundary-value
problem of [gle-model.h](gle-model.h), discretised by the implicit midpoint
rule on a **fixed logarithmic mesh**. This is the branch-tracing workhorse
behind [gle-continuation.h](gle-continuation.h)'s driver: it parameterises
solutions by the meniscus rise $\Delta$ — along which the Fig. 4b branch has
**no fold** — and treats the capillary number $\mathrm{Ca}$ and the total arc
length $s_{\mathrm{end}}$ as unknowns.

## Why collocation, and why this parameterisation

Single shooting (see [gle-shoot.h](gle-shoot.h)) is ideal for fast single
solves on the lower branch, but it cannot trace the full bifurcation diagram:

- near the saddle-node the branch makes a hairpin in the $(\omega_0,
  \mathrm{Ca})$ plane narrower than the shooting map can resolve — the
  forward flow *compresses* branch separation like $e^{-s}$ while its noise
  is *amplified* like $e^{+s}$;
- deep on the upper branch the trajectory carries a film section of length
  $\sim \Delta$ whose exponential dichotomy makes shooting ill-conditioned
  in *both* directions.

A banded LU factorisation of the collocation system is the classical cure
(it is equivalent to a stabilised multiple-shooting march; cf. AUTO), and a
**fixed** mesh makes the extended system self-consistent between Newton
iterations — the mesh re-adaptation of scipy's `solve_bvp` inside a
continuation loop is exactly what broke the historical Python attempts
(issues #14, #16 of this repository).

## Discretisation

Nodes $s_i = s_0\,(s_{\mathrm{end}}/s_0)^{\tau_i}$, $\tau_i = i/N$ fixed
(log-uniform: the mesh resolves the slip scale and dilates with the unknown
domain length). State $y_i = (h, \theta, \omega, \zeta)_i$. Cell residuals
(implicit midpoint, second order):

$$ y_{i+1} - y_i - (s_{i+1}-s_i)\,
   \mathbf{f}\!\left(\tfrac{1}{2}(y_i + y_{i+1})\right) = 0 . $$

Boundary and closure conditions:

- $h_0 = h_0^{\mathrm{bc}}$, $\theta_0 = \theta_e$,
  $\zeta_0 = s_0\cos\theta_e$ (contact line);
- $\omega_N = \sqrt{2(1 - \sin\theta_N)}$ (static-meniscus manifold);
- $h_N = H_{\mathrm{match}}$ (defines $s_{\mathrm{end}}$);
- $\zeta_N + \omega_N = \Delta^{*}$ (the target meniscus rise).

Unknowns: $4(N{+}1)$ node states $+\;\mathrm{Ca}\;+\;s_{\mathrm{end}}$;
equations: $4N$ collocation $+\,6$ conditions. Square.

## Linear algebra

The Jacobian is block-bidiagonal (bandwidth 7) with two dense **border
columns** ($\partial/\partial \mathrm{Ca}$, $\partial/\partial
s_{\mathrm{end}}$, both formed by finite differences of the full residual)
and two **border rows** (the $h_N$ and $\Delta$ conditions). It is solved by
the standard bordering algorithm: banded LU with partial pivoting on the
square band part, three back-solves, then a $2\times2$ reduced system.

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#ifndef GLE_COLLOCATE_H
#define GLE_COLLOCATE_H

#include <stdlib.h>
#include <string.h>
#include "gle-continuation.h"

/**
## Banded LU with partial pivoting

Minimal LAPACK-style band factorisation (`dgbtrf`-lite). The matrix is
stored in band format: `ab[(2*kl + ku + 1) x n]`, entry $(i,j)$ at
`ab[kl + ku + i - j + ldab*j]` for $\max(0,j-ku) \le i \le
\min(n-1,j+kl)$; the extra `kl` superdiagonals hold pivoting fill.
*/
typedef struct {
  int n, kl, ku, ldab;
  double *ab;
  int *ipiv;
} gle_band_matrix;

static inline int gle_band_alloc (gle_band_matrix *m, int n, int kl, int ku) {
  m->n = n;
  m->kl = kl;
  m->ku = ku;
  m->ldab = 2*kl + ku + 1;
  m->ab = (double *) calloc ((size_t) m->ldab*n, sizeof (double));
  m->ipiv = (int *) calloc (n, sizeof (int));
  return (m->ab && m->ipiv) ? 0 : 1;
}

static inline void gle_band_free (gle_band_matrix *m) {
  free (m->ab);
  free (m->ipiv);
  m->ab = NULL;
  m->ipiv = NULL;
}

static inline void gle_band_zero (gle_band_matrix *m) {
  memset (m->ab, 0, (size_t) m->ldab*m->n*sizeof (double));
}

static inline double *gle_band_at (gle_band_matrix *m, int i, int j) {
  return &m->ab[m->kl + m->ku + i - j + m->ldab*j];
}

/**
### gle_band_factor()

In-place LU with row partial pivoting (rows only move within the band's
`kl` fill region, as in `dgbtrf`).

#### Returns
`0` on success, `j+1` if a zero pivot was met in column `j`.
*/
static inline int gle_band_factor (gle_band_matrix *m) {
  int n = m->n, kl = m->kl, ku = m->ku, ldab = m->ldab;
  double *ab = m->ab;
  for (int j = 0; j < n; j++) {
    int pmax = (j + kl < n - 1 ? j + kl : n - 1);
    /* pivot search in column j, rows j..pmax */
    int piv = j;
    double amax = fabs (ab[kl + ku + ldab*j]);   /* (j,j) */
    for (int i = j + 1; i <= pmax; i++) {
      double a = fabs (ab[kl + ku + i - j + ldab*j]);
      if (a > amax) {
	amax = a;
	piv = i;
      }
    }
    m->ipiv[j] = piv;
    if (amax == 0.0)
      return j + 1;
    int jmax = (j + kl + ku < n - 1 ? j + kl + ku : n - 1);
    if (piv != j)
      for (int c = j; c <= jmax; c++) {
	double *a1 = gle_band_at (m, j, c), *a2 = gle_band_at (m, piv, c);
	double t = *a1; *a1 = *a2; *a2 = t;
      }
    double pivval = ab[kl + ku + ldab*j];
    for (int i = j + 1; i <= pmax; i++) {
      double *lij = gle_band_at (m, i, j);
      *lij /= pivval;
      double lv = *lij;
      if (lv != 0.0)
	for (int c = j + 1; c <= jmax; c++)
	  *gle_band_at (m, i, c) -= lv*(*gle_band_at (m, j, c));
    }
  }
  return 0;
}

/**
### gle_band_solve()

Back-substitution for one right-hand side (in place).
*/
static inline void gle_band_solve (gle_band_matrix *m, double *b) {
  int n = m->n, kl = m->kl, ku = m->ku;
  for (int j = 0; j < n; j++) {
    int piv = m->ipiv[j];
    if (piv != j) {
      double t = b[j]; b[j] = b[piv]; b[piv] = t;
    }
    int imax = (j + kl < n - 1 ? j + kl : n - 1);
    for (int i = j + 1; i <= imax; i++)
      b[i] -= (*gle_band_at (m, i, j))*b[j];
  }
  for (int j = n - 1; j >= 0; j--) {
    int imin = (j - kl - ku > 0 ? j - kl - ku : 0);
    b[j] /= *gle_band_at (m, j, j);
    for (int i = imin; i < j; i++)
      b[i] -= (*gle_band_at (m, i, j))*b[j];
  }
}

/**
## The collocation problem
*/
typedef struct {
  int N;              /* cells; N+1 nodes                                  */
  double *tau;        /* fixed node positions in [0,1]                     */
  double tau_split;   /* mesh grading: log below, uniform above            */
  double s_split;     /* physical arc length of the grading switch         */
  double *y;          /* 4*(N+1) node states                               */
  double Ca;          /* unknown: capillary number                         */
  double s_end;       /* unknown: total arc length                         */
  double Delta;       /* achieved meniscus rise (zeta_N + omega_N)         */
  /* workspace */
  double *res;        /* 4N+4 band-part residual                           */
  double *colCa, *colS; /* border columns                                  */
  double *ytmp;
  gle_band_matrix band;
} GLECollocation;

/**
### gle_colloc_node_s()

The fixed mesh mapping $\tau \mapsto s$: **logarithmic** from $s_0$ to
`s_split` for $\tau \le$ `tau_split` (resolving the slip-to-mesoscale
decades near the contact line), then **uniform** from `s_split` to the
unknown $s_{\mathrm{end}}$ (resolving the film and the meniscus foot, whose
widths are $\mathcal{O}(\sqrt{\mathrm{Ca}})$ — a pure log mesh starves this
region of nodes once the upper-branch film lengthens). Only the uniform part
dilates with $s_{\mathrm{end}}$.
*/
static inline double gle_colloc_node_s (const GLECollocation *c,
					const GLEParams *p, int i,
					double s_end) {
  double s0 = gle_s0 (p);
  double t = c->tau[i];
  if (t <= c->tau_split)
    return s0*pow (c->s_split/s0, t/c->tau_split);
  return c->s_split
    + (s_end - c->s_split)*(t - c->tau_split)/(1.0 - c->tau_split);
}

/**
### gle_colloc_alloc(), gle_colloc_free()

Workspace management. `N` cells give a $4N{+}4$ band system with
`kl = ku = 7`.
*/
static inline int gle_colloc_alloc (GLECollocation *c, int N) {
  memset (c, 0, sizeof *c);
  c->N = N;
  c->tau = (double *) malloc ((N + 1)*sizeof (double));
  c->y = (double *) malloc (4*(N + 1)*sizeof (double));
  c->res = (double *) malloc ((4*N + 4)*sizeof (double));
  c->colCa = (double *) malloc ((4*N + 4)*sizeof (double));
  c->colS = (double *) malloc ((4*N + 4)*sizeof (double));
  c->ytmp = (double *) malloc (4*(N + 1)*sizeof (double));
  if (!c->tau || !c->y || !c->res || !c->colCa || !c->colS || !c->ytmp)
    return 1;
  for (int i = 0; i <= N; i++)
    c->tau[i] = (double) i/N;
  c->tau_split = 0.45;
  c->s_split = 0.3;
  return gle_band_alloc (&c->band, 4*N + 4, 7, 7);
}

static inline void gle_colloc_free (GLECollocation *c) {
  free (c->tau); free (c->y); free (c->res);
  free (c->colCa); free (c->colS); free (c->ytmp);
  gle_band_free (&c->band);
}

/**
### gle_colloc_band_residual()

The $4N{+}4$ "band part" of the residual at node states `y`, parameters
(`Ca` inside `p`, domain length `s_end`):

- rows $0..2$: contact-line conditions on $h_0, \theta_0, \zeta_0$;
- rows $3\,..\,4N+2$: the $4N$ midpoint collocation residuals;
- row $4N+3$: the static-manifold condition at node $N$.

The two border conditions ($h_N = H$ and $\zeta_N + \omega_N = \Delta^{*}$)
are evaluated separately by `gle_colloc_border_residual()`.
*/
static inline void gle_colloc_band_residual (const GLECollocation *c,
					     const GLEParams *p,
					     const double *y, double s_end,
					     double *res) {
  int N = c->N;
  double s0 = gle_s0 (p);
  res[0] = y[0] - gle_h0 (p);
  res[1] = y[1] - p->theta_mic;
  res[2] = y[3] - s0*cos (p->theta_mic);
  for (int i = 0; i < N; i++) {
    double si = gle_colloc_node_s (c, p, i, s_end);
    double si1 = gle_colloc_node_s (c, p, i + 1, s_end);
    double ds = si1 - si;
    double ym[4], fm[4];
    for (int k = 0; k < 4; k++)
      ym[k] = 0.5*(y[4*i + k] + y[4*(i + 1) + k]);
    if (gle_rhs (p, ym, fm)) {
      /* out-of-domain midpoint: large smooth penalty steers Newton back */
      for (int k = 0; k < 4; k++)
	res[3 + 4*i + k] = 1.0e3*(ym[1] < 0.5*p->theta_mic ? -1.0 : 1.0);
      continue;
    }
    for (int k = 0; k < 4; k++)
      res[3 + 4*i + k] = y[4*(i + 1) + k] - y[4*i + k] - ds*fm[k];
  }
  res[4*N + 3] = y[4*N + 2] - gle_static_curvature (y[4*N + 1]);
}

static inline void gle_colloc_border_residual (const GLECollocation *c,
					       const GLEParams *p,
					       const double *y,
					       double Delta_target,
					       double rb[2]) {
  int N = c->N;
  (void) p;
  rb[0] = y[4*N + 0] - p->H_match;
  rb[1] = y[4*N + 3] + y[4*N + 2] - Delta_target;
}

/**
### gle_colloc_assemble()

Builds the band Jacobian (analytic in the node states: per-cell $4\times4$
blocks by finite differences of the *local* midpoint RHS — local FD carries
no global error amplification) and the two border columns (full-residual
finite differences in $\mathrm{Ca}$ and $s_{\mathrm{end}}$).
*/
static inline void gle_colloc_assemble (GLECollocation *c, GLEParams *p,
					double Delta_target,
					double rb[2], double rbCa[2],
					double rbS[2]) {
  int N = c->N;
  int n = 4*N + 4;
  gle_band_zero (&c->band);

  gle_colloc_band_residual (c, p, c->y, c->s_end, c->res);
  gle_colloc_border_residual (c, p, c->y, Delta_target, rb);

  /* rows 0..2: identity-like on y_0 */
  *gle_band_at (&c->band, 0, 0) = 1.0;
  *gle_band_at (&c->band, 1, 1) = 1.0;
  *gle_band_at (&c->band, 2, 3) = 1.0;

  /* collocation rows */
  for (int i = 0; i < N; i++) {
    double si = gle_colloc_node_s (c, p, i, c->s_end);
    double si1 = gle_colloc_node_s (c, p, i + 1, c->s_end);
    double ds = si1 - si;
    double ym[4], fm[4];
    for (int k = 0; k < 4; k++)
      ym[k] = 0.5*(c->y[4*i + k] + c->y[4*(i + 1) + k]);
    int bad = gle_rhs (p, ym, fm);
    /* d f / d ym by local central differences */
    double dfdy[4][4];
    for (int k2 = 0; k2 < 4; k2++) {
      double dm = 1.0e-7*fmax (fabs (ym[k2]), 1.0e-6);
      double yp[4], yq[4], fp[4], fq[4];
      memcpy (yp, ym, sizeof yp);
      memcpy (yq, ym, sizeof yq);
      yp[k2] += dm;
      yq[k2] -= dm;
      int b1 = gle_rhs (p, yp, fp), b2 = gle_rhs (p, yq, fq);
      for (int k1 = 0; k1 < 4; k1++)
	dfdy[k1][k2] = (bad || b1 || b2) ? 0.0 : (fp[k1] - fq[k1])/(2.0*dm);
    }
    for (int k1 = 0; k1 < 4; k1++) {
      int row = 3 + 4*i + k1;
      for (int k2 = 0; k2 < 4; k2++) {
	double dk = (k1 == k2 ? 1.0 : 0.0);
	*gle_band_at (&c->band, row, 4*i + k2) +=
	  -dk - 0.5*ds*dfdy[k1][k2];
	*gle_band_at (&c->band, row, 4*(i + 1) + k2) +=
	  dk - 0.5*ds*dfdy[k1][k2];
      }
    }
  }

  /* far-field manifold row */
  *gle_band_at (&c->band, n - 1, 4*N + 2) = 1.0;
  *gle_band_at (&c->band, n - 1, 4*N + 1) =
    -(-cos (c->y[4*N + 1]))/fmax (gle_static_curvature (c->y[4*N + 1]),
				  1.0e-12);
  /* d/dtheta sqrt(2(1-sin th)) = -cos th / sqrt(2(1-sin th)) */

  /* border columns: full-residual FD in Ca and s_end */
  double dCa = 1.0e-7*fmax (fabs (p->Ca), 1.0e-6);
  double saveCa = p->Ca;
  p->Ca = saveCa + dCa;
  gle_colloc_band_residual (c, p, c->y, c->s_end, c->colCa);
  double rbp[2];
  gle_colloc_border_residual (c, p, c->y, Delta_target, rbp);
  rbCa[0] = (rbp[0] - rb[0])/dCa;
  rbCa[1] = (rbp[1] - rb[1])/dCa;
  for (int i = 0; i < n; i++)
    c->colCa[i] = (c->colCa[i] - c->res[i])/dCa;
  p->Ca = saveCa;

  double dS = 1.0e-7*fmax (fabs (c->s_end), 1.0);
  gle_colloc_band_residual (c, p, c->y, c->s_end + dS, c->colS);
  gle_colloc_border_residual (c, p, c->y, Delta_target, rbp);
  rbS[0] = (rbp[0] - rb[0])/dS;
  rbS[1] = (rbp[1] - rb[1])/dS;
  for (int i = 0; i < n; i++)
    c->colS[i] = (c->colS[i] - c->res[i])/dS;
}

/**
### gle_colloc_border_rows()

The two border rows' coefficients w.r.t. the node states are sparse: the
$h_N$ condition touches `y[4N]`, the $\Delta$ condition touches
`y[4N+2]` and `y[4N+3]`. Given the three band back-solves, the bordering
algorithm reduces to a $2\times2$ solve; this helper evaluates
$\mathbf{b}^{T} x$ for a border row on a full-length vector.
*/
static inline double gle_colloc_border_dot (int which, int N,
					    const double *x) {
  return which == 0 ? x[4*N + 0] : x[4*N + 2] + x[4*N + 3];
}

/**
### gle_colloc_solve()

Damped Newton on the bordered system at fixed `Delta_target`. `c->y`,
`c->Ca`, `c->s_end` must hold a starting guess (typically the previous
branch point). On success, `c->Delta` is updated and `p->Ca = c->Ca`.

#### Returns
`0` on convergence, `1` on failure (state left unspecified).
*/
static inline int gle_colloc_solve (GLECollocation *c, GLEParams *p,
				    double Delta_target, int *iters_out) {
  int N = c->N, n = 4*N + 4;
  const int maxit = 30;
  const double tol = 1.0e-10;

  for (int it = 0; it < maxit; it++) {
    double rb[2], rbCa[2], rbS[2];
    p->Ca = c->Ca;
    gle_colloc_assemble (c, p, Delta_target, rb, rbCa, rbS);

    double rnorm = 0.0;
    for (int i = 0; i < n; i++)
      rnorm = fmax (rnorm, fabs (c->res[i]));
    rnorm = fmax (rnorm, fmax (fabs (rb[0]), fabs (rb[1])));
    if (rnorm < tol) {
      c->Delta = c->y[4*N + 3] + c->y[4*N + 2];
      if (iters_out)
	*iters_out = it;
      return 0;
    }

    if (gle_band_factor (&c->band))
      return 1;

    /* three back-solves: residual and the two border columns */
    double *xr = c->res, *xc = c->colCa, *xs = c->colS;
    gle_band_solve (&c->band, xr);
    gle_band_solve (&c->band, xc);
    gle_band_solve (&c->band, xs);

    /* bordered 2x2 reduction:
       [ b0.(xc)  b0.(xs) ] [dCa]   [ rb0 - b0.xr ]
       [ b1.(xc)  b1.(xs) ] [dS ] = [ rb1 - b1.xr ]   (all with signs below)
       where b_k are the border-row coefficient vectors; the band solves
       give A^{ -1 } columns, so dx = -xr - dCa*(-xc)... — worked out with
       xr = A^{-1} r, xc = A^{-1} c_Ca, xs = A^{-1} c_S:
       band rows:  A dx + c_Ca dCa + c_S dS = -r
                => dx = -xr - xc dCa - xs dS
       border row k: b_k.dx + rbX_k dCa + rbS_k dS = -rb_k
                => (rbCa_k - b_k.xc) dCa + (rbS_k - b_k.xs) dS
                   = -rb_k + b_k.xr                                       */
    double M00 = rbCa[0] - gle_colloc_border_dot (0, N, xc);
    double M01 = rbS[0] - gle_colloc_border_dot (0, N, xs);
    double M10 = rbCa[1] - gle_colloc_border_dot (1, N, xc);
    double M11 = rbS[1] - gle_colloc_border_dot (1, N, xs);
    double q0 = -rb[0] + gle_colloc_border_dot (0, N, xr);
    double q1 = -rb[1] + gle_colloc_border_dot (1, N, xr);
    double det = M00*M11 - M01*M10;
    if (!isfinite (det) || fabs (det) < 1.0e-300)
      return 1;
    double dCa = ( M11*q0 - M01*q1)/det;
    double dS = (-M10*q0 + M00*q1)/det;

    /* damping: limit relative moves of Ca and s_end, and keep the state
       physical (theta in (0,pi), h > 0) by fractional steps */
    double damp = 1.0;
    if (fabs (dCa) > 0.2*fmax (fabs (c->Ca), 1.0e-6))
      damp = fmin (damp, 0.2*fmax (fabs (c->Ca), 1.0e-6)/fabs (dCa));
    if (fabs (dS) > 0.2*fmax (fabs (c->s_end), 1.0))
      damp = fmin (damp, 0.2*fmax (fabs (c->s_end), 1.0)/fabs (dS));

    for (int trial = 0; trial < 8; trial++) {
      int ok = 1;
      for (int i = 0; i <= N; i++) {
	double hn = c->y[4*i + 0] - damp*(xr[4*i + 0] + xc[4*i + 0]*dCa
					  + xs[4*i + 0]*dS);
	double tn = c->y[4*i + 1] - damp*(xr[4*i + 1] + xc[4*i + 1]*dCa
					  + xs[4*i + 1]*dS);
	if (hn <= 0.0 || tn <= -0.5*M_PI || tn >= M_PI) {
	  ok = 0;
	  break;
	}
      }
      if (ok)
	break;
      damp *= 0.5;
    }

    for (int i = 0; i < n; i++)
      c->ytmp[i] = c->y[i] - damp*(xr[i] + xc[i]*dCa + xs[i]*dS);
    memcpy (c->y, c->ytmp, n*sizeof (double));
    c->Ca += damp*dCa;
    c->s_end += damp*dS;
    if (c->Ca <= 0.0 || c->s_end <= gle_s0 (p)*2.0)
      return 1;
  }
  return 1;
}

/**
### gle_colloc_seed_from_shoot()

Populates the mesh from a converged shooting solution: re-integrates the
trajectory with `gle_shoot_residual()`'s sampler, then interpolates
$(h,\theta,\omega,\zeta)$ onto the log mesh in $s$. Sets `c->Ca` and
`c->s_end` from the shot.

#### Returns
`0` on success.
*/
typedef struct {
  double *s;
  double *y4;
  long n, cap;
} gle_seed_buf;

static void gle_seed_sampler (void *ctx, double s, const double y[4]) {
  gle_seed_buf *b = (gle_seed_buf *) ctx;
  if (b->n == b->cap) {
    b->cap = b->cap ? 2*b->cap : 16384;
    b->s = (double *) realloc (b->s, b->cap*sizeof (double));
    b->y4 = (double *) realloc (b->y4, 4*b->cap*sizeof (double));
  }
  b->s[b->n] = s;
  for (int k = 0; k < 4; k++)
    b->y4[4*b->n + k] = y[k];
  b->n++;
}

static inline int gle_colloc_seed_from_shoot (GLECollocation *c,
					      GLEParams *p,
					      const GLESolution *sol) {
  gle_seed_buf buf = { NULL, NULL, 0, 0 };
  GLESolution tmp;
  gle_shoot_residual (p, sol->omega0, &tmp, gle_seed_sampler, &buf);
  if (buf.n < 2) {
    free (buf.s);
    free (buf.y4);
    return 1;
  }
  c->Ca = p->Ca;
  c->s_end = buf.s[buf.n - 1];
  long j = 0;
  for (int i = 0; i <= c->N; i++) {
    double si = gle_colloc_node_s (c, p, i, c->s_end);
    while (j < buf.n - 2 && buf.s[j + 1] < si)
      j++;
    double t = (si - buf.s[j])/fmax (buf.s[j + 1] - buf.s[j], 1e-300);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    for (int k = 0; k < 4; k++)
      c->y[4*i + k] = (1.0 - t)*buf.y4[4*j + k] + t*buf.y4[4*(j + 1) + k];
  }
  free (buf.s);
  free (buf.y4);
  return 0;
}

/**
### gle_colloc_march()

Traces the full Fig. 4b branch by marching the target meniscus rise
$\Delta^{*}$ — monotone along the entire branch, so the march needs no fold
handling at all. Secant predictor in $\Delta$ for all unknowns
$(y, \mathrm{Ca}, s_{\mathrm{end}})$, damped-Newton collocation corrector,
adaptive $\mathrm{d}\Delta$ (halve on failure, grow $1.4\times$ on fast
convergence).

Writes one CSV row per accepted point when `csv` is non-`NULL` (same schema
as `gle_branch_csv_header()`), tracks the running $\mathrm{Ca}$ maximum, and
refines the fold location by a quadratic fit of $\mathrm{Ca}(\Delta)$
through the maximum and its neighbours.

#### Parameters
- `c`: seeded collocation problem (see `gle_colloc_seed_from_shoot()`).
- `p`: model parameters.
- `Delta_max`: march until the meniscus rise exceeds this.
- `dDelta0`: initial step in $\Delta$.
- `max_points`: point budget.
- `csv`: optional output stream.
- `fold_Ca`, `fold_Delta`: optional fold estimate outputs.
- `verbose`: progress on `stderr`.

#### Returns
The number of accepted branch points.
*/
static inline int gle_colloc_march (GLECollocation *c, GLEParams *p,
				    double Delta_max, double dDelta0,
				    int max_points, FILE *csv,
				    double *fold_Ca, double *fold_Delta,
				    int verbose) {
  int N = c->N, n = 4*N + 4;
  int npts = 0;
  double dD = dDelta0;
  const double dD_min = 1.0e-7, dD_max = 0.05;

  double *y_good = (double *) malloc (n*sizeof (double));
  double *y_prev = (double *) malloc (n*sizeof (double));
  if (!y_good || !y_prev) {
    free (y_good); free (y_prev);
    return 0;
  }

  /* consistency solve at the seed's own Delta */
  double D = c->y[4*N + 3] + c->y[4*N + 2];
  int iters;
  if (gle_colloc_solve (c, p, D, &iters)) {
    if (verbose)
      fprintf (stderr, "gle_colloc_march: consistency solve failed\n");
    free (y_good); free (y_prev);
    return 0;
  }

  memcpy (y_good, c->y, n*sizeof (double));
  memcpy (y_prev, c->y, n*sizeof (double));
  double Ca_good = c->Ca, S_good = c->s_end, D_good = D;
  double Ca_prev = c->Ca, S_prev = c->s_end, D_prev = D;

  /* fold bookkeeping: full (Delta, Ca) history for the post-fit */
  double *hist_D = (double *) malloc (max_points*sizeof (double));
  double *hist_Ca = (double *) malloc (max_points*sizeof (double));

#define GLE_COLLOC_EMIT()						\
  do {									\
    double th_min = c->y[1];						\
    for (int i = 1; i <= N; i++)					\
      if (c->y[4*i + 1] < th_min)					\
	th_min = c->y[4*i + 1];						\
    double sa = 1.0 - 0.5*c->Delta*c->Delta;				\
    double th_app = (fabs (sa) <= 1.0 ? asin (sa) : NAN);		\
    if (hist_D) {							\
      hist_D[npts] = c->Delta;						\
      hist_Ca[npts] = c->Ca;						\
    }									\
    if (csv) {								\
      GLEBranchPoint b;							\
      b.Ca = c->Ca; b.Delta = c->Delta; b.omega0 = c->y[2];		\
      b.theta_app = th_app; b.theta_min = th_min;			\
      b.s_end = c->s_end; b.residual = 0.0; b.iters = iters;		\
      gle_branch_csv_row (csv, npts, &b);				\
      fflush (csv);							\
    }									\
    npts++;								\
  } while (0)

  GLE_COLLOC_EMIT ();

  while (npts < max_points && D_good < Delta_max) {
    double D_new = D_good + dD;
    /* secant predictor for all unknowns */
    double r = (D_good > D_prev ? (D_new - D_good)/(D_good - D_prev) : 0.0);
    for (int i = 0; i < n; i++)
      c->y[i] = y_good[i] + r*(y_good[i] - y_prev[i]);
    c->Ca = Ca_good + r*(Ca_good - Ca_prev);
    c->s_end = S_good + r*(S_good - S_prev);

    if (gle_colloc_solve (c, p, D_new, &iters)) {
      dD *= 0.5;
      if (verbose >= 2)
	fprintf (stderr, "    colloc fail at Delta = %.5f, dDelta -> %.3e\n",
		 D_new, dD);
      /* restore last good state (also on final exit, so the caller sees a
	 converged solution, not the failed iterate) */
      memcpy (c->y, y_good, n*sizeof (double));
      c->Ca = Ca_good;
      c->s_end = S_good;
      c->Delta = D_good;
      if (dD < dD_min) {
	if (verbose)
	  fprintf (stderr, "gle_colloc_march: step underflow at "
		   "Delta = %.5f (Ca = %.6e)\n", D_good, Ca_good);
	break;
      }
      continue;
    }

    /* accept */
    memcpy (y_prev, y_good, n*sizeof (double));
    Ca_prev = Ca_good; S_prev = S_good; D_prev = D_good;
    memcpy (y_good, c->y, n*sizeof (double));
    Ca_good = c->Ca; S_good = c->s_end; D_good = D_new;

    GLE_COLLOC_EMIT ();

    if (verbose && npts % 20 == 0)
      fprintf (stderr, "  point %4d: Ca = %.6e  Delta = %.4f  "
	       "dDelta = %.3g  iters = %d\n",
	       npts, c->Ca, c->Delta, dD, iters);

    if (iters <= 5 && dD < dD_max)
      dD *= 1.4;
  }

  /* fold refinement: quadratic vertex of Ca(Delta) around the history
     maximum */
  if ((fold_Ca || fold_Delta) && hist_D && npts >= 3) {
    int im = 0;
    for (int i = 1; i < npts; i++)
      if (hist_Ca[i] > hist_Ca[im])
	im = i;
    int i1 = (im > 0 ? im - 1 : im), i3 = (im < npts - 1 ? im + 1 : im);
    double d1 = hist_D[i1], d2 = hist_D[im], d3 = hist_D[i3];
    double c1 = hist_Ca[i1], c2 = hist_Ca[im], c3 = hist_Ca[i3];
    double fCa = c2, fD = d2;
    double denom = (d1 - d2)*(d1 - d3)*(d2 - d3);
    if (fabs (denom) > 0.0) {
      double A = (d3*(c2 - c1) + d2*(c1 - c3) + d1*(c3 - c2))/denom;
      double B = (d3*d3*(c1 - c2) + d2*d2*(c3 - c1) + d1*d1*(c2 - c3))/denom;
      if (A != 0.0) {
	fD = -B/(2.0*A);
	double C0 = c1 - A*d1*d1 - B*d1;
	fCa = A*fD*fD + B*fD + C0;
      }
    }
    if (fold_Ca) *fold_Ca = fCa;
    if (fold_Delta) *fold_Delta = fD;
    if (verbose)
      fprintf (stderr, "gle_colloc_march: fold at Ca* = %.6e, "
	       "Delta* = %.4f\n", fCa, fD);
  }
#undef GLE_COLLOC_EMIT
  free (hist_D);
  free (hist_Ca);
  free (y_good);
  free (y_prev);
  return npts;
}

#endif /* GLE_COLLOCATE_H */
