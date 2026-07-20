/**
# gle-slip-reference.h — hooks for independent cutoff references

Defines the two reference sources used by the automatic Chan cutoff workflow:
the Scott--Hocking one-phase branch and interpolation from an independently
generated two-phase table. The one-phase branch consumes a converged frozen
solution of Scott's public integral problem and its endpoint asymptotics. The
finite-viscosity branch consumes the frozen two-phase output of
`gle-ode/reference-generator/`; no reference solve or third-party dependency
enters this header.

Include [gle-slip-closure.h](gle-slip-closure.h), not this hook header directly.
*/

#ifndef GLE_SLIP_REFERENCE_H
#define GLE_SLIP_REFERENCE_H

#ifndef GLE_SLIP_CLOSURE_H
# error "include gle-slip-closure.h instead of gle-slip-reference.h"
#endif

#include "gle-slip-scott-data.h"
#include "gle-slip-table-data.h"

/**
### gle_cutoff_scott_hocking_reference()

Evaluates the arbitrary-angle, `mu_r == 0` Scott--Hocking reference obtained by
solving the singular integral problem of Julian F. Scott, "Calculation of a
key function in the asymptotic description of moving contact lines",
*Q. J. Mech. Appl. Math.* **73**, 279--291 (2020):

- DOI: https://doi.org/10.1093/qjmam/hbaa012
- open author manuscript: https://hal.science/hal-03227614v1

The public reference generator in `gle-ode/reference-generator/scott_hocking.py`
solves Scott's equations (2.1)--(2.3), evaluates (2.11), and freezes a dense
nonuniform table. Each node is compared with coarser and wider-domain solves;
direct solves at every interior cell midpoint independently check the runtime
interpolation. The implementation applies a shape-preserving cubic Hermite
interpolation (PCHIP) to the regularised quantities

$$
R_0=Q_i-\ln(\alpha/3),\quad 0\leq\alpha\leq\pi/2,
\qquad
R_\pi=Q_i-\frac{\pi}{\pi-\alpha},\quad
\pi/2\leq\alpha\leq\pi.
$$

The analytic Hocking anchor $Q_i(\pi/2)=\gamma_E-\ln 2$ joins the two intervals.
Their other endpoints use Scott's published asymptotics:

$$
Q_i\sim\ln(\alpha/3)+0.156\alpha^2\quad(\alpha\to0),\qquad
Q_i\sim\frac{\pi}{\pi-\alpha}+\gamma_E-\ln2-2\quad(\alpha\to\pi).
$$

The lower endpoint tangent enforces the coefficient $0.156$ explicitly. This
is a numerically converged Scott--Hocking reference, not an analytic
arbitrary-angle formula: only the right-angle anchor is analytic. The frozen
70-node table has maximum node sensitivity $6.62\times10^{-5}$ in $Q_i$; 71
off-node solves give maximum interpolation discrepancy $4.25\times10^{-5}$.
It reproduces Scott's published Table 1 to $3.11\times10^{-6}$. The Chan
convention is $Q=1+Q_i$ and hence $\ln c_\lambda=\ln(\sin\alpha)-Q_i$.
*/
static inline double gle_pchip_endpoint_slope (double width0, double width1,
						double secant0,
						double secant1) {
  double slope = ((2.0*width0 + width1)*secant0 - width0*secant1)/
    (width0 + width1);
  if (secant0 == 0.0 || (slope > 0.0) != (secant0 > 0.0))
    return 0.0;
  if ((secant0 > 0.0) != (secant1 > 0.0) &&
      fabs (slope) > 3.0*fabs (secant0))
    return 3.0*secant0;
  return slope;
}

/** Evaluate a generated Fritsch--Carlson PCHIP without external dependencies. */
static inline double gle_pchip_eval (const double *angle,
				      const double *value, int count,
				      double query, int enforce_small_angle) {
  if (!angle || !value || count < 3 || count > GLE_SCOTT_UPPER_COUNT)
    return NAN;
  double width[GLE_SCOTT_UPPER_COUNT - 1];
  double secant[GLE_SCOTT_UPPER_COUNT - 1];
  double tangent[GLE_SCOTT_UPPER_COUNT];
  for (int i = 0; i < count - 1; i++) {
    width[i] = angle[i + 1] - angle[i];
    secant[i] = (value[i + 1] - value[i])/width[i];
  }
  tangent[0] = gle_pchip_endpoint_slope (width[0], width[1],
					  secant[0], secant[1]);
  for (int i = 1; i < count - 1; i++) {
    if ((secant[i - 1] > 0.0) == (secant[i] > 0.0) &&
	secant[i - 1] != 0.0 && secant[i] != 0.0) {
      double weight1 = 2.0*width[i] + width[i - 1];
      double weight2 = width[i] + 2.0*width[i - 1];
      tangent[i] = (weight1 + weight2)/
	(weight1/secant[i - 1] + weight2/secant[i]);
    }
    else
      tangent[i] = 0.0;
  }
  tangent[count - 1] = gle_pchip_endpoint_slope (
    width[count - 2], width[count - 3],
    secant[count - 2], secant[count - 3]);

  if (enforce_small_angle) {
    tangent[0] = 0.0;
    /* For the first Hermite interval, this tangent makes the quadratic
	 coefficient of R_0 exactly Scott's published 0.156. */
    tangent[1] = 3.0*value[1]/width[0] - 0.156*width[0];
  }

  int i = 0;
  while (i < count - 2 && query > angle[i + 1])
    i++;
  double t = (query - angle[i])/width[i];
  double t2 = t*t, t3 = t2*t;
  return (2.0*t3 - 3.0*t2 + 1.0)*value[i] +
    (t3 - 2.0*t2 + t)*width[i]*tangent[i] +
    (-2.0*t3 + 3.0*t2)*value[i + 1] +
    (t3 - t2)*width[i]*tangent[i + 1];
}

static inline int
gle_cutoff_scott_hocking_reference (double theta_e,
				      GLECutoffResult *out) {
  if (!isfinite (theta_e) || theta_e <= 0.0 || theta_e >= M_PI) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_SCOTT_HOCKING,
			     GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }

  double qi = theta_e <= 0.5*M_PI ?
    log (theta_e/3.0) + gle_pchip_eval (
      gle_scott_lower_theta, gle_scott_lower_regular,
      GLE_SCOTT_LOWER_COUNT, theta_e, 1) :
    M_PI/(M_PI - theta_e) + gle_pchip_eval (
      gle_scott_upper_theta, gle_scott_upper_regular,
      GLE_SCOTT_UPPER_COUNT, theta_e, 0);
  return gle_cutoff_from_Q (theta_e, 1.0 + qi,
			    GLE_CUTOFF_SCOTT_HOCKING, 0, out);
}

/**
### gle_cutoff_reference_table()

Interpolates the matching quantity $Q$ with a local cubic Lagrange polynomial
in $\theta_e$ and a local quartic Lagrange polynomial in
$\log_{10}\mu_r$, then reconstructs $c$. The frozen data cover the
$\mu_r\leq1$ half-plane; for $\mu_r>1$ the exact phase-exchange identity

$$
Q(\theta_e,\mu_r)=Q(\pi-\theta_e,1/\mu_r)
$$

maps the query into that half-plane. The function never extrapolates. A valid
physical query outside the frozen grid reports `GLE_CUTOFF_UNAVAILABLE`, so
the automatic policy may select its explicitly labelled Luo--Gao fallback.
Invalid physical input reports `GLE_CUTOFF_DOMAIN`; malformed frozen data
report `GLE_CUTOFF_NUMERIC` and are never hidden by the fallback.
*/
static inline int gle_reference_bracket (const double *grid, int count,
					 double query, int *index,
					 double *weight) {
  if (!grid || !index || !weight || count < 2 || !isfinite (query))
    return GLE_CUTOFF_NUMERIC;
  for (int i = 0; i < count; i++)
    if (!isfinite (grid[i]) || (i && !(grid[i] > grid[i - 1])))
      return GLE_CUTOFF_NUMERIC;

  /* Degree/radian and log10 round trips can move an exact endpoint by a few
     ulps. Clamping only within this roundoff band is not extrapolation. */
  double scale = fmax (1.0, fmax (fabs (grid[0]), fabs (grid[count - 1])));
  double tol = 64.0*DBL_EPSILON*scale;
  if (query < grid[0]) {
    if (query < grid[0] - tol)
      return GLE_CUTOFF_UNAVAILABLE;
    query = grid[0];
  }
  if (query > grid[count - 1]) {
    if (query > grid[count - 1] + tol)
      return GLE_CUTOFF_UNAVAILABLE;
    query = grid[count - 1];
  }

  if (query == grid[count - 1]) {
    *index = count - 2;
    *weight = 1.0;
    return GLE_CUTOFF_OK;
  }
  int i = 0;
  while (i < count - 2 && query > grid[i + 1])
    i++;
  double width = grid[i + 1] - grid[i];
  double w = (query - grid[i])/width;
  if (!isfinite (width) || width <= 0.0 || !isfinite (w) ||
      w < 0.0 || w > 1.0)
    return GLE_CUTOFF_NUMERIC;
  *index = i;
  *weight = w;
  return GLE_CUTOFF_OK;
}

enum {
  GLE_REFERENCE_THETA_STENCIL = GLE_SLIP_TABLE_THETA_STENCIL_NODES,
  GLE_REFERENCE_LOGM_STENCIL = GLE_SLIP_TABLE_LOGM_STENCIL_NODES,
  GLE_REFERENCE_MAX_STENCIL = 5
};

#if GLE_SLIP_TABLE_THETA_STENCIL_NODES > 5 || \
    GLE_SLIP_TABLE_LOGM_STENCIL_NODES > 5
# error "generated reference-table stencil exceeds the runtime workspace"
#endif

static inline int gle_reference_lagrange_weights (const double *grid,
					   int count, int requested_count,
					   int cell_index,
					   double query, int *start,
					   int *stencil_count,
					   double weight[GLE_REFERENCE_MAX_STENCIL]) {
  if (!grid || !start || !stencil_count || !weight || count < 2 ||
	      requested_count < 2 ||
	      requested_count > GLE_REFERENCE_MAX_STENCIL ||
	      cell_index < 0 || cell_index >= count - 1 || !isfinite (query))
    return GLE_CUTOFF_NUMERIC;
  int n = count < requested_count ? count : requested_count;
  int first = cell_index - (n/2 - 1);
  if (first < 0)
    first = 0;
  if (first > count - n)
    first = count - n;
  for (int local_i = 0; local_i < n; local_i++) {
    int i = first + local_i;
    double numerator = 1.0, denominator = 1.0;
    for (int local_j = 0; local_j < n; local_j++) {
      int j = first + local_j;
      if (i == j)
	continue;
      numerator *= query - grid[j];
      denominator *= grid[i] - grid[j];
    }
    weight[local_i] = numerator/denominator;
    if (!isfinite (weight[local_i]))
      return GLE_CUTOFF_NUMERIC;
  }
  *start = first;
  *stencil_count = n;
  return GLE_CUTOFF_OK;
}

static inline int gle_cutoff_reference_table (double theta_e, double mu_r,
					       GLECutoffResult *out) {
  if (!out || !isfinite (theta_e) || theta_e <= 0.0 || theta_e >= M_PI ||
      !isfinite (mu_r) || mu_r < 0.0) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_REFERENCE_TABLE,
			     GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  if (mu_r == 0.0) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_REFERENCE_TABLE,
			     GLE_CUTOFF_UNAVAILABLE);
    return GLE_CUTOFF_UNAVAILABLE;
  }

  double theta_deg = theta_e*180.0/M_PI;
  double log_m = log10 (mu_r);
  if (log_m > 0.0) {
    theta_deg = 180.0 - theta_deg;
    log_m = -log_m;
  }
  if (!isfinite (theta_deg) || !isfinite (log_m)) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_REFERENCE_TABLE,
			     GLE_CUTOFF_NUMERIC);
    return GLE_CUTOFF_NUMERIC;
  }

  int itheta, im;
  double wt, wm;
  int status = gle_reference_bracket (gle_slip_table_theta_deg,
				      GLE_SLIP_TABLE_THETA_COUNT,
				      theta_deg, &itheta, &wt);
  if (status == GLE_CUTOFF_OK)
    status = gle_reference_bracket (gle_slip_table_log10_m,
				    GLE_SLIP_TABLE_LOGM_COUNT,
				    log_m, &im, &wm);
  if (status != GLE_CUTOFF_OK) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_REFERENCE_TABLE, status);
    return status;
  }

  int theta_start, logm_start, theta_count, logm_count;
  double theta_weight[GLE_REFERENCE_MAX_STENCIL];
  double logm_weight[GLE_REFERENCE_MAX_STENCIL];
  status = gle_reference_lagrange_weights (gle_slip_table_theta_deg,
					   GLE_SLIP_TABLE_THETA_COUNT,
					   GLE_REFERENCE_THETA_STENCIL,
					   itheta, theta_deg,
					   &theta_start, &theta_count,
					   theta_weight);
  if (status == GLE_CUTOFF_OK)
    status = gle_reference_lagrange_weights (gle_slip_table_log10_m,
					     GLE_SLIP_TABLE_LOGM_COUNT,
					     GLE_REFERENCE_LOGM_STENCIL,
					     im, log_m,
					     &logm_start, &logm_count,
					     logm_weight);
  if (status != GLE_CUTOFF_OK) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_REFERENCE_TABLE, status);
    return status;
  }

  double Q = 0.0;
  for (int local_i = 0; local_i < theta_count; local_i++)
    for (int local_j = 0; local_j < logm_count; local_j++) {
      double node = gle_slip_table_Q[theta_start + local_i]
	[logm_start + local_j];
      if (!isfinite (node)) {
	gle_cutoff_result_reset (out, GLE_CUTOFF_REFERENCE_TABLE,
				 GLE_CUTOFF_NUMERIC);
	return GLE_CUTOFF_NUMERIC;
      }
      Q += theta_weight[local_i]*logm_weight[local_j]*node;
    }
  if (!isfinite (Q)) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_REFERENCE_TABLE,
			     GLE_CUTOFF_NUMERIC);
    return GLE_CUTOFF_NUMERIC;
  }
  return gle_cutoff_from_Q (theta_e, Q, GLE_CUTOFF_REFERENCE_TABLE, 0, out);
}

#endif /* GLE_SLIP_REFERENCE_H */
