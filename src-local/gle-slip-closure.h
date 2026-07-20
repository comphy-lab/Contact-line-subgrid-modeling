/**
# gle-slip-closure.h — case-level cutoff closure for the Chan GLE

Resolves the Cox matching constant once per case and converts it into the
constant $c_\lambda$ required by [gle-model-chan.h](gle-model-chan.h):

$$
Q=\ln\!\left(\frac{\sin\theta_e}{c_\lambda}\right)+1,
\qquad
c_\lambda=\sin\theta_e\,\exp(1-Q).
$$

The automatic policy tries, in order, the Scott--Hocking one-phase reference
interpolation, the corrected right-angle branch, a reference-table hook, and
the explicitly approximate Luo--Gao formula. The one-phase reference and
frozen table are isolated in
[gle-slip-reference.h](gle-slip-reference.h), keeping their data provenance
separate from the model dispatcher.
*/

#ifndef GLE_SLIP_CLOSURE_H
#define GLE_SLIP_CLOSURE_H

#include <float.h>
#include "gle-model.h"

/**
## Closure status and provenance
*/
enum gle_cutoff_status {
  GLE_CUTOFF_OK = 0,
  GLE_CUTOFF_NOT_USED = 1,
  GLE_CUTOFF_UNAVAILABLE = 2,
  GLE_CUTOFF_DOMAIN = 3,
  GLE_CUTOFF_NUMERIC = 4
};

typedef struct {
  double Q;
  double log_c;
  double c;
  int method;          /* resolved enum gle_cutoff_method                    */
  int status;          /* enum gle_cutoff_status                             */
  int luo_gao_approximation; /* non-zero only for Luo--Gao fallback          */
} GLECutoffResult;

static inline void gle_cutoff_result_reset (GLECutoffResult *out, int method,
					     int status) {
  if (!out)
    return;
  out->Q = out->log_c = out->c = NAN;
  out->method = method;
  out->status = status;
  out->luo_gao_approximation = 0;
}

static inline int gle_cutoff_from_Q (double theta_e, double Q, int method,
				      int luo_gao_approximation,
				      GLECutoffResult *out) {
  if (!out || !isfinite (theta_e) || theta_e <= 0.0 || theta_e >= M_PI ||
      !isfinite (Q)) {
    gle_cutoff_result_reset (out, method, GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  double log_c = log (sin (theta_e)) + 1.0 - Q;
  double c = exp (log_c);
  if (!isfinite (log_c) || !isfinite (c) || c <= 0.0) {
    gle_cutoff_result_reset (out, method, GLE_CUTOFF_NUMERIC);
    return GLE_CUTOFF_NUMERIC;
  }
  out->Q = Q;
  out->log_c = log_c;
  out->c = c;
  out->method = method;
  out->status = GLE_CUTOFF_OK;
  out->luo_gao_approximation = luo_gao_approximation;
  return GLE_CUTOFF_OK;
}

#include "gle-slip-reference.h"

/**
### gle_cutoff_scott_hocking()

Public named entry point for the numerically converged Scott--Hocking
one-phase reference. The analytic Hocking right-angle anchor, generated table
and public provenance live in [gle-slip-reference.h](gle-slip-reference.h).
*/
static inline int gle_cutoff_scott_hocking (double theta_e,
					    GLECutoffResult *out) {
  return gle_cutoff_scott_hocking_reference (theta_e, out);
}

/**
### gle_cutoff_manual()

Records a caller-supplied Chan coefficient while still exposing its compatible
$Q$ and $\log c$ for provenance.
*/
static inline int gle_cutoff_manual (double theta_e, double c,
				      GLECutoffResult *out) {
  if (!isfinite (theta_e) || theta_e <= 0.0 || theta_e >= M_PI ||
      !isfinite (c) || c <= 0.0) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_MANUAL, GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  double Q = log (sin (theta_e)/c) + 1.0;
  return gle_cutoff_from_Q (theta_e, Q, GLE_CUTOFF_MANUAL, 0, out);
}

/**
### gle_cutoff_corrected_right_angle()

Evaluates the corrected equal-slip right-angle branch. The underlying helper
uses viscosity-ratio inversion symmetry to remain well conditioned for large
ratios.
*/
static inline int gle_cutoff_corrected_right_angle (double theta_e,
					     double mu_r,
					     GLECutoffResult *out) {
  double tol = 64.0*DBL_EPSILON*fmax (1.0, fabs (theta_e));
  if (!isfinite (theta_e) || fabs (theta_e - 0.5*M_PI) > tol ||
      !isfinite (mu_r) || mu_r < 0.0) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_CORRECTED_RIGHT_ANGLE,
			     GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  double c = gle_slip_prefactor_right_angle (mu_r);
  if (!isfinite (c) || c <= 0.0) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_CORRECTED_RIGHT_ANGLE,
			     GLE_CUTOFF_NUMERIC);
    return GLE_CUTOFF_NUMERIC;
  }
  return gle_cutoff_from_Q (theta_e, 1.0 - log (c),
			    GLE_CUTOFF_CORRECTED_RIGHT_ANGLE, 0, out);
}

/**
### gle_luo_gao_matching_log_term()

Evaluates the logarithmic last term of Luo & Gao (2025), Eq. (4.10), without
loss of precision at either $x=\Omega/B\to0$ or $x\to1$. The first limit uses
$\operatorname{atanh}(x)/x$; the second uses
$\ln[(B+\Omega)/(B-\Omega)]=2\ln(1+x)-\ln(4AC/B^2)$.
The discriminant and the potentially cancelling coefficient
$(2AE-BD)/(BD)$ are formed in scaled form, using `fma()` when its ratios are
representable and logarithms otherwise.
*/
static inline int
gle_luo_gao_matching_log_term (const GLELuoGaoCoefficients *q,
				 double *term) {
  if (!q || !term || !isfinite (q->a) || !isfinite (q->b) ||
      !isfinite (q->c) || !isfinite (q->d) || !isfinite (q->e) ||
      q->a <= 0.0 || q->b <= 0.0 || q->c <= 0.0 || q->d <= 0.0 ||
      q->e <= 0.0)
    return 1;

  /* z = 4AC/B^2 lies in (0,1] on the physical branch. Retain log(z):
     when z is too small to represent, it still gives the finite x -> 1
     logarithm below. */
  double log_z = log (4.0) + log (q->a) + log (q->c) - 2.0*log (q->b);
  double log_scale = 1.0 + fabs (log (q->a)) + fabs (log (q->b)) +
    fabs (log (q->c));
  double roundoff = 128.0*DBL_EPSILON*log_scale;
  if (!isfinite (log_z) || log_z > roundoff)
    return 1;
  if (log_z > 0.0)
    log_z = 0.0;
  double z = exp (log_z);
  double disc_scaled = -expm1 (log_z); /* 1 - z, accurate when z -> 1 */
  if (disc_scaled < 0.0 && disc_scaled >= -roundoff)
    disc_scaled = 0.0;
  if (disc_scaled > 1.0 && disc_scaled <= 1.0 + roundoff)
    disc_scaled = 1.0;
  if (!isfinite (z) || !isfinite (disc_scaled) || disc_scaled < 0.0 ||
      disc_scaled > 1.0)
    return 1;

  double x = sqrt (disc_scaled);
  /* r = (2AE-BD)/(BD). Ratios plus one fused subtraction avoid the
     cancellation in the common r ~= 0 case without overflowing AE or BD. */
  double a_over_b = q->a/q->b, e_over_d = q->e/q->d;
  double r = NAN;
  if (isfinite (a_over_b) && isfinite (e_over_d) &&
      isfinite (2.0*a_over_b))
    r = fma (2.0*a_over_b, e_over_d, -1.0);
  if (!isfinite (r)) {
    double log_ratio = log (2.0) + log (q->a) + log (q->e) -
      log (q->b) - log (q->d);
    if (!isfinite (log_ratio) || log_ratio > log (DBL_MAX))
      return 1;
    r = (log_ratio < log (DBL_MIN) ? -1.0 : expm1 (log_ratio));
  }

  if (x <= 32.0*DBL_EPSILON) {
    double x2 = x*x;
    *term = r*(1.0 + x2/3.0 + x2*x2/5.0);
  }
  else if (x < 0.9)
    *term = r*(atanh (x)/x);
  else {
    /* This remains finite when x rounds to one because log_z was retained. */
    double log_ratio = 2.0*log1p (x) - log_z;
    *term = r*log_ratio/(2.0*x);
  }
  return !isfinite (*term);
}

/**
### gle_cutoff_luo_gao_approx()

Evaluates Luo & Gao (2025), Eq. (4.10), as an explicitly approximate closure.
The one-phase reduction uses their Eq. (4.11) directly. The logarithmic ratio
in the two-phase expression is evaluated with `log1p()` and its regular limit
is used when the discriminant vanishes.
*/
static inline int gle_cutoff_luo_gao_approx (double theta_e, double mu_r,
					      GLECutoffResult *out) {
  if (!isfinite (theta_e) || theta_e <= 0.0 || theta_e >= M_PI ||
      !isfinite (mu_r) || mu_r < 0.0) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_LUO_GAO_APPROX,
			     GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }

  double Q;
  if (mu_r == 0.0) {
    double sth = sin (theta_e);
    double ratio = gle_f2 (theta_e)/(2.0*sth*sth);
    if (!isfinite (ratio) || ratio <= 0.0) {
      gle_cutoff_result_reset (out, GLE_CUTOFF_LUO_GAO_APPROX,
			       GLE_CUTOFF_NUMERIC);
      return GLE_CUTOFF_NUMERIC;
    }
    Q = log (ratio) + 1.0;
  }
  else {
    /* The matching constant is invariant under phase exchange
       (theta, M) -> (pi - theta, 1/M). Canonicalising to M <= 1 before
       forming Luo--Gao's raw coefficients prevents the M^2 terms from
       overflowing at otherwise valid large viscosity ratios. Reconstruct
       c below with the caller's original angle. */
    double theta_match = theta_e;
    double mu_match = mu_r;
    if (mu_match > 1.0) {
      theta_match = M_PI - theta_match;
      mu_match = 1.0/mu_match;
    }
    GLELuoGaoCoefficients q;
    if (gle_luo_gao_coefficients (theta_match, mu_match, &q) ||
	q.a <= 0.0 || q.b <= 0.0 || q.c <= 0.0 || q.d <= 0.0) {
      gle_cutoff_result_reset (out, GLE_CUTOFF_LUO_GAO_APPROX,
			       GLE_CUTOFF_NUMERIC);
      return GLE_CUTOFF_NUMERIC;
    }
    double term;
    if (gle_luo_gao_matching_log_term (&q, &term)) {
      gle_cutoff_result_reset (out, GLE_CUTOFF_LUO_GAO_APPROX,
			       GLE_CUTOFF_NUMERIC);
      return GLE_CUTOFF_NUMERIC;
    }
    Q = 1.0 + 0.5*(log (q.a) - log (q.c)) + term;
  }
  return gle_cutoff_from_Q (theta_e, Q, GLE_CUTOFF_LUO_GAO_APPROX,
			    1, out);
}

/**
### gle_cutoff_resolve()

Resolves the requested Chan cutoff policy. `GLE_CUTOFF_AUTO` falls back only
when an exact/reference hook reports that data is unavailable; numerical errors
from an installed reference are not hidden by the approximate fallback.
*/
static inline int gle_cutoff_resolve (GLEParams *p, GLECutoffResult *out) {
  if (!p || !out)
    return GLE_CUTOFF_DOMAIN;
  if (!isfinite (p->mu_r) || p->mu_r < 0.0) {
    gle_cutoff_result_reset (out, p->cutoff_method, GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  int st;
  switch (p->cutoff_method) {
  case GLE_CUTOFF_MANUAL:
    st = gle_cutoff_manual (p->theta_mic, p->c_slip, out);
    break;
  case GLE_CUTOFF_SCOTT_HOCKING:
    if (p->mu_r != 0.0) {
      gle_cutoff_result_reset (out, p->cutoff_method, GLE_CUTOFF_DOMAIN);
      return GLE_CUTOFF_DOMAIN;
    }
    st = gle_cutoff_scott_hocking (p->theta_mic, out);
    break;
  case GLE_CUTOFF_CORRECTED_RIGHT_ANGLE:
    st = gle_cutoff_corrected_right_angle (p->theta_mic, p->mu_r, out);
    break;
  case GLE_CUTOFF_REFERENCE_TABLE:
    st = gle_cutoff_reference_table (p->theta_mic, p->mu_r, out);
    break;
  case GLE_CUTOFF_LUO_GAO_APPROX:
    st = gle_cutoff_luo_gao_approx (p->theta_mic, p->mu_r, out);
    break;
  case GLE_CUTOFF_AUTO: {
    if (p->mu_r == 0.0) {
      st = gle_cutoff_scott_hocking (p->theta_mic, out);
      if (st == GLE_CUTOFF_OK)
	break;
      if (st != GLE_CUTOFF_UNAVAILABLE)
	return st;
    }
    double tol = 64.0*DBL_EPSILON*fmax (1.0, fabs (p->theta_mic));
    if (fabs (p->theta_mic - 0.5*M_PI) <= tol) {
      st = gle_cutoff_corrected_right_angle (p->theta_mic, p->mu_r, out);
      if (st == GLE_CUTOFF_OK)
	break;
      return st;
    }
    st = gle_cutoff_reference_table (p->theta_mic, p->mu_r, out);
    if (st == GLE_CUTOFF_OK)
      break;
    if (st != GLE_CUTOFF_UNAVAILABLE)
      return st;
    st = gle_cutoff_luo_gao_approx (p->theta_mic, p->mu_r, out);
    if (st != GLE_CUTOFF_OK)
      return st;
    break;
  }
  default:
    gle_cutoff_result_reset (out, p->cutoff_method, GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  if (st == GLE_CUTOFF_OK)
    p->c_slip = out->c;
  return st;
}

/**
### gle_model_prepare()

Resolves a Chan cutoff once after all case parameters are known. The direct
Luo--Gao model deliberately reports `GLE_CUTOFF_NOT_USED` and leaves
`c_slip` irrelevant.
*/
static inline int gle_model_prepare (GLEParams *p, GLECutoffResult *out) {
  if (!p || !out)
    return GLE_CUTOFF_DOMAIN;
  if (gle_model_common_validate (p)) {
    gle_cutoff_result_reset (out, p->cutoff_method, GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  if (p->model == GLE_MODEL_CHAN)
    return gle_cutoff_resolve (p, out);
  if (p->model == GLE_MODEL_LUO_GAO) {
    gle_cutoff_result_reset (out, p->cutoff_method, GLE_CUTOFF_NOT_USED);
    return GLE_CUTOFF_OK;
  }
  gle_cutoff_result_reset (out, p->cutoff_method, GLE_CUTOFF_DOMAIN);
  return GLE_CUTOFF_DOMAIN;
}

/**
### gle_model_prepare_copy()

Copies a caller-owned parameter set and resolves the model on the copy. Public
solver entry points use this once at their boundary, then pass the prepared
copy through all inner iterations. This both preserves the caller's inputs and
keeps reference-table work out of the RHS hot path.
*/
static inline int gle_model_prepare_copy (const GLEParams *input,
					   GLEParams *prepared,
					   GLECutoffResult *out) {
  if (!input || !prepared || !out) {
    gle_cutoff_result_reset (out, GLE_CUTOFF_MANUAL, GLE_CUTOFF_DOMAIN);
    return GLE_CUTOFF_DOMAIN;
  }
  *prepared = *input;
  return gle_model_prepare (prepared, out);
}

static inline const char *gle_cutoff_status_name (int status) {
  switch (status) {
  case GLE_CUTOFF_OK: return "ok";
  case GLE_CUTOFF_NOT_USED: return "not_used";
  case GLE_CUTOFF_UNAVAILABLE: return "unavailable";
  case GLE_CUTOFF_DOMAIN: return "domain_error";
  case GLE_CUTOFF_NUMERIC: return "numeric_error";
  default: return "invalid";
  }
}

#endif /* GLE_SLIP_CLOSURE_H */
