/**
# gle-model-luo-gao.h — Luo--Gao slippery-wedge GLE

Implements Eq. (3.16) of Luo & Gao, *J. Fluid Mech.* **1019**, A52
(2025), with the sign converted to this repository's convention that positive
`Ca` denotes a receding contact line. Their local slippery-wedge approximation
incorporates constant Navier slip directly and therefore uses no fitted scalar
$c_\lambda$.

Include [gle-model.h](gle-model.h), not this implementation header directly.
*/

#ifndef GLE_MODEL_LUO_GAO_H
#define GLE_MODEL_LUO_GAO_H

#ifndef GLE_MODEL_H
# error "include gle-model.h instead of gle-model-luo-gao.h"
#endif

/**
## Luo--Gao coefficient set

The five elementary functions are those of Luo & Gao (2025), Eq. (3.17):

$$
\begin{aligned}
A &= M f_1(\theta)f_2(\pi-\theta)+f_1(\pi-\theta)f_2(\theta),\\
B &= 2\sin^2\theta[Mf_1(\theta)+f_1(\pi-\theta)]
   +2(M+1)f_2(\theta)f_2(\pi-\theta),\\
C &= 4\sin^2\theta[Mf_2(\theta)+f_2(\pi-\theta)],\\
D &= 2\sin\theta[M^2f_1(\theta)+2Mf_3(\theta)+f_1(\pi-\theta)],\\
E &= 4\sin\theta[M^2f_2(\theta)+M\pi+f_2(\pi-\theta)].
\end{aligned}
$$

In particular, $E$ has one external factor of $\sin\theta$ and an $M^2$
coefficient on $f_2(\theta)$. These source-sensitive powers are covered by
the core regression tests. The coefficients are also used by the explicit
matching approximation in [gle-slip-closure.h](gle-slip-closure.h).
*/
typedef struct {
  double a, b, c, d, e;
} GLELuoGaoCoefficients;

static inline int gle_luo_gao_coefficients (double th, double mu_r,
					     GLELuoGaoCoefficients *q) {
  if (!q || !isfinite (th) || !isfinite (mu_r) || mu_r < 0.0 ||
      th <= -0.5*M_PI || th >= M_PI || (mu_r > 0.0 && th <= 0.0))
    return 1;
  double sth = sin (th), s2 = sth*sth;
  double f1t = gle_f1 (th), f2t = gle_f2 (th);
  double f1p = gle_f1 (M_PI - th), f2p = gle_f2 (M_PI - th);
  double f3t = gle_f3 (th);
  q->a = mu_r*f1t*f2p + f1p*f2t;
  q->b = 2.0*s2*(mu_r*f1t + f1p)
    + 2.0*(mu_r + 1.0)*f2t*f2p;
  q->c = 4.0*s2*(mu_r*f2t + f2p);
  q->d = 2.0*sth*(mu_r*mu_r*f1t + 2.0*mu_r*f3t + f1p);
  q->e = 4.0*sth*(mu_r*mu_r*f2t + mu_r*M_PI + f2p);
  return !(isfinite (q->a) && isfinite (q->b) && isfinite (q->c) &&
	   isfinite (q->d) && isfinite (q->e));
}

/**
### gle_luo_gao_one_phase_factor()

Returns the one-phase local slip factor

$$F_{\rm LG}(\theta)=\frac{2\sin^3\theta}
 {\theta-\sin\theta\cos\theta},$$

using its exact even extension and the removable value $F_{\rm LG}(0)=3$.
*/
static inline double gle_luo_gao_one_phase_factor (double th) {
  double ta = fabs (th);
  if (!isfinite (ta) || ta >= M_PI)
    return NAN;
  if (ta == 0.0)
    return 3.0;
  double sth = sin (ta);
  return 2.0*sth*sth*sth/gle_f2 (ta);
}

/**
### gle_luo_gao_viscous_gradient()

Evaluates the slip-resolved viscous contribution

$$
\left.\frac{\mathrm{d}\omega}{\mathrm{d}s}\right|_{\rm visc}
=\frac{\mathrm{Ca}}{h}\,
\frac{D h\sin^2\theta+E\lambda\sin^3\theta}
 {A h^2+B\lambda h\sin\theta+C\lambda^2\sin^2\theta}.
$$

Unlike Chan's case-constant $c_\lambda$, all five coefficients are evaluated
at the current local angle on every right-hand-side call. For `mu_r == 0`, the
analytically reduced one-phase expression has an exact even extension and is
therefore valid for the small negative angles in a Landau--Levich tail. At
finite `mu_r`, Eq. (3.16) is retained only on its published domain
$0<\theta<\pi$; a zero or negative local angle is rejected rather than
silently extrapolating the two-phase wedge. The positive-side small-slope
limit remains $3\,\mathrm{Ca}/[h(h+3\lambda)]$.

#### Returns
`0` on success and `1` for a non-finite or singular state.
*/
static inline int gle_luo_gao_viscous_gradient (const GLEParams *p, double hf,
						 double th, double *visc) {
  if (!p || !visc || !isfinite (hf) || hf <= 0.0 ||
      !isfinite (p->slip) || p->slip <= 0.0)
    return 1;
  if (!isfinite (p->Ca) || !isfinite (p->mu_r) || p->mu_r < 0.0)
    return 1;

  if (p->mu_r == 0.0) {
    double flg = gle_luo_gao_one_phase_factor (th);
    double den = hf*(hf + flg*p->slip);
    if (!isfinite (flg) || !isfinite (den) || den <= 0.0)
      return 1;
    *visc = p->Ca*flg/den;
    return !isfinite (*visc);
  }

  if (th <= 0.0)
    return 1;

  GLELuoGaoCoefficients q;
  if (gle_luo_gao_coefficients (th, p->mu_r, &q))
    return 1;
  double sth = sin (th), s2 = sth*sth;
  double num = q.d*hf*s2 + q.e*p->slip*s2*sth;
  double den = q.a*hf*hf + q.b*p->slip*hf*sth
    + q.c*p->slip*p->slip*s2;
  if (!isfinite (num) || !isfinite (den) || den == 0.0)
    return 1;
  *visc = p->Ca/hf*num/den;
  return !isfinite (*visc);
}

#endif /* GLE_MODEL_LUO_GAO_H */
