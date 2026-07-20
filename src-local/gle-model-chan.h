/**
# gle-model-chan.h — Chan et al. constant-cutoff GLE

Implements the viscous curvature gradient of the modified Generalised
Lubrication Equation used by Chan et al. (2013, 2020),

$$
\left.\frac{\mathrm{d}\omega}{\mathrm{d}s}\right|_{\rm visc}
 = \frac{3\,\mathrm{Ca}\,M(\theta,\mu_r)}
        {h\,(h+c_\lambda\lambda)}.
$$

The scalar `c_slip` is constant along one trajectory. It may be supplied
manually or resolved once from $(\theta_e,\mu_r)$ by
[gle-slip-closure.h](gle-slip-closure.h); those are two policies for this same
equation, not distinct GLEs.

Include [gle-model.h](gle-model.h), not this implementation header directly.
*/

#ifndef GLE_MODEL_CHAN_H
#define GLE_MODEL_CHAN_H

#ifndef GLE_MODEL_H
# error "include gle-model.h instead of gle-model-chan.h"
#endif

/**
### gle_chan_viscous_gradient()

Evaluates the Chan et al. viscous contribution to
$\mathrm{d}\omega/\mathrm{d}s$.

#### Returns
`0` on success and `1` if a parameter, mobility, or denominator is outside the
finite physical domain.
*/
static inline int gle_chan_viscous_gradient (const GLEParams *p, double hf,
					      double th, double *visc) {
  if (!p || !visc || !isfinite (hf) || hf <= 0.0 ||
      !isfinite (p->slip) || p->slip <= 0.0 ||
      !isfinite (p->c_slip) || p->c_slip <= 0.0 ||
      !isfinite (p->Ca) || !isfinite (p->mu_r) || p->mu_r < 0.0)
    return 1;
  double mob = gle_mobility (th, p->mu_r);
  double den = hf*(hf + p->c_slip*p->slip);
  if (!isfinite (mob) || !isfinite (den) || den <= 0.0)
    return 1;
  *visc = 3.0*p->Ca*mob/den;
  return !isfinite (*visc);
}

#endif /* GLE_MODEL_CHAN_H */
