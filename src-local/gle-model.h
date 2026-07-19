/**
# gle-model.h — Generalized Lubrication Equation: physics layer

Defines the parameter set and the right-hand side of the Generalized
Lubrication Equation (GLE) for a moving contact line, in the arc-length
formulation of [Snoeijer (2006)](https://doi.org/10.1063/1.2171190) with the
two-fluid Huh–Scriven mobility of Chan, Snoeijer & Eggers (2013) as reviewed by
[Snoeijer & Andreotti (2013)](https://doi.org/10.1146/annurev-fluid-011212-140734).

The state vector is $y = (h, \theta, \omega, \zeta)$ with $s$ the arc length
along the liquid–gas interface measured from the contact line:

$$
\frac{\mathrm{d}h}{\mathrm{d}s} = \sin\theta, \qquad
\frac{\mathrm{d}\theta}{\mathrm{d}s} = \omega, \qquad
\frac{\mathrm{d}\omega}{\mathrm{d}s} =
   \frac{3\,\mathrm{Ca}\; M(\theta,\mu_r)}{h\,(h+3\lambda)} + G(\theta), \qquad
\frac{\mathrm{d}\zeta}{\mathrm{d}s} = \cos\theta .
$$

Here $h$ is the film thickness measured normal to the plate, $\theta$ the local
interface inclination with respect to the plate, $\omega = \mathrm{d}\theta/
\mathrm{d}s$ the interface curvature, and $\zeta$ the distance travelled along
the plate (used to reconstruct the contact-line elevation). All lengths are
non-dimensionalized by the capillary length $\ell_\gamma = \sqrt{\gamma/\rho g}$
unless `grav` is set to zero, in which case the length unit is free.

## Sign conventions

$\mathrm{Ca} = \eta U/\gamma > 0$ corresponds to a **receding** contact line
(plate withdrawn from the bath — the dip-coating configuration of Fig. 4b of
Snoeijer & Andreotti 2013); $\mathrm{Ca} < 0$ is an advancing line (plate
plunged in). This follows Snoeijer (2006), where the depth-averaged liquid
velocity in the plate frame obeys $U/U^* = +1$ for receding motion. The
consistency check is the flat-film limit $\theta \to 0$, $\omega' \to 0$ of the
vertical-plate geometry, which yields the gravity-driven film thickness
$h_{\mathrm{film}} = \sqrt{3\,\mathrm{Ca}}\,\ell_\gamma$.

The gravity term is
$$G(\theta) = -\,g^{*}\cos\theta \quad \text{(vertical plate)}, \qquad
  G(\theta) = +\,g^{*}\sin\theta \quad \text{(horizontal plate)},$$
with $g^{*} = (\ell_\gamma/\ell_{\mathrm{unit}})^2$ stored in `grav`
($g^{*} = 1$ in capillary-length units, $0$ disables gravity). In the
vertical-plate geometry, $s$ runs from the contact line *down* towards the
bath, so $z(s) = \Delta - \zeta(s)$ and the static far field obeys the
hydrostatic balance $\omega = z$.

## The Huh–Scriven mobility

$$
M(\theta,\mu_r) = \frac{2\sin^3\theta\,\left[\mu_r^2 f_1(\theta)
   + 2\mu_r f_3(\theta) + f_1(\pi-\theta)\right]}
  {3\,\left[\mu_r f_1(\theta) f_2(\pi-\theta)
   \;+\; f_1(\pi-\theta) f_2(\theta)\right]}
$$

with $f_1(\theta) = \theta^2 - \sin^2\theta$, $f_2(\theta) = \theta -
\sin\theta\cos\theta$, $f_3(\theta) = \theta(\pi - \theta) + \sin^2\theta$ and
$\mu_r$ the gas/liquid viscosity ratio. The **plus** sign in the denominator is
essential: the historical C ports of this repository carried a minus sign there
(flagged in review on PRs #9, #10 and #13, never fixed), which flips the sign
of the whole viscous term at small $\mu_r$. In the one-fluid limit,
$M(\theta,0) = 2\sin^3\theta/[3 f_2(\theta)] = \cos\theta\, F(\theta)$ with
$F(\theta)$ of Snoeijer (2006) Eq. (15), and $M \to 1$ as $\theta \to 0$,
recovering classical lubrication $h''' = 3\,\mathrm{Ca}/h^2$.

The function is *not* named `f`: in Basilisk, `f` is conventionally the VoF
volume-fraction field (see issue #4 of this repository).

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#ifndef GLE_MODEL_H
#define GLE_MODEL_H

#include <math.h>
#include <stddef.h>

/**
## Geometry selector
*/
enum gle_geometry {
  GLE_PLATE_VERTICAL = 0,   /* dip-coating: plate vertical, bath below      */
  GLE_PLATE_HORIZONTAL = 1  /* plate horizontal, gravity normal to plate    */
};

/**
### GLEParams

All physical and numerical parameters of a GLE boundary-value problem.

#### Fields
- `Ca`: capillary number ($>0$ receding, $<0$ advancing).
- `mu_r`: gas/liquid viscosity ratio $\mu_r$ ($0$ = one-fluid limit).
- `slip`: Navier slip length $\lambda$ (in the working length unit).
- `theta_mic`: microscopic contact angle $\theta_e$ imposed at the inner
  boundary (radians).
- `grav`: gravity prefactor $g^{*}$ ($1$ in capillary-length units, $0$ off).
- `geometry`: one of `enum gle_geometry`.
- `s0`: inner starting arc length ($\le 0$ selects the default $s_0=\lambda$).
  The curvature diverges logarithmically at the contact line even with slip,
  so integration starts at a small but finite $s_0$.
- `h0`: film thickness at $s_0$ ($\le 0$ selects the wedge value
  $s_0 \sin\theta_e$).
- `H_match`: film thickness at which the outer boundary condition is applied.
- `smax_cap`: hard cap on total arc length (safety for lost trajectories).
- `rtol`, `atol`: relative/absolute local error tolerances of the integrator.
- `max_steps`: integrator step budget per trajectory.
*/
typedef struct {
  double Ca;
  double mu_r;
  double slip;
  double theta_mic;
  double grav;
  int geometry;

  double s0;
  double h0;
  double H_match;
  double smax_cap;

  double rtol;
  double atol;
  long max_steps;
} GLEParams;

/**
### gle_default_params()

Returns a parameter set matching the receding-plate (dip-coating) problem in
capillary-length units, with tolerances tight enough for continuation work.
*/
static inline GLEParams gle_default_params (void) {
  GLEParams p;
  p.Ca = 1.0e-3;
  p.mu_r = 0.0;
  p.slip = 1.0e-6;
  p.theta_mic = 51.5*M_PI/180.0;
  p.grav = 1.0;
  p.geometry = GLE_PLATE_VERTICAL;
  p.s0 = -1.0;         /* default: lambda */
  p.h0 = -1.0;         /* default: s0*sin(theta_mic) */
  p.H_match = 5.0;
  p.smax_cap = 100.0;
  p.rtol = 1.0e-10;
  p.atol = 1.0e-12;
  p.max_steps = 2000000;
  return p;
}

/**
### gle_s0(), gle_h0()

Resolved inner boundary values (apply the defaults documented above).
*/
static inline double gle_s0 (const GLEParams *p) {
  return (p->s0 > 0.0 ? p->s0 : p->slip);
}

static inline double gle_h0 (const GLEParams *p) {
  return (p->h0 > 0.0 ? p->h0 : gle_s0 (p)*sin (p->theta_mic));
}

/**
### gle_f1(), gle_f2(), gle_f3()

The Huh–Scriven auxiliary functions. `gle_f1` and `gle_f2` suffer catastrophic
cancellation for small arguments ($f_1 \sim \theta^4/3$, $f_2 \sim
2\theta^3/3$), so both switch to Taylor series below $\theta = 0.02$:

$$f_1 = \frac{\theta^4}{3}\left(1 - \frac{2\theta^2}{15}
        + \frac{\theta^4}{189}\right) + \mathcal{O}(\theta^{10}), \qquad
  f_2 = \frac{2\theta^3}{3}\left(1 - \frac{\theta^2}{5}
        + \frac{2\theta^4}{105}\right) + \mathcal{O}(\theta^{9}).$$
*/
static inline double gle_f1 (double th) {
  if (fabs (th) < 0.02) {
    double t2 = th*th;
    return t2*t2/3.0*(1.0 - 2.0*t2/15.0 + t2*t2/189.0);
  }
  double sth = sin (th);
  return th*th - sth*sth;
}

static inline double gle_f2 (double th) {
  if (fabs (th) < 0.02) {
    double t2 = th*th;
    return 2.0*th*t2/3.0*(1.0 - t2/5.0 + 2.0*t2*t2/105.0);
  }
  return th - sin (th)*cos (th);
}

static inline double gle_f3 (double th) {
  double sth = sin (th);
  return th*(M_PI - th) + sth*sth;
}

/**
### gle_mobility()

The two-fluid Huh–Scriven mobility $M(\theta,\mu_r)$ defined in the header
notes, with the corrected `+` sign in the denominator. Valid for $\theta \in
(0,\pi)$; for $\mu_r = 0$ the reduced one-fluid expression
$2\sin^3\theta/[3f_2(\theta)]$ is used directly (it is also the exact
$\mu_r \to 0$ limit of the general form). $M \to 1$ as $\theta \to 0$.

#### Parameters
- `th`: interface angle $\theta$ (radians), $0 < \theta < \pi$.
- `mu_r`: gas/liquid viscosity ratio.

Negative angles are handled by the **even extension** $M(\theta) =
M(|\theta|)$, with $M(0) = 1$: on the upper branch of the dip-coating
problem, the film joins the meniscus through an *oscillatory* tail in which
$\theta$ legitimately dips slightly below zero (the classical dimple
oscillations of Landau–Levich-type films). There $|\theta| \ll 1$ and the
even extension is exactly the classical-lubrication limit; the one-fluid
$M(\theta) = 2\sin^3\theta/[3f_2(\theta)]$ is even in $\theta$ analytically.

#### Returns
$M(\theta,\mu_r)$, or `NAN` if $|\theta| \ge \pi$.
*/
static inline double gle_mobility (double th, double mu_r) {
  th = fabs (th);
  if (th >= M_PI)
    return NAN;
  if (th == 0.0)
    return 1.0;
  double sth = sin (th);
  if (mu_r == 0.0)
    return 2.0*sth*sth*sth/(3.0*gle_f2 (th));
  double num = 2.0*sth*sth*sth*
    (mu_r*mu_r*gle_f1 (th) + 2.0*mu_r*gle_f3 (th) + gle_f1 (M_PI - th));
  double den = 3.0*
    (mu_r*gle_f1 (th)*gle_f2 (M_PI - th) + gle_f1 (M_PI - th)*gle_f2 (th));
  return num/den;
}

/**
### gle_rhs()

Right-hand side of the GLE system. Writes
$\mathrm{d}y/\mathrm{d}s$ into `dyds[4]` for state `y[4]` $= (h, \theta,
\omega, \zeta)$.

#### Returns
- `0` on success;
- `1` if the state left the physical domain ($h \le 0$ or $\theta \notin
  (-\pi/2,\pi)$ — small negative angles are allowed for the oscillatory film
  tail, see `gle_mobility()`), in which case `dyds` is not usable. The
  integrator treats this as a step failure and retries with a smaller step,
  so trajectories that genuinely leave the domain terminate cleanly.
*/
static inline int gle_rhs (const GLEParams *p, const double y[4],
			   double dyds[4]) {
  double hf = y[0], th = y[1], om = y[2];
  if (hf <= 0.0 || th <= -0.5*M_PI || th >= M_PI)
    return 1;
  double mob = gle_mobility (th, p->mu_r);
  double visc = 3.0*p->Ca*mob/(hf*(hf + 3.0*p->slip));
  double gravity = (p->geometry == GLE_PLATE_VERTICAL ?
		    -p->grav*cos (th) : p->grav*sin (th));
  dyds[0] = sin (th);
  dyds[1] = om;
  dyds[2] = visc + gravity;
  dyds[3] = cos (th);
  return 0;
}

/**
### gle_static_curvature()

The first integral of the *static* vertical-plate meniscus. Static solutions
($\mathrm{Ca}=0$, vertical plate, capillary-length units) conserve
$\omega^2/2 + \sin\theta$; the branch connected to a flat bath
($\omega \to 0$, $\theta \to \pi/2$) therefore satisfies

$$\omega = \sqrt{2\,(1 - \sin\theta)}.$$

This is used as the outer boundary condition of the dip-coating problem: at
$h = H_{\mathrm{match}}$ the viscous term has decayed like
$3\,\mathrm{Ca}\,M/h$ and the trajectory must have landed on the static
meniscus manifold. It is also the classical meniscus-rise relation $\Delta =
\sqrt{2(1-\sin\theta_{\mathrm{app}})}$ (Landau & Lifshitz 1984), Eq. (4) of
Snoeijer & Andreotti (2013).
*/
static inline double gle_static_curvature (double th) {
  double one_m = 1.0 - sin (th);
  return (one_m > 0.0 ? sqrt (2.0*one_m) : 0.0);
}

#endif /* GLE_MODEL_H */
