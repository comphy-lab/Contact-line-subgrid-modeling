/**
# Landau–Levich plate coating with a GLE subgrid contact-line model

2D two-phase simulation of a plate sliding tangentially to a liquid bath
(the DNS side of the multiscale contact-line strategy), demonstrating the
**per-timestep GLE coupling**: every iteration, the event `gle_boundary`

1. samples the interface curvature near the contact line at the grid scale
   from the DNS,
2. solves the subgrid Generalized Lubrication Equation from the slip length
   $\lambda$ up to the grid size $\Delta$ with that curvature as its outer
   boundary condition ([gle-basilisk.h](../src-local/gle-basilisk.h)),
3. imposes the resulting *apparent* contact angle through the
   height-function boundary condition of Basilisk's [contact.h]().

The historical `contact-fixed.h` patch (height-preferred normals +
per-step height refresh) is now upstream in Basilisk's `contact.h` and has
been archived.

The DNS therefore never needs to resolve the sub-grid wedge: the GLE
carries the viscous bending from $\lambda$ (~nm) to $\Delta$, and the DNS
carries it from $\Delta$ to the macroscopic scale — the division of labour
of Snoeijer & Andreotti, *Annu. Rev. Fluid Mech.* 45 (2013).

The coupling uses the local relative speed of the plate and contact line.
The curvature and interface position are sampled in the finest interfacial
cell nearest the plate. Grid convergence of that sample remains a
problem-specific production requirement.

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#include "navier-stokes/centered.h"
#define FILTERED 1
#include "two-phase.h"
#include "navier-stokes/conserving.h"
#include "tension.h"
#include "reduced.h"
#include "contact.h"
#include "gle-basilisk.h"

#define MINlevel 3
#define tsnap (1e-2)

/**
## Parameters

Non-dimensionalisation follows [contactline.c](contactline.c) (viscous
units, `f.sigma = 1`). The GLE microscopic parameters: `lambda_slip` is the
Navier slip length and `theta_mic` the microscopic contact angle, both
handed to the subgrid solver.
*/
#define fErr (1e-3)
#define VelErr (1e-3)

double hf, tmax, Ldomain, Ca, mu_r, rho_r, lc, u_c, t_c, l_c, lr;
double lambda_slip, theta_mic;
int MAXlevel;

/**
## Boundary conditions

The plate is the bottom boundary, moving tangentially with speed `Ca`. The
contact angle imposed on the height functions is `theta_gle`, refreshed
every time step by the GLE event.
*/
u.t[bottom] = dirichlet(Ca);
uf.t[bottom] = dirichlet(Ca);
u.n[bottom] = dirichlet(0.0);
uf.n[bottom] = dirichlet(0.0);
p[bottom] = neumann(0.0);

u.t[right] = neumann(0.);
uf.t[right] = neumann(0.);
u.n[right] = neumann(0.);
uf.n[right] = neumann(0.0);
p[right] = dirichlet(0.0);

u.t[left] = neumann(0.);
uf.t[left] = neumann(0.);
u.n[left] = neumann(0.);
uf.n[left] = neumann(0.0);
p[left] = dirichlet(0.0);

u.t[top] = neumann(0.);
uf.t[top] = neumann(0.);
u.n[top] = neumann(0.);
uf.n[top] = neumann(0.0);
p[top] = dirichlet(0.0);

double theta_gle;
vector hei[];
hei.t[bottom] = contact_angle (theta_gle);

/**
`gle_theta_state` is a spatially constant dumped scalar. Basilisk snapshots
store fields rather than arbitrary C globals, so this preserves the dynamic
contact angle across restart. The contact-line velocity history is rebuilt
from the first post-restart sample; that first sample safely falls back to
the plate capillary number.
*/
scalar gle_theta_state[];
static double gle_previous_x = 0., gle_previous_t = 0.;
static bool gle_have_previous = false;
static unsigned long gle_missing_samples = 0;
static double gle_last_Delta = nodata;

#if GLE_TEST_MODE
static bool gle_test_candidate_found = false;
static bool gle_test_angle_changed = false;
#endif

static void gle_store_theta_state (void) {
  foreach()
    gle_theta_state[] = theta_gle;
  boundary ({gle_theta_state});
}

static void gle_restore_theta_state (void) {
  double restored_theta = -HUGE;
  foreach (reduction(max:restored_theta))
    if (isfinite (gle_theta_state[]) && gle_theta_state[] > 0. &&
	gle_theta_state[] < pi)
      restored_theta = max (restored_theta, gle_theta_state[]);
  if (restored_theta > 0.)
    theta_gle = restored_theta;
  else if (pid() == 0)
    fprintf (ferr, "warning: restart has no valid GLE angle; using theta_mic\n");
  gle_have_previous = false;
}

int main() {
#if GLE_TEST_MODE
  tmax = 1e-3;
  MAXlevel = 5;
#else
  tmax = 1e2;
  MAXlevel = 10;
#endif
  mu_r = 2e-2;
  rho_r = 1e-3;
  Ca = 5e-3;
  t_c = 1.9e-1;
  l_c = 1.4e-2;
  u_c = l_c/t_c;
  lc = 2.7e-3;
  lr = 1;

  /* GLE microscopic parameters (DNS length units) */
  /* Chan et al.'s two-fluid matching coefficient is available in closed
     form at a right angle, so the demonstration uses that documented case. */
  theta_mic = 0.5*pi;
  theta_gle = theta_mic;

  Ldomain = lr > 1 ? 32 : 32*lr;
  hf = 0.5*Ldomain;
  lambda_slip = 1e-3*Ldomain/(1 << MAXlevel);   /* lambda << Delta */

  fprintf(ferr, "Level %d tmax %g, hf %3.2f, lambda %g\n",
	  MAXlevel, tmax, hf, lambda_slip);

  L0 = Ldomain;
  X0 = -hf; Y0 = 0.;
  init_grid (1 << MINlevel);

#if !GLE_TEST_MODE
  char comm[80];
  sprintf (comm, "mkdir -p intermediate");
  system(comm);
#endif

  rho1 = 1e0; mu1 = 1e0;
  rho2 = rho1*rho_r; mu2 = mu1*mu_r;

  G.x = -10*(t_c*t_c)/l_c;
  f.sigma = 1.0;

  /* contact.h only refreshes height vectors associated with the VOF field. */
  f.height = hei;
#if GLE_TEST_MODE
  assert (f.height.x.i == hei.x.i && f.height.y.i == hei.y.i);
#endif

  run();
}

event init(t = 0){
  bool restored = false;
#if !GLE_TEST_MODE
  restored = restore (file = "restart");
#endif
  if (!restored) {
    /* `refine()` repeats until the condition is false. Scaling the strip
       with the current `Delta` therefore reaches `MAXlevel` from the coarse
       startup mesh while retaining only a narrow interface/wall band. */
    refine ((fabs (x) <= 2.*Delta || y <= 2.*Delta) && level < MAXlevel);
    fraction (f, -x);
    gle_store_theta_state ();
  }
  else
    gle_restore_theta_state ();
}

/**
### gle_boundary()

The coupling event. The curvature is sampled in the interfacial cell
closest to the plate; the subgrid GLE then supplies the apparent angle at
the grid scale. If the GLE has no steady solution at the instantaneous
parameters (beyond the entrainment transition), the previous angle is kept
— the DNS then resolves the ensuing film dynamics itself.
*/
scalar KAPPA[];
scalar CLPOS[];
event gle_boundary (i++) {
  curvature (f, KAPPA);
  position (f, CLPOS, (coord){1., 0.});

  /* Lexicographic reductions select the candidate deterministically under
     OpenMP/MPI: nearest wall, finest local cell, then smallest interface x.
     `interfacial()` includes exactly grid-aligned VOF interfaces. */
  double ycl = HUGE;
  foreach (reduction(min:ycl))
    if (interfacial (point, f) && KAPPA[] != nodata && CLPOS[] != nodata &&
	y < ycl)
      ycl = y;

  double Delta_cl = HUGE;
  if (ycl < HUGE)
    foreach (reduction(min:Delta_cl))
      if (interfacial (point, f) && KAPPA[] != nodata && CLPOS[] != nodata &&
	  y == ycl && Delta < Delta_cl)
	Delta_cl = Delta;

  double xcl = HUGE;
  if (Delta_cl < HUGE)
    foreach (reduction(min:xcl))
      if (interfacial (point, f) && KAPPA[] != nodata && CLPOS[] != nodata &&
	  y == ycl && Delta == Delta_cl && CLPOS[] < xcl)
	xcl = CLPOS[];

  double kappa_cl = -HUGE;
  if (xcl < HUGE)
    foreach (reduction(max:kappa_cl))
      if (interfacial (point, f) && KAPPA[] != nodata && CLPOS[] != nodata &&
	  y == ycl && Delta == Delta_cl && CLPOS[] == xcl &&
	  KAPPA[] > kappa_cl)
	kappa_cl = KAPPA[];

  if (xcl < HUGE && kappa_cl > -HUGE) {
    gle_missing_samples = 0;
    gle_last_Delta = Delta_cl;
#if GLE_TEST_MODE
    gle_test_candidate_found = true;
#endif

    double Ca_local = Ca;
    if (gle_have_previous && t > gle_previous_t) {
      double contact_line_speed = (xcl - gle_previous_x)/(t - gle_previous_t);
      Ca_local = Ca - contact_line_speed;
    }
    gle_previous_x = xcl;
    gle_previous_t = t;
    gle_have_previous = true;

    GLEParams gp = gle_default_params ();
    gp.Ca = Ca_local;              /* plate speed minus line speed */
    gp.mu_r = mu_r;                /* DNS gas/liquid viscosity ratio */
    gp.slip = lambda_slip;
    gp.c_slip = gle_slip_prefactor_right_angle (mu_r);
    gp.theta_mic = theta_mic;
    gp.grav = 0.0;                 /* negligible below the grid scale */
    gp.smax_cap = 10.0*Ldomain;
    /* Here h = y and the GLE arclength points from the contact line toward
       the bath, opposite to Basilisk's normal orientation for `f = -x`.
       Consequently omega_GLE = -kappa_Basilisk for this geometry. */
    double omega_dns = -kappa_cl;
#if GLE_TEST_MODE
    double old_theta = theta_gle;
#endif
    double th = gle_dns_apparent_angle (&gp, omega_dns, Delta_cl,
					theta_gle);
    if (isfinite (th) && th > 0.01 && th < pi - 0.01) {
      theta_gle = th;
      gle_store_theta_state ();
#if GLE_TEST_MODE
      if (fabs (theta_gle - old_theta) > 1e-12)
	gle_test_angle_changed = true;
#endif
    }
  }
  else {
    gle_have_previous = false;
    gle_missing_samples++;
    if (pid() == 0 &&
	(gle_missing_samples <= 3 || gle_missing_samples % 100 == 0))
      fprintf (ferr,
	       "warning: GLE coupling found no valid contact-line sample "
	       "at i=%d t=%g (miss %lu)\n",
	       i, t, gle_missing_samples);
  }
}

event adapt(i++) {
  adapt_wavelet((scalar *){f, u.x, u.y},
    (double[]){fErr, VelErr, VelErr},
    MAXlevel, MINlevel);
}

/**
## Outputs
*/
#if !GLE_TEST_MODE
event writingFiles (t = 0; t += tsnap; t <= tmax + tsnap) {
  gle_store_theta_state ();
  dump (file = "restart");
  char nameOut[80];
  sprintf (nameOut, "intermediate/snapshot-%5.4f", t);
  dump (file = nameOut);
}

event logWriting (i++) {
  double ke = 0.;
  foreach (reduction(+:ke))
    ke += sq(Delta)*(sq(u.x[]) + sq(u.y[]))*rho(f[]);
  static FILE * fp;
  if (i == 0) {
    fprintf (ferr, "i dt t ke theta_gle\n");
    fp = fopen ("log", "w");
    fprintf (fp, "i dt t ke theta_gle\n");
  } else
    fp = fopen ("log", "a");
  fprintf (fp, "%d %g %g %g %g\n", i, dt, t, ke, theta_gle*180.0/pi);
  fclose(fp);
  fprintf (ferr, "%d %g %g %g %g\n", i, dt, t, ke, theta_gle*180.0/pi);
}
#else
/**
## Bounded coupling regression

`GLE_TEST_MODE` runs only two iterations and performs no dumps. It checks the
four seam contracts that a compile-only test misses.
*/
event gle_test_stop (i = 2) {
  assert (f.height.x.i == hei.x.i && f.height.y.i == hei.y.i);
  assert (gle_test_candidate_found);
  assert (gle_last_Delta != nodata && gle_last_Delta > 0. &&
	  gle_last_Delta < Ldomain/(1 << MINlevel));
  assert (gle_test_angle_changed);
  fprintf (ferr, "GLE_TEST_MODE: coupling regression passed\n");
  return 1;
}
#endif
