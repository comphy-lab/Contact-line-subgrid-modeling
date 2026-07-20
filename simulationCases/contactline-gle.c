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

## Scope

This case is the **integration seam demonstration**, superseding the static
contact angle of [contactline.c](contactline.c). Production use requires
two problem-specific choices documented in the README roadmap: the signed
capillary number should be built from the *local* contact-line speed
(plate speed minus interface speed), and the curvature sample should be
grid-converged (here: the interfacial cell nearest the contact line).

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

int main() {
  tmax = 1e2;
  MAXlevel = 10;
  mu_r = 2e-2;
  rho_r = 1e-3;
  Ca = 5e-3;
  t_c = 1.9e-1;
  l_c = 1.4e-2;
  u_c = l_c/t_c;
  lc = 2.7e-3;
  lr = 1;

  /* GLE microscopic parameters (DNS length units) */
  theta_mic = 60.0*pi/180.0;
  theta_gle = theta_mic;

  Ldomain = lr > 1 ? 32 : 32*lr;
  hf = 0.5*Ldomain;
  lambda_slip = 1e-3*Ldomain/(1 << MAXlevel);   /* lambda << Delta */

  fprintf(ferr, "Level %d tmax %g, hf %3.2f, lambda %g\n",
	  MAXlevel, tmax, hf, lambda_slip);

  L0 = Ldomain;
  X0 = -hf; Y0 = 0.;
  init_grid (1 << MINlevel);

  char comm[80];
  sprintf (comm, "mkdir -p intermediate");
  system(comm);

  rho1 = 1e0; mu1 = 1e0;
  rho2 = rho1*rho_r; mu2 = mu1*mu_r;

  G.x = -10*(t_c*t_c)/l_c;
  f.sigma = 1.0;

  run();
}

event init(t = 0){
  if(!restore (file = "restart")) {
    refine(((x < 1e-1 && x > -1e-1) || (y < 1e-1)) && level < MAXlevel);
    fraction (f, -x);
  }
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
event gle_boundary (i++) {
  curvature (f, KAPPA);
  /* Two passes keep the min-y selection and its curvature consistent
     under parallel reductions: ties at equal y resolve deterministically
     to the largest curvature. */
  double ycl = HUGE;
  foreach (reduction(min:ycl))
    if (f[] > 0.1 && f[] < 0.9 && KAPPA[] != nodata && y < ycl)
      ycl = y;
  double kappa_cl = -HUGE;
  if (ycl < HUGE)
    foreach (reduction(max:kappa_cl))
      if (f[] > 0.1 && f[] < 0.9 && KAPPA[] != nodata && y == ycl &&
	  KAPPA[] > kappa_cl)
	kappa_cl = KAPPA[];
  if (ycl < HUGE && kappa_cl > -HUGE) {
    GLEParams gp = gle_default_params ();
    gp.Ca = Ca;                    /* receding plate: Ca > 0 */
    gp.mu_r = mu_r;                /* DNS gas/liquid viscosity ratio */
    gp.slip = lambda_slip;
    gp.theta_mic = theta_mic;
    gp.grav = 0.0;                 /* negligible below the grid scale */
    gp.smax_cap = 10.0*Ldomain;
    double Delta_grid = Ldomain/(1 << MAXlevel);
    /* kappa handoff: kappa_cl (Basilisk's curvature(), sampled at the
       interfacial cell nearest the plate) is passed straight through as
       the GLE's outer curvature target. This assumes Basilisk's
       curvature() sign convention matches the GLE's d(theta)/ds > 0
       toward-the-bath convention -- production use must verify the sign,
       and should build Ca from the LOCAL contact-line speed (plate speed
       minus interface speed), not the fixed plate Ca used here. */
    double th = gle_dns_apparent_angle (&gp, kappa_cl, Delta_grid,
					theta_gle);
    if (isfinite (th) && th > 0.01 && th < pi - 0.01)
      theta_gle = th;
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
event writingFiles (t = 0; t += tsnap; t <= tmax + tsnap) {
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
