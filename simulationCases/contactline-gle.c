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

#ifndef GLE_TEST_MODE
# define GLE_TEST_MODE 0
#endif
#ifndef GLE_RESTART_TEST_MODE
# define GLE_RESTART_TEST_MODE 0
#endif
#if GLE_TEST_MODE && GLE_RESTART_TEST_MODE
# error "GLE_TEST_MODE and GLE_RESTART_TEST_MODE are mutually exclusive"
#endif

/* The restart regression keeps the production time-based stop schedule while
   making the DNS deliberately small. */
#ifndef GLE_RESTART_TEST_TMAX
# define GLE_RESTART_TEST_TMAX 2e-3
#endif
#ifndef GLE_RESTART_TEST_CHECKPOINT
# define GLE_RESTART_TEST_CHECKPOINT 1e-3
#endif

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
The subgrid equation and Chan cutoff policy are compile-time case choices.
For example, compile the direct alternative with

~~~c
qcc ... -DGLE_RUNTIME_MODEL=GLE_MODEL_LUO_GAO contactline-gle.c ...
~~~

The case is prepared once because $c(\theta_e,M)$ is fixed by the microscopic
inputs, not by the instantaneous interface profile. Only the relative
capillary number changes in the per-timestep copy.
*/
#ifndef GLE_RUNTIME_MODEL
# define GLE_RUNTIME_MODEL GLE_MODEL_CHAN
#endif
#ifndef GLE_RUNTIME_CUTOFF
# define GLE_RUNTIME_CUTOFF GLE_CUTOFF_AUTO
#endif

static GLEParams gle_case_params;
static GLECutoffResult gle_case_cutoff;

static void gle_prepare_case_model (void) {
  gle_case_params = gle_default_params ();
  gle_case_params.model = GLE_RUNTIME_MODEL;
  gle_case_params.cutoff_method = GLE_RUNTIME_CUTOFF;
  gle_case_params.mu_r = mu_r;
  gle_case_params.slip = lambda_slip;
  gle_case_params.theta_mic = theta_mic;
  gle_case_params.grav = 0.0;
  gle_case_params.smax_cap = 10.0*Ldomain;
  int status = gle_model_prepare (&gle_case_params, &gle_case_cutoff);
  if (status != GLE_CUTOFF_OK) {
    if (pid() == 0)
      fprintf (ferr,
	       "error: cannot prepare GLE model=%s cutoff=%s: %s\n",
	       gle_model_name (gle_case_params.model),
	       gle_cutoff_method_name (gle_case_params.cutoff_method),
	       gle_cutoff_status_name (status));
    exit (2);
  }
  if (pid() == 0) {
    fprintf (ferr, "GLE model=%s", gle_model_name (gle_case_params.model));
    if (gle_case_params.model == GLE_MODEL_CHAN)
      fprintf (ferr,
	       ", cutoff=%s, c=%g, Q=%g, luo_gao_approximation=%s",
	       gle_cutoff_method_name (gle_case_cutoff.method),
	       gle_case_params.c_slip, gle_case_cutoff.Q,
	       gle_case_cutoff.luo_gao_approximation ? "yes" : "no");
    fprintf (ferr, "\n");
  }
}

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
The four `gle_*_state` fields are spatially constant dumped scalars. Basilisk
snapshots store named fields rather than arbitrary C globals, so these preserve
both the dynamic contact angle and the preceding contact-line sample across a
restart. Older snapshots without the three history fields remain readable: the
first valid post-restart sample only rebuilds the velocity history and leaves
the restored angle unchanged.
*/
scalar gle_theta_state[];
scalar gle_previous_x_state[];
scalar gle_previous_t_state[];
scalar gle_history_valid_state[];
static double gle_previous_x = 0., gle_previous_t = 0.;
static bool gle_have_previous = false;
static bool gle_needs_history_sample = false;
static bool gle_was_restored = false;
static bool gle_first_restored_sample = false;
static unsigned long gle_missing_samples = 0;
static double gle_last_Delta = nodata;

#if GLE_TEST_MODE
static bool gle_test_candidate_found = false;
static bool gle_test_angle_changed = false;
#endif

#if GLE_RESTART_TEST_MODE
static void gle_restart_test_record (double xcl, double previous_x,
				     double previous_t,
				     double contact_line_speed,
				     double Ca_local, bool used_history,
				     bool sample_only) {
  if (pid() != 0)
    return;
  static bool started = false;
  FILE * fp = fopen ("gle-restart-trace.tsv",
		    started || gle_was_restored ? "a" : "w");
  if (!fp) {
    perror ("gle-restart-trace.tsv");
    exit (2);
  }
  fprintf (fp, "%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
	   "%d %d %d %d\n",
	   iter, t, xcl, previous_x, previous_t, contact_line_speed, Ca_local,
	   theta_gle, used_history, sample_only, gle_was_restored,
	   gle_first_restored_sample);
  fclose (fp);
  started = true;
}
#endif

static void gle_store_restart_state (void) {
  foreach() {
    gle_theta_state[] = theta_gle;
    gle_previous_x_state[] = gle_previous_x;
    gle_previous_t_state[] = gle_previous_t;
    gle_history_valid_state[] = gle_have_previous ? 1. : 0.;
  }
  boundary ({gle_theta_state, gle_previous_x_state, gle_previous_t_state,
	     gle_history_valid_state});
}

static void gle_restore_restart_state (void) {
  double restored_theta = -HUGE;
  double restored_x = -HUGE, restored_t = -HUGE, restored_valid = -HUGE;
  foreach (reduction(max:restored_theta) reduction(max:restored_x)
	   reduction(max:restored_t) reduction(max:restored_valid)) {
    if (isfinite (gle_theta_state[]) && gle_theta_state[] > 0. &&
	gle_theta_state[] < pi)
      restored_theta = max (restored_theta, gle_theta_state[]);
    if (isfinite (gle_previous_x_state[]))
      restored_x = max (restored_x, gle_previous_x_state[]);
    if (isfinite (gle_previous_t_state[]))
      restored_t = max (restored_t, gle_previous_t_state[]);
    if (isfinite (gle_history_valid_state[]))
      restored_valid = max (restored_valid, gle_history_valid_state[]);
  }
  if (restored_theta > 0.)
    theta_gle = restored_theta;
  else if (pid() == 0)
    fprintf (ferr, "warning: restart has no valid GLE angle; using theta_mic\n");

  if (restored_valid > 0.5 && restored_x > -HUGE && restored_t > -HUGE &&
      restored_t <= t + 1e-12*max (1., fabs (t))) {
    gle_previous_x = restored_x;
    gle_previous_t = restored_t;
    gle_have_previous = true;
    gle_needs_history_sample = false;
  }
  else {
    gle_have_previous = false;
    gle_needs_history_sample = true;
    if (pid() == 0)
      fprintf (ferr,
	       "warning: restart has no valid GLE velocity history; "
	       "sampling once before the next GLE solve\n");
  }
  gle_was_restored = true;
  gle_first_restored_sample = true;
}

int main() {
#if GLE_TEST_MODE
  tmax = 1e-3;
  MAXlevel = 5;
#elif GLE_RESTART_TEST_MODE
  tmax = GLE_RESTART_TEST_TMAX;
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
  gle_prepare_case_model ();

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
    gle_store_restart_state ();
  }
  else
    gle_restore_restart_state ();
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

#if GLE_RESTART_TEST_MODE
    double previous_x = gle_previous_x, previous_t = gle_previous_t;
#endif
    bool used_history = gle_have_previous && t > gle_previous_t;
    bool sample_only = gle_needs_history_sample ||
      (gle_have_previous && t <= gle_previous_t);
    double contact_line_speed = nodata, Ca_local = nodata;
    if (!sample_only) {
      contact_line_speed = used_history ?
	(xcl - gle_previous_x)/(t - gle_previous_t) : 0.;
      Ca_local = Ca - contact_line_speed;
    }
    gle_previous_x = xcl;
    gle_previous_t = t;
    gle_have_previous = true;
    gle_needs_history_sample = false;

    if (!sample_only) {
      GLEParams gp = gle_case_params;
      gp.Ca = Ca_local;            /* plate speed minus line speed */
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
#if GLE_TEST_MODE
	if (fabs (theta_gle - old_theta) > 1e-12)
	  gle_test_angle_changed = true;
#endif
      }
    }
    gle_store_restart_state ();
#if GLE_RESTART_TEST_MODE
    gle_restart_test_record (xcl, previous_x, previous_t,
			     contact_line_speed, Ca_local, used_history,
			     sample_only);
#endif
    gle_first_restored_sample = false;
  }
  else {
    gle_have_previous = false;
    gle_needs_history_sample = true;
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
  gle_store_restart_state ();
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

#if GLE_RESTART_TEST_MODE
/** Deliberately interrupt the bounded restart regression at a scheduled time. */
event gle_restart_test_interrupt (t = GLE_RESTART_TEST_CHECKPOINT) {
  if (getenv ("GLE_RESTART_TEST_INTERRUPT")) {
    gle_store_restart_state ();
    dump (file = "restart");
    if (pid() == 0)
      fprintf (ferr, "GLE_RESTART_TEST: interrupted i=%d t=%.17g\n", i, t);
    return 1;
  }
}
#endif

/** Terminate the production schedule exactly at its declared horizon. */
event stop (t = tmax) {
  gle_store_restart_state ();
  dump (file = "restart");
#if GLE_RESTART_TEST_MODE
  if (pid() == 0)
    fprintf (ferr, "GLE_RESTART_TEST: production stop i=%d t=%.17g\n", i, t);
#endif
  return 1;
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
