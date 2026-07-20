/**
# test-gle-core.c — focused GLE parameter and allocation regressions

Small dependency-free executable covering the runtime model dispatcher,
case-level cutoff provenance, strict parsing, and the failure modes that must
be rejected before a solve.

Compile and run from `gle-ode/`:

```bash
cc -O2 -std=c99 -Wall -Wextra -Werror -pedantic -I../src-local \
  test-gle-core.c -lm -o test-gle-core
./test-gle-core
```
*/

#include <math.h>
#include <stdio.h>

/* Exercise the strict-C99 path even on libc implementations that expose
   M_PI by default. gle-model.h must supply the guarded fallback. */
#ifdef M_PI
# undef M_PI
#endif

#include "gle-params.h"
#include "gle-basilisk.h"
#include "gle-collocate.h"

/* This is the complete positional initialiser accepted before runtime model
   selection was added. The warning suppression is deliberately narrow: the
   omitted trailing fields are the compatibility behaviour under test. */
#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wmissing-field-initializers"
#elif defined(__GNUC__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
static const GLEParams legacy_positional = {
  0.0123, 0.04, 0.005, 1.7, 0.9, 0.5, GLE_PLATE_HORIZONTAL,
  0.006, 0.007, 8.0, 9.0, GLE_OUTER_OMEGA_ZERO,
  1.0e-7, 2.0e-8, 12345
};
#if defined(__clang__)
# pragma clang diagnostic pop
#elif defined(__GNUC__)
# pragma GCC diagnostic pop
#endif

static int check_close (const char *name, double got, double want,
			double tol) {
  if (!isfinite (got) || fabs (got - want) > tol) {
    fprintf (stderr, "%s: got %.17g, want %.17g +/- %.3g\n",
	     name, got, want, tol);
    return 1;
  }
  return 0;
}

static void count_sampler (void *ctx, double s, const double y[4]) {
  (void) s;
  (void) y;
  (*(int *) ctx)++;
}

int main (void) {
  int failed = 0;
  const double gamma_E = 0.57721566490153286061;
  GLEParams p = gle_default_params ();
  GLECutoffResult cutoff;

  failed += check_close ("legacy c_slip default", p.c_slip, 3.0, 0.0);
  if (p.model != GLE_MODEL_CHAN || p.cutoff_method != GLE_CUTOFF_MANUAL) {
    fprintf (stderr, "default runtime selectors changed unexpectedly\n");
    failed++;
  }
  failed += check_close ("legacy positional Ca", legacy_positional.Ca,
			  0.0123, 0.0);
  failed += check_close ("legacy positional s0", legacy_positional.s0,
			  0.006, 0.0);
  failed += check_close ("legacy positional H_match",
			  legacy_positional.H_match, 8.0, 0.0);
  failed += check_close ("legacy positional rtol", legacy_positional.rtol,
			  1.0e-7, 0.0);
  if (legacy_positional.geometry != GLE_PLATE_HORIZONTAL ||
      legacy_positional.outer_bc != GLE_OUTER_OMEGA_ZERO ||
      legacy_positional.max_steps != 12345 ||
      legacy_positional.model != GLE_MODEL_CHAN ||
      legacy_positional.cutoff_method != GLE_CUTOFF_MANUAL) {
    fprintf (stderr, "legacy positional aggregate layout changed\n");
    failed++;
  }
  failed += check_close ("right-angle one-fluid c_slip",
			 gle_slip_prefactor_right_angle (0.0),
			 exp (log (2.0) - gamma_E), 5.0e-15);
  failed += check_close ("right-angle c_slip at M=0.02",
			 gle_slip_prefactor_right_angle (0.02),
			 1.2092544572784334, 1.0e-13);
  failed += check_close ("right-angle c_slip at M=1",
			 gle_slip_prefactor_right_angle (1.0),
			 2.052069628256686, 1.0e-13);
  failed += check_close ("right-angle c_slip at M=10",
			 gle_slip_prefactor_right_angle (10.0),
			 1.4761693355225698, 1.0e-13);
  failed += check_close ("right-angle large-ratio c_slip",
			 gle_slip_prefactor_right_angle (1.0e8),
			 1.1229190126700019, 1.0e-13);
  failed += check_close ("right-angle phase-exchange symmetry",
			 gle_slip_prefactor_right_angle (0.1),
			 gle_slip_prefactor_right_angle (10.0), 0.0);
  if (!isnan (gle_slip_prefactor_right_angle (-1.0))) {
    fprintf (stderr, "negative viscosity ratio was accepted\n");
    failed++;
  }

  /* Numerically converged Scott--Hocking one-phase reference: analytic
     right-angle anchor, frozen integral-equation nodes, and case-level
     mutation of c_slip. */
  p = gle_default_params ();
  p.theta_mic = 0.5*M_PI;
  p.cutoff_method = GLE_CUTOFF_SCOTT_HOCKING;
  if (gle_model_prepare (&p, &cutoff) != GLE_CUTOFF_OK ||
      cutoff.method != GLE_CUTOFF_SCOTT_HOCKING ||
      cutoff.luo_gao_approximation) {
    fprintf (stderr, "Scott--Hocking reference closure did not resolve\n");
    failed++;
  }
  else {
    failed += check_close ("Scott--Hocking exact right-angle Q",
			   cutoff.Q, 1.0 + gamma_E - log (2.0), 5.0e-15);
    failed += check_close ("Scott--Hocking exact right-angle c",
			   cutoff.c, exp (log (2.0) - gamma_E), 5.0e-15);
    failed += check_close ("Scott closure updates c_slip",
			   p.c_slip, cutoff.c, 0.0);
  }
  if (gle_cutoff_scott_hocking_reference (0.1, &cutoff) !=
	GLE_CUTOFF_OK)
    failed++;
  else
    failed += check_close ("Scott generated Q_i node at alpha=0.1",
			   cutoff.Q - 1.0, -3.3996431057228507, 5.0e-14);
  if (gle_cutoff_scott_hocking (3.0, &cutoff) != GLE_CUTOFF_OK)
    failed++;
  else
    failed += check_close ("Scott generated Q_i node at alpha=3.0",
			   cutoff.Q - 1.0, 20.084722540229595, 5.0e-13);
  if (gle_cutoff_scott_hocking (0.025, &cutoff) != GLE_CUTOFF_OK)
    failed++;
  else
    failed += check_close ("Scott small-angle endpoint checkpoint",
			   cutoff.Q - 1.0, -4.787394462666471, 5.0e-13);
  if (gle_cutoff_scott_hocking (2.95, &cutoff) != GLE_CUTOFF_OK)
    failed++;
  else {
    double qi_295 = cutoff.Q - 1.0;
    double large_asymptote = M_PI/(M_PI - 2.95) + gamma_E -
	log (2.0) - 2.0;
    failed += check_close ("Scott generated Q_i node at alpha=2.95",
			   qi_295, 14.300831418480071, 2.0e-12);
    if (fabs (qi_295 - large_asymptote) > 3.0e-2) {
      fprintf (stderr, "regularised Scott interpolation lost large-angle asymptote\n");
      failed++;
    }
  }
  if (gle_cutoff_scott_hocking (0.5*(3.12 + M_PI), &cutoff) !=
	GLE_CUTOFF_OK)
    failed++;
  else
    failed += check_close ("Scott large-angle endpoint checkpoint",
			   cutoff.Q - 1.0, 288.87187914660444, 2.0e-10);

  p = gle_default_params ();
  p.theta_mic = 1.0;
  p.mu_r = 0.0;
  p.cutoff_method = GLE_CUTOFF_AUTO;
  if (gle_model_prepare (&p, &cutoff) != GLE_CUTOFF_OK ||
      cutoff.method != GLE_CUTOFF_SCOTT_HOCKING ||
      cutoff.luo_gao_approximation) {
    fprintf (stderr, "auto did not select the Scott--Hocking M=0 branch\n");
    failed++;
  }

  /* Auto gives the analytic branches precedence, then uses the frozen
     two-phase Q table without extrapolation. */
  p = gle_default_params ();
  p.theta_mic = 0.5*M_PI;
  p.mu_r = 0.02;
  p.cutoff_method = GLE_CUTOFF_AUTO;
  if (gle_model_prepare (&p, &cutoff) != GLE_CUTOFF_OK ||
      cutoff.method != GLE_CUTOFF_CORRECTED_RIGHT_ANGLE ||
      cutoff.luo_gao_approximation)
    {
      fprintf (stderr, "auto did not select corrected right-angle closure\n");
      failed++;
    }
  else
    failed += check_close ("auto right-angle c_slip", p.c_slip,
			   gle_slip_prefactor_right_angle (p.mu_r), 0.0);

  for (int i = 0; i < GLE_SLIP_TABLE_THETA_COUNT; i++)
    for (int j = 0; j < GLE_SLIP_TABLE_LOGM_COUNT; j++) {
      double theta_node = gle_slip_table_theta_deg[i]*M_PI/180.0;
      double ratio_node = pow (10.0, gle_slip_table_log10_m[j]);
      if (gle_cutoff_reference_table (theta_node, ratio_node, &cutoff) !=
	  GLE_CUTOFF_OK || cutoff.method != GLE_CUTOFF_REFERENCE_TABLE ||
	  cutoff.luo_gao_approximation) {
	fprintf (stderr, "frozen table node did not resolve\n");
	failed++;
      }
      else {
	failed += check_close ("frozen table node Q", cutoff.Q,
			       gle_slip_table_Q[i][j], 5.0e-13);
	failed += check_close ("table reconstructs log(c)", cutoff.log_c,
			       1.0 + log (sin (theta_node)) - cutoff.Q,
			       5.0e-15);
      }
    }

  /* This is an independent converged FEM solve at a table-cell centre, not a
     value reconstructed from the C table. It is also the worst checkpoint in
     the committed 4-by-5 interpolation audit. */
  double theta_check_deg = 137.5, logm_check = -1.625;
  if (gle_cutoff_reference_table (theta_check_deg*M_PI/180.0,
				  pow (10.0, logm_check), &cutoff) !=
	GLE_CUTOFF_OK)
    failed++;
  else
    failed += check_close ("table vs independent FEM checkpoint", cutoff.Q,
			   1.4207002726888973, 1.0e-3);

  GLECutoffResult table_direct, table_exchanged;
  if (gle_cutoff_reference_table (0.7, 0.1, &table_direct) != GLE_CUTOFF_OK ||
      gle_cutoff_reference_table (M_PI - 0.7, 10.0, &table_exchanged) !=
	GLE_CUTOFF_OK)
    failed++;
  else
    failed += check_close ("table phase-exchange symmetry",
			   table_direct.Q, table_exchanged.Q, 5.0e-14);
  if (gle_cutoff_reference_table (0.7, 1.0, &table_direct) != GLE_CUTOFF_OK ||
      gle_cutoff_reference_table (M_PI - 0.7, 1.0, &table_exchanged) !=
	GLE_CUTOFF_OK)
    failed++;
  else
    failed += check_close ("table equal-viscosity angle symmetry",
			   table_direct.Q, table_exchanged.Q,
			   16.0*DBL_EPSILON);

  p = gle_default_params ();
  p.theta_mic = 1.0;
  p.mu_r = 0.02;
  p.cutoff_method = GLE_CUTOFF_AUTO;
  if (gle_model_prepare (&p, &cutoff) != GLE_CUTOFF_OK ||
      cutoff.method != GLE_CUTOFF_REFERENCE_TABLE ||
      cutoff.luo_gao_approximation ||
      !isfinite (p.c_slip) || p.c_slip <= 0.0) {
    fprintf (stderr, "auto did not select the finite-M reference table\n");
    failed++;
  }

  /* Valid physical cases outside the frozen grid remain distinguishable from
     bad input. Auto alone may route them to the labelled approximation. */
  double theta_below = (gle_slip_table_theta_deg[0] - 1.0)*M_PI/180.0;
  double theta_above =
    (gle_slip_table_theta_deg[GLE_SLIP_TABLE_THETA_COUNT - 1] + 1.0)*
    M_PI/180.0;
  double ratio_below = pow (10.0, gle_slip_table_log10_m[0] - 0.1);
  double ratio_above = pow (10.0, -gle_slip_table_log10_m[0] + 0.1);
  if (gle_cutoff_reference_table (theta_below, 0.1, &cutoff) !=
	GLE_CUTOFF_UNAVAILABLE || cutoff.status != GLE_CUTOFF_UNAVAILABLE ||
      gle_cutoff_reference_table (theta_above, 0.1, &cutoff) !=
	GLE_CUTOFF_UNAVAILABLE ||
      gle_cutoff_reference_table (1.0, ratio_below, &cutoff) !=
	GLE_CUTOFF_UNAVAILABLE ||
      gle_cutoff_reference_table (1.0, ratio_above, &cutoff) !=
	GLE_CUTOFF_UNAVAILABLE ||
      gle_cutoff_reference_table (1.0, 0.0, &cutoff) !=
	GLE_CUTOFF_UNAVAILABLE) {
    fprintf (stderr, "reference table extrapolated outside its domain\n");
    failed++;
  }
  if (gle_cutoff_reference_table (-0.1, 0.1, &cutoff) !=
	GLE_CUTOFF_DOMAIN ||
      gle_cutoff_reference_table (1.0, -0.1, &cutoff) !=
	GLE_CUTOFF_DOMAIN) {
    fprintf (stderr, "reference table mislabelled invalid input\n");
    failed++;
  }

  p = gle_default_params ();
  p.theta_mic = theta_below;
  p.mu_r = 0.1;
  p.cutoff_method = GLE_CUTOFF_AUTO;
  if (gle_model_prepare (&p, &cutoff) != GLE_CUTOFF_OK ||
      cutoff.method != GLE_CUTOFF_LUO_GAO_APPROX ||
      !cutoff.luo_gao_approximation) {
    fprintf (stderr, "auto did not label the out-of-table Luo--Gao fallback\n");
    failed++;
  }

  GLEParams unavailable_case = gle_default_params ();
  unavailable_case.theta_mic = 20.0*M_PI/180.0;
  unavailable_case.mu_r = 0.1;
  unavailable_case.cutoff_method = GLE_CUTOFF_REFERENCE_TABLE;
  double unavailable_s = gle_s0 (&unavailable_case);
  double unavailable_y[4] = {
    gle_h0 (&unavailable_case), unavailable_case.theta_mic, 0.0, 0.0
  };
  int sampler_calls = 0;
  if (gle_integrate (&unavailable_case, &unavailable_s, unavailable_y, -1.0,
		     unavailable_case.smax_cap, count_sampler,
		     &sampler_calls) != GLE_ERR_DOMAIN || sampler_calls != 0) {
    fprintf (stderr, "unavailable closure reached the integration sampler\n");
    failed++;
  }

  double y[4] = { 0.2, 0.25*M_PI, 0.0, 0.0 }, dyds[4];
  p = gle_default_params ();
  if (!gle_rhs (NULL, y, dyds) || !gle_rhs (&p, NULL, dyds) ||
      !gle_rhs (&p, y, NULL)) {
    fprintf (stderr, "gle_rhs accepted a null pointer\n");
    failed++;
  }
  p.Ca = 0.01;
  p.grav = 0.0;
  p.c_slip = 1.25;
  if (gle_rhs (&p, y, dyds)) {
    fprintf (stderr, "gle_rhs unexpectedly rejected a physical state\n");
    failed++;
  }
  else {
    double want = 3.0*p.Ca*gle_mobility (y[1], p.mu_r)/
	(y[0]*(y[0] + p.c_slip*p.slip));
    failed += check_close ("c_slip enters RHS", dyds[2], want, 1.0e-15);
  }

  /* Public numerical entry points resolve AUTO once on a local copy. The raw
     RHS intentionally does not: this first proves that stale c=3 changes the
     physics, then checks that step, integration, residual and shooting APIs
     all reproduce the explicitly prepared case without mutating the caller. */
  GLEParams auto_case = gle_default_params ();
  auto_case.theta_mic = 0.5*M_PI;
  auto_case.mu_r = 0.02;
  auto_case.cutoff_method = GLE_CUTOFF_AUTO;
  auto_case.Ca = 1.0e-3;
  auto_case.grav = 0.0;
  auto_case.slip = 3.0e-2;
  auto_case.smax_cap = 6.0e-2;
  auto_case.outer_bc = GLE_OUTER_OMEGA_ZERO;
  GLEParams prepared_case = auto_case;
  if (gle_model_prepare (&prepared_case, &cutoff) != GLE_CUTOFF_OK)
    failed++;
  else {
    double state[4] = { gle_h0 (&auto_case), auto_case.theta_mic,
			0.0, gle_s0 (&auto_case)*cos (auto_case.theta_mic) };
    double raw_stale[4], raw_prepared[4];
    if (gle_rhs (&auto_case, state, raw_stale) ||
	gle_rhs (&prepared_case, state, raw_prepared) ||
	fabs (raw_stale[2] - raw_prepared[2]) < 1.0e-6) {
      fprintf (stderr, "stale c=3 did not expose the raw-RHS hazard\n");
      failed++;
    }

    double step_auto[4], step_prepared[4];
    double err_auto = gle_rkck_step (&auto_case, state, 1.0e-5, step_auto);
    double err_prepared = gle_rkck_step (&prepared_case, state, 1.0e-5,
					 step_prepared);
    failed += check_close ("AUTO prepared RK error", err_auto,
			   err_prepared, 0.0);
    for (int k = 0; k < 4; k++)
      failed += check_close ("AUTO prepared RK state", step_auto[k],
			     step_prepared[k], 0.0);

    double s_auto = gle_s0 (&auto_case), s_prepared = s_auto;
    double int_auto[4], int_prepared[4];
    for (int k = 0; k < 4; k++)
      int_auto[k] = int_prepared[k] = state[k];
    int st_auto = gle_integrate (&auto_case, &s_auto, int_auto, -1.0,
				 auto_case.smax_cap, NULL, NULL);
    int st_prepared = gle_integrate (&prepared_case, &s_prepared,
				     int_prepared, -1.0,
				     prepared_case.smax_cap, NULL, NULL);
    if (st_auto != st_prepared)
      failed++;
    failed += check_close ("AUTO prepared integration s", s_auto,
			   s_prepared, 0.0);
    for (int k = 0; k < 4; k++)
      failed += check_close ("AUTO prepared integration state", int_auto[k],
			     int_prepared[k], 0.0);

    GLESolution residual_auto, residual_prepared;
    double R_auto = gle_shoot_residual (&auto_case, 0.0, &residual_auto,
					NULL, NULL);
    double R_prepared = gle_shoot_residual (&prepared_case, 0.0,
					    &residual_prepared, NULL, NULL);
    failed += check_close ("AUTO prepared shooting residual", R_auto,
			   R_prepared, 0.0);

    GLESolution shot_auto, shot_prepared;
    int shot_st_auto = gle_shoot (&auto_case, 0.0, &shot_auto);
    int shot_st_prepared = gle_shoot (&prepared_case, 0.0, &shot_prepared);
    if (shot_st_auto || shot_st_prepared) {
      fprintf (stderr, "AUTO prepared shooting regression did not converge\n");
      failed++;
    }
    else {
      failed += check_close ("AUTO prepared shooting omega0",
			     shot_auto.omega0, shot_prepared.omega0, 0.0);
      failed += check_close ("AUTO prepared shooting theta_end",
			     shot_auto.theta_end, shot_prepared.theta_end, 0.0);
    }
    if (auto_case.c_slip != 3.0) {
      fprintf (stderr, "public solver mutated caller-owned cutoff\n");
      failed++;
    }
  }

  /* The retained continuation API prepares once for the whole march and its
     solve-Ca corrector preserves the historical p->Ca writeback contract. */
  GLEParams cont_auto = gle_default_params ();
  cont_auto.mu_r = 0.02;
  cont_auto.cutoff_method = GLE_CUTOFF_AUTO;
  GLEParams cont_prepared = cont_auto;
  if (gle_model_prepare (&cont_prepared, &cutoff) != GLE_CUTOFF_OK)
    failed++;
  else {
    GLEContOpts opts = gle_default_cont_opts ();
    opts.max_points = 2;
    GLEBranchPoint branch_auto[2], branch_prepared[2];
    int n_auto = gle_continuation (&cont_auto, &opts, branch_auto, NULL,
				   NULL, NULL);
    int n_prepared = gle_continuation (&cont_prepared, &opts,
				       branch_prepared, NULL, NULL, NULL);
    if (n_auto != 2 || n_prepared != 2) {
      fprintf (stderr, "AUTO prepared legacy continuation failed\n");
      failed++;
    }
    else
      for (int i = 0; i < 2; i++) {
	failed += check_close ("AUTO prepared continuation Ca",
			       branch_auto[i].Ca, branch_prepared[i].Ca, 0.0);
	failed += check_close ("AUTO prepared continuation omega0",
			       branch_auto[i].omega0,
			       branch_prepared[i].omega0, 0.0);
      }
    if (cont_auto.c_slip != 3.0) {
      fprintf (stderr, "continuation mutated caller-owned cutoff\n");
      failed++;
    }

    if (n_auto == 2 && n_prepared == 2) {
      GLEParams ca_auto = gle_default_params ();
      ca_auto.mu_r = 0.02;
      ca_auto.cutoff_method = GLE_CUTOFF_AUTO;
      GLEParams ca_prepared = ca_auto;
      if (gle_model_prepare (&ca_prepared, &cutoff) != GLE_CUTOFF_OK)
	failed++;
      else {
	GLESolution ca_sol_auto, ca_sol_prepared;
	int ca_st_auto = gle_solve_ca (&ca_auto, branch_auto[0].omega0,
				      opts.Ca_start, &ca_sol_auto);
	int ca_st_prepared = gle_solve_ca (&ca_prepared,
					  branch_prepared[0].omega0,
					  opts.Ca_start, &ca_sol_prepared);
	if (ca_st_auto || ca_st_prepared) {
	  fprintf (stderr, "AUTO prepared solve-Ca corrector failed\n");
	  failed++;
	}
	else {
	  failed += check_close ("AUTO prepared solve-Ca writeback",
				 ca_auto.Ca, ca_prepared.Ca, 0.0);
	  failed += check_close ("AUTO prepared solve-Ca residual",
				 ca_sol_auto.residual,
				 ca_sol_prepared.residual, 0.0);
	}
      }
    }
  }

  /* At M=0 the direct Luo--Gao GLE reduces analytically to its local
     one-phase factor. A Chan state with c=F_LG(theta) coincides at that one
     state, while the direct model updates F_LG along the trajectory. */
  p.model = GLE_MODEL_LUO_GAO;
  p.mu_r = 0.0;
  double flg = gle_luo_gao_one_phase_factor (y[1]);
  if (gle_rhs (&p, y, dyds)) {
    fprintf (stderr, "direct Luo--Gao RHS rejected its one-phase state\n");
    failed++;
  }
  else {
    double want = p.Ca*flg/(y[0]*(y[0] + flg*p.slip));
    failed += check_close ("Luo--Gao one-phase reduction", dyds[2], want,
			   1.0e-15);
  }
  p.model = GLE_MODEL_CHAN;
  p.c_slip = flg;
  double chan_at_state;
  if (gle_rhs (&p, y, dyds))
    failed++;
  chan_at_state = dyds[2];
  p.model = GLE_MODEL_LUO_GAO;
  if (gle_rhs (&p, y, dyds))
    failed++;
  else
    failed += check_close ("Chan/Luo--Gao local M=0 identity",
			   dyds[2], chan_at_state, 1.0e-15);

  p.mu_r = 0.02;
  y[1] = 0.0;
  if (!gle_rhs (&p, y, dyds)) {
    fprintf (stderr, "finite-M Luo--Gao accepted theta=0 outside its domain\n");
    failed++;
  }
  y[1] = -0.01;
  if (!gle_rhs (&p, y, dyds)) {
    fprintf (stderr, "finite-M Luo--Gao accepted negative theta\n");
    failed++;
  }
  p.mu_r = 0.0;
  if (gle_rhs (&p, y, dyds)) {
    fprintf (stderr, "one-phase Luo--Gao rejected its exact even extension\n");
    failed++;
  }
  else {
    double local_factor = gle_luo_gao_one_phase_factor (y[1]);
    failed += check_close ("one-phase Luo--Gao negative-angle extension",
			   dyds[2], p.Ca*local_factor/
			   (y[0]*(y[0] + local_factor*p.slip)), 1.0e-15);
  }

  /* Model preparation also validates the gravity-free subgrid parameter set.
     The Basilisk entry point defensively resolves an otherwise-unprepared
     automatic cutoff before doing any integration. */
  p = gle_default_params ();
  p.model = GLE_MODEL_LUO_GAO;
  p.grav = 0.0;
  if (gle_model_prepare (&p, &cutoff) != GLE_CUTOFF_OK ||
      cutoff.status != GLE_CUTOFF_NOT_USED) {
    fprintf (stderr, "valid gravity-free Luo--Gao model did not prepare\n");
    failed++;
  }
  p.slip = 0.0;
  if (gle_model_prepare (&p, &cutoff) != GLE_CUTOFF_DOMAIN) {
    fprintf (stderr, "model preparation accepted zero slip\n");
    failed++;
  }

  p = gle_default_params ();
  p.theta_mic = 0.5*M_PI;
  p.mu_r = 0.02;
  p.grav = 0.0;
  p.cutoff_method = GLE_CUTOFF_AUTO;
  double inner_grid = gle_h0 (&p);
  double caller_c = p.c_slip;
  if (!isnan (gle_dns_apparent_angle (&p, 0.0, inner_grid, p.theta_mic)) ||
      p.c_slip != caller_c) {
    fprintf (stderr, "Basilisk entry point did not defensively prepare cutoff\n");
    failed++;
  }

  /* Primary-source coefficient audit. In particular, Eq. (3.17) has one
     external sin(theta) and M^2 multiplying f2(theta) in E. */
  double theta_audit = 0.7, ratio_audit = 0.2;
  GLELuoGaoCoefficients lg;
  if (gle_luo_gao_coefficients (theta_audit, ratio_audit, &lg)) {
    fprintf (stderr, "Luo--Gao coefficient construction failed\n");
    failed++;
  }
  else {
    double sth = sin (theta_audit), s2 = sth*sth;
    /* Independent 80-digit evaluation of Luo--Gao Eqs. (3.17) and (4.10),
       rounded here to binary64. These literal anchors prevent a mutually
       consistent transcription error in the implementation and its tests. */
    failed += check_close ("Luo--Gao finite-M coefficient A", lg.a,
			   1.1936272780789502, 1.0e-13);
    failed += check_close ("Luo--Gao finite-M coefficient B", lg.b,
			   6.0758139001465701, 1.0e-13);
    failed += check_close ("Luo--Gao finite-M coefficient C", lg.c,
			   4.9399779761821755, 1.0e-13);
    failed += check_close ("Luo--Gao finite-M coefficient D", lg.d,
			   8.2447109935300596, 1.0e-13);
    failed += check_close ("Luo--Gao finite-M coefficient E", lg.e,
			   9.2018174738990376, 1.0e-13);
    double expected_e = 4.0*sth*(ratio_audit*ratio_audit*
	gle_f2 (theta_audit) + ratio_audit*M_PI +
	gle_f2 (M_PI - theta_audit));
    failed += check_close ("Luo--Gao Eq. 3.17 E coefficient", lg.e,
			   expected_e, 1.0e-13);
    failed += check_close ("Luo--Gao no-slip Cox identity",
			   lg.d*s2/lg.a,
			   3.0*gle_mobility (theta_audit, ratio_audit),
			   1.0e-13);
  }

  GLECutoffResult lg_match, lg_match_sym;
  if (gle_cutoff_luo_gao_approx (theta_audit, ratio_audit, &lg_match) ||
      gle_cutoff_luo_gao_approx (M_PI - theta_audit, 1.0/ratio_audit,
				 &lg_match_sym)) {
    fprintf (stderr, "Luo--Gao finite-M matching closure failed\n");
    failed++;
  }
  else {
    failed += check_close ("Luo--Gao finite-M Q", lg_match.Q,
			   -0.35916973691609338, 1.0e-13);
    failed += check_close ("Luo--Gao finite-M c", lg_match.c,
			   2.5079135452414427, 1.0e-13);
    failed += check_close ("Luo--Gao Q phase-exchange symmetry",
			   lg_match_sym.Q, lg_match.Q, 1.0e-13);
  }

  GLECutoffResult lg_extreme, lg_extreme_exchanged;
  double theta_extreme = 20.0*M_PI/180.0;
  double ratio_extreme = DBL_MAX;
  if (gle_cutoff_luo_gao_approx (theta_extreme, ratio_extreme, &lg_extreme) ||
      gle_cutoff_luo_gao_approx (M_PI - theta_extreme, 1.0/ratio_extreme,
				 &lg_extreme_exchanged)) {
    fprintf (stderr, "extreme-ratio Luo--Gao fallback overflowed\n");
    failed++;
  }
  else {
    failed += check_close ("extreme Luo--Gao phase-exchange Q",
			   lg_extreme.Q, lg_extreme_exchanged.Q, 2.0e-13);
    failed += check_close ("extreme Luo--Gao phase-exchange c",
			   lg_extreme.c, lg_extreme_exchanged.c, 2.0e-13);
    failed += check_close ("extreme Luo--Gao original-angle reconstruction",
			   lg_extreme.log_c,
			   1.0 + log (sin (theta_extreme)) - lg_extreme.Q,
			   5.0e-15);
  }

  GLEParams lg_state = gle_default_params ();
  double y_lg[4] = { 0.2, theta_audit, 0.0, 0.0 }, dyds_lg[4];
  lg_state.model = GLE_MODEL_LUO_GAO;
  lg_state.mu_r = ratio_audit;
  lg_state.slip = 0.03;
  lg_state.Ca = 0.01;
  lg_state.grav = 0.0;
  if (gle_rhs (&lg_state, y_lg, dyds_lg)) {
    fprintf (stderr, "Luo--Gao finite-M RHS anchor was rejected\n");
    failed++;
  }
  else
    failed += check_close ("Luo--Gao finite-M viscous RHS", dyds_lg[2],
			   0.51874305933760114, 1.0e-13);

  if (gle_luo_gao_coefficients (theta_audit, 0.0, &lg))
    failed++;
  else {
    double sth = sin (theta_audit), s2 = sth*sth;
    double film = 0.2, lambda = 0.03;
    double rational = (lg.d*film*s2 + lg.e*lambda*s2*sth)/
	(lg.a*film*film + lg.b*lambda*film*sth +
	 lg.c*lambda*lambda*s2);
    double local_factor = gle_luo_gao_one_phase_factor (theta_audit);
    failed += check_close ("Luo--Gao full-coefficient M=0 reduction",
			   rational, local_factor/(film + local_factor*lambda),
			   1.0e-13);
  }

  GLELuoGaoCoefficients zero_disc = { 1.0, 2.0, 1.0, 1.0, 2.0 };
  double log_term;
  if (gle_luo_gao_matching_log_term (&zero_disc, &log_term))
    failed++;
  else
    failed += check_close ("Luo--Gao Q Omega/B -> 0", log_term, 1.0,
			   1.0e-15);
  GLELuoGaoCoefficients scaled_zero_disc =
    { 1.0e200, 2.0e200, 1.0e200, 1.0e200, 2.0e200 };
  if (gle_luo_gao_matching_log_term (&scaled_zero_disc, &log_term)) {
    fprintf (stderr, "scaled Luo--Gao discriminant overflowed\n");
    failed++;
  }
  else
    failed += check_close ("scaled Luo--Gao Q Omega/B -> 0", log_term, 1.0,
			   1.0e-15);
  GLELuoGaoCoefficients unit_ratio =
    { 1.0e-100, 1.0, 1.0e-100, 1.0, 1.0 };
  if (gle_luo_gao_matching_log_term (&unit_ratio, &log_term)) {
    fprintf (stderr, "Luo--Gao Q Omega/B -> 1 was singular\n");
    failed++;
  }
  else {
    double expected = (2.0e-100 - 1.0)/2.0*
	(2.0*log (2.0) - log (4.0e-200));
    failed += check_close ("Luo--Gao Q Omega/B -> 1", log_term, expected,
			   1.0e-13);
  }

  /* Chan resolves one case-constant c; Luo--Gao evaluates a different local
     slip factor as theta changes along the same trajectory. */
  p = gle_default_params ();
  p.Ca = 0.01;
  p.grav = 0.0;
  p.c_slip = 1.7;
  double c_case = p.c_slip;
  y[1] = 0.4;
  failed += gle_rhs (&p, y, dyds);
  y[1] = 1.2;
  failed += gle_rhs (&p, y, dyds);
  failed += check_close ("Chan c remains case-constant", p.c_slip,
			 c_case, 0.0);
  if (gle_luo_gao_one_phase_factor (0.4) ==
      gle_luo_gao_one_phase_factor (1.2)) {
    fprintf (stderr, "Luo--Gao local slip factor did not vary with angle\n");
    failed++;
  }

  p.model = 99;
  if (!gle_rhs (&p, y, dyds)) {
    fprintf (stderr, "invalid runtime model was accepted by gle_rhs\n");
    failed++;
  }

  double x;
  long n;
  if (gle_parse_double ("nan", &x) || gle_parse_double ("1e9999", &x) ||
      gle_parse_double ("1.0junk", &x) || !gle_parse_double ("1.25", &x)) {
    fprintf (stderr, "strict floating-point parsing regression\n");
    failed++;
  }
  if (gle_parse_long ("3.5", &n) || gle_parse_long ("2x", &n) ||
      !gle_parse_long ("2500", &n)) {
    fprintf (stderr, "strict integer parsing regression\n");
    failed++;
  }
  int selector;
  if (!gle_parse_model ("chan", &selector) || selector != GLE_MODEL_CHAN ||
      !gle_parse_model ("luo-gao", &selector) ||
      selector != GLE_MODEL_LUO_GAO || gle_parse_model ("mystery", &selector)) {
    fprintf (stderr, "runtime model parsing regression\n");
    failed++;
  }
  if (!gle_parse_cutoff_method ("scott_hocking", &selector) ||
      selector != GLE_CUTOFF_SCOTT_HOCKING ||
      !gle_parse_cutoff_method ("corrected_right_angle", &selector) ||
      selector != GLE_CUTOFF_CORRECTED_RIGHT_ANGLE ||
      gle_parse_cutoff_method ("mystery", &selector)) {
    fprintf (stderr, "cutoff method parsing regression\n");
    failed++;
  }

  GLECollocation c;
  if (!gle_colloc_alloc (&c, 0) || !gle_colloc_alloc (&c, -1)) {
    fprintf (stderr, "invalid collocation mesh was accepted\n");
    failed++;
  }

  /* Direct collocation residual/Jacobian APIs also prepare one local copy.
     Their outputs must match an explicit preparation, while the prepared-only
     kernel demonstrates that stale c=3 would otherwise change the system. */
  GLECollocation coll_auto, coll_prepared;
  if (gle_colloc_alloc (&coll_auto, 2) ||
      gle_colloc_alloc (&coll_prepared, 2)) {
    fprintf (stderr, "collocation preparation regression allocation failed\n");
    failed++;
  }
  else {
    GLEParams cp_auto = gle_default_params ();
    cp_auto.theta_mic = 1.0;
    cp_auto.mu_r = 0.02;
    cp_auto.cutoff_method = GLE_CUTOFF_AUTO;
    cp_auto.slip = 3.0e-2;
    cp_auto.Ca = 1.0e-3;
    GLEParams cp_prepared = cp_auto;
    if (gle_model_prepare (&cp_prepared, &cutoff) != GLE_CUTOFF_OK)
      failed++;
    else {
      coll_auto.s_split = coll_prepared.s_split = 4.0e-2;
      coll_auto.s_end = coll_prepared.s_end = 8.0e-2;
      coll_auto.Ca = coll_prepared.Ca = cp_auto.Ca;
      for (int i = 0; i <= 2; i++) {
	double ss = gle_colloc_node_s (&coll_auto, &cp_auto, i,
					      coll_auto.s_end);
	double ds = ss - gle_s0 (&cp_auto);
	double vals[4] = {
	  gle_h0 (&cp_auto) + ds*sin (cp_auto.theta_mic),
	  cp_auto.theta_mic, 0.0,
	  gle_s0 (&cp_auto)*cos (cp_auto.theta_mic) +
	    ds*cos (cp_auto.theta_mic)
	};
	for (int k = 0; k < 4; k++)
	  coll_auto.y[4*i + k] = coll_prepared.y[4*i + k] = vals[k];
      }
      double stale_res[12], prepared_raw_res[12], public_auto_res[12],
	public_prepared_res[12];
      gle_colloc_band_residual_prepared (&coll_auto, &cp_auto, coll_auto.y,
					 coll_auto.s_end, stale_res);
      gle_colloc_band_residual_prepared (&coll_prepared, &cp_prepared,
					 coll_prepared.y,
					 coll_prepared.s_end,
					 prepared_raw_res);
      if (fabs (stale_res[5] - prepared_raw_res[5]) < 1.0e-8) {
	fprintf (stderr, "stale c=3 did not expose collocation hazard\n");
	failed++;
      }
      gle_colloc_band_residual (&coll_auto, &cp_auto, coll_auto.y,
				coll_auto.s_end, public_auto_res);
      gle_colloc_band_residual (&coll_prepared, &cp_prepared,
				coll_prepared.y, coll_prepared.s_end,
				public_prepared_res);
      for (int i = 0; i < 12; i++)
	failed += check_close ("AUTO prepared collocation residual",
			       public_auto_res[i], public_prepared_res[i], 0.0);

      double rb_auto[2], rbca_auto[2], rbs_auto[2];
      double rb_prepared[2], rbca_prepared[2], rbs_prepared[2];
      gle_colloc_assemble (&coll_auto, &cp_auto, 0.2, rb_auto, rbca_auto,
			   rbs_auto);
      gle_colloc_assemble (&coll_prepared, &cp_prepared, 0.2, rb_prepared,
			   rbca_prepared, rbs_prepared);
      for (int i = 0; i < 12; i++) {
	failed += check_close ("AUTO prepared collocation assembly residual",
			       coll_auto.res[i], coll_prepared.res[i], 0.0);
	failed += check_close ("AUTO prepared collocation Ca column",
			       coll_auto.colCa[i], coll_prepared.colCa[i], 0.0);
	failed += check_close ("AUTO prepared collocation s column",
			       coll_auto.colS[i], coll_prepared.colS[i], 0.0);
      }
      for (int i = 0; i < 2; i++) {
	failed += check_close ("AUTO prepared collocation border", rb_auto[i],
			       rb_prepared[i], 0.0);
	failed += check_close ("AUTO prepared collocation Ca border",
			       rbca_auto[i], rbca_prepared[i], 0.0);
	failed += check_close ("AUTO prepared collocation s border",
			       rbs_auto[i], rbs_prepared[i], 0.0);
      }
      if (cp_auto.c_slip != 3.0) {
	fprintf (stderr, "collocation mutated caller-owned cutoff\n");
	failed++;
      }
    }
    gle_colloc_free (&coll_auto);
    gle_colloc_free (&coll_prepared);
  }

  GLECollocation seed_bad;
  if (gle_colloc_alloc (&seed_bad, 2))
    failed++;
  else {
    GLEParams seed_params = gle_default_params ();
    seed_params.mu_r = 0.02;
    seed_params.cutoff_method = GLE_CUTOFF_AUTO;
    seed_params.smax_cap = 1.1*gle_s0 (&seed_params);
    GLESolution incomplete = { 0 };
    if (!gle_colloc_seed_from_shoot (&seed_bad, &seed_params, &incomplete)) {
      fprintf (stderr, "collocation seed accepted an incomplete trajectory\n");
      failed++;
    }
    gle_colloc_free (&seed_bad);
  }

  /* Seeded direct solve and the public Delta-march wrapper both use one
     prepared copy for all Newton/Jacobian evaluations. */
  GLEParams march_auto = gle_default_params ();
  march_auto.mu_r = 0.02;
  march_auto.cutoff_method = GLE_CUTOFF_AUTO;
  march_auto.Ca = 1.0e-6;
  GLEParams march_prepared = march_auto;
  if (gle_model_prepare (&march_prepared, &cutoff) != GLE_CUTOFF_OK)
    failed++;
  else {
    GLESolution march_shot_auto, march_shot_prepared;
    int shot_a = gle_shoot (&march_auto,
	gle_static_curvature (march_auto.theta_mic, march_auto.grav),
	&march_shot_auto);
    int shot_b = gle_shoot (&march_prepared,
	gle_static_curvature (march_prepared.theta_mic, march_prepared.grav),
	&march_shot_prepared);
    GLECollocation march_c_auto, march_c_prepared;
    int alloc_a = gle_colloc_alloc (&march_c_auto, 100);
    int alloc_b = gle_colloc_alloc (&march_c_prepared, 100);
    if (shot_a || shot_b || alloc_a || alloc_b) {
      fprintf (stderr, "collocation march regression setup failed\n");
      failed++;
    }
    else if (gle_colloc_seed_from_shoot (&march_c_auto, &march_auto,
					 &march_shot_auto) ||
	     gle_colloc_seed_from_shoot (&march_c_prepared, &march_prepared,
					 &march_shot_prepared)) {
      fprintf (stderr, "collocation march regression seed failed\n");
      failed++;
    }
    else {
      int N = march_c_auto.N;
      double D_auto = march_c_auto.y[4*N + 3] +
	march_c_auto.y[4*N + 2]/march_auto.grav;
      double D_prepared = march_c_prepared.y[4*N + 3] +
	march_c_prepared.y[4*N + 2]/march_prepared.grav;
      int it_auto = -1, it_prepared = -1;
      int solve_auto = gle_colloc_solve (&march_c_auto, &march_auto,
					  D_auto, &it_auto);
      int solve_prepared = gle_colloc_solve (&march_c_prepared,
					      &march_prepared,
					      D_prepared, &it_prepared);
      if (solve_auto || solve_prepared) {
	fprintf (stderr, "AUTO prepared direct collocation solve failed\n");
	failed++;
      }
      else {
	failed += check_close ("AUTO prepared direct collocation Ca",
			       march_c_auto.Ca, march_c_prepared.Ca, 0.0);
	failed += check_close ("AUTO prepared direct collocation Delta",
			       march_c_auto.Delta, march_c_prepared.Delta, 0.0);
	int nm_auto = gle_colloc_march (
	  &march_c_auto, &march_auto, D_auto + 0.01, 0.001, 0.01, 1,
	  NULL, NULL, NULL, 0);
	int nm_prepared = gle_colloc_march (
	  &march_c_prepared, &march_prepared, D_prepared + 0.01, 0.001,
	  0.01, 1, NULL, NULL, NULL, 0);
	if (nm_auto != 1 || nm_prepared != 1) {
	  fprintf (stderr, "AUTO prepared collocation march failed\n");
	  failed++;
	}
	else {
	  failed += check_close ("AUTO prepared collocation march Ca",
				 march_c_auto.Ca, march_c_prepared.Ca, 0.0);
	  failed += check_close ("AUTO prepared collocation march Delta",
				 march_c_auto.Delta, march_c_prepared.Delta, 0.0);
	}
      }
      if (march_auto.c_slip != 3.0) {
	fprintf (stderr, "collocation solve mutated caller-owned cutoff\n");
	failed++;
      }
    }
    if (!alloc_a)
      gle_colloc_free (&march_c_auto);
    if (!alloc_b)
      gle_colloc_free (&march_c_prepared);
  }

  c.N = 2;
  p = gle_default_params ();
  p.grav = 1.0;
  if (gle_colloc_march (&c, &p, 2.0, 0.0, 0.02, 10, NULL,
			NULL, NULL, 0) != 0 ||
      gle_colloc_march (&c, &p, 2.0, 0.01, 0.02, 0, NULL,
			NULL, NULL, 0) != 0) {
    fprintf (stderr, "invalid continuation controls were accepted\n");
    failed++;
  }

  if (failed) {
    fprintf (stderr, "test-gle-core: %d check(s) failed\n", failed);
    return 1;
  }
  puts ("test-gle-core: all checks passed");
  return 0;
}
