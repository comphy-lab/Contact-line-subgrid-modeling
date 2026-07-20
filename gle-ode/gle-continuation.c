/**
# gle-continuation.c — trace the dip-coating bifurcation diagram

Standalone driver reproducing the theory curve of Fig. 4b of Snoeijer &
Andreotti, *Annu. Rev. Fluid Mech.* 45 (2013): the steady meniscus branch
$\Delta(\mathrm{Ca})$ from $\mathrm{Ca} \to 0$, through the saddle-node fold
at $\mathrm{Ca}^{*}$ (maximum plate speed, $\theta_{\mathrm{app}} \to 0$,
$\Delta \to \sqrt{2}$), and up the high-$\Delta$ branch towards the critical
Landau–Levich asymptote.

Pipeline: a robust shooting solve seeds the branch at small `Ca_start`
([src-local/gle-shoot.h](../src-local/gle-shoot.h)); the fixed-mesh
collocation solver then marches the target meniscus rise $\Delta^{*}$ —
monotone along the whole branch, so the fold needs no special handling
([src-local/gle-collocate.h](../src-local/gle-collocate.h)).

## Usage

```bash
./gle-continuation [file.params] [key=value ...]
# e.g. reproduce the Fig. 4b theory curve:
./gle-continuation fig4b.params branch_out=branch.csv
```

Driver-specific keys: `branch_out` (CSV path, default `gle-branch.csv`),
`mesh_N` (collocation cells, default 2500), `dDelta` (initial march step in
$\Delta$, default $2\times10^{-3}$).

On completion the fold $(\mathrm{Ca}^{*}, \Delta^{*})$ and the last computed
point (whose $\mathrm{Ca}$ approaches the Landau–Levich critical speed as
$\Delta$ grows) are printed.

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "gle-params.h"
#include "gle-collocate.h"

int main (int argc, char *argv[]) {
  GLEParams p = gle_default_params ();
  GLECutoffResult cutoff;
  gle_cutoff_result_reset (&cutoff, p.cutoff_method, GLE_CUTOFF_UNAVAILABLE);
  GLEContOpts opts = gle_default_cont_opts ();
  const char *out_path = "gle-branch.csv";
  int mesh_N = 2500;
  double dDelta0 = 2.0e-3;
  double dDelta_cap = 0.02;
  int driver_bad = 0;

  for (int i = 1; i < argc; i++) {
    if (!strncmp (argv[i], "branch_out=", 11)) {
      out_path = argv[i] + 11;
      if (!out_path[0]) {
	fprintf (stderr, "gle-continuation: branch_out must not be empty\n");
	driver_bad = 1;
      }
      argv[i] = (char *) "";
    }
    else if (!strncmp (argv[i], "mesh_N=", 7)) {
      long v;
      if (!gle_parse_long (argv[i] + 7, &v) || v < 2 ||
	  v > (INT_MAX - 4)/4) {
	fprintf (stderr, "gle-continuation: mesh_N must be an integer in [2, %d]\n",
		 (INT_MAX - 4)/4);
	driver_bad = 1;
      }
      else
	mesh_N = (int) v;
      argv[i] = (char *) "";
    }
    else if (!strncmp (argv[i], "dDelta=", 7)) {
      if (!gle_parse_double (argv[i] + 7, &dDelta0) || dDelta0 <= 0.0) {
	fprintf (stderr, "gle-continuation: dDelta must be finite and positive\n");
	driver_bad = 1;
      }
      argv[i] = (char *) "";
    }
    else if (!strncmp (argv[i], "dDelta_max=", 11)) {
      if (!gle_parse_double (argv[i] + 11, &dDelta_cap) ||
	  dDelta_cap <= 0.0) {
	fprintf (stderr,
		 "gle-continuation: dDelta_max must be finite and positive\n");
	driver_bad = 1;
      }
      argv[i] = (char *) "";
    }
  }
  int ac = 1;
  for (int i = 1; i < argc; i++)
    if (argv[i][0] != '\0')
      argv[ac++] = argv[i];

  if (gle_params_load (ac, argv, &p, &opts))
    driver_bad = 1;
  if (gle_params_validate (&p, "gle-continuation") ||
      gle_cont_opts_validate (&opts, "gle-continuation"))
    driver_bad = 1;
  if (!driver_bad && gle_params_prepare (&p, &cutoff, "gle-continuation"))
    driver_bad = 1;
  if (p.geometry != GLE_PLATE_VERTICAL ||
      p.outer_bc != GLE_OUTER_STATIC_MENISCUS) {
    fprintf (stderr, "gle-continuation: collocation requires geometry=vertical "
	     "and outer_bc=manifold\n");
    driver_bad = 1;
  }
  if (dDelta0 > dDelta_cap) {
    fprintf (stderr, "gle-continuation: dDelta must not exceed dDelta_max\n");
    driver_bad = 1;
  }
  if (driver_bad)
    return 2;

  fprintf (stderr,
	   "gle-continuation: model = %s, theta_e = %g deg, slip = %.3e, "
	   "mu_r = %g, H_match = %g, mesh_N = %d\n",
	   gle_model_name (p.model), p.theta_mic*180.0/M_PI, p.slip, p.mu_r,
	   p.H_match, mesh_N);
  if (p.model == GLE_MODEL_CHAN)
    fprintf (stderr,
	     "gle-continuation: c_method = %s -> %s, c_slip = %.10g, "
	     "Q = %.10g, luo_gao_approximation = %s\n",
	     gle_cutoff_method_name (p.cutoff_method),
	     gle_cutoff_method_name (cutoff.method), p.c_slip, cutoff.Q,
	     cutoff.luo_gao_approximation ? "yes" : "no");
  else
    fprintf (stderr, "gle-continuation: Chan cutoff = not_used\n");

  /* --- seed: shooting solve on the quasi-static lower branch --- */
  GLESolution sol;
  p.Ca = opts.Ca_start;
  if (gle_shoot (&p, gle_static_curvature (p.theta_mic, p.grav), &sol)) {
    fprintf (stderr, "gle-continuation: seeding shot failed at Ca = %g\n",
	     p.Ca);
    return 1;
  }

  GLECollocation c;
  if (gle_colloc_alloc (&c, mesh_N)) {
    fprintf (stderr, "gle-continuation: out of memory (mesh_N = %d)\n",
	     mesh_N);
    return 1;
  }
  if (gle_colloc_seed_from_shoot (&c, &p, &sol)) {
    fprintf (stderr, "gle-continuation: mesh seeding failed\n");
    gle_colloc_free (&c);
    return 1;
  }

  FILE *csv = fopen (out_path, "w");
  if (!csv) {
    fprintf (stderr, "gle-continuation: cannot write '%s'\n", out_path);
    gle_colloc_free (&c);
    return 1;
  }
  gle_branch_csv_header (csv);

  double fold_Ca = NAN, fold_Delta = NAN;
  int n = gle_colloc_march (&c, &p, opts.Delta_max, dDelta0, dDelta_cap,
			    opts.max_points, csv, &fold_Ca, &fold_Delta,
			    opts.verbose);
  fclose (csv);

  if (n < 3) {
    fprintf (stderr, "gle-continuation: branch tracing failed (n = %d)\n", n);
    gle_colloc_free (&c);
    return 1;
  }

  printf ("gle_model     = %s\n", gle_model_name (p.model));
  printf ("c_method      = %s\n",
	  cutoff.status == GLE_CUTOFF_NOT_USED ? "not_used" :
	  gle_cutoff_method_name (cutoff.method));
  printf ("c_status      = %s\n", gle_cutoff_status_name (cutoff.status));
  if (p.model == GLE_MODEL_CHAN) {
    printf ("c_luo_gao_approximation = %s\n",
	    cutoff.luo_gao_approximation ? "yes" : "no");
    printf ("c_slip        = %.10g\n", p.c_slip);
    printf ("Q             = %.10g\n", cutoff.Q);
  }
  printf ("points        = %d\n", n);
  printf ("fold_Ca       = %.10e\n", fold_Ca);
  printf ("fold_Delta    = %.10e\n", fold_Delta);
  printf ("last_Ca       = %.10e\n", c.Ca);
  printf ("last_Delta    = %.10e\n", c.Delta);
  printf ("branch        -> %s\n", out_path);

  gle_colloc_free (&c);
  return 0;
}
