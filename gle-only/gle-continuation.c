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
`mesh_N` (collocation cells, default 1200), `dDelta` (initial march step in
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
#include "gle-params.h"
#include "gle-collocate.h"

int main (int argc, char *argv[]) {
  GLEParams p = gle_default_params ();
  GLEContOpts opts = gle_default_cont_opts ();
  const char *out_path = "gle-branch.csv";
  int mesh_N = 1200;
  double dDelta0 = 2.0e-3;

  for (int i = 1; i < argc; i++) {
    if (!strncmp (argv[i], "branch_out=", 11)) {
      out_path = argv[i] + 11;
      argv[i] = (char *) "";
    }
    else if (!strncmp (argv[i], "mesh_N=", 7)) {
      mesh_N = atoi (argv[i] + 7);
      argv[i] = (char *) "";
    }
    else if (!strncmp (argv[i], "dDelta=", 7)) {
      dDelta0 = atof (argv[i] + 7);
      argv[i] = (char *) "";
    }
  }
  int ac = 1;
  for (int i = 1; i < argc; i++)
    if (argv[i][0] != '\0')
      argv[ac++] = argv[i];

  gle_params_load (ac, argv, &p, &opts);

  fprintf (stderr,
	   "gle-continuation: theta_e = %g deg, slip = %.3e, mu_r = %g, "
	   "H_match = %g, mesh_N = %d\n",
	   p.theta_mic*180.0/M_PI, p.slip, p.mu_r, p.H_match, mesh_N);

  /* --- seed: shooting solve on the quasi-static lower branch --- */
  GLESolution sol;
  p.Ca = opts.Ca_start;
  if (gle_shoot (&p, gle_static_curvature (p.theta_mic), &sol)) {
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
    return 1;
  }

  FILE *csv = fopen (out_path, "w");
  if (!csv) {
    fprintf (stderr, "gle-continuation: cannot write '%s'\n", out_path);
    return 1;
  }
  gle_branch_csv_header (csv);

  double fold_Ca = NAN, fold_Delta = NAN;
  int n = gle_colloc_march (&c, &p, opts.Delta_max, dDelta0,
			    opts.max_points, csv, &fold_Ca, &fold_Delta,
			    opts.verbose);
  fclose (csv);

  if (n < 3) {
    fprintf (stderr, "gle-continuation: branch tracing failed (n = %d)\n", n);
    gle_colloc_free (&c);
    return 1;
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
