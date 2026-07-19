/**
# gle-params.h — key=value runtime parameters for the GLE drivers

Minimal runtime-parameter layer following the CoMPhy convention (parameter
files of `key = value` lines, `#` comments, CLI `key=value` overrides). The
drivers in [gle-only/](../gle-only/) call `gle_params_load()` with `argc/argv`;
the first non-`key=value` argument is treated as a parameter file path.

Angles are given in **degrees** in parameter files (`theta_mic_deg`) and
converted here; every other quantity is in capillary-length units as defined
in [gle-model.h](gle-model.h).

## Author
Vatsal Sanjay
Email: vatsal.sanjay@comphy-lab.org
CoMPhy Lab, Department of Physics, Durham University
Last updated: Jul 20, 2026
*/

#ifndef GLE_PARAMS_H
#define GLE_PARAMS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gle-model.h"
#include "gle-continuation.h"

/**
### gle_kv_apply()

Applies one `key=value` pair to the parameter structs. Unknown keys warn on
`stderr` (they do not abort: parameter files may be shared between drivers).

#### Returns
`1` if the key was recognised, `0` otherwise.
*/
static int gle_kv_apply (GLEParams *p, GLEContOpts *o, const char *key,
			 const char *val) {
  double x = atof (val);
#define GLE_KEY(name, target)			\
  if (!strcmp (key, name)) { target = x; return 1; }
  GLE_KEY ("Ca", p->Ca);
  GLE_KEY ("mu_r", p->mu_r);
  GLE_KEY ("slip", p->slip);
  GLE_KEY ("grav", p->grav);
  GLE_KEY ("s0", p->s0);
  GLE_KEY ("h0", p->h0);
  GLE_KEY ("H_match", p->H_match);
  GLE_KEY ("smax_cap", p->smax_cap);
  GLE_KEY ("rtol", p->rtol);
  GLE_KEY ("atol", p->atol);
#undef GLE_KEY
  if (!strcmp (key, "theta_mic_deg")) {
    p->theta_mic = x*M_PI/180.0;
    return 1;
  }
  if (!strcmp (key, "geometry")) {
    if (!strcmp (val, "vertical")) p->geometry = GLE_PLATE_VERTICAL;
    else if (!strcmp (val, "horizontal")) p->geometry = GLE_PLATE_HORIZONTAL;
    else fprintf (stderr, "gle-params: unknown geometry '%s'\n", val);
    return 1;
  }
  if (!strcmp (key, "max_steps")) {
    p->max_steps = atol (val);
    return 1;
  }
  if (o) {
#define GLE_KEY_O(name, target)			\
    if (!strcmp (key, name)) { target = x; return 1; }
    GLE_KEY_O ("Ca_start", o->Ca_start);
    GLE_KEY_O ("alpha0", o->alpha0);
    GLE_KEY_O ("alpha_min", o->alpha_min);
    GLE_KEY_O ("alpha_max", o->alpha_max);
    GLE_KEY_O ("Delta_max", o->Delta_max);
    GLE_KEY_O ("Ca_stop_min", o->Ca_stop_min);
#undef GLE_KEY_O
    if (!strcmp (key, "max_points")) {
      o->max_points = atoi (val);
      return 1;
    }
    if (!strcmp (key, "verbose")) {
      o->verbose = atoi (val);
      return 1;
    }
  }
  return 0;
}

/**
### gle_params_load()

Populates defaults, then applies (in order): an optional parameter file given
as a bare CLI argument, then every CLI `key=value` override left to right.
`o` may be `NULL` for drivers without continuation options.
*/
static void gle_params_load (int argc, char *argv[], GLEParams *p,
			     GLEContOpts *o) {
  for (int i = 1; i < argc; i++) {
    char *eq = strchr (argv[i], '=');
    if (!eq) {                        /* parameter file */
      FILE *fp = fopen (argv[i], "r");
      if (!fp) {
	fprintf (stderr, "gle-params: cannot open '%s'\n", argv[i]);
	exit (1);
      }
      char line[512];
      while (fgets (line, sizeof line, fp)) {
	char *hash = strchr (line, '#');
	if (hash) *hash = '\0';
	char key[128], val[128];
	if (sscanf (line, " %127[^= \t] = %127s", key, val) == 2)
	  if (!gle_kv_apply (p, o, key, val))
	    fprintf (stderr, "gle-params: unknown key '%s' in %s\n",
		     key, argv[i]);
      }
      fclose (fp);
    }
  }
  for (int i = 1; i < argc; i++) {    /* CLI overrides win */
    char *eq = strchr (argv[i], '=');
    if (eq) {
      *eq = '\0';
      if (!gle_kv_apply (p, o, argv[i], eq + 1))
	fprintf (stderr, "gle-params: unknown key '%s'\n", argv[i]);
      *eq = '=';
    }
  }
}

#endif /* GLE_PARAMS_H */
