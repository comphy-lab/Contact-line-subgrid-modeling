/**
# gle-params.h — key=value runtime parameters for the GLE drivers

Minimal runtime-parameter layer following the CoMPhy convention (parameter
files of `key = value` lines, `#` comments, CLI `key=value` overrides). The
drivers in [gle-ode/](../gle-ode/) call `gle_params_load()` with `argc/argv`;
the first non-`key=value` argument is treated as a parameter file path. Any
further bare (non-`key=value`) argument is ignored with a warning on
`stderr`.

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
### gle_bad_numeric()

Reports a value that failed to parse as a number, and returns `1` so the
caller treats the key as *recognised but ignored* rather than unknown (an
unparsable value should not also trigger a spurious "unknown key" warning).
The assignment is skipped: `gle_kv_apply()`'s caller simply keeps whatever
default or earlier value the field already held.
*/
static int gle_bad_numeric (const char *val, const char *key) {
  fprintf (stderr, "gle-params: bad numeric value '%s' for key '%s'\n",
	   val, key);
  return 1;
}

/**
### gle_parse_long()

Strict integer parse of `val` via `strtol()` with an `endptr` check (the
whole string must be consumed, and at least one digit must have been read).

#### Returns
`1` and writes `*out` on success, `0` on a malformed value.
*/
static int gle_parse_long (const char *val, long *out) {
  char *endp;
  long v = strtol (val, &endp, 10);
  if (endp == val || *endp != '\0')
    return 0;
  *out = v;
  return 1;
}

/**
### gle_kv_apply()

Applies one `key=value` pair to the parameter structs. Unknown keys warn on
`stderr` (they do not abort: parameter files may be shared between drivers).
Numeric values are parsed with `strtod()`/`strtol()` and checked against
`endptr`; a value that does not fully parse (e.g. `Ca=abc`) is reported via
`gle_bad_numeric()` and the assignment is skipped, but the key still counts
as recognised (see `gle_bad_numeric()`).

#### Returns
`1` if the key was recognised, `0` otherwise.
*/
static int gle_kv_apply (GLEParams *p, GLEContOpts *o, const char *key,
			 const char *val) {
  char *endp;
  double x = strtod (val, &endp);
  int x_ok = (endp != val && *endp == '\0');
#define GLE_KEY(name, target)						\
  if (!strcmp (key, name)) {						\
    if (!x_ok) return gle_bad_numeric (val, key);			\
    target = x;								\
    return 1;								\
  }
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
    if (!x_ok)
      return gle_bad_numeric (val, key);
    p->theta_mic = x*M_PI/180.0;
    return 1;
  }
  if (!strcmp (key, "geometry")) {
    if (!strcmp (val, "vertical")) p->geometry = GLE_PLATE_VERTICAL;
    else if (!strcmp (val, "horizontal")) p->geometry = GLE_PLATE_HORIZONTAL;
    else fprintf (stderr, "gle-params: unknown geometry '%s'\n", val);
    return 1;
  }
  if (!strcmp (key, "outer_bc")) {
    if (!strcmp (val, "manifold")) p->outer_bc = GLE_OUTER_STATIC_MENISCUS;
    else if (!strcmp (val, "omega_zero")) p->outer_bc = GLE_OUTER_OMEGA_ZERO;
    else fprintf (stderr, "gle-params: unknown outer_bc '%s'\n", val);
    return 1;
  }
  if (!strcmp (key, "max_steps")) {
    long v;
    if (!gle_parse_long (val, &v))
      return gle_bad_numeric (val, key);
    p->max_steps = v;
    return 1;
  }
  if (o) {
#define GLE_KEY_O(name, target)					\
    if (!strcmp (key, name)) {						\
      if (!x_ok) return gle_bad_numeric (val, key);			\
      target = x;							\
      return 1;								\
    }
    GLE_KEY_O ("Ca_start", o->Ca_start);
    GLE_KEY_O ("alpha0", o->alpha0);
    GLE_KEY_O ("alpha_min", o->alpha_min);
    GLE_KEY_O ("alpha_max", o->alpha_max);
    GLE_KEY_O ("Delta_max", o->Delta_max);
    GLE_KEY_O ("Ca_stop_min", o->Ca_stop_min);
#undef GLE_KEY_O
    if (!strcmp (key, "max_points")) {
      long v;
      if (!gle_parse_long (val, &v))
	return gle_bad_numeric (val, key);
      o->max_points = (int) v;
      return 1;
    }
    if (!strcmp (key, "verbose")) {
      long v;
      if (!gle_parse_long (val, &v))
	return gle_bad_numeric (val, key);
      o->verbose = (int) v;
      return 1;
    }
  }
  return 0;
}

/**
### gle_params_load()

Populates defaults, then applies (in order): an optional parameter file given
as the first bare CLI argument, then every CLI `key=value` override left to
right. Only the first bare argument is treated as a parameter file path; any
subsequent bare argument is reported to `stderr` as an ignored extra
positional argument rather than being opened. `o` may be `NULL` for drivers
without continuation options.
*/
static void gle_params_load (int argc, char *argv[], GLEParams *p,
			     GLEContOpts *o) {
  int have_file = 0;
  for (int i = 1; i < argc; i++) {
    char *eq = strchr (argv[i], '=');
    if (!eq) {                        /* parameter file */
      if (have_file) {
	fprintf (stderr,
		 "gle-params: ignoring extra positional argument '%s'\n",
		 argv[i]);
	continue;
      }
      have_file = 1;
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
