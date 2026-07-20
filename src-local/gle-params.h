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

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gle-model.h"
#include "gle-slip-closure.h"
#include "gle-continuation.h"

/**
### gle_bad_numeric()

Reports a value that failed to parse as a number.  A malformed recognised
value is fatal to the driver's parameter-loading pass: silently retaining a
default makes a mistyped physical parameter particularly difficult to spot.
*/
static int gle_bad_numeric (const char *val, const char *key) {
  fprintf (stderr, "gle-params: bad numeric value '%s' for key '%s'\n",
	   val, key);
  return -1;
}

/**
### gle_parse_double()

Strict finite floating-point parse.  The entire string must be consumed;
overflow, underflow, `nan`, and `inf` are rejected.
*/
static int gle_parse_double (const char *val, double *out) {
  char *endp;
  errno = 0;
  double v = strtod (val, &endp);
  if (endp == val || *endp != '\0' || errno == ERANGE || !isfinite (v))
    return 0;
  *out = v;
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
  errno = 0;
  long v = strtol (val, &endp, 10);
  if (endp == val || *endp != '\0' || errno == ERANGE)
    return 0;
  *out = v;
  return 1;
}

/** Trim leading and trailing ASCII/C-locale whitespace in place. */
static char *gle_trim_whitespace (char *text) {
  while (*text && isspace ((unsigned char) *text))
    text++;
  char *end = text + strlen (text);
  while (end > text && isspace ((unsigned char) end[-1]))
    end--;
  *end = '\0';
  return text;
}

/**
### gle_parse_model(), gle_parse_cutoff_method()

Parse the named runtime physics selectors. Canonical spellings use underscores;
hyphenated Luo--Gao spellings and descriptive cutoff aliases are accepted for
command-line convenience. Output always uses the canonical names returned by
`gle_model_name()` and `gle_cutoff_method_name()`.
*/
static int gle_parse_model (const char *val, int *out) {
  if (!strcmp (val, "chan") || !strcmp (val, "chan_etal"))
    *out = GLE_MODEL_CHAN;
  else if (!strcmp (val, "luo_gao") || !strcmp (val, "luo-gao"))
    *out = GLE_MODEL_LUO_GAO;
  else
    return 0;
  return 1;
}

static int gle_parse_cutoff_method (const char *val, int *out) {
  if (!strcmp (val, "manual"))
    *out = GLE_CUTOFF_MANUAL;
  else if (!strcmp (val, "auto"))
    *out = GLE_CUTOFF_AUTO;
  else if (!strcmp (val, "scott_hocking"))
    *out = GLE_CUTOFF_SCOTT_HOCKING;
  else if (!strcmp (val, "corrected_right_angle"))
    *out = GLE_CUTOFF_CORRECTED_RIGHT_ANGLE;
  else if (!strcmp (val, "reference_table") || !strcmp (val, "table"))
    *out = GLE_CUTOFF_REFERENCE_TABLE;
  else if (!strcmp (val, "luo_gao_approx") ||
	   !strcmp (val, "luo-gao-approx"))
    *out = GLE_CUTOFF_LUO_GAO_APPROX;
  else
    return 0;
  return 1;
}

/**
### gle_kv_apply()

Applies one `key=value` pair to the parameter structs. Unknown keys warn on
`stderr` (they do not abort: parameter files may be shared between drivers).
Numeric values are parsed with `gle_parse_double()`/`gle_parse_long()`.

#### Returns
`1` if the key was recognised and applied, `0` if it is unknown, `-1` if a
recognised value is malformed.
*/
static int gle_kv_apply (GLEParams *p, GLEContOpts *o, const char *key,
			 const char *val) {
  double x = 0.0;
  int x_ok = gle_parse_double (val, &x);
#define GLE_KEY(name, target)						\
  if (!strcmp (key, name)) {						\
    if (!x_ok) return gle_bad_numeric (val, key);			\
    target = x;								\
    return 1;								\
  }
  GLE_KEY ("Ca", p->Ca);
  GLE_KEY ("mu_r", p->mu_r);
  GLE_KEY ("slip", p->slip);
  GLE_KEY ("c_slip", p->c_slip);
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
  if (!strcmp (key, "gle_model")) {
    if (!gle_parse_model (val, &p->model)) {
      fprintf (stderr, "gle-params: unknown gle_model '%s'\n", val);
      return -1;
    }
    return 1;
  }
  if (!strcmp (key, "c_method")) {
    if (!gle_parse_cutoff_method (val, &p->cutoff_method)) {
      fprintf (stderr, "gle-params: unknown c_method '%s'\n", val);
      return -1;
    }
    return 1;
  }
  if (!strcmp (key, "geometry")) {
    if (!strcmp (val, "vertical")) p->geometry = GLE_PLATE_VERTICAL;
    else if (!strcmp (val, "horizontal")) p->geometry = GLE_PLATE_HORIZONTAL;
    else {
      fprintf (stderr, "gle-params: unknown geometry '%s'\n", val);
      return -1;
    }
    return 1;
  }
  if (!strcmp (key, "outer_bc")) {
    if (!strcmp (val, "manifold")) p->outer_bc = GLE_OUTER_STATIC_MENISCUS;
    else if (!strcmp (val, "omega_zero")) p->outer_bc = GLE_OUTER_OMEGA_ZERO;
    else {
      fprintf (stderr, "gle-params: unknown outer_bc '%s'\n", val);
      return -1;
    }
    return 1;
  }
  if (!strcmp (key, "max_steps")) {
    long v;
    if (!gle_parse_long (val, &v))
      return gle_bad_numeric (val, key);
    p->max_steps = v;
    return 1;
  }
  /* Continuation-only keys remain recognised when `o == NULL`, allowing a
     shared parameter file to be passed to the single-solve driver quietly. */
#define GLE_KEY_O(name, field)						\
  if (!strcmp (key, name)) {						\
    if (!x_ok) return gle_bad_numeric (val, key);			\
    if (o) o->field = x;						\
    return 1;								\
  }
  GLE_KEY_O ("Ca_start", Ca_start);
  GLE_KEY_O ("alpha0", alpha0);
  GLE_KEY_O ("alpha_min", alpha_min);
  GLE_KEY_O ("alpha_max", alpha_max);
  GLE_KEY_O ("Delta_max", Delta_max);
  GLE_KEY_O ("Ca_stop_min", Ca_stop_min);
#undef GLE_KEY_O
  if (!strcmp (key, "max_points")) {
    long v;
    if (!gle_parse_long (val, &v))
      return gle_bad_numeric (val, key);
    if (v < INT_MIN || v > INT_MAX) {
      fprintf (stderr, "gle-params: integer value '%s' is out of range for key '%s'\n",
	       val, key);
      return -1;
    }
    if (o)
      o->max_points = (int) v;
    return 1;
  }
  if (!strcmp (key, "verbose")) {
    long v;
    if (!gle_parse_long (val, &v))
      return gle_bad_numeric (val, key);
    if (v < INT_MIN || v > INT_MAX) {
      fprintf (stderr, "gle-params: integer value '%s' is out of range for key '%s'\n",
	       val, key);
      return -1;
    }
    if (o)
      o->verbose = (int) v;
    return 1;
  }
  return 0;
}

/**
### gle_params_validate()

Validates the shared physical and integrator parameters before a driver
allocates workspaces or starts a solve. Non-positive `s0` and `h0` retain
their documented "derive from slip and wedge" meaning; their resolved values
are checked here.

#### Returns
`0` when the parameter set is usable, `1` after writing a diagnostic to
`stderr` otherwise.
*/
static inline int gle_params_validate (const GLEParams *p,
				       const char *driver) {
#define GLE_REQUIRE(cond, message)                                      \
  do {                                                                  \
    if (!(cond)) {                                                      \
      fprintf (stderr, "%s: invalid parameters: %s\n", driver, message); \
      return 1;                                                         \
    }                                                                   \
  } while (0)
  GLE_REQUIRE (isfinite (p->Ca), "Ca must be finite");
  GLE_REQUIRE (isfinite (p->mu_r) && p->mu_r >= 0.0,
	       "mu_r must be finite and non-negative");
  GLE_REQUIRE (isfinite (p->slip) && p->slip > 0.0,
	       "slip must be finite and positive");
  GLE_REQUIRE (p->model == GLE_MODEL_CHAN ||
	       p->model == GLE_MODEL_LUO_GAO,
	       "gle_model selector is invalid");
  GLE_REQUIRE (p->cutoff_method >= GLE_CUTOFF_MANUAL &&
	       p->cutoff_method <= GLE_CUTOFF_LUO_GAO_APPROX,
	       "c_method selector is invalid");
  GLE_REQUIRE (p->model != GLE_MODEL_CHAN ||
	       p->cutoff_method != GLE_CUTOFF_MANUAL ||
	       (isfinite (p->c_slip) && p->c_slip > 0.0),
	       "manual Chan c_slip must be finite and positive");
  GLE_REQUIRE (isfinite (p->theta_mic) && p->theta_mic > 0.0 &&
	       p->theta_mic < M_PI,
	       "theta_mic_deg must lie strictly between 0 and 180");
  GLE_REQUIRE (isfinite (p->grav) && p->grav >= 0.0,
	       "grav must be finite and non-negative");
  GLE_REQUIRE (p->geometry == GLE_PLATE_VERTICAL ||
	       p->geometry == GLE_PLATE_HORIZONTAL,
	       "geometry selector is invalid");
  GLE_REQUIRE (isfinite (p->s0) && isfinite (p->h0),
	       "s0 and h0 must be finite");
  GLE_REQUIRE (isfinite (gle_s0 (p)) && gle_s0 (p) > 0.0,
	       "resolved s0 must be positive");
  GLE_REQUIRE (isfinite (gle_h0 (p)) && gle_h0 (p) > 0.0,
	       "resolved h0 must be positive");
  GLE_REQUIRE (isfinite (p->H_match) && p->H_match > gle_h0 (p),
	       "H_match must be finite and exceed the inner film thickness");
  GLE_REQUIRE (isfinite (p->smax_cap) && p->smax_cap > gle_s0 (p),
	       "smax_cap must be finite and exceed the starting arc length");
  GLE_REQUIRE (p->outer_bc == GLE_OUTER_STATIC_MENISCUS ||
	       p->outer_bc == GLE_OUTER_OMEGA_ZERO,
	       "outer_bc selector is invalid");
  GLE_REQUIRE (p->outer_bc != GLE_OUTER_STATIC_MENISCUS || p->grav > 0.0,
	       "grav must be positive for static-meniscus matching");
  GLE_REQUIRE (isfinite (p->rtol) && p->rtol > 0.0,
	       "rtol must be finite and positive");
  GLE_REQUIRE (isfinite (p->atol) && p->atol > 0.0,
	       "atol must be finite and positive");
  GLE_REQUIRE (p->max_steps > 0, "max_steps must be positive");
#undef GLE_REQUIRE
  return 0;
}

/**
### gle_params_prepare()

Applies the selected case-level cutoff policy after all file and command-line
overrides are known. For the direct Luo--Gao model the returned provenance is
`not_used`; no Chan cutoff is evaluated.

#### Returns
`0` when the model is ready to solve, otherwise `1` after a concise diagnostic.
*/
static inline int gle_params_prepare (GLEParams *p, GLECutoffResult *cutoff,
				      const char *driver) {
  int status = gle_model_prepare (p, cutoff);
  if (status == GLE_CUTOFF_OK)
    return 0;
  fprintf (stderr,
	   "%s: cannot prepare gle_model=%s with c_method=%s: %s\n",
	   driver, gle_model_name (p ? p->model : -1),
	   gle_cutoff_method_name (p ? p->cutoff_method : -1),
	   gle_cutoff_status_name (status));
  return 1;
}

/**
### gle_cont_opts_validate()

Validates the shooting-continuation controls loaded through the shared
parameter layer.
*/
static inline int gle_cont_opts_validate (const GLEContOpts *o,
					  const char *driver) {
#define GLE_REQUIRE(cond, message)                                      \
  do {                                                                  \
    if (!(cond)) {                                                      \
      fprintf (stderr, "%s: invalid parameters: %s\n", driver, message); \
      return 1;                                                         \
    }                                                                   \
  } while (0)
  GLE_REQUIRE (isfinite (o->Ca_start) && o->Ca_start > 0.0,
	       "Ca_start must be finite and positive");
  GLE_REQUIRE (isfinite (o->alpha_min) && o->alpha_min > 0.0 &&
	       isfinite (o->alpha0) && o->alpha0 >= o->alpha_min &&
	       isfinite (o->alpha_max) && o->alpha_max >= o->alpha0,
	       "require 0 < alpha_min <= alpha0 <= alpha_max");
  GLE_REQUIRE (o->max_points >= 3, "max_points must be at least 3");
  GLE_REQUIRE (isfinite (o->Delta_max) && o->Delta_max > 0.0,
	       "Delta_max must be finite and positive");
  GLE_REQUIRE (isfinite (o->Ca_stop_min) && o->Ca_stop_min >= 0.0,
	       "Ca_stop_min must be finite and non-negative");
  GLE_REQUIRE (o->verbose >= 0, "verbose must be non-negative");
#undef GLE_REQUIRE
  return 0;
}

/**
### gle_params_load()

Populates defaults, then applies (in order): an optional parameter file given
as the first bare CLI argument, then every CLI `key=value` override left to
right. Only the first bare argument is treated as a parameter file path; any
subsequent bare argument is reported to `stderr` as an ignored extra
positional argument rather than being opened. `o` may be `NULL` for drivers
without continuation options. File entries are split at the first `=` and the
complete trimmed right-hand side is parsed; empty recognised values and
trailing non-comment tokens are errors rather than silent defaults.

#### Returns
`0` on success, `1` if a recognised value is malformed. Unknown keys remain
warnings so a parameter file can be shared between drivers.
*/
static inline int gle_params_load (int argc, char *argv[], GLEParams *p,
				   GLEContOpts *o) {
  int have_file = 0;
  int bad = 0;
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
      long line_number = 0;
      while (fgets (line, sizeof line, fp)) {
	line_number++;
	char *hash = strchr (line, '#');
	if (hash) *hash = '\0';
	char *entry = gle_trim_whitespace (line);
	if (!entry[0])
	  continue;
	char *separator = strchr (entry, '=');
	if (!separator) {
	  fprintf (stderr, "gle-params: malformed line %ld in %s "
		   "(expected key=value)\n", line_number, argv[i]);
	  bad = 1;
	  continue;
	}
	*separator = '\0';
	char *key = gle_trim_whitespace (entry);
	char *val = gle_trim_whitespace (separator + 1);
	if (!key[0]) {
	  fprintf (stderr, "gle-params: empty key on line %ld in %s\n",
		   line_number, argv[i]);
	  bad = 1;
	  continue;
	}
	/* Pass the complete trimmed RHS. The strict scalar and selector parsers
	   then reject both an empty value and trailing, non-comment tokens. */
	int applied = gle_kv_apply (p, o, key, val);
	if (applied < 0)
	  bad = 1;
	else if (!applied)
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
      int applied = gle_kv_apply (p, o, argv[i], eq + 1);
      if (applied < 0)
	bad = 1;
      else if (!applied)
	fprintf (stderr, "gle-params: unknown key '%s'\n", argv[i]);
      *eq = '=';
    }
  }
  return bad;
}

#endif /* GLE_PARAMS_H */
