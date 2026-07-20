/**
# test-gle-core.c — focused GLE parameter and allocation regressions

Small dependency-free executable covering the failure modes that must be
rejected before a solve: the explicit microscopic cutoff coefficient, strict
finite parsing, invalid collocation sizes, and zero continuation steps.

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
#include "gle-collocate.h"

static int check_close (const char *name, double got, double want,
			double tol) {
  if (!isfinite (got) || fabs (got - want) > tol) {
    fprintf (stderr, "%s: got %.17g, want %.17g +/- %.3g\n",
	     name, got, want, tol);
    return 1;
  }
  return 0;
}

int main (void) {
  int failed = 0;
  const double gamma_E = 0.57721566490153286061;
  GLEParams p = gle_default_params ();

  failed += check_close ("legacy c_slip default", p.c_slip, 3.0, 0.0);
  failed += check_close ("right-angle one-fluid c_slip",
			 gle_slip_prefactor_right_angle (0.0),
			 exp (log (2.0) - gamma_E), 5.0e-15);
  failed += check_close ("right-angle c_slip at M=0.02",
			 gle_slip_prefactor_right_angle (0.02),
			 1.1580405939091214, 5.0e-3);
  failed += check_close ("right-angle c_slip at M=1",
			 gle_slip_prefactor_right_angle (1.0),
			 2.052069628256686, 5.0e-3);
  failed += check_close ("right-angle c_slip at M=10",
			 gle_slip_prefactor_right_angle (10.0),
			 6.404211775495213, 5.0e-3);
  failed += check_close ("right-angle large-ratio c_slip",
			 gle_slip_prefactor_right_angle (1.0e8), 12.60, 2.0e-2);
  if (!isnan (gle_slip_prefactor_right_angle (-1.0))) {
    fprintf (stderr, "negative viscosity ratio was accepted\n");
    failed++;
  }

  double y[4] = { 0.2, 0.25*M_PI, 0.0, 0.0 }, dyds[4];
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

  GLECollocation c;
  if (!gle_colloc_alloc (&c, 0) || !gle_colloc_alloc (&c, -1)) {
    fprintf (stderr, "invalid collocation mesh was accepted\n");
    failed++;
  }
  c.N = 2;
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
