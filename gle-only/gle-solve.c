/**
# gle-solve.c — single GLE boundary-value solve

Standalone driver: solves the dip-coating GLE boundary-value problem at one
capillary number and writes the interface profile
$(s, h, \theta, \omega, z)$ to CSV. This is the C replacement for the
`solve_bvp`-based `GLE_solver.py` on the Python side, sharing its physics
exactly (see [src-local/gle-model.h](../src-local/gle-model.h)).

## Usage

```bash
./gle-solve [file.params] [key=value ...]
# e.g.
./gle-solve fig4b.params Ca=0.005 profile_out=profile.csv
```

Recognised driver-specific keys: `profile_out` (CSV path, default
`gle-profile.csv`), `omega0_guess` (shooting seed; default = static meniscus
curvature).

Output columns: `s,h,theta_deg,omega,z` with $z = \Delta - \zeta(s)$ the
elevation above the bath.

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

/**
## Profile capture

The sampler records every accepted integration step; $z$ is reconstructed
after the solve, once $\Delta$ is known.
*/
typedef struct {
  double *s, *hf, *th, *om, *zt;
  long n, cap;
} profile_buf;

static void profile_sampler (void *ctx, double s, const double y[4]) {
  profile_buf *b = (profile_buf *) ctx;
  if (b->n == b->cap) {
    b->cap = b->cap ? 2*b->cap : 65536;
    b->s = (double *) realloc (b->s, b->cap*sizeof (double));
    b->hf = (double *) realloc (b->hf, b->cap*sizeof (double));
    b->th = (double *) realloc (b->th, b->cap*sizeof (double));
    b->om = (double *) realloc (b->om, b->cap*sizeof (double));
    b->zt = (double *) realloc (b->zt, b->cap*sizeof (double));
  }
  b->s[b->n] = s;
  b->hf[b->n] = y[0];
  b->th[b->n] = y[1];
  b->om[b->n] = y[2];
  b->zt[b->n] = y[3];
  b->n++;
}

int main (int argc, char *argv[]) {
  GLEParams p = gle_default_params ();
  const char *out_path = "gle-profile.csv";
  double omega0_guess = 0.0;

  /* driver-specific keys are peeled off before the shared loader runs */
  for (int i = 1; i < argc; i++) {
    if (!strncmp (argv[i], "profile_out=", 12)) {
      out_path = argv[i] + 12;
      argv[i] = (char *) "";
    }
    else if (!strncmp (argv[i], "omega0_guess=", 13)) {
      omega0_guess = atof (argv[i] + 13);
      argv[i] = (char *) "";
    }
  }
  /* compact argv (empty strings removed) */
  int ac = 1;
  for (int i = 1; i < argc; i++)
    if (argv[i][0] != '\0')
      argv[ac++] = argv[i];

  gle_params_load (ac, argv, &p, NULL);

  if (omega0_guess == 0.0)
    omega0_guess = gle_static_curvature (p.theta_mic);

  GLESolution sol;
  if (gle_shoot (&p, omega0_guess, &sol)) {
    fprintf (stderr,
	     "gle-solve: no converged solution at Ca = %g "
	     "(theta_e = %g deg, slip = %g)\n",
	     p.Ca, p.theta_mic*180.0/M_PI, p.slip);
    return 1;
  }

  /* re-integrate the converged solution, recording the profile */
  profile_buf buf = { NULL, NULL, NULL, NULL, NULL, 0, 0 };
  gle_shoot_residual (&p, sol.omega0, &sol, profile_sampler, &buf);

  FILE *fp = fopen (out_path, "w");
  if (!fp) {
    fprintf (stderr, "gle-solve: cannot write '%s'\n", out_path);
    return 1;
  }
  fprintf (fp, "s,h,theta_deg,omega,z\n");
  for (long i = 0; i < buf.n; i++)
    fprintf (fp, "%.12e,%.12e,%.12e,%.12e,%.12e\n",
	     buf.s[i], buf.hf[i], buf.th[i]*180.0/M_PI, buf.om[i],
	     sol.Delta - buf.zt[i]);
  fclose (fp);

  printf ("Ca            = %.10e\n", p.Ca);
  printf ("theta_mic     = %.6f deg\n", p.theta_mic*180.0/M_PI);
  printf ("slip          = %.3e\n", p.slip);
  printf ("omega0        = %.12e\n", sol.omega0);
  printf ("Delta         = %.12e\n", sol.Delta);
  printf ("theta_app     = %.6f deg\n", sol.theta_app*180.0/M_PI);
  printf ("theta_min     = %.6f deg\n", sol.theta_min*180.0/M_PI);
  printf ("s_end         = %.6e\n", sol.s_end);
  printf ("residual      = %.3e\n", sol.residual);
  printf ("profile       -> %s (%ld points)\n", out_path, buf.n);
  return 0;
}
