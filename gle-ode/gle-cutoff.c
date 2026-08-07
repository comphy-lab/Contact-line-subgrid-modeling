/**
# gle-cutoff.c — evaluate the case-level Chan cutoff

Small dependency-free driver for the question ``given theta_e and M, what is
c?'' It resolves the same named closure used by the GLE and Basilisk paths,
without solving an interface boundary-value problem.

Only the model selector, microscopic angle, viscosity ratio, cutoff selector
and (for `manual`) supplied coefficient are relevant. Outer-boundary,
gravity, integration and slip-length settings are deliberately not validated
by this helper.

## Usage

```bash
./gle-cutoff [file.params] [key=value ...]
# e.g.
./gle-cutoff theta_mic_deg=60 mu_r=0.1 c_method=auto
```

The default policy is `auto`: Scott--Hocking at $M=0$, the corrected
right-angle branch, the frozen two-phase table, then the explicitly labelled
Luo--Gao approximation only outside the table domain.
*/

#include <stdio.h>
#include "gle-params.h"

int main (int argc, char *argv[]) {
  GLEParams p = gle_default_params ();
  GLECutoffResult cutoff;
  p.model = GLE_MODEL_CHAN;
  p.cutoff_method = GLE_CUTOFF_AUTO;

  if (gle_params_load (argc, argv, &p, NULL))
    return 2;
  if (p.model != GLE_MODEL_CHAN) {
    fprintf (stderr,
             "gle-cutoff: c is a Chan-model matching constant; "
             "direct Luo--Gao does not use c\n");
    return 2;
  }
  int status = gle_cutoff_resolve (&p, &cutoff);
  if (status != GLE_CUTOFF_OK) {
    fprintf (stderr,
             "gle-cutoff: cannot resolve c_method=%s for theta=%g deg, "
             "mu_r=%g: %s\n",
             gle_cutoff_method_name (p.cutoff_method),
             p.theta_mic*180.0/M_PI, p.mu_r,
             gle_cutoff_status_name (status));
    return 2;
  }

  printf ("theta_mic_deg = %.12g\n", p.theta_mic*180.0/M_PI);
  printf ("mu_r          = %.12g\n", p.mu_r);
  printf ("c_method      = %s\n", gle_cutoff_method_name (cutoff.method));
  printf ("luo_gao_approximation = %s\n",
	  cutoff.luo_gao_approximation ? "yes" : "no");
  printf ("Q             = %.17g\n", cutoff.Q);
  printf ("log_c         = %.17g\n", cutoff.log_c);
  printf ("c             = %.17g\n", cutoff.c);
  return 0;
}
