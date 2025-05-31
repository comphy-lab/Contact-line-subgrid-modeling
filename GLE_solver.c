#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "src-local/gle_solver.h"

/**
 * ## Mathematical Equations
 *
 * The system solves
 *   dh/ds = sin(theta)
 *   dtheta/ds = omega
 *   domega/ds = 3*Ca*f(theta,mu_r)/(h*(h+3*lambda_slip)) - cos(theta)
 */

double f1(double theta) {
  return theta * theta - sin(theta) * sin(theta);
}

double f2(double theta) {
  return theta - sin(theta) * cos(theta);
}

double f3(double theta) {
  return theta * (M_PI - theta) + sin(theta) * sin(theta);
}

double f_func(double theta, double mu_r) {
  double num = 2.0 * pow(sin(theta), 3.0) *
               (mu_r * mu_r * f1(theta) + 2.0 * mu_r * f3(theta) +
                f1(M_PI - theta));
  double den = 3.0 * (mu_r * f1(theta) * f2(M_PI - theta) -
                      f1(M_PI - theta) * f2(theta));
  return num / den;
}

void gle_rhs(double s, const double y[3], double dyds[3],
             const gle_params *p) {
  (void)s; /* unused */
  double h = y[0];
  double theta = y[1];
  double omega = y[2];

  dyds[0] = sin(theta);
  dyds[1] = omega;
  dyds[2] = 3.0 * p->Ca * f_func(theta, p->mu_r) /
            (h * (h + 3.0 * p->lambda_slip)) - cos(theta);
}

static void rk4_step(double s, double h, double theta, double omega,
                     double ds, const gle_params *p,
                     double *h_out, double *theta_out, double *omega_out) {
  double y[3] = {h, theta, omega};
  double k1[3], k2[3], k3[3], k4[3], yt[3];

  gle_rhs(s, y, k1, p);

  for (int i = 0; i < 3; ++i) yt[i] = y[i] + 0.5 * ds * k1[i];
  gle_rhs(s + 0.5 * ds, yt, k2, p);

  for (int i = 0; i < 3; ++i) yt[i] = y[i] + 0.5 * ds * k2[i];
  gle_rhs(s + 0.5 * ds, yt, k3, p);

  for (int i = 0; i < 3; ++i) yt[i] = y[i] + ds * k3[i];
  gle_rhs(s + ds, yt, k4, p);

  *h_out = h + ds / 6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
  *theta_out = theta + ds / 6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
  *omega_out = omega + ds / 6.0 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
}

int integrate_GLE(const gle_params *p, double s_start, double s_end,
                  int steps, double h0, double theta0, double omega0,
                  const char *csv_path) {
  FILE *fp = fopen(csv_path, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", csv_path);
    return 1;
  }

  fprintf(fp, "s,h,theta\n");

  double ds = (s_end - s_start) / steps;
  double s = s_start;
  double h = h0;
  double theta = theta0;
  double omega = omega0;

  fprintf(fp, "%g,%g,%g\n", s, h, theta);

  for (int i = 0; i < steps; ++i) {
    double hn, thetan, omegan;
    rk4_step(s, h, theta, omega, ds, p, &hn, &thetan, &omegan);
    s += ds;
    h = hn;
    theta = thetan;
    omega = omegan;
    fprintf(fp, "%g,%g,%g\n", s, h, theta);
  }

  fclose(fp);
  return 0;
}

int main(int argc, char *argv[]) {
  const char *csv = "gle_output.csv";
  if (argc > 1) csv = argv[1];

  gle_params params = {1.0, 1e-5, 1e-3};

  int ret = integrate_GLE(&params, 0.0, 4e-4, 1000,
                          params.lambda_slip, M_PI/6.0, 0.0,
                          csv);
  if (ret) {
    fprintf(stderr, "Integration failed\n");
    return 1;
  }
  return 0;
}
