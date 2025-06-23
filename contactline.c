// This is a 2D simulation for Landau-Levich coating with the plate moving with non-dimensional velocity Ca
//The box is infinitely large, domain size Ldomain will be reduced later on
#include "navier-stokes/centered.h"
#define FILTERED
#include "two-phase.h"
#include "navier-stokes/conserving.h"
#include "tension.h"
// #include "adapt_wavelet_limited_v2.h"
#include "reduced.h"
#include "contact-fixed.h"

#define MINlevel 3                                              // minimum level

#define tsnap (1e-2)

// Error tolerances
#define fErr (1e-3)                                 // error tolerance in VOF
#define KErr (1e-3)                                 // error tolerance in KAPPA
#define VelErr (1e-3)                            // error tolerances in velocity


double  hf, tmax, Ldomain, Ca, mu_r, rho_r, lc, u_c, t_c, l_c, lr;
// hf is the height of the interface
// tmax is the maximum time
// Ldomain is the domain size
// Ca is the capillary number
// mu_r is the viscosity ratio
// rho_r is the density ratio
// lc is the capillary length
// u_c is the characteristic velocity
// t_c is the characteristic time
// l_c is the characteristic length
int MAXlevel;

// boundary conditions
u.t[bottom] = dirichlet(Ca);
uf.t[bottom] = dirichlet(Ca);
u.n[bottom] = dirichlet(0.0);
uf.n[bottom] = dirichlet(0.0);
p[bottom] = neumann(0.0);

u.t[right] = neumann(0.);
uf.t[right] = neumann(0.);
u.n[right] = neumann(0.);
uf.n[right] = neumann(0.0);
p[right] = dirichlet(0.0);

u.t[left] = neumann(0.);
uf.t[left] = neumann(0.);
u.n[left] = neumann(0.);
uf.n[left] = neumann(0.0);
p[left] = dirichlet(0.0);

u.t[top] = neumann(0.);
uf.t[top] = neumann(0.);
u.n[top] = neumann(0.);
uf.n[top] = neumann(0.0);
p[top] = dirichlet(0.0);

vector h[];
h.t[bottom] = contact_angle(theta);

int main(int argc, char const *argv[]) {
 
  tmax = 1e5;
  MAXlevel = 10;
  mu_r = 2e-2;
  rho_r = 1e-3;
  Ca = 0;
  // t_c = 1.9e-10;
  t_c = 1.9e-1;
  // l_c = 1.4e-8;
  l_c = 1.4e-2;
  u_c = l_c/t_c;
  lc = 2.7e-3;
  // lr = lc/l_c;
  lr = 1;
  theta = pi/2;

  // lr is lc/l_\nu: the ratio of the capillary length to the viscous length, for water it is 1e5
  // 1000 cSt Si oil will have lc/l_\nu \approx 1e-1

  Ldomain = lr > 1 ? 32 : 32*lr; // if lr is larger than 1, then there won't be any issues with resolving lr.. issue comes when lr is smaller than 1... careful if lr is less than 1/32: then we would not resolve l_nu (which is 1)... so we need to make sure that lr is larger than 1/32
  hf = 0.5*Ldomain;

  fprintf(ferr, "Level %d tmax %g., hf %3.2f\n", MAXlevel, tmax,  hf);

  L0=Ldomain;
  X0=-hf; Y0=0.;
  init_grid (1 << (MAXlevel));

  char comm[80];
  sprintf (comm, "mkdir -p intermediate1");
  system(comm);

  rho1 = 1e0; mu1 = 1e0;
  rho2 = rho1*rho_r; mu2 = mu1*mu_r;

  G.x = -10*(t_c*t_c)/l_c;

  f.sigma = 1.0;

  run();
}

event init(t = 0){
  if(!restore (file = "restart1")) {
    refine(((x<1e-1 && x>-1e-1) || (y < 1e-1)) && level<MAXlevel);
    /**
    We can now initialize the volume fractions in the domain. */
    double DeltaMin = L0/pow(2,MAXlevel);
    fraction (f, -(x-0.5*DeltaMin)); //this line is breaking the simulations
    //fraction(f, -x);
  }
  // return 1;
}

//  #define Rrefine(x,y,x0) (sq(x-x0) + sq(y))

//  int refRegion(double x, double y, double z){
//    scalar pos[];
//    position(f, pos, {1,0});
//    double xmax = statsf(pos).max;
//    return Rrefine(x,y,xmax) < sq(1e-2) ? MAXlevel+3: 
//    Rrefine(x,y,xmax) < sq(4e-2) ? MAXlevel+2: 
//    Rrefine(x,y,xmax) < sq(8e-2) ? MAXlevel+1: 
//    Rrefine(x,y,xmax) < sq(1e-1) ? MAXlevel:  
//    Rrefine(x,y,xmax) < sq(5e-1) ? MAXlevel-1: 
//    Rrefine(x,y,xmax) < sq(1e0) ? MAXlevel-2: 
//    Rrefine(x,y,xmax) < sq(4e0) ? MAXlevel-3: 
//    MAXlevel-4; 
  
//    // return y < 1e-2 ? MAXlevel+1: y < 1e-1 ? MAXlevel: y < 1e0 ? MAXlevel-1: y < 4e0 ? MAXlevel-2: MAXlevel-3;
//  }


scalar KAPPA[];
event adapt(i++) {
  curvature(f, KAPPA);
  adapt_wavelet((scalar *){f, u.x, u.y},
    (double[]){fErr, VelErr, VelErr},
    MAXlevel, MINlevel);
  // if (t < 100*tsnap)
    // adapt_wavelet((scalar *){f, u.x, u.y, KAPPA},
    // (double[]){fErr, 1e-4, 1e-4, KErr},
    // MAXlevel, minlevel=MINlevel);
    //else {
   //adapt_wavelet_limited((scalar *){f, u.x, u.y, KAPPA},
     //(double[]){fErr, VelErr, VelErr, KErr},
     //refRegion, minlevel=MINlevel);
   //}
}

// Outputs
event writingFiles (t = 0; t += tsnap; t <= tmax + tsnap) {
  dump (file = "restart1");
  char nameOut[80];
  sprintf (nameOut, "intermediate1/snapshot-%5.4f", t);
  dump (file = nameOut);
}

event logWriting (i++) {
  // dump(file="dump_0.007");
  // if (i > 0) return 1;
  double ke = 0.;
  foreach (reduction(+:ke)){
    ke += sq(Delta)*(sq(u.x[]) + sq(u.y[]))*rho(f[]);
  }
  static FILE * fp;
  if (i == 0) {
    fprintf (ferr, "i dt t ke\n");
    fp = fopen ("log", "w");
    fprintf (fp, "i dt t ke\n");
    fprintf (fp, "%d %g %g %g\n", i, dt, t, ke);
    fclose(fp);
  } else {
    fp = fopen ("log", "a");
    fprintf (fp, "%d %g %g %g\n", i, dt, t, ke);
    fclose(fp);
  }
  fprintf (ferr, "%d %g %g %g\n", i, dt, t, ke);
}
