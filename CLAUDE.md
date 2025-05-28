# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python implementations for contact line subgrid modeling, focusing on solving the Generalized Lubrication Equations (GLE) and analyzing Huh-Scriven velocity fields near moving contact lines in fluid dynamics.

## Key Components

### GLE_solver.py
- Solves coupled ODEs for contact line dynamics using the Generalized Lubrication Equations
- Key functions:
  - `f1(theta)`, `f2(theta)`, `f3(theta)`: Helper functions for the GLE formulation
  - `f(theta, R)`: Main function combining the helpers with viscosity ratio R
  - `GLE(y)`: System of ODEs defining the contact line shape evolution
- Currently uses scipy's odeint but has convergence issues noted in git history

### huh_scriven_velocity.py
- Analyzes the Huh-Scriven velocity field near moving contact lines
- Computes velocity components in polar (Ur, Uphi) and Cartesian (Ux, Uy) coordinates
- Visualizes relative velocities between the fluid and moving plate

## Development Commands

```bash
# Run the GLE solver
python GLE_solver.py

# Run the Huh-Scriven velocity analysis
python huh_scriven_velocity.py
```

## Mathematical Context

The code implements solutions to the contact line dynamics problem where:
- Ca: Capillary number (ratio of viscous to surface tension forces)
- lambda_slip: Slip length parameter
- R: Viscosity ratio (μ_g/μ_l)
- theta: Contact angle
- h: Film thickness
- s: Arc length coordinate along the interface

## Known Issues

- GLE solver has convergence problems (noted in commit history)
- Current integration limits (s_range) may need adjustment for different parameter regimes
- Validation against analytical or experimental results still needed