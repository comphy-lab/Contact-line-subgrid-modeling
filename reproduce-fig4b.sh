#!/usr/bin/env bash
# Reproduce Fig. 4b of Snoeijer & Andreotti, Annu. Rev. Fluid Mech. 45 (2013):
# traces the dip-coating bifurcation branch with gle-continuation and overlays
# it on the digitized theory curve + experimental data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building gle-ode"
make -C "$REPO_ROOT/gle-ode"

mkdir -p "$REPO_ROOT/gle-ode/output"

echo "==> Tracing the fig. 4b branch"
(
  cd "$REPO_ROOT/gle-ode"
  ./gle-continuation fig4b.params branch_out=output/fig4b-branch.csv dDelta_max=0.015
)

BRANCH_CSV="$REPO_ROOT/gle-ode/output/fig4b-branch.csv"

if command -v uv >/dev/null 2>&1; then
  echo "==> Plotting reproduction figure"
  (cd "$REPO_ROOT" && uv run postProcess/plot-fig4b.py)
else
  echo "==> uv not found - to plot the reproduction figure, run:"
  echo "    uv run postProcess/plot-fig4b.py"
fi

echo
echo "Produced:"
echo "  $BRANCH_CSV"
echo "  $REPO_ROOT/img/fig4b-reproduction.png"
echo "  $REPO_ROOT/img/fig4b-reproduction.pdf"
