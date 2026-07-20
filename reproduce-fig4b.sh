#!/usr/bin/env bash
# Reproduce Fig. 4b of Snoeijer & Andreotti, Annu. Rev. Fluid Mech. 45 (2013):
# traces legacy and model-resolved dip-coating branches with gle-continuation.
# The reproduction plot uses independently fold-calibrated slip lengths; a
# separate plot compares Chan and Luo--Gao at one identical physical slip.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_N="${GLE_FIG4B_MESH_N:-2500}"
DDELTA="${GLE_FIG4B_DDELTA:-0.002}"
DDELTA_MAX="${GLE_FIG4B_DDELTA_MAX:-0.015}"

# Plotting may load a threaded numerical backend.  Four threads is ample and
# leaves the workstation responsive while the serial C continuations run.
plot_threads="${GLE_PLOT_THREADS:-4}"
if [[ ! "$plot_threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLE_PLOT_THREADS must be a positive integer" >&2
  exit 2
fi
if (( plot_threads > 4 )); then
  plot_threads=4
fi
export OPENBLAS_NUM_THREADS="$plot_threads"
export OMP_NUM_THREADS="$plot_threads"
export VECLIB_MAXIMUM_THREADS="$plot_threads"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to regenerate and validate the Figure 4b plots" >&2
  echo "install uv from https://docs.astral.sh/uv/ and rerun this script" >&2
  exit 2
fi

echo "==> Building gle-ode"
make -C "$REPO_ROOT/gle-ode"

mkdir -p "$REPO_ROOT/gle-ode/output"

trace_branch () {
  local label="$1"
  local params="$2"
  local output="$3"
  echo "==> Tracing $label"
  (
    cd "$REPO_ROOT/gle-ode"
    ./gle-continuation "$params" \
      "branch_out=output/$output" \
      "mesh_N=$MESH_N" \
      "dDelta=$DDELTA" \
      "dDelta_max=$DDELTA_MAX" \
      verbose=0
  )
}

trace_branch "legacy Chan branch (manual c=3)" \
  fig4b.params fig4b-legacy-branch.csv
trace_branch "fold-calibrated Chan/Scott branch" \
  fig4b-chan-scott-calibrated.params fig4b-chan-scott-calibrated-branch.csv
trace_branch "fold-calibrated direct Luo--Gao branch" \
  fig4b-luo-gao-calibrated.params fig4b-luo-gao-calibrated-branch.csv
trace_branch "fixed-slip Chan/Scott branch" \
  fig4b-chan-common-slip.params fig4b-chan-common-slip-branch.csv
trace_branch "fixed-slip direct Luo--Gao branch" \
  fig4b-luo-gao-common-slip.params fig4b-luo-gao-common-slip-branch.csv

echo "==> Plotting reproduction and model comparisons"
(cd "$REPO_ROOT" && uv run postProcess/test-plot-fig4b.py)
(cd "$REPO_ROOT" && uv run postProcess/plot-fig4b.py)
(cd "$REPO_ROOT" && uv run postProcess/plot-finite-m-closure-comparison.py)

for figure in \
  fig4b-reproduction.png fig4b-reproduction.pdf \
  fig4b-model-comparison.png fig4b-model-comparison.pdf \
  finite-m-closure-comparison.png finite-m-closure-comparison.pdf; do
  if [[ ! -s "$REPO_ROOT/img/$figure" ]]; then
    echo "error: plotting did not produce img/$figure" >&2
    exit 1
  fi
done

echo
echo "Produced:"
echo "  $REPO_ROOT/gle-ode/output/fig4b-legacy-branch.csv"
echo "  $REPO_ROOT/gle-ode/output/fig4b-chan-scott-calibrated-branch.csv"
echo "  $REPO_ROOT/gle-ode/output/fig4b-luo-gao-calibrated-branch.csv"
echo "  $REPO_ROOT/gle-ode/output/fig4b-chan-common-slip-branch.csv"
echo "  $REPO_ROOT/gle-ode/output/fig4b-luo-gao-common-slip-branch.csv"
echo "  $REPO_ROOT/img/fig4b-reproduction.png"
echo "  $REPO_ROOT/img/fig4b-reproduction.pdf"
echo "  $REPO_ROOT/img/fig4b-model-comparison.png"
echo "  $REPO_ROOT/img/fig4b-model-comparison.pdf"
echo "  $REPO_ROOT/img/finite-m-closure-comparison.png"
echo "  $REPO_ROOT/img/finite-m-closure-comparison.pdf"
