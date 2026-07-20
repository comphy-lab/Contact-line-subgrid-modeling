#!/bin/sh
set -eu

repo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
generator_dir="$repo_dir/gle-ode/reference-generator"
data_dir="$generator_dir/data"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/gle-reference.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT HUP INT TERM

# Keep linear-algebra backends within the local workstation budget.  Callers
# may lower this value, but the public evidence gate never needs more than four
# threads.
reference_threads=${GLE_REFERENCE_THREADS:-4}
case "$reference_threads" in
  *[!0-9]*|'')
    echo "GLE_REFERENCE_THREADS must be an integer from 1 to 4" >&2
    exit 2
    ;;
esac
if [ "$reference_threads" -lt 1 ]; then
  echo "GLE_REFERENCE_THREADS must be an integer from 1 to 4" >&2
  exit 2
fi
if [ "$reference_threads" -gt 4 ]; then
  reference_threads=4
fi
export OPENBLAS_NUM_THREADS="$reference_threads"
export OMP_NUM_THREADS="$reference_threads"
export VECLIB_MAXIMUM_THREADS="$reference_threads"

python3 -m unittest discover \
  -s "$generator_dir/tests" -p 'test_freeze_table.py' -v -b
python3 "$generator_dir/tests/test_scott_reference.py"
uv run "$generator_dir/tests/test_scott_solver.py"
uv run "$generator_dir/scott_hocking.py" verify
uv run "$generator_dir/tests/test_generator.py"

# Recompute the interpolation audit in isolation. This binds the committed
# evidence to the current freezer, rather than trusting a stored pass flag.
cp "$data_dir/two-phase-q.csv" "$test_dir/two-phase-q.csv"
cp "$data_dir/two-phase-q.manifest.json" \
  "$test_dir/two-phase-q.manifest.json"
cp "$data_dir/two-phase-q-checkpoints.csv" \
  "$test_dir/two-phase-q-checkpoints.csv"
cp "$data_dir/two-phase-q-checkpoints.manifest.json" \
  "$test_dir/two-phase-q-checkpoints.manifest.json"
cp "$data_dir/two-phase-q-interpolation-audit.csv" \
  "$test_dir/two-phase-q-interpolation-audit.csv"

python3 "$generator_dir/freeze_table.py" check \
  --table "$test_dir/two-phase-q.csv" \
  --checkpoints "$test_dir/two-phase-q-checkpoints.csv" \
  --tolerance-observed-Q 1e-3 --tolerance-error-budget-Q 3e-3 \
  --tolerance-right-angle-Q 1e-3 \
  --tolerance-symmetry-Q 1e-3 \
  --output "$test_dir/two-phase-q-interpolation-audit.csv" >/dev/null

cmp "$data_dir/two-phase-q-interpolation-audit.csv" \
  "$test_dir/two-phase-q-interpolation-audit.csv"
cmp "$data_dir/two-phase-q.manifest.json" \
  "$test_dir/two-phase-q.manifest.json"

python3 "$generator_dir/freeze_table.py" emit-c \
  --table "$test_dir/two-phase-q.csv" \
  --output "$test_dir/gle-slip-table-data.h" >/dev/null
cmp "$repo_dir/src-local/gle-slip-table-data.h" \
  "$test_dir/gle-slip-table-data.h"

echo "reference-table evidence: verified"
