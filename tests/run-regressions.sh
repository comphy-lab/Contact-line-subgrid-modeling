#!/bin/sh
set -eu

repo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/gle-regression.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT HUP INT TERM

"$repo_dir/gle-ode/gle-solve" "$repo_dir/gle-ode/fig4b.params" \
  Ca=1e-6 profile_out="$test_dir/profile.csv" >/dev/null

expect_failure () {
  if "$@" >"$test_dir/rejected.stdout" 2>"$test_dir/rejected.stderr"; then
    echo "expected command to reject invalid input: $*" >&2
    exit 1
  fi
}

expect_failure "$repo_dir/gle-ode/gle-solve" \
  "$repo_dir/gle-ode/fig4b.params" c_slip=0
expect_failure "$repo_dir/gle-ode/gle-continuation" \
  "$repo_dir/gle-ode/fig4b.params" max_points=0 branch_out="$test_dir/a.csv"
expect_failure "$repo_dir/gle-ode/gle-continuation" \
  "$repo_dir/gle-ode/fig4b.params" mesh_N=-1 branch_out="$test_dir/b.csv"
expect_failure "$repo_dir/gle-ode/gle-continuation" \
  "$repo_dir/gle-ode/fig4b.params" dDelta=0 branch_out="$test_dir/c.csv"

if command -v qcc >/dev/null 2>&1; then
  echo "qcc found - running bounded Basilisk coupling regression"
  (
    cd "$repo_dir/simulationCases"
    qcc -O2 -disable-dimensions -DGLE_TEST_MODE=1 \
      -I../src-local contactline-gle.c \
      -o "$test_dir/contactline-gle-test" -lm
    cd "$test_dir"
    ./contactline-gle-test
  )
else
  echo "qcc not found - skipping Basilisk coupling regression"
fi
