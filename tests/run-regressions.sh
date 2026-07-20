#!/bin/sh
set -eu

repo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/gle-regression.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT HUP INT TERM

"$repo_dir/gle-ode/gle-solve" "$repo_dir/gle-ode/fig4b.params" \
  Ca=1e-6 profile_out="$test_dir/profile.csv" >/dev/null
"$repo_dir/gle-ode/gle-cutoff" \
  theta_mic_deg=90 mu_r=0.02 c_method=auto >"$test_dir/cutoff.txt"
grep -q '^c_method      = corrected_right_angle$' "$test_dir/cutoff.txt"
"$repo_dir/gle-ode/gle-cutoff" \
  theta_mic_deg=60 mu_r=0.1 c_method=auto >"$test_dir/cutoff-table.txt"
grep -q '^c_method      = reference_table$' "$test_dir/cutoff-table.txt"
grep -q '^luo_gao_approximation = no$' "$test_dir/cutoff-table.txt"
# The closure helper depends only on theta_e, M and its closure selector; it
# must not inherit static-meniscus or integration validation from a GLE solve.
"$repo_dir/gle-ode/gle-cutoff" \
  theta_mic_deg=60 mu_r=0.1 c_method=auto grav=0 slip=0 smax_cap=0 \
  >"$test_dir/cutoff-closure-only.txt"
grep -q '^c_method      = reference_table$' \
  "$test_dir/cutoff-closure-only.txt"
"$repo_dir/gle-ode/gle-cutoff" \
  theta_mic_deg=20 mu_r=0.1 c_method=auto >"$test_dir/cutoff-fallback.txt"
grep -q '^c_method      = luo_gao_approx$' "$test_dir/cutoff-fallback.txt"
grep -q '^luo_gao_approximation = yes$' "$test_dir/cutoff-fallback.txt"
# Phase-exchange canonicalisation must keep the labelled fallback finite even
# when the caller's raw viscosity ratio would overflow an M^2 coefficient.
"$repo_dir/gle-ode/gle-cutoff" \
  theta_mic_deg=20 mu_r=1e200 c_method=auto \
  >"$test_dir/cutoff-fallback-extreme.txt"
grep -q '^c_method      = luo_gao_approx$' \
  "$test_dir/cutoff-fallback-extreme.txt"
grep -q '^c             = 0.025' \
  "$test_dir/cutoff-fallback-extreme.txt"
"$repo_dir/gle-ode/gle-solve" "$repo_dir/gle-ode/fig4b.params" \
  Ca=1e-6 gle_model=luo_gao c_slip=0 \
  profile_out="$test_dir/profile-luo-gao.csv" >/dev/null

expect_failure () {
  if "$@" >"$test_dir/rejected.stdout" 2>"$test_dir/rejected.stderr"; then
    echo "expected command to reject invalid input: $*" >&2
    exit 1
  fi
}

expect_failure "$repo_dir/gle-ode/gle-solve" \
  "$repo_dir/gle-ode/fig4b.params" c_slip=0
expect_failure "$repo_dir/gle-ode/gle-solve" \
  "$repo_dir/gle-ode/fig4b.params" gle_model=unknown
expect_failure "$repo_dir/gle-ode/gle-solve" \
  "$repo_dir/gle-ode/fig4b.params" c_method=unknown
expect_failure "$repo_dir/gle-ode/gle-cutoff" gle_model=luo_gao
expect_failure "$repo_dir/gle-ode/gle-cutoff" \
  theta_mic_deg=60 mu_r=-0.1 c_method=manual c_slip=3
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
    qcc -O2 -disable-dimensions -DGLE_TEST_MODE=1 \
      -DGLE_RUNTIME_MODEL=GLE_MODEL_LUO_GAO \
      -I../src-local contactline-gle.c \
      -o "$test_dir/contactline-gle-luo-gao-test" -lm
    cd "$test_dir"
    ./contactline-gle-test
    ./contactline-gle-luo-gao-test
  )
else
  echo "qcc not found - skipping Basilisk coupling regression"
fi
