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

expect_rejection () {
  expected_diagnostic=$1
  shift
  set +e
  "$@" >"$test_dir/rejected.stdout" 2>"$test_dir/rejected.stderr"
  rejection_status=$?
  set -e
  if [ "$rejection_status" -ne 2 ]; then
    echo "expected rejection status 2, got $rejection_status: $*" >&2
    cat "$test_dir/rejected.stderr" >&2
    exit 1
  fi
  if ! grep -Fq "$expected_diagnostic" "$test_dir/rejected.stderr"; then
    echo "missing rejection diagnostic '$expected_diagnostic': $*" >&2
    cat "$test_dir/rejected.stderr" >&2
    exit 1
  fi
}

expect_rejection "manual Chan c_slip must be finite and positive" \
  "$repo_dir/gle-ode/gle-solve" \
  "$repo_dir/gle-ode/fig4b.params" c_slip=0
expect_rejection "unknown gle_model 'unknown'" "$repo_dir/gle-ode/gle-solve" \
  "$repo_dir/gle-ode/fig4b.params" gle_model=unknown
expect_rejection "unknown c_method 'unknown'" "$repo_dir/gle-ode/gle-solve" \
  "$repo_dir/gle-ode/fig4b.params" c_method=unknown
expect_rejection "direct Luo--Gao does not use c" \
  "$repo_dir/gle-ode/gle-cutoff" gle_model=luo_gao
expect_rejection "cannot resolve c_method=manual" \
  "$repo_dir/gle-ode/gle-cutoff" \
  theta_mic_deg=60 mu_r=-0.1 c_method=manual c_slip=3
expect_rejection "max_points must be at least 3" \
  "$repo_dir/gle-ode/gle-continuation" \
  "$repo_dir/gle-ode/fig4b.params" max_points=0 branch_out="$test_dir/a.csv"
expect_rejection "mesh_N must be an integer" \
  "$repo_dir/gle-ode/gle-continuation" \
  "$repo_dir/gle-ode/fig4b.params" mesh_N=-1 branch_out="$test_dir/b.csv"
expect_rejection "dDelta must be finite and positive" \
  "$repo_dir/gle-ode/gle-continuation" \
  "$repo_dir/gle-ode/fig4b.params" dDelta=0 branch_out="$test_dir/c.csv"

printf '%s\n' 'mu_r =' >"$test_dir/empty-rhs.params"
expect_rejection "bad numeric value '' for key 'mu_r'" \
  "$repo_dir/gle-ode/gle-solve" "$test_dir/empty-rhs.params"
printf '%s\n' 'Ca = 1 junk' >"$test_dir/trailing-junk.params"
expect_rejection "bad numeric value '1 junk' for key 'Ca'" \
  "$repo_dir/gle-ode/gle-solve" "$test_dir/trailing-junk.params"

# A point-limited lower-branch trace has not bracketed the turning point, so
# its final sample must never be promoted to a fold.
set +e
"$repo_dir/gle-ode/gle-continuation" "$repo_dir/gle-ode/fig4b.params" \
  mesh_N=100 max_points=3 verbose=1 branch_out="$test_dir/truncated.csv" \
  >"$test_dir/truncated.txt" 2>"$test_dir/truncated.stderr"
truncated_status=$?
set -e
if [ "$truncated_status" -ne 1 ]; then
  echo "expected truncated continuation status 1, got $truncated_status" >&2
  cat "$test_dir/truncated.stderr" >&2
  exit 1
fi
grep -Fq 'fold not bracketed; preserved 3-point partial branch' \
  "$test_dir/truncated.stderr"
awk -F= '
  /^fold_Ca/ {
    value = tolower($2); gsub(/[[:space:]]/, "", value)
    if (value !~ /nan/) exit 1
    found = 1
  }
  END { if (!found) exit 1 }
' "$test_dir/truncated.txt"

# Trace through the fold and upper branch, then require every emitted nonlinear
# residual (including both border rows) to be finite and below the Newton gate.
"$repo_dir/gle-ode/gle-continuation" "$repo_dir/gle-ode/fig4b.params" \
  mesh_N=100 Delta_max=3.7 dDelta=0.01 dDelta_max=0.02 max_points=500 \
  verbose=0 branch_out="$test_dir/complete.csv" >"$test_dir/complete.txt"
awk -F= '
  /^fold_Ca/ {
    value = tolower($2); gsub(/[[:space:]]/, "", value)
    if (value ~ /nan|inf/ || value + 0 <= 0) exit 1
    found = 1
  }
  END { if (!found) exit 1 }
' "$test_dir/complete.txt"
awk -F, '
  NR == 1 {
    if ($8 != "residual") exit 1
    next
  }
  {
    value = tolower($8)
    if (value ~ /nan|inf/ || $8 + 0 < 0 || $8 + 0 >= 1.0e-10) exit 1
    last_delta = $3
    rows++
  }
  END { if (rows < 3 || last_delta < 3.7) exit 1 }
' "$test_dir/complete.csv"

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

"$repo_dir/tests/run-basilisk-restart-regression.sh"
