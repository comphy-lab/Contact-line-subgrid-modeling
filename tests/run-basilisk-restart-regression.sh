#!/bin/sh
set -eu

repo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

if ! command -v qcc >/dev/null 2>&1; then
  echo "qcc not found - skipping Basilisk restart regression"
  exit 0
fi

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/gle-basilisk-restart.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT HUP INT TERM
mkdir "$test_dir/baseline" "$test_dir/restarted"

(
  cd "$repo_dir/simulationCases"
  qcc -O2 -disable-dimensions -DGLE_RESTART_TEST_MODE=1 \
    -I../src-local contactline-gle.c \
    -o "$test_dir/contactline-gle-restart-test" -lm
)

(
  cd "$test_dir/baseline"
  "$test_dir/contactline-gle-restart-test" >stdout 2>stderr
)

(
  cd "$test_dir/restarted"
  GLE_RESTART_TEST_INTERRUPT=1 \
    "$test_dir/contactline-gle-restart-test" >first.stdout 2>first.stderr
  "$test_dir/contactline-gle-restart-test" >second.stdout 2>second.stderr
)

grep -q 'GLE_RESTART_TEST: production stop .* t=0.002' \
  "$test_dir/baseline/stderr"
grep -q 'GLE_RESTART_TEST: interrupted .* t=0.001' \
  "$test_dir/restarted/first.stderr"
if grep -q 'GLE_RESTART_TEST: production stop' \
     "$test_dir/restarted/first.stderr"; then
  echo "interrupted run reached the production stop unexpectedly" >&2
  exit 1
fi
grep -q 'GLE_RESTART_TEST: production stop .* t=0.002' \
  "$test_dir/restarted/second.stderr"

# Columns 2--10 contain the physical sample and closure inputs. They must
# match the uninterrupted trajectory; columns 11--12 only identify the
# restarted process and its first post-restore sample.
awk '
function abs(x) { return x < 0 ? -x : x }
function nearly_equal(a, b) {
  return abs(a - b) <= 5e-12*(1 + abs(a) + abs(b))
}
FNR == NR {
  for (column = 2; column <= 10; column++)
    baseline[$1, column] = $column
  baseline_rows++
  next
}
{
  if (!(($1, 2) in baseline)) {
    print "restart trace has no uninterrupted row for iteration " $1 > "/dev/stderr"
    failed = 1
    next
  }
  for (column = 2; column <= 10; column++)
    if (!nearly_equal($column, baseline[$1, column])) {
      print "restart mismatch at iteration " $1 ", column " column > "/dev/stderr"
      failed = 1
    }
  if ($11 == 1 && !found_restored) {
    found_restored = 1
    if ($9 != 1 || $10 != 0 || $12 != 1) {
      print "first restored solve did not use the dumped velocity history" > "/dev/stderr"
      failed = 1
    }
  }
  restart_rows++
}
END {
  if (!found_restored) {
    print "restart trace contains no post-restore coupling sample" > "/dev/stderr"
    failed = 1
  }
  if (restart_rows != baseline_rows) {
    print "interrupted and uninterrupted trace lengths differ" > "/dev/stderr"
    failed = 1
  }
  exit failed
}
' "$test_dir/baseline/gle-restart-trace.tsv" \
  "$test_dir/restarted/gle-restart-trace.tsv"

echo "Basilisk production-stop/restart regression passed"
