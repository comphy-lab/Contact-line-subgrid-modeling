#!/bin/sh
set -eu

# Read a commit message on stdin and match the opt-out marker literally.
grep -Fq -- '[skip-deploy]'
