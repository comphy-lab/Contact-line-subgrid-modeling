# Top-level Makefile for the GLE contact-line solver.
#
# Delegates the actual build to gle-ode/ (the standalone, dependency-free
# C solver + continuation drivers). No machine-local paths are assumed.

.PHONY: all test clean

all:
	$(MAKE) -C gle-ode

test: all
	$(MAKE) -C gle-ode test
	sh tests/run-regressions.sh

clean:
	$(MAKE) -C gle-ode clean
