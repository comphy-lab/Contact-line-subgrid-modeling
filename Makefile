# Top-level Makefile for the GLE contact-line solver.
#
# Delegates the actual build to gle-ode/ (the standalone, dependency-free
# C solver + continuation drivers). No machine-local paths are assumed.

.PHONY: all test clean

all:
	$(MAKE) -C gle-ode

test: all
	cd gle-ode && ./gle-solve fig4b.params Ca=1e-6
	@if which qcc >/dev/null 2>&1; then \
		echo "qcc found - compiling simulationCases/contactline-gle.c"; \
		cd simulationCases && qcc -O2 -disable-dimensions -I../src-local \
			-o /tmp/contactline-gle-test contactline-gle.c -lm && \
		rm -f /tmp/contactline-gle-test; \
	else \
		echo "qcc not found - skipping Basilisk compile check"; \
	fi

clean:
	$(MAKE) -C gle-ode clean
