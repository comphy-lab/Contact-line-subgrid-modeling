# gle-ode: standalone GLE solver + pseudo-arclength continuation.
# No external dependencies: any C99 compiler with libm.

CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
INC = -I../src-local

all: gle-solve gle-continuation

gle-solve: gle-solve.c ../src-local/gle-model.h ../src-local/gle-integrate.h \
           ../src-local/gle-shoot.h ../src-local/gle-params.h
	$(CC) $(CFLAGS) $(INC) -o $@ gle-solve.c -lm

gle-continuation: gle-continuation.c ../src-local/gle-model.h \
                  ../src-local/gle-integrate.h ../src-local/gle-shoot.h \
                  ../src-local/gle-continuation.h ../src-local/gle-collocate.h \
                  ../src-local/gle-params.h
	$(CC) $(CFLAGS) $(INC) -o $@ gle-continuation.c -lm

clean:
	rm -f gle-solve gle-continuation *.o

.PHONY: all clean
