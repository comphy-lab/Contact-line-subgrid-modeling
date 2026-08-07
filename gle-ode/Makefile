# gle-ode: standalone GLE solver + pseudo-arclength continuation.
# The C runtime targets need only a C99 compiler and libm. The in-repo
# reference-generator/ is a separate development tool with pinned Python deps.

CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
INC = -I../src-local
MODEL_HEADERS = ../src-local/gle-model.h ../src-local/gle-model-chan.h \
	../src-local/gle-model-luo-gao.h ../src-local/gle-slip-closure.h \
	../src-local/gle-slip-reference.h ../src-local/gle-slip-scott-data.h \
	../src-local/gle-slip-table-data.h

all: gle-cutoff gle-solve gle-continuation

gle-cutoff: gle-cutoff.c $(MODEL_HEADERS) ../src-local/gle-params.h
	$(CC) $(CFLAGS) $(INC) -o $@ gle-cutoff.c -lm

gle-solve: gle-solve.c $(MODEL_HEADERS) ../src-local/gle-integrate.h \
           ../src-local/gle-shoot.h ../src-local/gle-params.h
	$(CC) $(CFLAGS) $(INC) -o $@ gle-solve.c -lm

gle-continuation: gle-continuation.c $(MODEL_HEADERS) \
                  ../src-local/gle-integrate.h ../src-local/gle-shoot.h \
                  ../src-local/gle-continuation.h ../src-local/gle-collocate.h \
                  ../src-local/gle-params.h
	$(CC) $(CFLAGS) $(INC) -o $@ gle-continuation.c -lm

test-gle-core: test-gle-core.c $(MODEL_HEADERS) \
               ../src-local/gle-integrate.h ../src-local/gle-shoot.h \
               ../src-local/gle-basilisk.h \
               ../src-local/gle-continuation.h ../src-local/gle-collocate.h \
               ../src-local/gle-params.h
	$(CC) $(CFLAGS) $(INC) -o $@ test-gle-core.c -lm

test: all test-gle-core
	./test-gle-core

clean:
	rm -f gle-cutoff gle-solve gle-continuation test-gle-core *.o

.PHONY: all test clean
