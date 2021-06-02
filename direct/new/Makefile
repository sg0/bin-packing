CFLAGS = -g -Wall -qopenmp -std=gnu99 -DUSE_OMP_WTIME -DDEBUG=2 -O3 #-DUSE_PTHREAD_MTX
LDFLAGS = -g -qopenmp -lm
CC = icc
MKDIR_P = mkdir -p
BINDIR = bins

.PHONY: bindir

all: bindir binaries

bindir:
	$(MKDIR_P) $(BINDIR)

binaries:  $(BINDIR)/bp_serial $(BINDIR)/bp_omp_par $(BINDIR)/bp_omp_task $(BINDIR)/bp_cilk_spawn $(BINDIR)/bp_cilk_for

$(BINDIR)/bp_serial: bin_packing_serial.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_serial bin_packing_serial.c $(LDFLAGS)
$(BINDIR)/bp_omp_par: bin_packing_omp_par.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_omp_par bin_packing_omp_par.c $(LDFLAGS)
$(BINDIR)/bp_omp_task: bin_packing_omp_task.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_omp_task bin_packing_omp_task.c $(LDFLAGS)
$(BINDIR)/bp_cilk_spawn: bin_packing_cilk_spawn.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_cilk_spawn bin_packing_cilk_spawn.c -lcilkrts $(LDFLAGS)
$(BINDIR)/bp_cilk_for: bin_packing_cilk_for.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_cilk_for bin_packing_cilk_for.c -lcilkrts $(LDFLAGS)

clean:	
	-rm -rf $(BINDIR)
