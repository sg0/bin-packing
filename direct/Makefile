CFLAGS = -fopenmp -g -Wall -std=gnu99 -DDEBUG=2 -mfpmath=sse -Ofast -march=native -funroll-loops
LDFLAGS = -fopenmp -g -lm
CC = gcc
MKDIR_P = mkdir -p
BINDIR = bins

.PHONY: bindir

all: bindir binaries

bindir:
	$(MKDIR_P) $(BINDIR)

binaries: $(BINDIR)/bp_serial $(BINDIR)/bp_omp_par $(BINDIR)/bp_omp_task

$(BINDIR)/bp_serial: bin_packing_serial.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_serial bin_packing_serial.c $(LDFLAGS)
$(BINDIR)/bp_omp_par: bin_packing_omp_par.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_omp_par bin_packing_omp_par.c $(LDFLAGS)
$(BINDIR)/bp_omp_task: bin_packing_omp_task.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_omp_task bin_packing_omp_task.c $(LDFLAGS)

clean:	
	-rm -rf $(BINDIR)
