CFLAGS = -fopenmp -g -Wall -std=gnu99 -O3 -DDEBUG=2 -ffast-math
LDFLAGS = -fopenmp -g -lm
CC = gcc
MKDIR_P = mkdir -p
BINDIR = bins

.PHONY: bindir

all: bindir binaries

bindir:
	$(MKDIR_P) $(BINDIR)

binaries: $(BINDIR)/bp_pgdm $(BINDIR)/bp_pgdm2

$(BINDIR)/bp_pgdm: bin_packing_paradigm.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_pgdm bin_packing_paradigm.c $(LDFLAGS)
$(BINDIR)/bp_pgdm2: bin_packing_paradigm2.c
	$(CC) $(CFLAGS) -o $(BINDIR)/bp_pgdm bin_packing_paradigm2.c $(LDFLAGS)

clean:	
	-rm -rf $(BINDIR)
