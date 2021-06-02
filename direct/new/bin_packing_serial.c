// simple bin packing
// using best fit algorithm
// -- place the next item
// -- in the tightest bin so 
// -- that smallest empty space 
// -- is left

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <limits.h>

#include <omp.h>

/* BPP instance generator, functions taken from
 * Numerical Recipes in C and fortran code listed 
 * in paper -- 
 * Schwerin, Petra, and Gerhard Wäscher. 
 * "The Bin¿Packing Problem: A Problem Generator 
 * and Some Numerical Experiments with FFD Packing 
 * and MTP." International Transactions in 
 * Operational Research 4.5¿6 (1997): 377-389.
 * */
/* Taken from: http://www.cs.nmsu.edu/~cgiannel/ran0.C */
// "Minimal" random number generator of Park and Miller with
// Bays-Durham shuffle and added safeguards. Returns a uniform random
// deviate between 0.0 and 1.0. Set or reset idum to any integer value
// (except the unlikely value MASK) to initialize the sequence; idum
// must not be altered between calls for successive deviates in a sequence. 
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

int *wu = NULL, *dum = NULL, *q = NULL;

float
ran0 (int *idum)
{
  (*idum) ^= MASK;		// XORing with MASK allows use of 0 and
  int k = (*idum) / IQ;		//     other simple bit patterns for idum.
  *idum = IA * (*idum - k*IQ) - IR*k;	// Compute idum = (IA*idum) % IM without
  if (*idum < 0)
    *idum += IM;		//     overflows by Schrage's method.
  float ans = AM * (*idum);		// Convert idum to a floating result.
  *idum ^= MASK;		// Unmask before return.
  return ans;
}

// comparison function for stdlib qsort
int
comp (const void *elem1, const void *elem2)
{
  int f = *((int *) elem1);
  int s = *((int *) elem2);
  if (f > s)
    return 1;
  if (f < s)
    return -1;
  return 0;
}

/*
n: number of items to be packed, n <= 1000
c: capacity of the bin
vl: lower bound for the relative weight of items
    as a fraction of the bin capacity
v2: upper bound for the relative weight of items
    as a fraction of the bin capacity
0 < vl <= v2 <=1
dum: dummy-vector (integer) of size n
z: seed for initialising the random number gene
m: number of different item weights
wu: (0..m) item weights
q: (0..m) number of items with weight wu[i],
i = 0,..,m 
*/
void
bppgen (int n, int c, float v1, float v2, int *z, int *m)
{
  // test 0 < vl <= v2 <=1
  if (v1 < 0 || v1 > v2 || v2 > 1)
    return;

  // generate n pseudorandom numbers in [v1*c, v2*c]
  for (int i = 0; i < n; i++)
  {
      float zzz = ran0 (z);
      dum[i] = (int) ((v1 + (v2 - v1) * zzz) * (float) (c) + zzz);
  }

  // sort dummy vector in nonincreasing order
  qsort (dum, n, sizeof(int), comp);

  // sort items in wu, sum up repeated item weights
  int ii = 0;
  wu[ii] = dum[n - 1];
  q[ii] = 1;
  for (int i = (n - 2); i >= 0; i--)
    {
      if (dum[i] == wu[ii])
	q[ii] += 1;
      else
	{
	  ii += 1;
	  wu[ii] = dum[i];
	  q[ii] = 1;
	}
    }
  // register number of different weights
  *m = (ii + 1);
}

// current item in wu (weights) array waiting
// to be allocated to a bin
int *space_left_on_bin = NULL;
int *item_in_bin = NULL;
int *bins = NULL;

int bin_capacity = 0;
int max_num_bins = 0;
int max_items_per_bin = 0;

// iterate over all items
// find the index to the tightest bin
// and insert item ther
void
insert_item_to_bin (int num_items)
{
  for (int u = 0; u < num_items; u++)
    {
      int item = wu[u];
      int done = 0;
      // try to find the tightest bin
      for (int i = 0; ((i < max_num_bins) && !done); i++)
	{
	  if (space_left_on_bin[i] >= item)
	    {
		for (int j = 0; j < max_items_per_bin; j++)
		{
		  if (bins[i * max_num_bins + j] == 0) // check if a cell is empty
		    {
		      space_left_on_bin[i] -= item;
		      bins[i * max_num_bins + j] = item;
		      item_in_bin[item] = 1;
		      done = 1;
		      break; // break from innermost loop
		    }
		  else
		      continue;
		}
	    }
	}
    }
}

int
main (int argc, char *argv[])
{
  int n, C, m = 0;
  float v1, v2;

  if (argc != 5)
    {
      printf ("Usage: %s < C > < n > < v1 > < v2 >\n", argv[0]);
      exit (1);
    }
  else
    {
      // these are not random values, there are constants 
      // mentioned in the BPPGEN paper (p. 379)
      C = atoi (argv[1]);
      n = atoi (argv[2]);
      v1 = atof (argv[3]);
      v2 = atof (argv[4]);
    }

  // allocate wu, q and dum [0..n]
  wu = (int *) malloc (n * sizeof (int));
  q = (int *) malloc (n * sizeof (int));
  dum = (int *) malloc (n * sizeof (int));

  memset (wu, 0, (n * sizeof (int)));
  memset (q, 0, (n * sizeof (int)));
  memset (dum, 0, (n * sizeof (int)));

  // generate seed, as mentioned in p. 381
  // of the bppgen paper
  int Z = (long) (v1 * C + (int) v2 * (float)((C / 10)) + (int) n * (float)((C / 100)) + C);

  // generate items
  bppgen (n, C, v1, v2, &Z, &m);

  // check item weights
  printf ("Number of item weights: %d\n", n);
  printf ("Different item weights: %d\n", m);
#if DEBUG > 1
  printf ("Item weights: ");
  for (int i = 0; i < m; i++)
    printf (" %d ", wu[i]);
  printf ("\n");
#endif

  bin_capacity = C;
  // upper limit
  // bin size and number of bins
  max_items_per_bin = bin_capacity / wu[m - 1];
  // worst case, one element per bin
  max_num_bins = m;

  // initialize bin size
  // initially all bins are empty
  bins = malloc (sizeof (int) * max_num_bins * max_items_per_bin);
  memset (bins, 0, sizeof (int) * max_num_bins * max_items_per_bin);

  item_in_bin = (int *) malloc (sizeof (int) * bin_capacity);
  memset (item_in_bin, 0, sizeof (int) * bin_capacity);

  space_left_on_bin = (int *) malloc (sizeof (int) * max_num_bins);
  for (int i = 0; i < max_num_bins; i++)
    space_left_on_bin[i] = C;

  srand (time (NULL));

  double start_time = omp_get_wtime ();
  {
    // insert items into bin
    insert_item_to_bin( m );
  }
  double end_time = omp_get_wtime ();

  printf
    ("\n Maximum number of bins = %d\n Maximum items per bin  = %d\n Time taken = %f secs\n",
     max_num_bins, max_items_per_bin, (end_time - start_time));

#if DEBUG > 1
  int bin_count = 0;
  for (int i = 0; i < max_num_bins; i++)
    {
      if (space_left_on_bin[i] != C)
	{
	  bin_count++;
	  printf ("\n bin %d: ", i + 1);
	  for (int j = 0; j < max_items_per_bin; j++)
	    printf ("\t%d", bins[i * max_num_bins + j]);
	}
    }
  printf ("\nNumber of bins: %d\n", bin_count);
#endif

#if DEBUG > 0
  int check = 0;
  for (int i = 0; i < max_num_bins; i++)
    {
      if (space_left_on_bin[i] < 0)
	{
	  check = 1;
	  break;
	}
    }
  if (check)
    printf ("Validation: FAIL\n");
  else
    printf ("Validation: PASS\n");
#endif

  free (bins);
  free (q);
  free (wu);
  free (dum);
  free (space_left_on_bin);
  free (item_in_bin);

  return 0;
}
