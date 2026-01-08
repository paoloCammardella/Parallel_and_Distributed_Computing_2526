#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

double maxsum(int N, int LD, double *A, int NT)
{
  double sum = 0, curr_max = 0, max = 0;
  int id, start, end, i, j;

  omp_set_num_threads(NT);
#pragma omp parallel private(id, sum, curr_max, start, end, i, j)
  {
    id = omp_get_thread_num();
    start = id * N / NT;
    end = (id + 1) * N / NT;

    for (i = start; i < end; i++)
    {
      for (j = i; j < N; j++)
      {
        sum = sum + sqrt(A[i * LD + j]);
      }
      if (curr_max < sum)
      {
        curr_max = sum;
      }
      sum = 0;
    }

#pragma omp critical
    {
      if (max < curr_max)
      {
        max = curr_max;
      }
    }
  }
  return max;
}