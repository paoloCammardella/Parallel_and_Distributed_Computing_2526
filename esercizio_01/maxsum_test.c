#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

double get_cur_time();
int main()
{
  int NT, N, i, j, LD;
  double MAX, *A, t1, t2, save;
  double maxsum(int, int, double *, int);

  LD = 800;
  A = (double *)malloc(sizeof(double) * LD * LD);
  N = 800;
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      A[i * LD + j] = (rand() % 100);
    }
  }

  for (NT = 1; NT <= 8; NT = NT * 2)
  {

    printf("===============\n");
    printf(" Threads = %d \n", NT);

    t1 = get_cur_time();
    MAX = maxsum(N, LD, A, NT);
    t2 = get_cur_time();

    if (NT == 1)
      save = t2 - t1;
    printf("Max sum of modules with N = %d is %f \n", N, MAX);
    printf("Total time: %e , speedup: %f , efficiency: %f \n", t2 - t1, save / (t2 - t1),
           save / (t2 - t1) / NT);
  }
}

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

double get_cur_time()
{
  struct timezone time_zone;
  struct timeval time_value;

  double cur_time;

  gettimeofday(&time_value, &time_zone);
  cur_time = time_value.tv_sec + time_value.tv_usec / 1000000.0;

  return cur_time;
}