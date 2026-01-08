#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

void matmatijk(int ldA, int ldB, int ldC,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
    int i, k, j;
    for (i = 0; i < N1; ++i)
    {
        for (j = 0; j < N3; ++j)
        {
            for (k = 0; k < N2; ++k)
            {
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
            }
        }
    }
}

void matmatkji(int ldA, int ldB, int ldC,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
    int i, k, j;
    for (k = 0; k < N2; ++k)
    {
        for (j = 0; j < N3; ++j)
        {
            for (i = 0; i < N1; ++i)
            {
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
            }
        }
    }
}

void matmatikj(int ldA, int ldB, int ldC,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
    int i, k, j;
    for (i = 0; i < N1; ++i)
    {
        for (k = 0; k < N2; ++k)
        {
            for (j = 0; j < N3; ++j)
            {
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
            }
        }
    }
}

void matmatjik(int ldA, int ldB, int ldC,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
    int i, k, j;
    for (j = 0; j < N3; ++j)
    {
        for (i = 0; i < N1; ++i)
        {
            for (k = 0; k < N2; ++k)
            {
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
            }
        }
    }
}

void matmatkij(int ldA, int ldB, int ldC,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
    int i, k, j;
    for (k = 0; k < N2; ++k)
    {
        for (i = 0; i < N1; ++i)
        {
            for (j = 0; j < N3; ++j)
            {
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
            }
        }
    }
}

void matmatjki(int ldA, int ldB, int ldC,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
    int i, k, j;
    for (j = 0; j < N3; ++j)
    {
        for (k = 0; k < N2; ++k)
        {
            for (i = 0; i < N1; ++i)
            {
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
            }
        }
    }
}

void matmatblock(int ldA, int ldB, int ldC,
                 double *A, double *B, double *C,
                 int N1, int N2, int N3,
                 int dbA, int dbB, int dbC)
{
    int ii, jj, kk;

    for (ii = 0; ii < N1; ii += dbA)
    {
        for (jj = 0; jj < N3; jj += dbB)
        {
            for (kk = 0; kk < N2; kk += dbC)
            {

                double *Ablk = &A[ii * ldA + kk];
                double *Bblk = &B[kk * ldB + jj];
                double *Cblk = &C[ii * ldC + jj];

                matmatikj(ldA, ldB, ldC, Ablk, Bblk, Cblk, dbA, dbB, dbC);
            }
        }
    }
}

void matmatthread(int ldA, int ldB, int ldC, double *A, double *B, double *C,
                  int N1, int N2, int N3, int dbA, int dbB, int dbC, int NTROW, int NTCOL)
{

    int i, j, k, IDi, IDj, STARTi, STARTj;

    omp_set_num_threads(NTROW * NTCOL);

#pragma omp parallel private(IDi, IDj, STARTi, STARTj)
    {
        IDi = omp_get_thread_num() / NTCOL;
        IDj = omp_get_thread_num() % NTCOL;

        STARTi = IDi * (N1 / NTROW);
        STARTj = IDj * (N3 / NTCOL);

        double *local_mat_A = A + STARTi * ldA;
        double *local_mat_B = B + STARTj;
        double *local_mat_C = C + (STARTi * ldC) + STARTj;

        matmatblock(ldA, ldB, ldC, local_mat_A, local_mat_B, local_mat_C, N1 / NTROW, N2, N3 / NTCOL, dbA, dbB, dbC);
    }
}