#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
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

    /*Increasing with block index*/
    for (ii = 0; ii < N1; ii += dbA)
    { // blocks on A and C rows
        for (jj = 0; jj < N3; jj += dbB)
        { // blocks on B and C rows
            for (kk = 0; kk < N2; kk += dbC)
            { // blocks on A and B columns

                /*Defining when it starts the block*/
                double *Ablk = &A[ii * ldA + kk]; // block A(ii,kk)
                double *Bblk = &B[kk * ldB + jj]; // block B(kk,jj)
                double *Cblk = &C[ii * ldC + jj]; // block C(ii,jj)

                matmatikj(ldA, ldB, ldC, Ablk, Bblk, Cblk, dbA, dbB, dbC);
            }
        }
    }
}
