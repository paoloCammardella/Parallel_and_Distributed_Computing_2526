#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>

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

void matmatthread(int ldA, int ldB, int ldC,
                  double *A, double *B, double *C,
                  int N1, int N2, int N3,
                  int dbA, int dbB, int dbC,
                  int NTROW, int NTCOL)
{
    int IDi, IDj, STARTi, STARTj;
    omp_set_num_threads(NTROW * NTCOL);
#pragma omp parallel private(IDi, IDj, STARTi, STARTj)
    {
        IDi = omp_get_thread_num() / NTCOL;
        IDj = omp_get_thread_num() % NTCOL;

        STARTi = IDi * N1 / NTROW;
        STARTj = IDj * N3 / NTCOL;

        double *A_loc = A + STARTi * ldA;
        double *B_loc = B + STARTj;
        double *C_loc = C + STARTi * ldC + STARTj;

        matmatblock(ldA, ldB, ldC,
                    A_loc, B_loc, C_loc,
                    N1 / NTROW, N2, N3 / NTCOL,
                    dbA, dbB, dbC);
    }
}

void matmatdist(
    MPI_Comm Gridcom,
    int LDA, int LDB, int LDC,
    double *A, double *B, double *C,
    int N1, int N2, int N3,
    int DB1, int DB2, int DB3,
    int NTrow, int NTcol)
{
    int dims[2], periods[2], coordinates[2];
    MPI_Cart_get(Gridcom, 2, dims, periods, coordinates);

    int NProw = dims[0];
    int NPcol = dims[1];
    int myrow = coordinates[0];
    int mycol = coordinates[1];

    MPI_Comm row_comms;
    int fix_row[2] = {0, 1};
    MPI_Cart_sub(Gridcom, fix_row, &row_comms);

    MPI_Comm col_comms;
    int fix_col[2] = {0, 1};
    MPI_Cart_sub(Gridcom, fix_col, &col_comms);

    int locN1 = N1 / NProw;
    int locN2 = N2 / K2;
    int locN3 = N3 / NPcol;

    int a = NProw, b = NPcol, rest;
    while (b != 0)
    {
        rest = a % b;
        a = b;
        b = rest;
    }
    int K2 = (NProw / a) * NPcol;

    double *Acolumn = (double *)malloc(sizeof(double) * locN1 * locN2);
    double *Brow = (double *)malloc(sizeof(double) * locN2 * locN3);

    int curr_block_A = 0;
    int curr_block_B = 0;

    int k, i, j;
    for (k = 0; k < K2; k++)
    {
        int c = k % NPcol;
        int r = k % NProw;

        if (mycol == c)
        {
            for (i = 0; i < locN1; i++)
            {
                memcpy(&Acol[i * locN2], &A[i * LDA + curr_block_A * locN2], locN2 * sizeof(double));
            }
            ++curr_block_A;
        }

        if (myrow == r)
        {
            for (j = 0; j < DB2; j++)
            {
                memcpy(&Brow[i * locN3], &B[(curr_block_B * locN2 + i) * LDB], locN3 * sizeof(double));
            }
            ++curr_block_B;
        }

        MPI_Bcast(Acolumn, locN1 * DB2, MPI_DOUBLE, c, row_comms);
        MPI_Bcast(Brow, DB2 * locN3, MPI_DOUBLE, r, col_comms);

        matmatthread(
            locN2, locN3, LDC,
            Acolumn, Brow, C,
            locN1, DB2, locN3,
            DB1, DB2, DB3,
            NTrow, NTcol);
    }

    free(Acolumn);
    free(Brow);
    MPI_Comm_free(&row_comms);
    MPI_Comm_free(&col_comms);
}
