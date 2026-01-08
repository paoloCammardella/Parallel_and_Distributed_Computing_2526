#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double get_cur_time()
{
    struct timeval tv;
    struct timezone tz;
    double cur_time;

    gettimeofday(&tv, &tz);
    cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

    return cur_time;
}

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

int main()
{
    int ldA, ldB, ldC, N1, N2, N3, i, j, dbA, dbB, dbC, NTROW, NTCOL, NT;
    double *A, *B, *C, time, flops, gflops, t0, t1;

    ldA = 2500;
    ldB = 2500;
    ldC = 2500;

    N1 = 1024;
    N2 = 1024;
    N3 = 1024;

    dbA = 256;
    dbB = 256;
    dbC = 256;

    NTROW = 2;
    NTCOL = 2;

    A = (double *)malloc(sizeof(double) * ldA * ldA);
    B = (double *)malloc(sizeof(double) * ldB * ldB);
    C = (double *)malloc(sizeof(double) * ldC * ldC);

    /* initializing A*/
    for (i = 0; i < N1; i++)
    {
        for (j = 0; j < N1; j++)
        {
            A[i * ldA + j] = (rand() % 100);
        }
    }

    /* initializing B*/
    for (i = 0; i < N2; i++)
    {
        for (j = 0; j < N2; j++)
        {
            B[i * ldB + j] = (rand() % 100);
        }
    }

    /* initializing C*/
    for (i = 0; i < N3; i++)
    {
        for (j = 0; j < N3; j++)
        {
            C[i * ldC + j] = (rand() % 100);
        }
    }

    /* Computing time ijk*/
    t0 = get_cur_time();
    matmatijk(ldA, ldB, ldC, A, B, C, N1, N2, N3);
    t1 = get_cur_time();

    time = t1 - t0;
    flops = 2.0 * N1 * N2 * N3; /*2 flops (addizione e moltiplicazione) * 3 for*/
    gflops = flops / (time * 1e9);

    printf("IJK Tempo: %f s, GFLOP/s: %f\n", time, gflops);

    /* Computing time kji*/
    t0 = get_cur_time();
    matmatkji(ldA, ldB, ldC, A, B, C, N1, N2, N3);
    t1 = get_cur_time();

    time = t1 - t0;
    flops = 2.0 * N1 * N2 * N3; /*2 flops (addizione e moltiplicazione) * 3 for*/
    gflops = flops / (time * 1e9);

    printf("KJI Tempo: %f s, GFLOP/s: %f\n", time, gflops);

    /* Computing time ikj*/
    t0 = get_cur_time();
    matmatikj(ldA, ldB, ldC, A, B, C, N1, N2, N3);
    t1 = get_cur_time();

    time = t1 - t0;
    flops = 2.0 * N1 * N2 * N3; /*2 flops (addizione e moltiplicazione) * 3 for*/
    gflops = flops / (time * 1e9);

    printf("IKJ Tempo: %f s, GFLOP/s: %f\n", time, gflops);

    /* Computing time jik*/
    t0 = get_cur_time();
    matmatjik(ldA, ldB, ldC, A, B, C, N1, N2, N3);
    t1 = get_cur_time();

    time = t1 - t0;
    flops = 2.0 * N1 * N2 * N3; /*2 flops (addizione e moltiplicazione) * 3 for*/
    gflops = flops / (time * 1e9);

    printf("JIK Tempo: %f s, GFLOP/s: %f\n", time, gflops);

    /* Computing time kij*/
    t0 = get_cur_time();
    matmatkij(ldA, ldB, ldC, A, B, C, N1, N2, N3);
    t1 = get_cur_time();

    time = t1 - t0;
    flops = 2.0 * N1 * N2 * N3; /*2 flops (addizione e moltiplicazione) * 3 for*/
    gflops = flops / (time * 1e9);

    printf("KIJ Tempo: %f s, GFLOP/s: %f\n", time, gflops);

    /* Computing time jki*/
    t0 = get_cur_time();
    matmatjki(ldA, ldB, ldC, A, B, C, N1, N2, N3);
    t1 = get_cur_time();

    time = t1 - t0;
    flops = 2.0 * N1 * N2 * N3; /*2 flops (addizione e moltiplicazione) * 3 for*/
    gflops = flops / (time * 1e9);

    printf("JKI Tempo: %f s, GFLOP/s: %f\n", time, gflops);

    /* Computing time matmatblock*/
    t0 = get_cur_time();
    matmatblock(ldA, ldB, ldC, A, B, C, N1, N2, N3, dbA, dbB, dbC);
    t1 = get_cur_time();

    time = t1 - t0;
    flops = 2.0 * N1 * N2 * N3; /*2 flops (addizione e moltiplicazione) * 3 for*/
    gflops = flops / (time * 1e9);

    printf("Block Tempo: %f s, GFLOP/s: %f\n", time, gflops);
    double Tthread, speedup, eff;
    double Tblock = time;
    double Gblock = gflops;
    int c;

    int configs[4][2] = {{1, 1}, {1, 2}, {2, 2}, {2, 4}};

    for (c = 0; c < 4; ++c)
    {
        NTROW = configs[c][0];
        NTCOL = configs[c][1];
        NT = NTROW * NTCOL;

        t0 = get_cur_time();
        matmatthread(ldA, ldB, ldC, A, B, C,
                     N1, N2, N3,
                     dbA, dbB, dbC,
                     NTROW, NTCOL);
        t1 = get_cur_time();

        Tthread = t1 - t0;
        time = Tthread;
        gflops = flops / (time * 1e9);

        speedup = Tblock / Tthread;
        eff = speedup / NT;

        printf("Thread NT=%d (%dx%d): Tempo=%f s, Gflops=%f, Speedup=%f, Efficienza=%f\n",
               NT, NTROW, NTCOL, time, gflops, speedup, eff);
    }

    return 0;
}