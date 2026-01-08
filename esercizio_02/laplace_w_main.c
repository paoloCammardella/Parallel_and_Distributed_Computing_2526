#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
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

void laplace(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter);

int main(int argc, char *argv[])
{

	int nproc, myid, prev, next;
	int N, i, j, ifirst, iter, Niter, LD;
	float *A, *Anew, *daprev, *danext;
	MPI_Status status;
	double t1, t2;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	printf("hello from %d di %d processi \n", myid, nproc);
	sleep(1);

	N = 400;
	Niter = 8000;
	LD = 500;
	A = (float *)malloc(500 * 500 * sizeof(float));
	Anew = (float *)malloc(500 * 500 * sizeof(float));
	daprev = (float *)malloc(500 * sizeof(float));
	danext = (float *)malloc(500 * sizeof(float));

	for (i = 0; i < N / nproc; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i * LD + j] = 0.;
		}
	}
	if (myid == 0)
		for (j = 0; j < N; j++)
			A[0 * LD + j] = j;

	if (myid == nproc - 1)
		for (j = 0; j < N; j++)
			A[(N / nproc - 1) * LD + j] = N - 1 - j;

	ifirst = myid * N / nproc;
	for (i = 0; i < N / nproc; i++)
	{
		A[i * LD + 0] = ifirst + i;
		A[i * LD + N - 1] = N - 1 - A[i * LD + 0];
	}

	if (myid == 0)
		printf("\n esecuzione con N = %d  e %d iterazioni\n\n", N, Niter);

	t1 = get_cur_time();

	laplace(A, Anew, daprev, danext, N, LD, Niter);

	t2 = get_cur_time();

	if (myid == 0)
		printf("con %d processi, il tempo e' %f\n", nproc, t2 - t1);

	sleep(1);
	if (myid == 0)
		printf("prima  %d -->   %f  %f  \n", myid, A[1 * LD + 1], A[1 * LD + 398]);
	if (myid == 3)
		printf("centro %d -->   %f  %f  \n", myid, A[49 * LD + 199], A[49 * LD + 200]);
	if (myid == 4)
		printf("centro %d -->   %f  %f  \n", myid, A[00 * LD + 199], A[00 * LD + 200]);
	if (myid == 7)
		printf("ultima %d -->   %f  %f  \n", myid, A[48 * LD + 1], A[48 * LD + 398]);

	MPI_Finalize();
}

void laplace(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter)
{
	int nproc, myid, start, end, i, j, iter;
	float above, below, left, right;
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	start = (myid == 0) ? 1 : 0;
	end = (myid == nproc - 1) ? ((N / nproc) - 1) : (N / nproc);

	for (iter = 0; iter < Niter; ++iter)
	{

		if (myid != 0)
		{
			MPI_Sendrecv(&A[0], N, MPI_FLOAT, myid - 1, 0, daprev, N, MPI_FLOAT, myid - 1, 1, MPI_COMM_WORLD, &status);
		}
		if (myid != nproc - 1)
		{
			MPI_Sendrecv(&A[((N / nproc) - 1) * LD], N, MPI_FLOAT, myid + 1, 1, danext, N, MPI_FLOAT, myid + 1, 0, MPI_COMM_WORLD, &status);
		}

		for (i = start; i < end; ++i)
		{
			for (j = 1; j < N - 1; ++j)
			{
				if (i == 0)
				{
					above = (myid == 0) ? A[(i)*LD + j] : daprev[j];
				}
				else
				{
					above = A[(i - 1) * LD + j];
				}

				if (i == ((N / nproc) - 1))
				{
					below = (myid == nproc - 1) ? A[(i)*LD + j] : danext[j];
				}
				else
				{
					below = A[(i + 1) * LD + j];
				}

				B[i * LD + j] = 0.25f * (above + below + A[i * LD + (j - 1)] + A[i * LD + (j + 1)]);
			}
		}
		for (i = start; i < end; i++)
		{
			for (j = 1; j < N - 1; j++)
			{
				A[i * LD + j] = B[i * LD + j];
			}
		}
	}
}

void laplace_nb(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter)
{
	int nproc, myid, start, end, i, j, iter;
	float above, below, left, right;
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	start = (myid == 0) ? 1 : 0;
	end = (myid == nproc - 1) ? ((N / nproc) - 1) : (N / nproc);

	for (iter = 0; iter < Niter; ++iter)
	{
		MPI_Request req[4];
		int rq = 0;

		if (myid != 0)
		{
			MPI_Irecv(daprev, N, MPI_FLOAT, myid - 1, 20, MPI_COMM_WORLD, &req[rq++]);
			MPI_Isend(&A[0], N, MPI_FLOAT, myid - 1, 10, MPI_COMM_WORLD, &req[rq++]);
		}
		if (myid != nproc - 1)
		{
			MPI_Irecv(danext, N, MPI_FLOAT, myid + 1, 10, MPI_COMM_WORLD, &req[rq++]);
			MPI_Isend(&A[((N / nproc) - 1) * LD], N, MPI_FLOAT, myid + 1, 20, MPI_COMM_WORLD, &req[rq++]);
		}

		if (rq > 0)
		{
			MPI_Waitall(rq, req, MPI_STATUSES_IGNORE);
		}

		for (i = start; i < end; ++i)
		{
			for (j = 1; j < N - 1; ++j)
			{
				if (i == 0)
				{
					above = (myid == 0) ? A[(i)*LD + j] : daprev[j];
				}
				else
				{
					above = A[(i - 1) * LD + j];
				}

				if (i == ((N / nproc) - 1))
				{
					below = (myid == nproc - 1) ? A[(i)*LD + j] : danext[j];
				}
				else
				{
					below = A[(i + 1) * LD + j];
				}

				B[i * LD + j] = 0.25f * (above + below + A[i * LD + (j - 1)] + A[i * LD + (j + 1)]);
			}
		}
		for (i = start; i < end; i++)
		{
			for (j = 1; j < N - 1; j++)
			{
				A[i * LD + j] = B[i * LD + j];
			}
		}
	}
}