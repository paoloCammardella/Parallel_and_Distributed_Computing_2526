#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

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