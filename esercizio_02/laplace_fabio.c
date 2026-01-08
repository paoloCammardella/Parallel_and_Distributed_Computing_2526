```#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>


void laplace (float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter) {
        int nproc, myid;
        MPI_Status status;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        int localN, localStart, localEnd, i, j, iter;
        float up, down, left, right;

        localN = N / nproc;
        localStart = (myid == 0) ? 1 : 0;              /* skip global row 0 */
        localEnd = (myid == nproc-1) ? localN-1 : localN; /* skip global row N-1 */


        for (iter = 0; iter < Niter; iter++){
                /* Exchange rows */
                if (myid != 0) {
                        MPI_Sendrecv(&A[0*LD], N, MPI_FLOAT, myid-1, 10,
                                     daprev, N, MPI_FLOAT, myid-1, 20,
                                     MPI_COMM_WORLD, &status);
                }
                if (myid != nproc-1) {
                        MPI_Sendrecv(&A[(localN-1)*LD], N, MPI_FLOAT, myid+1, 20,
                                     danext, N, MPI_FLOAT, myid+1, 10,
                                     MPI_COMM_WORLD, &status);
                }
                /* Compute B */
                for (i = localStart; i < localEnd; i++){
                        for (j = 1; j < N-1; j++){

                                /* up */
                                if (i == 0) {
                                        up = (myid == 0) ? A[(i)*LD + j] : daprev[j];
                                } else {
                                        up = A[(i-1)*LD + j];
                                }

                                /* down */
                                if (i == localN-1) {
                                        down = (myid == nproc-1) ? A[(i)*LD + j] : danext[j];
                                } else {
                                        down = A[(i+1)*LD + j];
                                }

                                left  = A[i*LD + (j-1)];
                                right = A[i*LD + (j+1)];

                                B[i*LD + j] = 0.25f * (up + down + left + right);
                        }
                }

                /* copy B into A*/
                for (i = localStart; i < localEnd; i++){
                        for (j = 1; j < N-1; j++){
                                A[i*LD + j] = B[i*LD + j];
                        }
                }
        }
}

void laplace_nb (float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter) {
        int nproc, myid;
        MPI_Status status;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        int localN, localStart, localEnd, i, j, iter;
        float up, down, left, right;

        localN = N / nproc;
        localStart = (myid == 0) ? 1 : 0;              /* skip global row 0 */
        localEnd = (myid == nproc-1) ? localN-1 : localN; /* skip global row N-1 */

        for (iter = 0; iter < Niter; iter++){
                MPI_Request req[4]; 
                int rq = 0;

                /* Recv */
                /* Receive first: Make the process ready to receive*/
                if (myid != 0) {
                        MPI_Irecv(daprev, N, MPI_FLOAT, myid-1, 20, MPI_COMM_WORLD, &req[rq++]);
                }
                if (myid != nproc-1) {
                        MPI_Irecv(danext, N, MPI_FLOAT, myid+1, 10, MPI_COMM_WORLD, &req[rq++]);
                }

                /* Send */
                if (myid != 0) {
                        MPI_Isend(&A[0*LD], N, MPI_FLOAT, myid-1, 10, MPI_COMM_WORLD, &req[rq++]);
                }
                if (myid != nproc-1) {
                        MPI_Isend(&A[(localN-1)*LD], N, MPI_FLOAT, myid+1, 20, MPI_COMM_WORLD, &req[rq++]);
                }

                /* Wait */
                if (rq > 0){
                        MPI_Waitall(rq, req, MPI_STATUSES_IGNORE);
                }

                /* Compute B */
                for (i = localStart; i < localEnd; i++){
                        for (j = 1; j < N-1; j++){

                                /* up */
                                if (i == 0) {
                                        up = (myid == 0) ? A[(i)*LD + j] : daprev[j];
                                } else {
                                        up = A[(i-1)*LD + j];
                                }

                                /* down */
                                if (i == localN-1) {
                                        down = (myid == nproc-1) ? A[(i)*LD + j] : danext[j];
                                } else {
                                        down = A[(i+1)*LD + j];
                                }

                                left  = A[i*LD + (j-1)];
                                right = A[i*LD + (j+1)];

                                B[i*LD + j] = 0.25f * (up + down + left + right);
                        }
                }

                /* copy B into A*/
                for (i = localStart; i < localEnd; i++){
                        for (j = 1; j < N-1; j++){
                                A[i*LD + j] = B[i*LD + j];
                        }
                }
        }
}```