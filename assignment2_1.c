/************************************************************************************
 * FILE: gepp_0_serquential.c
 * DESCRIPTION:
 * program for Gaussian elimination with partial pivoting (blocked, loop unrolling parallel)
 * AUTHOR: Sicheng Liu
 * LAST REVISED: 15/04/2024
 *************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

#define BLOCK_SIZE 4
#define THRESHOLD 1.0e-16

typedef struct
{
    /* data */
    int original_row;
    int current_row;
} PivotingInfo;

void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);
void orig_gaussian_elimination(double **T, int n);
// void distrubuted_gaussian_elimination(double **b, double **LL, double **temp, int n);

int main(int agrc, char **agrv)
{
    double *a0, *b0, *d0, *e0, *f0, *LL0, *temp0; // auxiliary 1D for 2D matrix a
    double **a, **b, **d, **e, **f, **LL, **temp; // 2D matrix for sequential computation

    int n; // input size
    int indk;
    double c, amax, c1, c2, c3;

    struct timeval start_time_orig, end_time_orig, start_time_bloc, end_time_bloc;

    long seconds, microseconds;
    double elapsed, elapsed_bloc;
    int myid, sourceid; // process id
    int numprocs;       // total number of processes created
    double **A;         // 2D matrix of size M x N, created by process 0 for input/output data
    double **AK;        // 2D submatrix for every processes, just enough to hold their assigned row blocks
    double **AW;
    double *A0, *AK0, *AW0; // auxiliary 1D matrices to make rows of A & AK contiguously placed in memory
    int M, N, K;            // the sizes of A & AK, the value of K may be different for different processes
    int Nb, bn, block_size;
    int q, r;
    int ib, kb, ib_0, i, j, k, l;

    MPI_Status status;

    if (agrc == 3)
    {
        n = atoi(agrv[1]);
        block_size = atoi(agrv[2]);
        // printf("The matrix size:  %d * %d \n", n, n);
    }
    else
    {
        printf("Usage: %s n\n\n"
               " n: the matrix size\n\n",
               agrv[0]);
        return 1;
    }

    /*** Allocate contiguous memory for 2D matrices ***/
    a0 = (double *)malloc(n * n * sizeof(double));
    a = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        a[i] = a0 + i * n;
    }
    LL0 = (double *)malloc(n * n * sizeof(double));
    LL = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        LL[i] = LL0 + i * n;
    }

    temp0 = (double *)malloc(n * n * sizeof(double));
    temp = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        temp[i] = temp0 + i * n;
    }

    srand(time(0));
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (numprocs == 1)
    {
        printf("\n\nThe number of processes created is just 1 - a trivial problem!\n\n");

        MPI_Finalize();
        return 0;
    }

    M = n;
    N = n;
    // block_size = BLOCK_SIZE;

    Nb = N / block_size; // the total number of row blocks
    bn = block_size * M; // row block size

    /* All processes create a submatrix AK of size K X N */
    q = Nb / numprocs; // each process gets at least q row blocks
    r = Nb % numprocs; // remaining row blocks
    if (myid < r)      // one more row block for each of the first r processes
        kb = q + 1;
    else
        kb = q;
    K = kb * block_size; // number of rows in submatrix and may be different for different processes

    AK0 = (double *)malloc(M * K * sizeof(double));
    AK = (double **)malloc(M * sizeof(double *));
    AW0 = (double *)malloc(M * block_size * sizeof(double));
    AW = (double **)malloc(M * sizeof(double *));
    if (AK == NULL)
    {
        fprintf(stderr, "**AK out of memory\n");
        exit(1);
    }
    for (i = 0; i < M; i++)
        AK[i] = &AK0[i * K];

    for (i = 0; i < M; i++)
    {
        AW[i] = &AW0[i * block_size];
    }

    /* all processes update matrix AK */
    for (i = 0; i < M; i++)
        for (j = 0; j < K; j++)
            AK[i][j] = myid + 1;

    /* all processes update matrix AW */
    for (i = 0; i < M; i++)
        for (j = 0; j < block_size; j++)
            AW[i][j] = 0;
    MPI_Datatype column_type;
    MPI_Type_vector(M, block_size, N, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    MPI_Datatype column_type_local;
    MPI_Type_vector(M, block_size, K, MPI_DOUBLE, &column_type_local);
    MPI_Type_commit(&column_type_local);
    /*Store the origin matrix on machine 0 and then scatter to other machines*/
    // printf("myid: %d, numprocs: %d\n ", myid, numprocs);
    if (myid == 0)
    {
        // printf("Processor 0 is running first.\n");
        A0 = (double *)malloc(M * M * sizeof(double));
        A = (double **)malloc(M * sizeof(double *));
        for (i = 0; i < M; i++)
            A[i] = &A0[i * N];

        /*copy the matrix*/
        srand(1234); // 使用固定的种子值
        printf("Creating and initializing matrices...\n\n");
        for (i = 0; i < M; i++)
        {
            for (j = 0; j < N; j++)
            {
                A[i][j] = (double)rand() / RAND_MAX;
                // A[i][j] = (double)i*N+j;
                // A[i][j] = (double)rand() / RAND_MAX;
                a[i][j] = A[i][j];
            }
        }
        gettimeofday(&start_time_orig, 0);
        orig_gaussian_elimination(a, N);
        gettimeofday(&end_time_orig, 0);
        seconds = end_time_orig.tv_sec - start_time_orig.tv_sec;
        microseconds = end_time_orig.tv_usec - start_time_orig.tv_usec;

        elapsed = seconds + 1e-6 * microseconds;
        printf("Sequential calculation time: %f\n\n", elapsed);
        ib = 0;
        ib_0 = 0;
        /*Column block cyclic partitioning*/
        for (j = 0; j < q; j++)
        {
            // printf("ib: %d K: %d ", ib, K);
            // copy the first block to the 0 machine
            for (i = 0; i < M; i++)
            {
                for (k = ib, l = ib_0; k < ib + block_size; k++, l++)
                {
                    AK[i][l] = A[i][k];
                }
            }
            // send other blocks to other machines
            for (i = 1; i < numprocs; i++)
            {
                ib += block_size;
                MPI_Send(&A[0][ib], 1, column_type, i, 1, MPI_COMM_WORLD);
            }

            // leave one row block for process 0
            // for (i = 0; i < M; i++)
            // {
            //     for (j = ib; j < ib + block_size; j++)
            //     {
            //         AK[i][j] = A[i][j];
            //     }
            // }
            ib += block_size;
            ib_0 += block_size;
        }

        // send remaining blocks, one block to each processe with myid < r
        if (r > 0)
        {
            for (i = 0; i < M; i++)
            {
                // printf("ib: %d ", ib);
                for (j = ib, l = ib_0; j < ib + block_size; j++, l++)
                {
                    AK[i][l] = A[i][j];
                }
            }
            for (i = 1; i < r; i++)
            {
                ib += block_size;
                MPI_Send(&A[0][ib], 1, column_type, i, 1, MPI_COMM_WORLD);
            }
        }
    }

    else
    { // all other processes receive a submatrix from process 0

        ib = 0;
        for (i = 0; i < kb; i++)
        {
            MPI_Recv(&AK[0][ib], 1, column_type_local, 0, 1, MPI_COMM_WORLD, &status);
            ib += block_size;
        }
    }
    // printf("Before PP: Current id is %d \n", myid);
    // print_matrix(AK, M, K);
    if (myid == 0)
    {
        gettimeofday(&start_time_orig, 0);
    }
    ib = 0;
    int ib_green = block_size;
    PivotingInfo pivoting_info[block_size];
    MPI_Datatype kv_pivoting_info_type;
    MPI_Type_contiguous(2, MPI_INT, &kv_pivoting_info_type);
    MPI_Type_commit(&kv_pivoting_info_type);

    int count[block_size];
    int stride[block_size];
    for (i = 0; i < block_size; i++)
    {
        count[i] = i;
        stride[i] = i * block_size;
    }
    MPI_Datatype lower_tri;
    MPI_Type_indexed(block_size, count, stride, MPI_DOUBLE, &lower_tri);
    MPI_Type_commit(&lower_tri);

    for (k = 0; k < Nb; k++)
    {
        int start_column = (k / numprocs) * block_size;
        int end_column = start_column + block_size;
        // int start_row = myid * block_size + (k / numprocs) * numprocs * block_size;
        int start_row = k * block_size;
        int end_row = start_row + block_size;
        int start_row_pink = k * block_size;
        int end_row_pink = start_row_pink + block_size;

        if (myid == k % numprocs)
        {

            /*Partial pivoting*/

            // printf("working columns %d %d\n", start_column, end_column);
            // printf("Current id is %d, k is %d \n", myid, k);
            for (i = start_column, l = start_row; i < end_column; i++, l++)
            {
                amax = AK[l][i];
                indk = l;
                for (j = l + 1; j < M; j++)
                {
                    if (fabs(AK[j][i]) > fabs(amax))
                    {
                        amax = AK[j][i];
                        indk = j;
                    }
                }

                if (amax == 0.0)
                {
                    printf("Matrix is singular!\n");
                    exit(1);
                }

                // Swap rows if needed
                if (indk != l)
                {
                    // printf("myid: %d, k: %d, l: %d, indk: %d, amax: %f\n", myid, k, l, indk, amax);

                    for (j = 0; j < K; ++j)
                    {
                        c = AK[l][j];
                        AK[l][j] = AK[indk][j];
                        AK[indk][j] = c;
                    }
                }
                pivoting_info[i - start_column].original_row = l;
                pivoting_info[i - start_column].current_row = indk;

                /*Gaussian elimination*/
                for (j = l + 1; j < M; ++j)
                {
                    AK[j][i] = AK[j][i] / AK[l][i];
                }
                for (j = l + 1; j < M; ++j)
                {
                    c = AK[j][i];
                    for (int x = i + 1; x < end_column; ++x)
                    {
                        // c=b[i][l];
                        AK[j][x] -= c * AK[l][x];
                    }
                }
            }

            /*Brodcast pivoting information*/
            MPI_Bcast(pivoting_info, block_size, kv_pivoting_info_type, k % numprocs, MPI_COMM_WORLD);
        }

        else
        {
            MPI_Bcast(pivoting_info, block_size, kv_pivoting_info_type, k % numprocs, MPI_COMM_WORLD);

            // printf("in process %d, receive pivoting info for columns %d to %d\n", myid, start_column, end_column);

            for (i = 0; i < block_size; i++)
            {
                // printf("in process %d, orig row: %d, current row: %d\n", myid, pivoting_info[i].original_row, pivoting_info[i].current_row);
                // pivoting for other blocks
                if (pivoting_info[i].original_row != pivoting_info[i].current_row)
                {
                    for (j = 0; j < K; ++j)
                    {
                        c = AK[pivoting_info[i].original_row][j];
                        AK[pivoting_info[i].original_row][j] = AK[pivoting_info[i].current_row][j];
                        AK[pivoting_info[i].current_row][j] = c;
                    }
                }
            }
        }
        // printf("myid is %d Current AK is: \n",myid);
        // print_matrix(AK, M, K);

        /*pink part*/
        if (myid == k % numprocs)
        {
            // printf("equal %d\n", myid);
            sourceid = myid;
            // printf("%d ",ib);
            for (j = ib; j < ib + block_size; j++)
            {
                // int x = ((j-ib)<b) ? (j-ib):b;

                for (i = 0; i < j - ib; i++)
                {
                    AW[j][i] = AK[j][i + start_column];
                }
            }

            MPI_Bcast(&AW[ib][0], 1, lower_tri, k % numprocs, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Bcast(&AW[ib][0], 1, lower_tri, k % numprocs, MPI_COMM_WORLD);
            // printf("Receive AW:\n");
            // print_matrix(AW,N,block_size);
        }

        // MPI_Bcast(&AW[ib][0], 1, column_type, i % numprocs, MPI_COMM_WORLD);
        ib += block_size;

        double di[block_size - 1][block_size - 1]; // 声明一个2D数组来存储系数

        for (i = 1; i < block_size; i++)
        {
            for (j = 0; j < block_size - 1; j++)
            {
                if (j < i)
                {
                    di[i - 1][j] = AW[start_row_pink + i][j];
                }
                else
                {
                    di[i - 1][j] = 0;
                }
            }
        }

        // printf("myid: %d, k is %d, start row is %d, start row pink is %d, start column is: %d\n", myid, k, start_row, start_row_pink, start_column);
        // for (i = 1; i < block_size; i++)
        // {
        //     for (j = 0; j < i; j++)
        //     {
        //         printf("%f ", di[i - 1][j]);
        //     }
        //     printf("\n");
        // }

        if (myid <= (k % numprocs))
        {
            for (j = end_column; j < K; j++)
            {

                for (i = 0; i < block_size - 1; i++)
                {
                    for (l = 0; l < block_size - 1; l++)
                    {
                        AK[i + start_row_pink + 1][j] -= di[i][l] * AK[start_row_pink + l][j];
                    }
                    // printf("Before: %f ", AK[start_row_pink + 1][j]);
                    // 更新第 i+1 行的元素

                    // printf("After: %f \n", AK[start_row_pink + 1][j]);

                    // // 更新第 i+2 行的元素
                    // AK[start_row_pink + 2][j] -= di[1][0] * AK[start_row_pink][j] + di[1][1] * AK[start_row_pink + 1][j];

                    // // 更新第 i+3 行的元素
                    // AK[start_row_pink + 3][j] -= di[2][0] * AK[start_row_pink][j] + di[2][1] * AK[start_row_pink + 1][j] + di[2][2] * AK[start_row_pink + 2][j];
                }
            }
        }
        else
        {
            for (j = start_column; j < K; j++)
            {
                // for(i=0;i<block_size;i++){
                for (i = 0; i < block_size - 1; i++)
                {
                    for (l = 0; l < block_size - 1; l++)
                    {
                        AK[i + start_row_pink + 1][j] -= di[i][l] * AK[start_row_pink + l][j];
                    }
                }
            }
        }
        // printf("AK:");
        // print_matrix(AK, M, K);

        /*Green part*/
        MPI_Datatype column_type_green;
        MPI_Type_vector(M - (k + 1) * block_size, block_size, block_size, MPI_DOUBLE, &column_type_green);
        MPI_Type_commit(&column_type_green);

        if (myid == k % numprocs && k != Nb - 1)
        {
            sourceid = myid;

            for (j = ib_green; j < M; j++)
            {

                for (i = 0; i < block_size; i++)
                {
                    // printf("j is %d ,AK is %f\n",j,AK[j][i + start_column]);
                    AW[j][i] = AK[j][i + start_column];
                    // printf("%f \n", AW[j][i]);
                }
                // printf("\n");
            }
            // printf("\n");

            // MPI_Bcast(&AW[ib][0], 1, column_type, sourceid, MPI_COMM_WORLD);
        }
        // else
        // {
        //     MPI_Bcast(&AW[ib][0], 1, column_type, sourceid, MPI_COMM_WORLD);
        // }
        // printf("soucheid: %d , myid: %d , ib: %d\n", sourceid, myid, ib);
        MPI_Bcast(&AW[ib_green][0], 1, column_type_green, k % numprocs, MPI_COMM_WORLD);
        ib_green += block_size;
        // print_matrix(AW, M, block_size);
        if (myid <= (k % numprocs))
        {

            for (j = end_row; j < M; j++)
            {
                for (i = end_column; i < K; i++)
                {
                    for (l = 0; l < block_size; l++)
                    {
                        AK[j][i] -= AW[j][l] * AK[start_row + l][i];
                    }

                    // AK[j][i] -= AW[j][0] * AK[end_row - 4][i] + AW[j][1] * AK[end_row - 3][i] + AW[j][2] * AK[end_row - 2][i] + AW[j][3] * AK[end_row - 1][i];
                }
            }
        }
        else
        {
            for (j = end_row; j < M; j++)
            {
                for (i = start_column; i < K; i++)
                {

                    for (l = 0; l < block_size; l++)
                    {
                        AK[j][i] -= AW[j][l] * AK[start_row + l][i];
                    }
                }
            }
        }
    }

    if (myid == 0)
    {
        ib = 0;
        ib_0 = 0;
        /*Column block cyclic partitioning*/
        for (j = 0; j < q; j++)
        {
            // printf("ib: %d K: %d ", ib, K);
            // copy the first block to the 0 machine
            for (i = 0; i < M; i++)
            {
                for (k = ib, l = ib_0; k < ib + block_size; k++, l++)
                {
                    A[i][k] = AK[i][l];
                }
            }
            // send other blocks to other machines
            for (i = 1; i < numprocs; i++)
            {
                ib += block_size;
                MPI_Recv(&A[0][ib], 1, column_type, i, 1, MPI_COMM_WORLD, &status);
            }

            // leave one row block for process 0
            // for (i = 0; i < M; i++)
            // {
            //     for (j = ib; j < ib + block_size; j++)
            //     {
            //         AK[i][j] = A[i][j];
            //     }
            // }
            ib += block_size;
            ib_0 += block_size;
        }

        // send remaining blocks, one block to each processe with myid < r
        if (r > 0)
        {
            for (i = 0; i < M; i++)
            {
                // printf("ib: %d ", ib);
                for (j = ib, l = ib_0; j < ib + block_size; j++, l++)
                {
                    A[i][j] = AK[i][l];
                }
            }
            for (i = 1; i < r; i++)
            {
                ib += block_size;
                MPI_Recv(&A[0][ib], 1, column_type, i, 1, MPI_COMM_WORLD, &status);
            }
        }
    }

    else
    { // all other processes receive a submatrix from process 0

        ib = 0;
        for (i = 0; i < kb; i++)
        {
            MPI_Send(&AK[0][ib], 1, column_type_local, 0, 1, MPI_COMM_WORLD);
            ib += block_size;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0)
    {
        gettimeofday(&end_time_orig, 0);
        seconds = end_time_orig.tv_sec - start_time_orig.tv_sec;
        microseconds = end_time_orig.tv_usec - start_time_orig.tv_usec;

        elapsed = seconds + 1e-6 * microseconds;
        printf("MPI calculation time: %f\n\n", elapsed);
        printf("Final Matrix %d \n", myid);
        // print_matrix(A, M, N);

        if (test(A, a, N) == 0)
        {
            printf("Correct!");
        }
        else
        {
            printf("Incorrect!");
        }
    }

    MPI_Finalize();

    return 0;
}

void print_matrix(double **T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%6.3f ", T[i][j]);
            if ((j + 1) % 4 == 0)
                printf("  ");
        }
        printf("\n");
        if ((i + 1) % 4 == 0)
            printf("\n");
    }
    printf("\n\n");
    return;
}

int test(double **t1, double **t2, int rows)
{
    int i, j;
    int cnt;
    cnt = 0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            if ((t1[i][j] - t2[i][j]) * (t1[i][j] - t2[i][j]) > 1.0e-16)
            {
                cnt += 1;
            }
        }
    }

    return cnt;
}

/*
 * Function: orig_gaussian_elimination
 * Description: Gaussian elimination function without any optimization.
 * Input: **a - the input matrix.
 *        n - the size of matrix a.
 * Output: None
 * Returns: None
 */
void orig_gaussian_elimination(double **a, int n)
{
    int i, j, k;
    int indk;
    double c, amax, c1, c2, c3;
    for (i = 0; i < n - 1; i++)
    {
        // find and record k where |a(k,i)|=𝑚ax|a(j,i)|
        amax = a[i][i];
        indk = i;
        for (k = i + 1; k < n; k++)
        {
            if (fabs(a[k][i]) > fabs(amax))
            {
                amax = a[k][i];
                indk = k;
            }
        }

        // exit with a warning that a is singular
        if (amax == 0)
        {
            printf("matrix is singular!\n");
            exit(1);
        }
        else if (indk != i) // swap row i and row k
        {
            // printf("indk: %d, i: %d \n", indk, i);
            for (j = 0; j < n; j++)
            {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        }

        // store multiplier in place of A(k,i)
        for (k = i + 1; k < n; k++)
        {
            a[k][i] = a[k][i] / a[i][i];
        }

        // subtract multiple of row a(i,:) to zero out a(j,i)
        for (k = i + 1; k < n; k++)
        {
            c = a[k][i];
            for (j = i + 1; j < n; j++)
            {
                a[k][j] -= c * a[i][j];
            }
        }
    }
}
