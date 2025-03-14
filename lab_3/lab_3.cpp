// #include "vec_fun.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <mpi.h>

MPI_Comm GridComm;
MPI_Comm ColComm;
MPI_Comm RowComm;
MPI_Datatype col, coltype;
int GridCoords[2];
int size = 0;
int rank = 0;
int p1 = 2;
int p2 = 2;
int n1 = 10;
int n2 = 10;
int n3 = 10;

double rand_double() { return (double)rand() / RAND_MAX * 50.0 - 2.0; }

void matrix_mul(double *A, double *B, double *C, int n1, int n2, int n3) {
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n3; j++)
      for (int k = 0; k < n2; k++)
        C[i * n3 + j] += A[i * n2 + k] * B[k * n3 + j];
  }
}

void set_to_zero(double *A, int row_count, int col_count) {
  for (int i = 0; i < row_count; i++) {
    for (int j = 0; j < col_count; j++) {
      A[i * col_count + j] = 0;
    }
  }
}

void data_initialization(double *A, int row_count, int col_count) {
  for (int i = 0; i < row_count; i++) {
    for (int j = 0; j < col_count; j++) {
      A[i * col_count + j] = rand_double();
    }
  }
}

void print_matrix(double *A, int row_count, int col_count) {
  for (int i = 0; i < row_count; i++) {
    for (int j = 0; j < col_count; j++)
      printf("%7.4f ", A[i * col_count + j]);
    printf("\n");
  }
}

void initialize_process(double *&A, double *&B, double *&C, double *&A_subm,
                        double *&B_subm, double *&C_subm, int ABlockSize,
                        int BBlockSize) {
  A_subm = new double[n2 * ABlockSize];
  B_subm = new double[n2 * BBlockSize];
  C_subm = new double[ABlockSize * BBlockSize];
  if (rank == 0) {
    A = new double[n1 * n2];
    B = new double[n2 * n3];
    C = new double[n1 * n3];
    data_initialization(A, n1, n2);
    data_initialization(B, n2, n3);
    set_to_zero(C, n1, n3);
  }
  set_to_zero(C_subm, ABlockSize, BBlockSize);
}

void create_grid() {
  int dim[2];
  int period[2] = {0, 0};
  int arr[2];

  dim[0] = p1;
  dim[1] = p2;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, 0, &GridComm);
  MPI_Cart_coords(GridComm, rank, 2, GridCoords);
  arr[0] = 0;
  arr[1] = 1;
  MPI_Cart_sub(GridComm, arr, &RowComm);

  arr[0] = 1;
  arr[1] = 0;
  MPI_Cart_sub(GridComm, arr, &ColComm);
}

void print_vector(double *pVector, int Size, int ProcNum) {
  for (int i = 0; i < Size; i++)
    std::cout << pVector[i] << std::endl;
  printf("\n");
}

void print_vector(int *pVector, int Size, int ProcNum) {
  for (int i = 0; i < Size; i++)
    std::cout << pVector[i] << std::endl;
  printf("\n");
}

void distrib(double *A, double *B, double *A_subm, double *B_subm,
             int ABlockSize, int BBlockSize) {
  if (GridCoords[1] == 0) {
    MPI_Scatter(A, ABlockSize * n2, MPI_DOUBLE, A_subm, ABlockSize * n2,
                MPI_DOUBLE, 0, ColComm);
  }

  MPI_Bcast(A_subm, ABlockSize * n2, MPI_DOUBLE, 0, RowComm);

  if (GridCoords[0] == 0) {
    MPI_Scatter(B, 1, coltype, B_subm, n2 * BBlockSize, MPI_DOUBLE, 0, RowComm);
  }
  MPI_Bcast(B_subm, BBlockSize * n2, MPI_DOUBLE, 0, ColComm);
}

int main(int argc, char *argv[]) {
  int ABlockSize = n1 / p1;
  int BBlockSize = n3 / p2;

  auto start = std::chrono::high_resolution_clock::now();
  double *A = NULL;
  double *B = NULL;
  double *C = NULL;

  double *A_subm = NULL;
  double *B_subm = NULL;
  double *C_subm = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  create_grid();

  initialize_process(A, B, C, A_subm, B_subm, C_subm, ABlockSize, BBlockSize);

  /*if (rank == 0) {
    printf("Initial matrix A \n");
    print_matrix(A, n1, n2);
    printf("Initial matrix B \n");
    print_matrix(B, n2, n3);
  }*/
  if (rank == 0) {
    MPI_Type_vector(n2, BBlockSize, n3, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, BBlockSize * sizeof(double), &coltype);
    MPI_Type_commit(&coltype);
  }
  distrib(A, B, A_subm, B_subm, ABlockSize, BBlockSize);
  matrix_mul(A_subm, B_subm, C_subm, ABlockSize, n2, BBlockSize);

  MPI_Datatype block, blocktype;
  MPI_Type_vector(ABlockSize, BBlockSize, n3, MPI_DOUBLE, &block);
  MPI_Type_commit(&block);

  MPI_Type_create_resized(block, 0, BBlockSize * sizeof(double), &blocktype);
  MPI_Type_commit(&blocktype);

  int *displ = new int[p1 * p2];
  int *recvcount = new int[p1 * p2];
  int BlockCount = 0;
  int BlockSize = ABlockSize * BBlockSize;
  int NumCount = 0;
  int Written;
  int j = 0;
  while (NumCount < p1 * p2 * BlockSize) {
    Written = 0;
    for (int i = 0; i < n3; i += BBlockSize) {
      displ[j] = BlockCount;
      recvcount[j] = 1;
      j++;
      BlockCount++;

      Written++;
    }
    NumCount += Written * BlockSize;
    BlockCount += Written * (ABlockSize - 1);
  }

  MPI_Gatherv(C_subm, BlockSize, MPI_DOUBLE, C, recvcount, displ, blocktype, 0,
              MPI_COMM_WORLD);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  if (rank == 0) {
    printf("matrix C \n");

    print_matrix(C, n1, n3);
    std::cout << elapsed.count() << "Секунд" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
