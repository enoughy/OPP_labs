// #include "vec_fun.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <ostream>
#include <random>

MPI_Comm GridComm;
MPI_Comm ColComm;
MPI_Comm RowComm;
MPI_Datatype col, coltype;
MPI_Datatype A_rows;
MPI_Datatype block, blocktype;
int GridCoords[2];
int size = 0;
int rank = 0;
int p1;
int p2;
int n1 = 16 * 180;
int n2 = 16 * 180;
int n3 = 16 * 180;

void is_matrix_eq(double *A, double *B, int n1, int n2) {
  for (int i = 0; i < n1 * n2; i++) {
    if (A[i] != B[i]) {
      printf("not eq");
      return;
    }
  }
  printf("eq");
}
/*void matrix_mul(double *A, double *B, double *C, int n1, int n2, int n3) {
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n3; j++)
      for (int k = 0; k < n2; k++)
        C[i * n3 + j] += A[i * n2 + k] * B[k * n3 + j];
  }
}*/
void matrix_mul(const double *A, const double *B, double *C, int M, int K,
                int N) {
  for (int i = 0; i < M; ++i) {
    double *c = C + i * N;
    for (int k = 0; k < K; ++k) {
      const double *b = B + k * N;
      double a = A[i * K + k];
      for (int j = 0; j < N; ++j)
        c[j] += a * b[j];
    }
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
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100, 100);

  for (int i = 0; i < row_count; i++) {
    for (int j = 0; j < col_count; j++) {
      A[i * col_count + j] = dist(gen);
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
                        double *&B_subm, double *&C_subm, int A_block_size,
                        int B_block_size) {
  A_subm = new double[n2 * A_block_size];
  B_subm = new double[n2 * B_block_size];
  C_subm = new double[A_block_size * B_block_size];
  if (rank == 0) {
    A = new double[n1 * n2];
    B = new double[n2 * n3];
    C = new double[n1 * n3];
    data_initialization(A, n1, n2);
    data_initialization(B, n2, n3);
    set_to_zero(C, n1, n3);
  }
  set_to_zero(C_subm, A_block_size, B_block_size);
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
             int A_block_size, int B_block_size) {
  if (GridCoords[1] == 0) {
    MPI_Scatter(A, 1, A_rows, A_subm, A_block_size * n2, MPI_DOUBLE, 0,
                ColComm);
  }

  MPI_Bcast(A_subm, A_block_size * n2, MPI_DOUBLE, 0, RowComm);

  if (GridCoords[0] == 0) {
    MPI_Scatter(B, 1, coltype, B_subm, B_block_size * n2, MPI_DOUBLE, 0,
                RowComm);
  }
  MPI_Bcast(B_subm, B_block_size * n2, MPI_DOUBLE, 0, ColComm);
}

void gather_matrix(int A_block_size, int B_block_size, double *C_subm,
                   double *C) {
  int *displ = new int[p1 * p2];
  int *recvcount = new int[p1 * p2];
  int block_count = 0;
  int block_size =
      A_block_size * B_block_size; // размер подматрицы каждого процесса
  int num_count = 0;
  int written;
  int j = 0;
  while (num_count < p1 * p2) {
    written = 0;
    for (int i = 0; i < p2; i += 1) {
      displ[j] = block_count;
      recvcount[j] = 1;

      j++;
      block_count++;

      written++;
    }
    num_count += written;
    block_count += written * (A_block_size - 1);
  }

  MPI_Gatherv(C_subm, block_size, MPI_DOUBLE, C, recvcount, displ, blocktype, 0,
              MPI_COMM_WORLD);
}

void mpi_type_create(int A_block_size, int B_block_size) {
  if (rank == 0) {
    MPI_Type_vector(n2, B_block_size, n3, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, B_block_size * sizeof(double), &coltype);
    MPI_Type_commit(&coltype);

    MPI_Type_vector(A_block_size, n2, n2, MPI_DOUBLE, &A_rows);
    MPI_Type_commit(&A_rows);
  }

  MPI_Type_vector(A_block_size, B_block_size, n3, MPI_DOUBLE, &block);
  MPI_Type_commit(&block);

  MPI_Type_create_resized(block, 0, B_block_size * sizeof(double), &blocktype);
  MPI_Type_commit(&blocktype);
}

int main(int argc, char *argv[]) {

  double *A = NULL;
  double *B = NULL;
  double *C = NULL;

  double *A_subm = NULL;
  double *B_subm = NULL;
  double *C_subm = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  p1 = std::atoi(argv[1]);
  p2 = std::atoi(argv[2]);

  int A_block_size = n1 / p1;
  int B_block_size = n3 / p2;

  mpi_type_create(A_block_size, B_block_size);

  initialize_process(A, B, C, A_subm, B_subm, C_subm, A_block_size,
                     B_block_size);
  double *C_1 = NULL;
  if (rank == 0) {
    C_1 = new double[n1 * n3];
    set_to_zero(C_1, n1, n3);
    matrix_mul(A, B, C_1, n1, n2, n3);
  }

  create_grid();
  distrib(A, B, A_subm, B_subm, A_block_size, B_block_size);

  auto start = std::chrono::high_resolution_clock::now();
  matrix_mul(A_subm, B_subm, C_subm, A_block_size, n2, B_block_size);

  auto end = std::chrono::high_resolution_clock::now();

  gather_matrix(A_block_size, B_block_size, C_subm, C);

  std::chrono::duration<double> elapsed = end - start;

  if (rank == 0) {
    // printf("matrix C \n");
    //
    // print_matrix(C, n1, n3);
    //
    // printf("matrix A \n");
    // print_matrix(A, n1, n2);
    // printf("matrix B \n");
    // print_matrix(B, n2, n3);
    // is_matrix_eq(C, C_1, n2, n3);
    std::cout << elapsed.count() << std::endl;
  }

  MPI_Finalize();
  return 0;
}
