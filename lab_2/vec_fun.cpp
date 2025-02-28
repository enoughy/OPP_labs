#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#define MATRIX_SIZE 2

int rank, size;
std::vector<double> mult_matrix_vec(const std::vector<double> &local_A,
                                    const std::vector<double> &x, int rank,
                                    int size) {
  int rows_per_proc = x.size() / size;
  std::vector<double> local_result(rows_per_proc, 0.0);
  int N = x.size();

  for (int i = 0; i < rows_per_proc; i++) {
    for (int j = 0; j < x.size(); j++) {
      local_result[i] += local_A[(i * N) + j] * x[j];
    }
  }
  std::vector<double> result(x.size(), 0.0);
  MPI_Allgather(local_result.data(), local_result.size(), MPI_DOUBLE,
                result.data(), local_result.size(), MPI_DOUBLE, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return result;
}

/*double scalar_mult(const std::vector<double> &x, const std::vector<double> &y)
{ if (x.size() != y.size()) { throw std::invalid_argument("Размеры векторов не
совпадают Scalar mult" + std::to_string(x.size()) + " " +
                                std::to_string(y.size()));
  }

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = x.size();
  int local_n = n / size;
  int remainder = n % size;

  int start = rank * local_n + std::min(rank, remainder);
  int end = start + local_n + (rank < remainder ? 1 : 0);

  double local_sum = 0.0;
  for (int i = start; i < end; i++) {
    local_sum += x[i] * y[i];
  }

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  return global_sum;
}*/

double scalar_mult(const std::vector<double> &x, const std::vector<double> &y) {
  double sum = 0.0;
  for (int i = 0; i < x.size(); i++) {
    sum += x[i] * y[i];
  }
  return sum;
}
std::vector<double> subtractVectors(const std::vector<double> &a,
                                    const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Векторы должны быть одинаковой длины " +
                                std::to_string(a.size()) + " " +
                                std::to_string(b.size()));
  }

  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

void print_vector(std::vector<double> &x);

std::vector<double> y_n_calculate(std::vector<double> &A,
                                  std::vector<double> &x,
                                  std::vector<double> &b) {

  std::vector<double> temp = mult_matrix_vec(A, x, rank, size);
  std::vector<double> y_n = subtractVectors(temp, b);
  return y_n;
}

double t_calculate(std::vector<double> &A, std::vector<double> &y) {
  std::vector<double> Ay_n = mult_matrix_vec(A, y, rank, size);
  // if (rank == 0)
  // print_vector(Ay_n);

  double a_1 = scalar_mult(y, Ay_n);
  double a_2 = scalar_mult(Ay_n, Ay_n);
  // if (rank == 0)
  // std::cout << "a_1 " << a_1 << " " << "a_2 " << a_2 << std::endl;

  return a_1 / a_2;
}

double vectorNorm(const std::vector<double> &x) {
  double a = scalar_mult(x, x);
  return std::sqrt(a);
}

std::vector<double> x_n_calculate(const std::vector<double> &x, const double t,
                                  const std::vector<double> &y) {
  std::vector<double> yt(y.size());
  for (int i = 0; i < y.size(); i++) {
    yt[i] = y[i] * t;
  }
  return subtractVectors(x, yt);
}

double e_calculate(std::vector<double> &A, const std::vector<double> &x,
                   std::vector<double> &b) {
  std::vector<double> Ax = mult_matrix_vec(A, x, rank, size);
  // if (rank == 0)
  //  print_vector(Ax);
  std::vector<double> temp = subtractVectors(Ax, b);
  // if (rank == 0)
  // print_vector(temp);
  double a_1 = vectorNorm(subtractVectors(Ax, b));
  double a_2 = vectorNorm(b);
  // if (rank == 0)
  // std::cout << "a_1 " << a_1 << " a_2 " << a_2 << std::endl;

  return a_1 / a_2;
}

std::vector<double> generateRandomVector(size_t N, double min = 0.0,
                                         double max = 1.0) {
  std::vector<double> v(N);
  std::srand(std::time(nullptr));

  for (size_t i = 0; i < N; i++) {
    v[i] = min + (max - min) * (static_cast<double>(std::rand()) / RAND_MAX);
  }

  return v;
}

void print_vector(std::vector<double> &x) {
  for (int i = 0; i < x.size(); i++) {
    std::cout << x[i] << " ";
    if (i == x.size() - 1) {
      std::cout << std::endl;
    }
  }
}
std::vector<double> read_vector_from_file(const std::string &filename) {
  std::ifstream in_file(filename);
  if (!in_file) {
    std::cerr << "Ошибка при открытии файла для чтения!" << std::endl;
    return {};
  }

  std::vector<double> vec;
  double num;

  while (in_file >> num) {
    vec.push_back(num);
  }

  in_file.close();
  return vec;
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Узнаем номер процесса
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Узнаем количество процессов
  int N = 8000;                         // 40000
  int rows_per_proc = N / size;         // Число строк для каждого процесса
  std::vector<double> local_A(rows_per_proc * N);
  std::vector<double> A(N * N, 1.0);
  if (rank == 0) {

    for (size_t i = 0; i < N; i++) {
      A[i * N + i] = 2.0;
    }
  }
  MPI_Scatter(A.data(), rows_per_proc * N, MPI_DOUBLE, local_A.data(),
              rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> u(N);
  std::vector<double> b;
  if (rank == 0) {
    u = read_vector_from_file("vector.txt");
  }
  MPI_Bcast(u.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  b = mult_matrix_vec(local_A, u, rank, size);
  // if (rank == 0)
  // print_vector(b);
  std::vector<double> x(N, 0.0);

  double e = pow(0.1, 12);
  std::vector<double> x_n = x;
  auto start = std::chrono::high_resolution_clock::now();
  double e_n = e_calculate(local_A, x, b);
  // if (rank == 0)
  //  std::cout << e_n << std::endl;
  int i = 0;
  int count = 0;
  while (e_n >= e) {
    std::vector<double> y_n = y_n_calculate(local_A, x_n, b);
    // if (rank == 0) {
    //  std::cout << "y_n ";
    //  print_vector(y_n);
    // }
    double t_n = t_calculate(local_A, y_n);
    // if (rank == 0)
    //   std::cout << "t " << std::setprecision(10) << t_n << std::endl;

    x_n = x_n_calculate(x_n, t_n, y_n);
    // if (rank == 0) {
    //  std::cout << "x_n ";
    //  print_vector(x_n);
    //}
    e_n = e_calculate(local_A, x_n, b);
    if (rank == 0)
      std::cout << "e_n" << e_n << std::endl;
    i++;
  }

  std::cout << "i " << i << std::endl;
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  if (rank == 0) {
    std::cout << "Время выполнения: " << elapsed.count() << "секунд"
              << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  /*if (rank == 0) {
    print_vector(u);
    print_vector(x_n);
  }*/
  MPI_Finalize();
}
