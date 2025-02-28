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

int N = 8;
int rank, size;
void print_vector(std::vector<double> &x);
std::vector<double> mult_matrix_vec(const std::vector<double> &local_A,
                                    std::vector<double> local_x, int rank,
                                    int size) {
  int rows_per_proc = N / size;
  std::vector<double> local_result(rows_per_proc, 0.0);

  MPI_Status status;

  for (int step = 0; step < size; step++) {
    int receiver = ((unsigned int)(rank - 1)) % size;
    int sender = (rank + 1) % size;
    // if (rank == 2)
    //  std::cout << "local a " << local_A[0] << "step " << step << std::endl;

    // Умножение локальной части матрицы на текущую версию вектора
    for (int i = 0; i < rows_per_proc; i++) {
      for (int j = 0; j < rows_per_proc; j++) {
        local_result[i] +=
            local_A[i * N + ((j + (step + rank) * rows_per_proc) % N)] *
            local_x[j];
        /*std::cout << "i " << i << " "
                  << local_A[i * N + ((j + (step + rank) * rows_per_proc) % N)]
                  << "*" << local_x[j] << " Step " << step << " rank " << rank
                  << " res = " << local_result[i] << " "
                  << i * N + ((j + (step + rank) * rows_per_proc) % N)
                  << std::endl;*/
      }
    }
    // Передача вектора по кольцу
    // if (rank == 2)
    // std::cout << "!!!local a " << local_A[0] << "step " << step << std::endl;
    MPI_Sendrecv_replace(local_x.data(), rows_per_proc, MPI_DOUBLE, receiver, 0,
                         sender, 0, MPI_COMM_WORLD, &status);
  }
  // print_vector(local_result);

  // Собираем результат от всех процессов
  std::vector<double> result(N, 0.0);
  MPI_Allgather(local_result.data(), rows_per_proc, MPI_DOUBLE, result.data(),
                rows_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return result;
}

double scalar_mult(const std::vector<double> &x, const std::vector<double> &y) {
  double sum = 0.0;
  for (int i = 0; i < x.size(); i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

double scalar_mult_local(const std::vector<double> &local_x,
                         const std::vector<double> &local_y, int rank,
                         int size) {

  double local_sum = 0.0;

  for (int i = 0; i < local_x.size(); i++) {
    local_sum += local_x[i] * local_y[i];
  }

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  return global_sum;
}
/*std::vector<double> subtractVectors(const std::vector<double> &a,
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
}*/

std::vector<double> subtractVectors(const std::vector<double> &local_a,
                                    const std::vector<double> &local_b) {

  std::vector<double> local_result(local_a.size());

  for (size_t i = 0; i < local_a.size(); i++) {
    local_result[i] = local_a[i] - local_b[i];
  }
  std::vector<double> result(N, 0.0);
  MPI_Allgather(local_result.data(), local_result.size(), MPI_DOUBLE,
                result.data(), local_result.size(), MPI_DOUBLE, MPI_COMM_WORLD);
  return result;
}

void print_vector(std::vector<double> &x);

std::vector<double> y_n_calculate(std::vector<double> &A,
                                  std::vector<double> &x, std::vector<double> b,
                                  int rows_per_proc) {

  std::vector<double> temp = mult_matrix_vec(A, x, rank, size);
  // print_vector(temp);
  std::vector<double> local_temp(rows_per_proc);
  MPI_Scatter(temp.data(), rows_per_proc, MPI_DOUBLE, local_temp.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> y_n = subtractVectors(local_temp, b);
  return y_n;
}

double t_calculate(std::vector<double> &A, std::vector<double> &y,
                   int rows_per_proc) {
  std::vector<double> local_y(rows_per_proc);
  MPI_Scatter(y.data(), rows_per_proc, MPI_DOUBLE, local_y.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> Ay_n = mult_matrix_vec(A, local_y, rank, size);
  // if (rank == 0) {
  //  print_vector(Ay_n);
  //}
  std::vector<double> local_Ay_n(rows_per_proc);
  MPI_Scatter(Ay_n.data(), rows_per_proc, MPI_DOUBLE, local_Ay_n.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double a_1 = scalar_mult(y, Ay_n);
  double a_2 = scalar_mult(Ay_n, Ay_n);
  // if (rank == 0)
  //  std::cout << "a_1 " << a_1 << "a_2 " << a_2 << std::endl;
  return a_1 / a_2;
}

double vectorNorm_local(const std::vector<double> &x) {
  double a = scalar_mult_local(x, x, rank, size);
  return std::sqrt(a);
}

double vectorNorm(const std::vector<double> &x) {
  double a = scalar_mult(x, x);
  return std::sqrt(a);
}
std::vector<double> x_n_calculate(const std::vector<double> &x, const double t,
                                  const std::vector<double> &y,
                                  int rows_per_proc) {
  std::vector<double> yt(y.size());
  for (int i = 0; i < y.size(); i++) {
    yt[i] = y[i] * t;
  }
  std::vector<double> local_yt(rows_per_proc);
  MPI_Scatter(yt.data(), rows_per_proc, MPI_DOUBLE, local_yt.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> temp = subtractVectors(x, local_yt);
  std::vector<double> local_temp(rows_per_proc);
  MPI_Scatter(temp.data(), rows_per_proc, MPI_DOUBLE, local_temp.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return local_temp;
}

double e_calculate(std::vector<double> &A, std::vector<double> &x,
                   std::vector<double> &b, int rows_per_proc) {

  std::vector<double> Ax = mult_matrix_vec(A, x, rank, size);
  // if (rank == 0)
  //  print_vector(Ax);
  std::vector<double> local_Ax(rows_per_proc);

  // std::cout << rows_per_proc << " " << Ax.size() << std::endl;
  MPI_Scatter(Ax.data(), rows_per_proc, MPI_DOUBLE, local_Ax.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> temp = subtractVectors(local_Ax, b);
  /*if (rank == 0)
    print_vector(temp);*/
  double a_1 = vectorNorm(temp);
  // std::cout << "temp";
  // print_vector(temp);
  double a_2 = vectorNorm_local(b);
  // if (rank == 0)
  //  std::cout << "a_1 " << a_1 << " a_2 " << a_2 << std::endl;
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
  int rows_per_proc = N / size;         // Число строк для каждого процесса
  std::vector<double> local_A(rows_per_proc * N);
  std::vector<double> A(N * N, 1.0);
  if (rank == 0) {

    for (int i = 0; i < N; i++) {
      A[i * N + i] = 2.0;
    }
  }
  MPI_Scatter(A.data(), rows_per_proc * N, MPI_DOUBLE, local_A.data(),
              rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // print_vector(local_A);

  std::vector<double> u(N);
  std::vector<double> b;
  if (rank == 0) {
    u = read_vector_from_file("vector.txt");
  }
  // MPI_Bcast(u.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> local_u(rows_per_proc);
  MPI_Scatter(u.data(), rows_per_proc, MPI_DOUBLE, local_u.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // std::cout << rank << std::endl;
  // print_vector(local_u);
  // print_vector(local_A);

  b = mult_matrix_vec(local_A, local_u, rank, size);

  // print_vector(b);
  std::vector<double> local_b(rows_per_proc);
  MPI_Scatter(b.data(), rows_per_proc, MPI_DOUBLE, local_b.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(N, 0.0);
  std::vector<double> local_x(rows_per_proc);
  MPI_Scatter(x.data(), rows_per_proc, MPI_DOUBLE, local_x.data(),
              rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Рассылаем вектор x каждому процессу
  // std::vector<double> local_x(N);
  // MPI_Bcast(local_x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double e = pow(0.1, 12);
  std::vector<double> x_n = local_x;
  auto start = std::chrono::high_resolution_clock::now();
  double e_n = e_calculate(local_A, local_x, local_b, rows_per_proc);
  int count = 0;
  // if (rank == 0) {
  //  std::cout << "e_n" << e_n << std::endl;
  //}
  while (e_n >= e) {
    std::vector<double> y_n =
        y_n_calculate(local_A, x_n, local_b, rows_per_proc);
    // if (rank == 0) {
    // std::cout << "y_n ";
    // print_vector(y_n);
    //}
    double t_n = t_calculate(local_A, y_n, rows_per_proc);
    // if (rank == 0)
    //  std::cout << "t_n " << std::setprecision(10) << t_n << std::endl;

    x_n = x_n_calculate(x_n, t_n, y_n, rows_per_proc);
    // std::cout << "x_n " << rank << " ";
    //  print_vector(x_n);
    e_n = e_calculate(local_A, x_n, local_b, rows_per_proc);
    if (rank == 0)
      std::cout << "e_n " << e_n << std::endl;
    //  if (i > 4) {
    //   break;
    // }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Время выполнения: " << elapsed.count() << "секунд" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    print_vector(u);
  }
  print_vector(x_n);
  MPI_Finalize();
}
