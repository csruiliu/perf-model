#include <algorithm>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <mutex>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// Thread-safe Timer class using RAII
class Timer {
  public:
    Timer(const std::string &name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {
    }

    ~Timer() {
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);

        std::lock_guard<std::mutex> lock(get_mutex());
        std::vector<std::pair<std::string, double>> &timings = get_timings();

        double current_time = static_cast<double>(duration.count());
        // Check if this timing already exists and update it, or add new entry
        bool found = false;
        for (std::vector<std::pair<std::string, double>>::iterator it = timings.begin(); it != timings.end(); ++it) {
            if (it->first == name_) {
                // Keep the larger timing value
                it->second = std::max(it->second, current_time);
                found = true;
                break;
            }
        }

        if (!found) {
            timings.push_back(std::make_pair(name_, duration.count()));
        }
    }

    static void print_results() {
        std::lock_guard<std::mutex> lock(get_mutex());
        std::cout << "\n=== Timing Results ===" << std::endl;

        double total = 0;
        const std::vector<std::pair<std::string, double>> &timings = get_timings();

        // Calculate total time
        for (std::vector<std::pair<std::string, double>>::const_iterator it = timings.begin(); it != timings.end();
             ++it) {
            if (it->first == "Total Execution") {
                total = it->second;
                break;
            }
        }

        // If no "Total Execution" found, sum all timings
        if (total == 0) {
            std::cout << "No Total Execution, Sum all Timings" << std::endl;
            for (std::vector<std::pair<std::string, double>>::const_iterator it = timings.begin(); it != timings.end();
                 ++it) {
                total += it->second;
            }
        }

        // Print header
        std::cout << std::left << std::setw(30) << "Operation" << std::right << std::setw(8) << "Time" << std::setw(4)
                  << "" << std::setw(10) << "Percentage" << std::endl;
        std::cout << std::string(52, '-') << std::endl;

        // Print individual timings in insertion order
        for (std::vector<std::pair<std::string, double>>::const_iterator it = timings.begin(); it != timings.end();
             ++it) {
            std::string operation = it->first + ":";

            // Check if operation name is too long (more than 29 characters)
            if (operation.length() > 29) {
                // Generic break: try to break at space near the middle
                size_t break_pos = operation.find(' ', operation.length() / 2);
                std::string line1, line2;

                if (break_pos != std::string::npos && break_pos < 29) {
                    line1 = operation.substr(0, break_pos);
                    line2 = operation.substr(break_pos + 1);
                } else {
                    // If no good break point, just truncate first line
                    line1 = operation.substr(0, 26) + "...";
                    line2 = operation.substr(26);
                }

                // Print first line (no timing info)
                std::cout << std::left << std::setw(30) << line1 << std::endl;

                // Print second line with timing info
                std::cout << std::left << std::setw(30) << line2 << std::right << std::setw(6)
                          << static_cast<long>(it->second) << std::setw(4) << " ms" << std::setw(7) << std::fixed
                          << std::setprecision(1);
            } else {
                // Single line output for shorter names
                std::cout << std::left << std::setw(30) << operation << std::right << std::setw(6)
                          << static_cast<long>(it->second) << std::setw(4) << " ms" << std::setw(7) << std::fixed
                          << std::setprecision(1);
            }

            if (total > 0) {
                std::cout << "(" << (it->second / total * 100.0) << "%)";
            } else {
                std::cout << "(0.0%)";
            }
            std::cout << std::endl;
        }

        std::cout << std::string(52, '-') << std::endl;
        std::cout << std::left << std::setw(30) << "Total:" << std::right << std::setw(6) << static_cast<long>(total)
                  << std::setw(4) << " ms" << std::endl;
    }

    static void clear() {
        std::lock_guard<std::mutex> lock(get_mutex());
        get_timings().clear();
    }

  private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;

    static std::vector<std::pair<std::string, double>> &get_timings() {
        static std::vector<std::pair<std::string, double>> timings;
        return timings;
    }

    static std::mutex &get_mutex() {
        static std::mutex mutex;
        return mutex;
    }
};

// Convenient macro for timing scopes
#define TIME_SCOPE(name) Timer timer_##__LINE__(name)

namespace gemm_mpi {

enum class Precision {
    SINGLE = 'S',
    DOUBLE = 'D',
    HALF = 'H'
};

template <typename T> struct CublasTraits;

template <> struct CublasTraits<float> {
    static constexpr auto gemm_func = &cublasSgemm;
    static constexpr auto math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
    static constexpr auto mpi_type = MPI_FLOAT;
};

template <> struct CublasTraits<double> {
    static constexpr auto gemm_func = &cublasDgemm;
    static constexpr auto math_mode = CUBLAS_DEFAULT_MATH;
    static constexpr auto mpi_type = MPI_DOUBLE;
};

template <> struct CublasTraits<__half> {
    static constexpr auto gemm_func = &cublasHgemm;
    static constexpr auto math_mode = CUBLAS_TENSOR_OP_MATH;
    // MPI doesn't have a native data type for __half, use MPI_UINT16_T
    static constexpr auto mpi_type = MPI_UINT16_T;
};

template <typename T> class GemmMPI {
  private:
    int mpi_rank_;
    int mpi_size_;
    int N_;
    int local_N_;
    int device_id_;

    // Host matrices (rank 0 only)
    T *full_A_host_;
    T *B_host_;
    T *full_C_host_;

    // GPU matrices
    T *full_A_gpu_;
    T *B_gpu_;
    T *full_C_gpu_;
    T *local_A_gpu_;
    T *local_C_gpu_;

    cublasHandle_t cublas_handle_;

  public:
    GemmMPI(int N)
        : N_(N), full_A_host_(nullptr), B_host_(nullptr), full_C_host_(nullptr), full_A_gpu_(nullptr), B_gpu_(nullptr),
          full_C_gpu_(nullptr), local_A_gpu_(nullptr), local_C_gpu_(nullptr), cublas_handle_(nullptr) {

        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        local_N_ = N_ / mpi_size_;

        if (N_ % mpi_size_ != 0) {
            if (mpi_rank_ == 0) {
                std::cerr << "Matrix size N must be divisible by number of MPI processes" << std::endl;
            }
            MPI_Finalize();
            exit(1);
        }

        // Set GPU device based on rank (handle multiple ranks per GPU)
        int num_gpus;
        cudaError_t err = cudaGetDeviceCount(&num_gpus);
        handle_cuda_error(err, "cudaGetDeviceCount");

        if (num_gpus == 0) {
            std::cerr << "No CUDA-capable devices found" << std::endl;
            MPI_Finalize();
            exit(1);
        }

        // Set GPU device based on rank
        device_id_ = mpi_rank_ % num_gpus;
        err = cudaSetDevice(device_id_);
        handle_cuda_error(err, "cudaSetDevice");

        if (mpi_rank_ == 0) {
            std::cout << "Using " << num_gpus << " GPU(s) for " << mpi_size_ << " MPI ranks" << std::endl;
            std::cout << "Local matrix size per rank: " << local_N_ << "x" << N_ << std::endl;
        }
    }

    ~GemmMPI() {
        cleanup();
    }

    bool allocate_host_matrices() {
        TIME_SCOPE("Matrices Allocation and Initialization on Host");
        if (mpi_rank_ == 0) {
            std::cout << "Allocating Matrics on Host [Rank 0]" << std::endl;
            // Use regular malloc for host memory allocation
            full_A_host_ = static_cast<T *>(malloc(sizeof(T) * N_ * N_));
            if (full_A_host_ == nullptr) {
                std::cerr << "Failed to allocate memory for full_A_host_" << std::endl;
                return false;
            }

            B_host_ = static_cast<T *>(malloc(sizeof(T) * N_ * N_));
            if (B_host_ == nullptr) {
                std::cerr << "Failed to allocate memory for B_host_" << std::endl;
                free(full_A_host_);
                return false;
            }

            full_C_host_ = static_cast<T *>(malloc(sizeof(T) * N_ * N_));
            if (full_C_host_ == nullptr) {
                std::cerr << "Failed to allocate memory for full_C_host_" << std::endl;
                free(full_A_host_);
                free(B_host_);
                return false;
            }

            // Initialize matrices with random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

            for (int i = 0; i < N_ * N_; ++i) {
                full_A_host_[i] = static_cast<T>(dis(gen));
                B_host_[i] = static_cast<T>(dis(gen));
                full_C_host_[i] = static_cast<T>(dis(gen));
            }
            cudaDeviceSynchronize();
        }
        MPI_Barrier(MPI_COMM_WORLD); // Wait for rank 0 to finish
        return true;
    }

    bool allocate_gpu_matrices() {
        TIME_SCOPE("Matrices Allocation on GPU");
        if (mpi_rank_ == 0) {
            std::cout << "Allocating Matrics on GPU" << std::endl;
            // Allocate full matrices on rank 0
            cudaError_t err = cudaMalloc(&full_A_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc full_A_gpu");

            err = cudaMalloc(&B_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc B_gpu");

            err = cudaMalloc(&full_C_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc full_C_gpu");
        } else {
            // Non-root ranks only need B matrix
            cudaError_t err = cudaMalloc(&B_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc B_gpu");
        }

        // Allocate local matrices on all ranks
        cudaError_t err = cudaMalloc(&local_A_gpu_, sizeof(T) * local_N_ * N_);
        handle_cuda_error(err, "cudaMalloc local_A_gpu");

        err = cudaMalloc(&local_C_gpu_, sizeof(T) * local_N_ * N_);
        handle_cuda_error(err, "cudaMalloc local_C_gpu");

        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

    bool copy_matrices_host_gpu() {
        TIME_SCOPE("Copy Data from Host to GPU");
        if (mpi_rank_ == 0) {
            std::cout << "Copy Data from Host to GPU [Rank 0]" << std::endl;
            // Copy from host to device
            cudaError_t err = cudaMemcpy(full_A_gpu_, full_A_host_, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
            handle_cuda_error(err, "cudaMemcpy A host to device");

            err = cudaMemcpy(B_gpu_, B_host_, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
            handle_cuda_error(err, "cudaMemcpy B host to device");

            err = cudaMemcpy(full_C_gpu_, full_C_host_, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
            handle_cuda_error(err, "cudaMemcpy C host to device");

            cudaDeviceSynchronize();
        }
        MPI_Barrier(MPI_COMM_WORLD); // Wait for rank 0 to finish
        return true;
    }

    bool distribute_data() {
        TIME_SCOPE("Data Distribution");
        // Handle __half type with custom MPI communication
        MPI_Scatter(full_A_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type, local_A_gpu_, local_N_ * N_,
                    CublasTraits<T>::mpi_type, 0, MPI_COMM_WORLD);

        MPI_Bcast(B_gpu_, N_ * N_, CublasTraits<T>::mpi_type, 0, MPI_COMM_WORLD);

        MPI_Scatter(full_C_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type, local_C_gpu_, local_N_ * N_,
                    CublasTraits<T>::mpi_type, 0, MPI_COMM_WORLD);

        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

    bool compute_gemm(int repeats, T alpha = T(1.0), T beta = T(0.0)) {
        TIME_SCOPE("Compute GEMM");
        T *temp_C = nullptr;
        cudaError_t err = cudaMalloc(&temp_C, sizeof(T) * local_N_ * N_);
        handle_cuda_error(err, "cudaMalloc temp_C");

        cublasStatus_t status = cublasSetMatrix(local_N_, N_, sizeof(T), local_C_gpu_, local_N_, temp_C, local_N_);
        handle_cublas_error(status, "cublasSetMatrix");

        for (int r = 0; r < repeats; ++r) {
            status = CublasTraits<T>::gemm_func(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, local_N_, N_, N_, &alpha,
                                                local_A_gpu_, local_N_, B_gpu_, N_, &beta, temp_C, local_N_);
            handle_cublas_error(status, "gemm computation");
        }

        cudaDeviceSynchronize();

        status = cublasGetMatrix(local_N_, N_, sizeof(T), temp_C, local_N_, local_C_gpu_, local_N_);
        handle_cublas_error(status, "cublasGetMatrix");

        cudaFree(temp_C);
        MPI_Barrier(MPI_COMM_WORLD);

        return true;
    }

    bool gather_results() {
        TIME_SCOPE("Copy data from GPU to Host");
        if (mpi_rank_ == 0) {
            std::cout << "Gather Results: Copy data from GPU to Host" << std::endl;
            MPI_Gather(local_C_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type, full_C_host_, local_N_ * N_,
                       CublasTraits<T>::mpi_type, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gather(local_C_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type, nullptr, 0, CublasTraits<T>::mpi_type, 0,
                       MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        return true;
    }

    bool setup_cublas() {
        cublasStatus_t status = cublasCreate(&cublas_handle_);
        handle_cublas_error(status, "cublasCreate");

        status = cublasSetMathMode(cublas_handle_, CublasTraits<T>::math_mode);
        handle_cublas_error(status, "cublasSetMathMode");

        return true;
    }

  private:
    void cleanup() {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }

        cudaFree(full_A_gpu_);
        cudaFree(B_gpu_);
        cudaFree(full_C_gpu_);
        cudaFree(local_A_gpu_);
        cudaFree(local_C_gpu_);

        if (mpi_rank_ == 0) {
            free(full_A_host_);
            free(B_host_);
            free(full_C_host_);
            // cudaFreeHost(full_A_host_);
            // cudaFreeHost(B_host_);
            // cudaFreeHost(full_C_host_);
        }

        // Reset pointers
        full_A_gpu_ = B_gpu_ = full_C_gpu_ = local_A_gpu_ = local_C_gpu_ = nullptr;
        full_A_host_ = B_host_ = full_C_host_ = nullptr;
    }

    void handle_cuda_error(cudaError_t error, const std::string &operation) {
        if (error != cudaSuccess) {
            std::cerr << "Rank " << mpi_rank_ << ": " << operation << " failed: " << cudaGetErrorString(error)
                      << std::endl;
            MPI_Finalize();
            exit(1);
        }
    }

    void handle_cublas_error(cublasStatus_t status, const std::string &operation) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Rank " << mpi_rank_ << ": " << operation << " failed with cuBLAS error " << status
                      << std::endl;
            MPI_Finalize();
            exit(1);
        }
    }
};

} // namespace gemm_mpi

// Template function to eliminate code duplication
template <typename T> bool run_gemm_mpi(int N, int repeats, T alpha, T beta) {
    gemm_mpi::GemmMPI<T> gemm(N);

    return gemm.allocate_host_matrices() && gemm.allocate_gpu_matrices() && gemm.copy_matrices_host_gpu() &&
           gemm.distribute_data() && gemm.setup_cublas() && gemm.compute_gemm(repeats, alpha, beta) &&
           gemm.gather_results();
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc != 6) {
        if (mpi_rank == 0) {
            std::cout << "Usage: " << argv[0] << " <N> <repeats> <alpha> <beta> <precision(S/D/H)>" << std::endl;
            std::cout << "  N: Matrix size (NxN)" << std::endl;
            std::cout << "  repeats: Number of GEMM iterations" << std::endl;
            std::cout << "  alpha, beta: GEMM coefficients" << std::endl;
            std::cout << "  precision: S(ingle), D(ouble), or H(alf)" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);
    int repeats = std::atoi(argv[2]);
    double alpha = std::atof(argv[3]);
    double beta = std::atof(argv[4]);
    char precision = argv[5][0];

    if (mpi_rank == 0) {
        std::cout << "Starting distributed GEMM with:" << std::endl;
        std::cout << "  Matrix size: " << N << "x" << N << std::endl;
        std::cout << "  MPI processes: " << mpi_size << std::endl;
        std::cout << "  Repeats: " << repeats << std::endl;
        std::cout << "  Precision: " << precision << std::endl;
    }

    // Check number of available GPUs
    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus == 0) {
        std::cerr << "Rank " << mpi_rank << ": No CUDA-capable devices found: " << cudaGetErrorString(err) << std::endl;
        MPI_Finalize();
        return 1;
    }

    {
        TIME_SCOPE("Total Execution");
        try {
            bool success = false;
            switch (precision) {
            case 'S': {
                success = run_gemm_mpi<float>(N, repeats, static_cast<float>(alpha), static_cast<float>(beta));
                break;
            }
            case 'D': {
                success = run_gemm_mpi<double>(N, repeats, alpha, beta);
                break;
            }
            case 'H': {
                success = run_gemm_mpi<__half>(N, repeats, static_cast<__half>(alpha), static_cast<__half>(beta));
                break;
            }
            default:
                if (mpi_rank == 0) {
                    std::cerr << "Invalid precision. Use S, D, or H" << std::endl;
                }
                MPI_Finalize();
                return 1;
            }
        } catch (const std::exception &e) {
            std::cerr << "Rank " << mpi_rank << ": Exception: " << e.what() << std::endl;
            MPI_Finalize();
            return 1;
        }
    }

    if (mpi_rank == 0) {
        Timer::print_results();
    }

    MPI_Finalize();
    return 0;
}