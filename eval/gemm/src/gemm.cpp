#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <sys/time.h>


// C++11 compatible Timer class using RAII
class Timer {
  public:
    Timer(const std::string &name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {
    }

    ~Timer() {
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);

        std::vector<std::pair<std::string, double>> &timings = get_timings();

        // Check if this timing already exists and update it, or add new entry
        bool found = false;
        for (std::vector<std::pair<std::string, double>>::iterator it = timings.begin(); it != timings.end(); ++it) {
            if (it->first == name_) {
                it->second = duration.count();
                found = true;
                break;
            }
        }

        if (!found) {
            timings.push_back(std::make_pair(name_, duration.count()));
        }
    }

    static void print_results(size_t N, int repeats, double alpha, double beta) {
        std::cout << "\n=== Timing Results ===" << std::endl;

        double total = 0;
        double compute_time = 0;
        const std::vector<std::pair<std::string, double>> &timings = get_timings();

        // Calculate total time
        for (std::vector<std::pair<std::string, double>>::const_iterator it = timings.begin(); it != timings.end();
             ++it) {
            if (it->first == "Total Execution") {
                total = it->second;
            }
            if (it->first == "Compute GEMM") {
                compute_time = it->second;
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

        /*
         * Printout Performance Results
         */
        long long flops_per_op = 0;

        // Using a common calulation for FLOPs based on the GEMM operation
        flops_per_op += 2LL * N * N * N - (N * N);

        // 2. Scaling by alpha: alpha * (op(A) * op(B))
        //    If alpha != 1.0, we need N×N additional multiplications
        if (alpha != 1.0) {
            flops_per_op += N * N;
        }

        // 3. Scaling by beta: beta * C
        //    If beta != 0.0, we need N×N additional multiplications
        if (beta != 0.0) {
            flops_per_op += N * N;
        }

        // 4. Final addition: (alpha * op(A) * op(B)) + (beta * C)
        //    If beta != 0.0, we need N×N additional additions
        if (beta != 0.0) {
            flops_per_op += N * N;
        }

        long long total_flops = flops_per_op * repeats;
        double flops_per_second = total_flops / (compute_time / 1000.0);

        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Matrix size: " << N << " x " << N << std::endl;
        std::cout << "Repeats: " << repeats << std::endl;
        std::cout << "Total FLOPs: " << total_flops << std::endl;
        std::cout << "Total Computation Time: " << std::fixed << std::setprecision(3) << compute_time << " ms"
                  << std::endl;

        // Display in appropriate units
        if (flops_per_second >= 1e12) {
            std::cout << "Performance: " << std::fixed << std::setprecision(2) << flops_per_second / 1e12 << " TFLOPs"
                      << std::endl;
        } else if (flops_per_second >= 1e9) {
            std::cout << "Performance: " << std::fixed << std::setprecision(2) << flops_per_second / 1e9 << " GFLOPs"
                      << std::endl;
        } else {
            std::cout << "Performance: " << std::fixed << std::setprecision(2) << flops_per_second / 1e6 << " MFLOPs"
                      << std::endl;
        }
    }

    static void clear() {
        get_timings().clear();
    }

  private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;

    static std::vector<std::pair<std::string, double>> &get_timings() {
        static std::vector<std::pair<std::string, double>> timings;
        return timings;
    }
};

// Convenient macro for timing scopes and use line number
#define TIME_SCOPE(name) Timer timer_##__LINE__(name)

// Printout cuda error message
void handle_cuda_error(cudaError_t error, const std::string &operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

bool set_gpu_device(int gpu_id) {
    TIME_SCOPE("GPU Device Setup");

    // Check number of available GPUs
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    handle_cuda_error(err, "cudaGetDeviceCount");

    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return false;
    }

    if (gpu_id >= device_count || gpu_id < 0) {
        std::cerr << "Invalid GPU ID: " << gpu_id << ". Available GPUs: 0-" << (device_count - 1) << std::endl;
        return false;
    }

    // Set the device
    err = cudaSetDevice(gpu_id);
    handle_cuda_error(err, "cudaSetDevice");

    // Get and print device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, gpu_id);
    handle_cuda_error(err, "cudaGetDeviceProperties");

    std::cout << "\n=== GPU Device Information ===" << std::endl;
    std::cout << "Using GPU " << gpu_id << ": " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak Memory Bandwidth: " << std::fixed << std::setprecision(1)
              << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << " GB/s" << std::endl;
    std::cout << "==============================" << std::endl;

    return true;
}

namespace gemm {

enum class Precision {
    SINGLE = 'S',
    DOUBLE = 'D',
    HALF = 'H'
};

template <typename T> struct CublasTraits;

template <> struct CublasTraits<float> {
    static constexpr auto gemm_func = &cublasSgemm;
    static constexpr auto math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
};

template <> struct CublasTraits<double> {
    static constexpr auto gemm_func = &cublasDgemm;
    static constexpr auto math_mode = CUBLAS_DEFAULT_MATH;
};

template <> struct CublasTraits<__half> {
    static constexpr auto gemm_func = &cublasHgemm;
    static constexpr auto math_mode = CUBLAS_TENSOR_OP_MATH;
};

template <typename T> class Gemm {
  private:
    size_t N_;

    // Matrices on Host
    T *h_matrix;

    // Matrices on Device
    T *d_matrixA;
    T *d_matrixB;
    T *d_matrixC;

    cublasHandle_t cublas_handle_;

  public:
    Gemm(size_t N) : N_(N) {
    }

    ~Gemm() {
        cleanup();
    }

    size_t get_matrix_size() const {
        return sizeof(T) * N_ * N_;
    }

    bool compute_gemm_warmup(int repeats, T alpha = T(1.0), T beta = T(0.0)) {
        TIME_SCOPE("Compute GEMM Warm-up");
        for (int w = 0; w < 10; ++w) {
            cublasStatus_t status =
                CublasTraits<T>::gemm_func(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, N_, N_, N_, &alpha, d_matrixA, N_,
                                           d_matrixB, N_, &beta, d_matrixC, N_);
            handle_cublas_error(status, "gemm computation");
        }

        cudaDeviceSynchronize();
        return true;
    }

    bool compute_gemm(int repeats, T alpha = T(1.0), T beta = T(0.0)) {
        TIME_SCOPE("Compute GEMM");
        for (int r = 0; r < repeats; ++r) {
            cublasStatus_t status =
                CublasTraits<T>::gemm_func(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, N_, N_, N_, &alpha, d_matrixA, N_,
                                           d_matrixB, N_, &beta, d_matrixC, N_);
            handle_cublas_error(status, "gemm computation");
        }

        cudaDeviceSynchronize();
        return true;
    }

    bool gather_results() {
        TIME_SCOPE("Copy data from GPU to Host");
        std::cout << "Gather Results: Copy data from GPU to Host" << std::endl;
        cudaError_t err = cudaMemcpy(h_matrix, d_matrixC, get_matrix_size(), cudaMemcpyDeviceToHost);
        handle_cuda_error(err, "cudaMemcpy C host to device");

        cudaDeviceSynchronize();
        return true;
    }

    bool allocate_copy_host_matrices() {
        TIME_SCOPE("Matrices Allocation and Transfer");
        std::cout << "Allocating Matrices on Host and Copying to GPU" << std::endl;

        // Use regular malloc instead of pinned memory
        h_matrix = static_cast<T *>(malloc(sizeof(T) * N_ * N_));
        if (!h_matrix) {
            std::cerr << "malloc h_matrix failed" << std::endl;
            return false;
        }

        // Initialize random value generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // Initialize matrices and copying them to GPU
        for (long long i = 0; i < N_ * N_; ++i) {
            //h_matrix[i] = static_cast<T>(dis(gen));
            h_matrix[i] = static_cast<T>((float)rand() / RAND_MAX);
        }
        cudaError_t err = cudaMemcpy(d_matrixA, h_matrix, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
        handle_cuda_error(err, "cudaMemcpy A host to device");

        for (long long i = 0; i < N_ * N_; ++i) {
            //h_matrix[i] = static_cast<T>(dis(gen));
            h_matrix[i] = static_cast<T>((float)rand() / RAND_MAX);
        }
        err = cudaMemcpy(d_matrixB, h_matrix, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
        handle_cuda_error(err, "cudaMemcpy B host to device");

        for (long long i = 0; i < N_ * N_; ++i) {
            //h_matrix[i] = static_cast<T>(dis(gen));
            h_matrix[i] = static_cast<T>((float)rand() / RAND_MAX);
        }
        err = cudaMemcpy(d_matrixC, h_matrix, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
        handle_cuda_error(err, "cudaMemcpy C host to device");

        cudaDeviceSynchronize();
        return true;
    }

    bool allocate_gpu_matrices() {
        TIME_SCOPE("Matrices Allocation on GPU");
        std::cout << "Allocating Matrics on GPU" << std::endl;

        // Allocate matrices on device
        cudaError_t err = cudaMalloc(&d_matrixA, get_matrix_size());
        handle_cuda_error(err, "cudaMalloc d_matrixA");

        err = cudaMalloc(&d_matrixB, get_matrix_size());
        handle_cuda_error(err, "cudaMalloc d_matrixB");

        err = cudaMalloc(&d_matrixC, get_matrix_size());
        handle_cuda_error(err, "cudaMalloc d_matrixC");

        cudaDeviceSynchronize();
        return true;
    }

    bool setup_cublas() {
        cublasStatus_t status = cublasCreate(&cublas_handle_);
        handle_cublas_error(status, "cublasCreate");

        status = cublasSetMathMode(cublas_handle_, CublasTraits<T>::math_mode);
        handle_cublas_error(status, "cublasSetMathMode");

        return true;
    }

    void cleanup() {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }

        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_matrixC);

        free(h_matrix);
        // Reset pointers
        d_matrixA = d_matrixB = d_matrixC = nullptr;
        h_matrix = nullptr;
    }

  private:
    void handle_cublas_error(cublasStatus_t status, const std::string &operation) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << operation << " failed with cuBLAS error " << status << std::endl;
            exit(1);
        }
    }
};

} // namespace gemm

// Template function to eliminate code duplication
template <typename T> bool run_gemm(size_t N, int repeats, T alpha, T beta, const std::string &precision_name) {
    TIME_SCOPE("Total Execution");

    gemm::Gemm<T> gemm(N);

    return gemm.allocate_gpu_matrices() && gemm.allocate_copy_host_matrices() && gemm.setup_cublas() &&
           gemm.compute_gemm_warmup(repeats, alpha, beta) && gemm.compute_gemm(repeats, alpha, beta) &&
           gemm.gather_results();
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " <N> <repeats> <alpha> <beta> <precision(S/D/H)>" << std::endl;
        std::cout << "  GPU_ID: GPU device ID (0, 1, 2, ...)" << std::endl;
        std::cout << "  N: Matrix size (NxN)" << std::endl;
        std::cout << "  repeats: Number of GEMM iterations" << std::endl;
        std::cout << "  alpha, beta: GEMM coefficients" << std::endl;
        std::cout << "  precision: S(ingle), D(ouble), or H(alf)" << std::endl;
        return 1;
    }

    int gpu_id = std::atoi(argv[1]);
    size_t N = std::atoi(argv[2]);
    int repeats = std::atoi(argv[3]);
    double alpha = std::atof(argv[4]);
    double beta = std::atof(argv[5]);
    char precision = argv[6][0];

    // MODIFIED: Set GPU device before any CUDA operations
    if (!set_gpu_device(gpu_id)) {
        return 1;
    }

    std::cout << "Starting distributed GEMM with:" << std::endl;
    std::cout << "  GPU ID: " << gpu_id << std::endl;
    std::cout << "  Matrix size: " << N << "x" << N << std::endl;
    std::cout << "  Repeats: " << repeats << std::endl;
    std::cout << "  Precision: " << precision << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        bool success = false;
        switch (precision) {
        case 'S':
            success = run_gemm<float>(N, repeats, static_cast<float>(alpha), static_cast<float>(beta), "Single");
            break;
        case 'D':
            success = run_gemm<double>(N, repeats, alpha, beta, "Double");
            break;
        case 'H':
            success = run_gemm<__half>(N, repeats, static_cast<__half>(alpha), static_cast<__half>(beta), "Half");
            break;
        default:
            std::cerr << "Invalid precision. Use S, D, or H" << std::endl;
            return 1;
        }
        if (success) {
            std::cout << precision << " precision GEMM completed successfully" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    Timer::print_results(N, repeats, alpha, beta);
    return 0;
}