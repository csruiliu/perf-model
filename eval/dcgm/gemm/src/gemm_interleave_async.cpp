#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <random>
#include <memory>

// Precision control - change this type to switch precision
// 'float' for single precision
// 'double' for double precision
using precision_t = float; 

// Helper function to get precision name
template<typename T>
const char* get_precision_name();

template<>
const char* get_precision_name<float>() { return "SINGLE"; }

template<>
const char* get_precision_name<double>() { return "DOUBLE"; }

// Timer
class Timer {
    std::chrono::steady_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::steady_clock::now(); }
    double elapsed_ms() const { 
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); 
    }
};

// Execution context
template<typename T>
struct ExecutionContext {
    // Memory bandwidth operation matrices
    T** d_memory_matrices;
    T* d_temp_matrix;
    
    // Pure compute operation matrices
    T* d_compute_matrixA;
    T* d_compute_matrixB;
    T* d_compute_matrixC;
    
    // CUBLAS handle and parameters
    cublasHandle_t handle;
    T alpha, beta;
    
    // CUDA streams for async operations
    cudaStream_t memory_stream;
    cudaStream_t compute_stream;

    // Matric num
    size_t num_memory_matrices;

    // Matrix sizes
    size_t memory_matrix_size;
    size_t compute_matrix_size;
    
    // Matrix bytes
    size_t memory_matrix_bytes;
    size_t compute_matrix_bytes;

    // Statistics
    size_t total_memory_ops_count;
    size_t total_compute_ops_count;
    long long flops_per_gemm;

    ExecutionContext() : d_memory_matrices(nullptr), d_temp_matrix(nullptr),
                        d_compute_matrixA(nullptr), d_compute_matrixB(nullptr), d_compute_matrixC(nullptr),
                        num_memory_matrices(0), memory_matrix_size(0), compute_matrix_size(0),
                        memory_matrix_bytes(0), compute_matrix_bytes(0),
                        total_memory_ops_count(0), total_compute_ops_count(0), flops_per_gemm(0) {
        alpha = T(1.0);
        beta = T(0.1);
    }
};

long long calculate_gemm_gflops(size_t matrix_size) {
    // For GEMM: C = A * B, FLOPS = 2 * M * N * K (M=N=K for square matrices)
    long long flops_per_gemm = 2LL * matrix_size * matrix_size * matrix_size;
    return flops_per_gemm;
}

void generate_random_matrix(float* matrix, int size, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size * size; i++) {
        matrix[i] = dis(gen);
    }
}

void generate_random_matrix(double* matrix, int size, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    
    for (int i = 0; i < size * size; i++) {
        matrix[i] = dis(gen);
    }
}

// Template specializations for CUBLAS operations
template<typename T>
cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k, const T *alpha,
                          const T *A, int lda, const T *B, int ldb,
                          const T *beta, T *C, int ldc);

template<>
cublasStatus_t cublas_gemm<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n, int k, const float *alpha,
                                 const float *A, int lda, const float *B, int ldb,
                                 const float *beta, float *C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
cublasStatus_t cublas_gemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                  int m, int n, int k, const double *alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  const double *beta, double *C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename T>
bool initialize_context(ExecutionContext<T>& ctx, size_t memory_matrix_size, size_t compute_matrix_size, size_t num_memory_matrices) {
    size_t memory_matrix_bytes = memory_matrix_size * memory_matrix_size * sizeof(T);
    size_t compute_matrix_bytes = compute_matrix_size * compute_matrix_size * sizeof(T);

    ctx.memory_matrix_size = memory_matrix_size;
    ctx.compute_matrix_size = compute_matrix_size;
    ctx.memory_matrix_bytes = memory_matrix_bytes;
    ctx.compute_matrix_bytes = compute_matrix_bytes;
    ctx.num_memory_matrices = num_memory_matrices;
    ctx.flops_per_gemm = calculate_gemm_gflops(compute_matrix_size);

    // Create CUDA streams
    cudaError_t cuda_status = cudaStreamCreate(&ctx.memory_stream);
    if (cuda_status != cudaSuccess) {
        printf("ERROR: creating memory stream: %s\n", cudaGetErrorString(cuda_status));
        return false;
    }
    
    cuda_status = cudaStreamCreate(&ctx.compute_stream);
    if (cuda_status != cudaSuccess) {
        printf("ERROR: creating compute stream: %s\n", cudaGetErrorString(cuda_status));
        cudaStreamDestroy(ctx.memory_stream);
        return false;
    }

    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&ctx.handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: creating cuBLAS handle\n");
        cudaStreamDestroy(ctx.memory_stream);
        cudaStreamDestroy(ctx.compute_stream);
        return false;
    }

    // Set cuBLAS to use the compute stream
    cublasSetStream(ctx.handle, ctx.compute_stream);
    cublasSetMathMode(ctx.handle, CUBLAS_DEFAULT_MATH);

    // Allocate GPU memory for memory bandwidth operations
    ctx.d_memory_matrices = new T*[num_memory_matrices];
    for (int i = 0; i < num_memory_matrices; i++) {
        cudaMalloc(&ctx.d_memory_matrices[i], memory_matrix_bytes);
        
        // Initialize with some data
        T* h_temp = new T[memory_matrix_size * memory_matrix_size];
        generate_random_matrix(h_temp, memory_matrix_size, i * 1000);
        cudaMemcpy(ctx.d_memory_matrices[i], h_temp, memory_matrix_bytes, cudaMemcpyHostToDevice);
        delete[] h_temp;
    }

    // Allocate temporary matrix for memory operations
    cudaMalloc(&ctx.d_temp_matrix, memory_matrix_bytes);

    // Allocate GPU memory for pure compute operations (separate matrices)
    cudaMalloc(&ctx.d_compute_matrixA, compute_matrix_bytes);
    cudaMalloc(&ctx.d_compute_matrixB, compute_matrix_bytes);
    cudaMalloc(&ctx.d_compute_matrixC, compute_matrix_bytes);

    // Initialize compute matrices
    T* h_compute_matrix = new T[compute_matrix_size * compute_matrix_size];
    generate_random_matrix(h_compute_matrix, compute_matrix_size, 12345);
    cudaMemcpy(ctx.d_compute_matrixA, h_compute_matrix, compute_matrix_bytes, cudaMemcpyHostToDevice);
    
    generate_random_matrix(h_compute_matrix, compute_matrix_size, 54321);
    cudaMemcpy(ctx.d_compute_matrixB, h_compute_matrix, compute_matrix_bytes, cudaMemcpyHostToDevice);

    // Initialize result matrix to zeros
    cudaMemset(ctx.d_compute_matrixC, 0, compute_matrix_bytes);

    delete[] h_compute_matrix;

    printf("Initialized memory matrices: %d x %d x %d (total %.1f MB )\n", 
           num_memory_matrices + 1, memory_matrix_size, memory_matrix_size, (double)num_memory_matrices * memory_matrix_bytes / 1e6);
    printf("Initialized compute matrices: 3 x %d x %d (total %.1f MB)\n", 
           compute_matrix_size, compute_matrix_size, (double)3.0 * compute_matrix_bytes / 1e6);
    
    return true;
}

template<typename T>
void cleanup_context(ExecutionContext<T>& ctx) {
    // Synchronize streams before cleanup
    cudaStreamSynchronize(ctx.memory_stream);
    cudaStreamSynchronize(ctx.compute_stream);
    
    // Destroy streams
    cudaStreamDestroy(ctx.memory_stream);
    cudaStreamDestroy(ctx.compute_stream);

    // Destroy cuBLAS handle
    if (ctx.handle) {
        cublasDestroy(ctx.handle);
    }

    // Free memory matrices
    if (ctx.d_memory_matrices) {
        for (int i = 0; i < ctx.num_memory_matrices; i++) {
            cudaFree(ctx.d_memory_matrices[i]);
        }
        delete[] ctx.d_memory_matrices;
    }
    
    cudaFree(ctx.d_temp_matrix);
    cudaFree(ctx.d_compute_matrixA);
    cudaFree(ctx.d_compute_matrixB);
    cudaFree(ctx.d_compute_matrixC);
}

template<typename T>
void interleave_execution(size_t memory_matrix_size, size_t compute_matrix_size, size_t num_memory_matrices, size_t total_runtime_ms) {
    Timer total_timer;

    // Preparation
    printf("Using %s precision operations\n", get_precision_name<T>());

    ExecutionContext<T> ctx;
    if (!initialize_context(ctx, memory_matrix_size, compute_matrix_size, num_memory_matrices)) {
        printf("Failed to initialize execution context\n");
        return;
    }

    int matrix_index = 0;
    int cycle_memory_ops = 0;
    int cycle_compute_ops = 0;
    long cycle_total_ops = 0;
    int num_cycles = 0;
    int matrix_src = 0, matrix_dst = 1;
    cublasOperation_t op_a;
    cublasOperation_t op_b;

    Timer cycle_timer;
    Timer sync_timer;
    size_t cycle_runtime_ms = 1000;
    
    while (total_timer.elapsed_ms() < total_runtime_ms) {
        cycle_timer.reset();
    
        // Execute operations for one cycle (1 second)
        while (cycle_timer.elapsed_ms() < cycle_runtime_ms) {
            if (cycle_total_ops % 2 == 0) {
                switch (cycle_memory_ops % 3) {
                    case 0:
                        // matrix to temp memcpy
                        cudaMemcpyAsync(ctx.d_temp_matrix, ctx.d_memory_matrices[matrix_src], 
                                        ctx.memory_matrix_bytes, cudaMemcpyDeviceToDevice, ctx.memory_stream);
                        break;
                                        
                    case 1:
                        // temp to matrix memcpy
                        cudaMemcpyAsync(ctx.d_memory_matrices[matrix_dst], ctx.d_temp_matrix, 
                                        ctx.memory_matrix_bytes, cudaMemcpyDeviceToDevice, ctx.memory_stream);
                        break;
                        
                    case 2:
                        // Direct matrix-to-matrix copy
                        cudaMemcpyAsync(ctx.d_memory_matrices[matrix_dst], ctx.d_memory_matrices[matrix_src], 
                                        ctx.memory_matrix_bytes, cudaMemcpyDeviceToDevice, ctx.memory_stream);
                        break;
                }
                cycle_memory_ops++;
                cycle_total_ops++;
                matrix_src = (matrix_src + 1) % ctx.num_memory_matrices;
                matrix_dst = (matrix_dst + 1) % ctx.num_memory_matrices;
            } else {
                op_a = (cycle_compute_ops % 2 == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
                op_b = (cycle_compute_ops % 3 == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
                
                T alpha = ctx.alpha + static_cast<T>(cycle_compute_ops * 0.0001);
                T beta = ctx.beta + static_cast<T>(cycle_compute_ops * 0.0001);

                cublasStatus_t gemm_status = cublas_gemm<T>(ctx.handle, op_a, op_b,
                                                            compute_matrix_size, compute_matrix_size, compute_matrix_size,
                                                            &alpha, 
                                                            ctx.d_compute_matrixA, compute_matrix_size,
                                                            ctx.d_compute_matrixB, compute_matrix_size,
                                                            &beta, 
                                                            ctx.d_compute_matrixC, compute_matrix_size);
                if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                    printf("GEMM operation failed\n");
                    break;
                }
                cycle_compute_ops++;
                cycle_total_ops++;
            }
            
            // Sync every 250ms
            if (sync_timer.elapsed_ms() > cycle_runtime_ms / 10) {
                // Synchronize both streams at the end of each cycle for accurate timing
                cudaStreamSynchronize(ctx.memory_stream);
                cudaStreamSynchronize(ctx.compute_stream);
                cudaDeviceSynchronize();
                sync_timer.reset();
            }

        }
        num_cycles++;
        double total_cycle_time = cycle_timer.elapsed_ms();
        
        // Calculate actual metrics
        double total_memory_rw = (double)cycle_memory_ops * ctx.memory_matrix_bytes * 2;
        double total_bandwidth = total_memory_rw / (total_cycle_time / 1000.0) / 1e9;
        double total_flops = (double)cycle_compute_ops * ctx.flops_per_gemm;
        double total_gflops = total_flops / (total_cycle_time / 1000.0) / 1e9;
        
        printf("Cycle %d: Total GPU Memory Ops: %d, Total Computation: %d, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
                num_cycles, cycle_memory_ops, cycle_compute_ops, total_bandwidth, total_gflops, total_cycle_time);

        cycle_memory_ops = 0;
        cycle_compute_ops = 0;
    }

    double total_time = total_timer.elapsed_ms();
    printf("\nTotal runtime: %.2f seconds\n", total_time / 1000.0);
    
    cleanup_context(ctx);
}


int main(int argc, char* argv[]) {
    // Set GPU device
    cudaSetDevice(0);

    // Matrix sizes
    size_t memory_matrix_size = 2048;  // Size for GPU memory bandwidth operations
    size_t compute_matrix_size = 1024; // Size for pure compute operations
    size_t num_memory_matrices = 4;
    size_t total_runtime_ms = 60000;

    if (argc != 5) {
        printf("Usage: %s <num_memory_matrices> <memory_matrix_size> <compute_matrix_size> <total_runtime_ms> \n", argv[0]);
        return 1;
    }

    num_memory_matrices = atoi(argv[1]);
    memory_matrix_size = atoi(argv[2]);
    compute_matrix_size = atoi(argv[3]);
    total_runtime_ms = atoi(argv[4]);

    printf("==================================================\n");
    printf("GPU Memory Bandwidth & Pure Compute Benchmark\n");
    
    // cudaProfilerStart();
    interleave_execution<precision_t>(memory_matrix_size, compute_matrix_size, num_memory_matrices, total_runtime_ms);
    // cudaProfilerStop();
    return 0;
}