#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <cuda_profiler_api.h>

// Precision control - change this type to switch precision
using precision_t = float; 

// Helper function to get precision name
template<typename T>
const char* get_precision_name();

template<>
const char* get_precision_name<float>() { return "SINGLE"; }

template<>
const char* get_precision_name<double>() { return "DOUBLE"; }

// Simple but robust timer 
class Timer {
    std::chrono::steady_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::steady_clock::now(); }
    double elapsed_ms() const { 
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); 
    }
};

// Execution pattern configuration
enum OperationType {
    OP_MEMORY,
    OP_COMPUTE
};

struct ExecutionPhase {
    OperationType type;
    double duration_ms;
};

// Define your execution patterns here
ExecutionPhase EXEC_PATTERN[] = {
    {OP_MEMORY, 500.0},
    {OP_COMPUTE, 500.0}
};

template<typename T>
struct ExecutionContext {
    // Memory bandwidth operation matrices
    T** d_memory_matrices;
    T* d_temp_matrix;
    
    // Pure compute operation matrices
    T* d_compute_matrixA;
    T* d_compute_matrixB;
    T* d_compute_matrixC;

    // coefficients 
    T alpha, beta;

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
    
    // Cublas handle
    cublasHandle_t handle;

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

// Specializations for cuBLAS GEMM
template<typename T>
cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k, const T* alpha,
                          const T* A, int lda, const T* B, int ldb,
                          const T* beta, T* C, int ldc);

template<>
cublasStatus_t cublas_gemm<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n, int k, const float* alpha,
                                 const float* A, int lda, const float* B, int ldb,
                                 const float* beta, float* C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
cublasStatus_t cublas_gemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                  int m, int n, int k, const double* alpha,
                                  const double* A, int lda, const double* B, int ldb,
                                  const double* beta, double* C, int ldc) {
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

    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&ctx.handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: creating cuBLAS handle\n");
        return false;
    }
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

    printf("Initialized memory matrices: %d x %d x %d (total %.1f MB)\n", 
           num_memory_matrices + 1, memory_matrix_size, memory_matrix_size, (double)num_memory_matrices * memory_matrix_bytes / 1e6);
    printf("Initialized compute matrices: 3 x %d x %d (total %.1f MB)\n", 
           compute_matrix_size, compute_matrix_size, (double)3.0 * compute_matrix_bytes / 1e6);
    
    return true;
}

template<typename T>
void execute_pure_gpu_memory_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer memory_timer;
    int phase_memory_ops = 0;
    int matrix_src = 0, matrix_dst = 1;
    
    // Dummy variable for read operations
    T* d_dummy;
    cudaMalloc(&d_dummy, sizeof(T));

    while (memory_timer.elapsed_ms() < duration_ms) {
        switch (phase_memory_ops % 3) {
            case 0:
                // matrix to temp memcpy
                cudaMemcpy(ctx->d_temp_matrix, ctx->d_memory_matrices[matrix_src], 
                           ctx->memory_matrix_bytes, cudaMemcpyDeviceToDevice);
                break;
                                
            case 1:
                // temp to matrix memcpy
                cudaMemcpy(ctx->d_memory_matrices[matrix_dst], ctx->d_temp_matrix, 
                           ctx->memory_matrix_bytes, cudaMemcpyDeviceToDevice);
                break;
                
            case 2:
                // Direct matrix-to-matrix copy
                cudaMemcpy(ctx->d_memory_matrices[matrix_dst], ctx->d_memory_matrices[matrix_src], 
                           ctx->memory_matrix_bytes, cudaMemcpyDeviceToDevice);
                break;
        }
        
        phase_memory_ops++;
        matrix_src = (matrix_src + 1) % ctx->num_memory_matrices;
        matrix_dst = (matrix_dst + 1) % ctx->num_memory_matrices;
    }
    
    ctx->total_memory_ops_count += phase_memory_ops;
}

template<typename T>
void execute_pure_compute_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer compute_timer;
    int phase_compute = 0;
    cublasOperation_t op_a;
    cublasOperation_t op_b;
    while (compute_timer.elapsed_ms() < duration_ms) {
        op_a = (phase_compute % 2 == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
        op_b = (phase_compute % 3 == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;

        T alpha = ctx->alpha + static_cast<T>(phase_compute * 0.0001);
        T beta = ctx->beta + static_cast<T>(phase_compute * 0.0001);
        // Pure GEMM computation - no memory transfers, just computation
        cublas_gemm<T>(ctx->handle, op_a, op_b,
                      ctx->compute_matrix_size, ctx->compute_matrix_size, ctx->compute_matrix_size,
                      &alpha,
                      ctx->d_compute_matrixA, ctx->compute_matrix_size,
                      ctx->d_compute_matrixB, ctx->compute_matrix_size,
                      &beta,
                      ctx->d_compute_matrixC, ctx->compute_matrix_size);
        
        cudaDeviceSynchronize();
        phase_compute++;
    }
    
    ctx->total_compute_ops_count += phase_compute;
}

template<typename T>
void execute_cycle(ExecutionContext<T>* ctx, ExecutionPhase* pattern, int pattern_length) {
    ctx->total_memory_ops_count = 0;
    ctx->total_compute_ops_count = 0;
    
    for (int i = 0; i < pattern_length; i++) {
        if (pattern[i].type == OP_MEMORY) {
            execute_pure_gpu_memory_phase<T>(ctx, pattern[i].duration_ms);
        } else {
            execute_pure_compute_phase<T>(ctx, pattern[i].duration_ms);
        }
    }
}

template<typename T>
void cleanup_context(ExecutionContext<T>& ctx) {
    for (int i = 0; i < ctx.num_memory_matrices; i++) {
        cudaFree(ctx.d_memory_matrices[i]);
    }
    delete[] ctx.d_memory_matrices;
    
    cudaFree(ctx.d_temp_matrix);
    cudaFree(ctx.d_compute_matrixA);
    cudaFree(ctx.d_compute_matrixB);
    cudaFree(ctx.d_compute_matrixC);
    cublasDestroy(ctx.handle);
}

template<typename T>
void bursty_execution(size_t memory_matrix_size, size_t compute_matrix_size, size_t num_memory_matrices, size_t total_runtime_ms) {
    Timer total_timer;

    // Preparation
    printf("Using %s precision operations\n", get_precision_name<T>());

    // Choose your execution pattern here
    ExecutionPhase* current_pattern = EXEC_PATTERN;
    int pattern_length = sizeof(EXEC_PATTERN) / sizeof(ExecutionPhase);

    printf("Using execution pattern with %d phases:\n", pattern_length);
    for (int i = 0; i < pattern_length; i++) {
        printf("  Phase %d: %s (%.0fms)\n", i + 1,
               current_pattern[i].type == OP_MEMORY ? "GPU Memory Ops" : "GPU Compute",
               current_pattern[i].duration_ms);
    }
    printf("\n");

    ExecutionContext<T> ctx;
    if (!initialize_context(ctx, memory_matrix_size, compute_matrix_size, num_memory_matrices)) {
        printf("Failed to initialize execution context\n");
        return;
    }

    double preparation_time = total_timer.elapsed_ms();
    printf("\nTotal preparation time: %.2f seconds\n", preparation_time / 1000.0);

    // Main execution loop
    total_timer.reset();

    int cycle = 0;
    Timer cycle_timer;

    while (total_timer.elapsed_ms() < total_runtime_ms) {
        printf("Cycle %d: ", ++cycle);
        cycle_timer.reset();    
        execute_cycle(&ctx, current_pattern, pattern_length);
        double cycle_time = cycle_timer.elapsed_ms();

        // Calculate actual metrics
        double total_memory_rw = (double)ctx.total_memory_ops_count * ctx.memory_matrix_bytes * 2.0;
        double total_bandwidth = total_memory_rw / (cycle_time / 1000.0) / 1e9;
        double total_flops = (double)ctx.total_compute_ops_count * ctx.flops_per_gemm;
        double total_gflops = total_flops / (cycle_time / 1000.0) / 1e9;
        
        printf("Total GPU Memory Ops: %d, Total Computation: %d, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
               ctx.total_memory_ops_count, ctx.total_compute_ops_count, total_bandwidth, total_gflops, cycle_time);
    }
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
    bursty_execution<precision_t>(memory_matrix_size, compute_matrix_size, num_memory_matrices, total_runtime_ms);
    // cudaProfilerStop();
    return 0;
}