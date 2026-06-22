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

// Constants
static const int NUM_MEMORY_MATRICES = 4;  // Multiple matrices for GPU memory operations
static const int NUM_MEMORY_STREAMS = 6;
static const int NUM_COMPUTE_STREAMS = 6;
static const double TOTAL_RUNTIME_MS = 60000.0;

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

template<typename T>
__global__ void consistent_copy_kernel(const T* __restrict__ src, T* __restrict__ dst, size_t elements_per_thread) {
    size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    
    if (std::is_same<T, float>::value) {
        // Process 4 floats per thread (16 bytes)
        const float4* src4 = reinterpret_cast<const float4*>(src + base_idx);
        float4* dst4 = reinterpret_cast<float4*>(dst + base_idx);
        dst4[0] = src4[0];  // Single memory operation, 16 bytes
    } else if (std::is_same<T, double>::value) {
        // Process 2 doubles per thread (16 bytes)
        const double2* src2 = reinterpret_cast<const double2*>(src + base_idx);
        double2* dst2 = reinterpret_cast<double2*>(dst + base_idx);
        dst2[0] = src2[0];  // Single memory operation, 16 bytes
    }
}

template<typename T>
__global__ void consistent_read_write_kernel(T* __restrict__ data, size_t elements_per_thread) {
    size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    
    if (std::is_same<T, float>::value) {
        // Read and write back - consistent 32 bytes per thread
        float4* data4 = reinterpret_cast<float4*>(data + base_idx);
        float4 temp = data4[0];  // Read 16 bytes
        data4[0] = temp;         // Write 16 bytes
    } else if (std::is_same<T, double>::value) {
        // Read and write back - consistent 32 bytes per thread
        double2* data2 = reinterpret_cast<double2*>(data + base_idx);
        double2 temp = data2[0];  // Read 16 bytes
        data2[0] = temp;          // Write 16 bytes
    }
}

// Consistent memory operation helper class
template<typename T>
struct MemoryOp {
    static const size_t BYTES_PER_OP = 16;  // Always 16 bytes per thread for reads/writes
    static const size_t ELEMENTS_PER_THREAD = BYTES_PER_OP / sizeof(T);
    
    static void launch_copy_operation(const T* src, T* dst, size_t total_elements, cudaStream_t stream) {
        size_t total_threads = total_elements / ELEMENTS_PER_THREAD;
        
        const int block_size = 256;
        const int grid_size = (total_threads + block_size - 1) / block_size;
        
        consistent_copy_kernel<T><<<grid_size, block_size, 0, stream>>>(
            src, dst, ELEMENTS_PER_THREAD
        );
    }
    
    static void launch_read_write_operation(T* data, size_t total_elements, cudaStream_t stream) {
        size_t total_threads = total_elements / ELEMENTS_PER_THREAD;
        
        const int block_size = 256;
        const int grid_size = (total_threads + block_size - 1) / block_size;
        
        consistent_read_write_kernel<T><<<grid_size, block_size, 0, stream>>>(
            data, ELEMENTS_PER_THREAD
        );
    }
};

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
    {OP_MEMORY, 250.0},
    {OP_COMPUTE, 750.0}
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
    
    // Streams
    cudaStream_t memory_streams[NUM_MEMORY_STREAMS];
    cudaStream_t compute_streams[NUM_COMPUTE_STREAMS];
    
    // cuBLAS handles
    cublasHandle_t cublas_handles[NUM_COMPUTE_STREAMS];
    
    // coefficients 
    T alpha, beta;

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
                        memory_matrix_size(0), compute_matrix_size(0),
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

template<typename T>
bool initialize_context(ExecutionContext<T>& ctx, size_t memory_matrix_size, size_t compute_matrix_size) {
    size_t memory_matrix_bytes = memory_matrix_size * memory_matrix_size * sizeof(T);
    size_t compute_matrix_bytes = compute_matrix_size * compute_matrix_size * sizeof(T);

    ctx.memory_matrix_size = memory_matrix_size;
    ctx.compute_matrix_size = compute_matrix_size;
    ctx.memory_matrix_bytes = memory_matrix_bytes;
    ctx.compute_matrix_bytes = compute_matrix_bytes;
    ctx.flops_per_gemm = calculate_gemm_gflops(compute_matrix_size);

    // Create streams with high priority for memory operations
    for (int i = 0; i < NUM_MEMORY_STREAMS; i++) {
        cudaError_t err = cudaStreamCreateWithPriority(&ctx.memory_streams[i], cudaStreamNonBlocking, 0);
        if (err != cudaSuccess) {
            printf("Failed to create memory stream %d: %s\n", i, cudaGetErrorString(err));
            return false;
        }
    }

    // Create compute streams
    for (int i = 0; i < NUM_COMPUTE_STREAMS; i++) {
        cudaError_t err = cudaStreamCreate(&ctx.compute_streams[i]);
        if (err != cudaSuccess) {
            printf("Failed to create compute stream %d: %s\n", i, cudaGetErrorString(err));
            return false;
        }
    }

    // Create cuBLAS handles
    for (int i = 0; i < NUM_COMPUTE_STREAMS; i++) {
        cublasCreate(&ctx.cublas_handles[i]);
        cublasSetStream(ctx.cublas_handles[i], ctx.compute_streams[i]);
        cublasSetMathMode(ctx.cublas_handles[i], CUBLAS_DEFAULT_MATH);
    }
    
    // Allocate GPU memory for memory bandwidth operations
    ctx.d_memory_matrices = new T*[NUM_MEMORY_MATRICES];
    for (int i = 0; i < NUM_MEMORY_MATRICES; i++) {
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

    // Prefault memory and ensure proper allocation
    for (int i = 0; i < NUM_MEMORY_STREAMS; i++) {
        cudaMemset(ctx.d_memory_matrices[i], 0, memory_matrix_bytes);
    }

    // Warm up GPU for consistent performance
    cudaDeviceSynchronize();
    
    printf("Initialized memory matrices: %d x %d x %d (total %.1f MB )\n", 
           NUM_MEMORY_MATRICES + 1, memory_matrix_size, memory_matrix_size, (double)NUM_MEMORY_MATRICES * memory_matrix_bytes / 1e6);
    printf("Initialized compute matrices: 3 x %d x %d (total %.1f MB)\n", 
           compute_matrix_size, compute_matrix_size, (double)3.0 * compute_matrix_bytes / 1e6);

    return true;
}

template<typename T>
void execute_pure_gpu_memory_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer memory_timer;
    
    int operation_type = 0;
    int phase_memory_ops = 0;
    
    const size_t matrix_elements = ctx->memory_matrix_size * ctx->memory_matrix_size;
    const size_t matrix_bytes = matrix_elements * sizeof(T);
    // const size_t elements_per_stream = matrix_elements / NUM_MEMORY_STREAMS;
    // const size_t elemnets_bytes = matrix_bytes / NUM_MEMORY_STREAMS;
    
    while (memory_timer.elapsed_ms() < duration_ms) {
        operation_type = phase_memory_ops % 4;

        switch (operation_type) {
            case 0:
                for (int stream_id = 0; stream_id < NUM_MEMORY_STREAMS && stream_id < NUM_MEMORY_MATRICES - 1; stream_id++) {
                    int src_matrix = (phase_memory_ops + stream_id) % NUM_MEMORY_MATRICES;
                    int dst_matrix = (phase_memory_ops + stream_id + 1) % NUM_MEMORY_MATRICES;
                    
                    size_t offset_bytes = stream_id * matrix_bytes;
                    
                    cudaMemcpyAsync(
                        reinterpret_cast<char*>(ctx->d_memory_matrices[dst_matrix]) + offset_bytes,
                        reinterpret_cast<const char*>(ctx->d_memory_matrices[src_matrix]) + offset_bytes,
                        matrix_bytes,
                        cudaMemcpyDeviceToDevice,
                        ctx->memory_streams[stream_id]
                    );
                }
                break;
            case 1:
                for (int stream_id = 0; stream_id < NUM_MEMORY_STREAMS && stream_id < NUM_MEMORY_MATRICES - 1; stream_id++) {
                    int src_matrix = stream_id;
                    int dst_matrix = stream_id + 1;
                    
                    cudaMemcpyAsync(
                        ctx->d_memory_matrices[dst_matrix],
                        ctx->d_memory_matrices[src_matrix],
                        matrix_bytes,
                        cudaMemcpyDeviceToDevice,
                        ctx->memory_streams[stream_id]
                    );
                }
                break;
            case 2:
                for (int stream_id = 0; stream_id < NUM_MEMORY_STREAMS && stream_id < NUM_MEMORY_MATRICES - 1; stream_id++) {
                    int src_matrix = (phase_memory_ops + stream_id) % NUM_MEMORY_MATRICES;
                    int dst_matrix = (phase_memory_ops + stream_id + 1) % NUM_MEMORY_MATRICES;
                    size_t offset = stream_id * matrix_elements;
                    
                    MemoryOp<T>::launch_copy_operation(
                        ctx->d_memory_matrices[src_matrix] + offset,
                        ctx->d_memory_matrices[dst_matrix] + offset,
                        matrix_elements,
                        ctx->memory_streams[stream_id]
                    );
                }
                break;
            case 3:
                // Read-write operations
                for (int stream_id = 0; stream_id < NUM_MEMORY_STREAMS; stream_id++) {
                    int matrix_idx = (phase_memory_ops + stream_id) % NUM_MEMORY_MATRICES;
                    size_t offset = stream_id * matrix_elements;
                    
                    MemoryOp<T>::launch_read_write_operation(
                        ctx->d_memory_matrices[matrix_idx] + offset,
                        matrix_elements,
                        ctx->memory_streams[stream_id]
                    );
                }
                break;
        }

        // Synchronize all streams before next iteration
        //for (int i = 0; i < NUM_MEMORY_STREAMS; i++) {
        //   cudaStreamSynchronize(ctx->memory_streams[i]);
        //}
        cudaDeviceSynchronize();
        phase_memory_ops++;
    }

    ctx->total_memory_ops_count += phase_memory_ops;
}

template<typename T>
void execute_pure_compute_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer compute_timer;
    int phase_compute_ops = 0;
    cublasOperation_t op_a;
    cublasOperation_t op_b;
    while (compute_timer.elapsed_ms() < duration_ms) {
        op_a = (phase_compute_ops % 2 == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
        op_b = (phase_compute_ops % 3 == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;

        T alpha = ctx->alpha + static_cast<T>(phase_compute_ops * 0.0001);
        T beta = ctx->beta + static_cast<T>(phase_compute_ops * 0.0001);

        // Launch GEMM operations on all compute streams
        for (int stream_id = 0; stream_id < NUM_COMPUTE_STREAMS; stream_id++) {
            // Pure GEMM computation - no memory transfers, just computation
            cublas_gemm<T>(ctx->cublas_handles[stream_id], op_a, op_b, 
                           ctx->compute_matrix_size, ctx->compute_matrix_size, ctx->compute_matrix_size,
                           &alpha,
                           ctx->d_compute_matrixA, ctx->compute_matrix_size,
                           ctx->d_compute_matrixB, ctx->compute_matrix_size,
                           &beta,
                           ctx->d_compute_matrixC, ctx->compute_matrix_size);
        }

        // Synchronize all compute streams
        //for (int i = 0; i < NUM_COMPUTE_STREAMS; i++) {
        //    cudaStreamSynchronize(ctx->compute_streams[i]);
        //}
        cudaDeviceSynchronize();
        phase_compute_ops++;
    }
    
    ctx->total_compute_ops_count += phase_compute_ops;
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
void cleanup_gpu_context(ExecutionContext<T>& ctx) {
    // Destroy streams
    for (int i = 0; i < NUM_MEMORY_STREAMS; i++) {
        cudaStreamDestroy(ctx.memory_streams[i]);
    }
    for (int i = 0; i < NUM_COMPUTE_STREAMS; i++) {
        cudaStreamDestroy(ctx.compute_streams[i]);
    }
    
    // Destroy cuBLAS handles
    for (int i = 0; i < NUM_COMPUTE_STREAMS; i++) {
        cublasDestroy(ctx.cublas_handles[i]);
    }
    
    // Free memory
    if (ctx.d_memory_matrices) {
        for (int i = 0; i < NUM_MEMORY_MATRICES; i++) {
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
void bursty_execution(int memory_matrix_size, int compute_matrix_size) {
    Timer total_timer;

    // Preparation
    printf("Using %s precision operations\n", get_precision_name<T>());
    
    // Choose your execution pattern here
    ExecutionPhase* current_pattern = EXEC_PATTERN;
    int pattern_length = sizeof(EXEC_PATTERN) / sizeof(ExecutionPhase);

    printf("Using execution pattern with %d phases:\n", pattern_length);
    for (int i = 0; i < pattern_length; i++) {
        printf("  Phase %d: %s (%.0fms)\n", i + 1,
               current_pattern[i].type == OP_MEMORY ? "Pure GPU Memory Ops" : "Pure Compute",
               current_pattern[i].duration_ms);
    }
    printf("\n");

    ExecutionContext<T> ctx;
    if (!initialize_context(ctx, memory_matrix_size, compute_matrix_size)) {
        printf("Failed to initialize execution context\n");
        return;
    }

    double cycle_duration = 0;
    for (int i = 0; i < pattern_length; i++) {
        cycle_duration += current_pattern[i].duration_ms;
    }

    double preparation_time = total_timer.elapsed_ms();
    printf("\nTotal preparation time: %.2f seconds\n", preparation_time / 1000.0);

    // Main execution loop
    total_timer.reset();

    int cycle = 0;
    Timer cycle_timer;

    while (total_timer.elapsed_ms() < TOTAL_RUNTIME_MS) {
        printf("Cycle %d: ", ++cycle);
        cycle_timer.reset();    
        execute_cycle(&ctx, current_pattern, pattern_length);
        double cycle_time = cycle_timer.elapsed_ms();

        // Calculate actual metrics
        double total_memory_data = (double)ctx.total_memory_ops_count * ctx.memory_matrix_bytes * 2.0 * NUM_MEMORY_STREAMS;
        double total_bandwidth = total_memory_data / (cycle_time / 1000.0) / 1e9;
        double total_flops = (double)ctx.total_compute_ops_count * ctx.flops_per_gemm * NUM_COMPUTE_STREAMS;
        double total_gflops = total_flops / (cycle_time / 1000.0) / 1e9;
        
        printf("Total GPU Memory Ops: %d, Total Computation: %d, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
               ctx.total_memory_ops_count, ctx.total_compute_ops_count, total_bandwidth, total_gflops, cycle_time);
    }

    double total_time = total_timer.elapsed_ms();
    printf("\nTotal runtime: %.2f seconds\n", total_time / 1000.0);
}

int main(int argc, char* argv[]) {
    // Set GPU device
    cudaSetDevice(0);

    // Matrix sizes
    int memory_matrix_size = 2048;  // Size for GPU memory bandwidth operations
    int compute_matrix_size = 1024; // Size for pure compute operations

    if (argc >= 2) {
        memory_matrix_size = atoi(argv[1]);
    }
    if (argc >= 3) {
        compute_matrix_size = atoi(argv[2]);
    }
    
    printf("==================================================\n");
    printf("GPU Memory Bandwidth & Pure Compute Benchmark\n");

    // cudaProfilerStart();
    bursty_execution<precision_t>(memory_matrix_size, compute_matrix_size);
    // cudaProfilerStop();
    return 0;
}