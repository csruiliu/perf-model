#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>


// Host memory allocation wrapper
void* host_alloc(size_t size, bool use_pinned) {
    if (use_pinned) {
        void* ptr;
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "Pinned host allocation failed: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        return ptr;
    } else {
        return malloc(size);
    }
}


void host_free(void* ptr, bool use_pinned) {
    if (use_pinned) {
        cudaFreeHost(ptr);
    } else {
        free(ptr);
    }
}


// Timer class for accurate timing measurements
struct Timer {
    bool use_cuda_events;
    cudaEvent_t start_event, stop_event;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_chrono, stop_chrono;

    Timer(bool use_cuda) : use_cuda_events(use_cuda) {
        if (use_cuda_events) {
            cudaEventCreate(&start_event);
            cudaEventCreate(&stop_event);
        }
    }

    ~Timer() {
        if (use_cuda_events) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
    }

    void record_start() {
        if (use_cuda_events) {
            cudaEventRecord(start_event);
        } else {
            start_chrono = std::chrono::high_resolution_clock::now();
        }
    }

    void record_stop() {
        if (use_cuda_events) {
            cudaEventRecord(stop_event);
        } else {
            stop_chrono = std::chrono::high_resolution_clock::now();
        }
    }
    
    double elapsed() {
        if (use_cuda_events) {
            cudaEventSynchronize(stop_event);
            float ms;
            cudaEventElapsedTime(&ms, start_event, stop_event);
            return ms / 1000.0;  // Convert to seconds
        } else {
            return std::chrono::duration<double>(stop_chrono - start_chrono).count();
        }
    }
};


void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --size <GB>          Data size in GB (default: 32)" << std::endl;
    std::cout << "  --copies <N>         Number of transfers (default: 10)" << std::endl;
    std::cout << "  --pinned <0|1>       Use pinned memory: 0=pageable, 1=pinned (default: 1)" << std::endl;
    std::cout << "  --cuda-events <0|1>  Timing method: 0=chrono, 1=CUDA events (default: 1)" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << program_name << " --size 16 --copies 20 --pinned 1 --cuda-events 1" << std::endl;
}


int main(int argc, char** argv) {
    // Default parameters
    size_t size_gb = 32;
    int num_copies = 10;
    bool use_pinned = false;
    bool use_cuda_events = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size_gb = std::atol(argv[++i]);
        } else if (strcmp(argv[i], "--copies") == 0 && i + 1 < argc) {
            num_copies = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pinned") == 0 && i + 1 < argc) {
            use_pinned = (std::atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--cuda-events") == 0 && i + 1 < argc) {
            use_cuda_events = (std::atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            print_usage(argv[0]);
            return -1;
        }
    }

    const size_t SIZE = size_gb * 1024UL * 1024 * 1024;

    // Validate parameters
    if (size_gb == 0 || num_copies <= 0) {
        std::cerr << "Error: Invalid parameters" << std::endl;
        print_usage(argv[0]);
        return -1;
    }

    // Print configuration
    std::cout << "=== PCIe Bandwidth Benchmark ===" << std::endl;
    std::cout << "Data size: " << size_gb << " GB (" << SIZE << " bytes)" << std::endl;
    std::cout << "Number of transfers: " << num_copies << std::endl;
    std::cout << "Memory type: " << (use_pinned ? "Pinned (page-locked)" : "Pageable") << std::endl;
    std::cout << "Timing method: " << (use_cuda_events ? "CUDA Events" : "std::chrono") << std::endl;
    std::cout << std::endl;

    // Set CUDA device
    cudaSetDevice(0);

    // Get and print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl << std::endl;

    // Allocate host memory
    void* x_h = host_alloc(SIZE, use_pinned);
    if (!x_h) {
        std::cerr << "Host allocation failed" << std::endl;
        return -1;
    }

    // Initialize host memory (optional but recommended for realistic testing)
    memset(x_h, 0, SIZE);

    // Allocate device memory
    void* x_d;
    cudaMalloc(&x_d, SIZE);

    // Warm-up transfer (important for accurate measurements)
    std::cout << "Performing warm-up transfer..." << std::endl;
    cudaMemcpy(x_d, x_h, SIZE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    std::cout << "Warm-up complete\n" << std::endl;

    // Benchmark loop
    Timer timer(use_cuda_events);
    double total_time = 0.0;
    double min_bandwidth = 1e9;
    double max_bandwidth = 0.0;
    
    std::cout << "Starting benchmark..." << std::endl;
    for (int i = 0; i < num_copies; ++i) {
        timer.record_start();
        cudaMemcpy(x_d, x_h, SIZE, cudaMemcpyHostToDevice);
        timer.record_stop();

        const double copy_time = timer.elapsed();
        total_time += copy_time;

        const double bandwidth = (SIZE / (1024.0 * 1024 * 1024)) / copy_time;
        min_bandwidth = std::min(min_bandwidth, bandwidth);
        max_bandwidth = std::max(max_bandwidth, bandwidth);

        printf("Transfer %2d: %7.2f GB/s (%.3f s)\n", i + 1, bandwidth, copy_time);
    }

    // Print summary statistics
    std::cout << "\n=== Results ===" << std::endl;
    printf("Total time:        %.3f s\n", total_time);
    printf("Average bandwidth: %.2f GB/s\n", (num_copies * SIZE / (1024.0 * 1024 * 1024)) / total_time);
    printf("Min bandwidth:     %.2f GB/s\n", min_bandwidth);
    printf("Max bandwidth:     %.2f GB/s\n", max_bandwidth);

    // Cleanup
    cudaFree(x_d);
    host_free(x_h, use_pinned);
    cudaDeviceReset();

    return 0;
}  