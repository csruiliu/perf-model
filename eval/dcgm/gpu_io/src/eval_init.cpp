#include <stdio.h>
#include <thread>
#include <chrono>

#include <cuda_runtime.h>

int main(int argc, char *argv[]) {

  // Initialize CUDA context (no explicit init function in CUDA)
  // Lazy context initialization
  // using cudaFree(0) after a cudaSetDevice() forces CUDA runtime to establish a context
  cudaError_t err_set = cudaSetDevice(0);
  cudaError_t err_free = cudaFree(0);
  if (err_set != cudaSuccess) {
      fprintf(stderr, "CUDA set device failed: %s\n", cudaGetErrorString(err_set));
      return 1;
  }

  if (err_free != cudaSuccess) {
    fprintf(stderr, "CUDA free failed: %s\n", cudaGetErrorString(err_free));
    return 1;
  }

  std::this_thread::sleep_for(std::chrono::seconds(10));

  // Destroy all allocations and reset all state on the current device in the current process.
  cudaDeviceReset();

  return 0;
}

