#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <iostream>
#include <ctime>
#include <omp.h>

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024

// write a kernel that computes the minimum of an array of integers
__global__ void findMinFixpointKernel(int *arr, int *size, int *min) {
    // global thread identifier
    unsigned int unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    // thread identifier within the block (used to access share memory)
    unsigned int thread_id = threadIdx.x;
    // declare chunk of shared memory
    __shared__ int minChunk[BLOCK_SIZE];

    // load elements into shared memory only if within bounds
    if (unique_id < *size) {
        minChunk[thread_id] = arr[unique_id];
    } else {
        // make sure that values out of bounds are set to a large value
        minChunk[thread_id] = INT_MAX;  
    }

    // update the global minimum if a smaller value is found within the block
    if(thread_id < *size && minChunk[thread_id] < *min) {
        *min = minChunk[thread_id];
    }
}

__global__ void findMinKernel(int *arr, int *size, int *min) {
    unsigned int unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thread_id = threadIdx.x;
    __shared__ int minChunk[BLOCK_SIZE];

    // load elements into shared memory only if within bounds
    if (unique_id < *size) {
        minChunk[thread_id] = arr[unique_id];
    } else {
        // make sure that values out of bounds are set to a large value
        minChunk[thread_id] = INT_MAX;  
    }

    __syncthreads();

    // perform reduction to find the minimum in the current block
    // this has complexity O(log(n)) where n is the number of elements in the block
    // since the reduction pattern amounts to organizing the elements in a binary tree
    // the stride is reduced by half at each iteration, and memory accesses are more coalesced
    // at the later iterations
    # pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            if (minChunk[thread_id] > minChunk[thread_id + s]) {
                minChunk[thread_id] = minChunk[thread_id + s];
            }
        }
        __syncthreads();
    }

    // perform atomic update of global minimum from local minima of each block
    // which is found in the first element of the shared memory (after reduction)
    if (thread_id == 0) {
        atomicMin(min, minChunk[0]);
    }
}


void kernel_wrapper(std::vector<int> &arr)
{
    int *d_arr;
    int *d_min;
    int *d_size;
    int min_value = 1000;
    int prev_min_value = 1001;
    int iters = 0;
    int size = arr.size();

    // measure time for memory allocation and data transfer
    cudaError_t err;
    time_t start_mem, end_mem;
    start_mem = clock();
    // initialize cuda context
    cudaFree(0);
    // allocate memory on the device
    err = cudaMalloc((void **)&d_arr, arr.size() * sizeof(int));
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error0: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **)&d_min, sizeof(int));
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error1: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **)&d_size, sizeof(int));
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error2: %s\n", cudaGetErrorString(err));
    }
    // initialize all the device variables
    err = cudaMemcpy(d_arr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error3: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_min, &min_value, sizeof(int), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error4: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error5: %s\n", cudaGetErrorString(err));
    }
    end_mem = clock();
    double time_taken_mem = double(end_mem - start_mem) / double(CLOCKS_PER_SEC);
    std::cout << "Time taken for memory allocation: \n" << time_taken_mem << std::endl;

    // ----------------- FIXPOINT MODEL ----------------- //
    time_t start, end;
    start = clock();
    // loop until min_value converges to a fixpoint => does not change between iterations
    while(min_value < prev_min_value) {
        // update the previous value
        prev_min_value = min_value;
        findMinFixpointKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_arr, d_size, d_min);
        if(cudaGetLastError() != cudaSuccess) {
            printf("Fixpoint kernel Error: %s\n", cudaGetErrorString(err));
        }
        // copy the result back to the host for comparison
        err = cudaMemcpy(&min_value, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaGetLastError() != cudaSuccess) {
            printf("Memcpy Error: %s\n", cudaGetErrorString(err));
        }
        iters++;
    }
    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "Time taken by fixpoint iteration: " << time_taken << std::endl;
    std::cout << "Number of fixpoint iterations: " << iters << std::endl;
    printf("Fixpoint computed minimum value is %d\n", min_value);

    // ----------------- PARALLEL REDUCTION KERNEL ----------------- //
    // measure time
    start = clock();
    // reset min_value
    min_value = 1000;
    err = cudaMemcpy(d_min, &min_value, sizeof(int), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error6: %s\n", cudaGetErrorString(err));
    }
    findMinKernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(d_arr, d_size, d_min);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Reduction kernel Error: %s\n", cudaGetErrorString(err));
    }
    // copy the result back to the host
    err = cudaMemcpy(&min_value, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Memcpy Error: %s\n", cudaGetErrorString(err));
    }
    end = clock();
    time_taken = double(end - start) / double(CLOCKS_PER_SEC);

    // copy the result back to the host
    cudaMemcpy(&min_value, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_min);
    cudaFree(d_size);

    std::cout << "Time taken by parallel reduction: " << time_taken << std::endl;
    std::cout << "Parallel reduction computed minimum value is " << min_value << std::endl;
}

// serial CPU function for comparison
int findMin(std::vector<int> arr) {
    int min = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

int main(int argc, char **argv) {
    // read the size of the array from the command line
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " <input dimension> " << std::endl;
        exit(1);
    }
    int n = atoi(argv[1]);

    // generate array of random integers of size n
    // Initialize a random number generator
    int min = 2;
    int max = 1000;
    // initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(min, max);
    std::vector<int> arr(n);
    
    // if openmp is enabled, set the number of threads
    # ifdef _OPENMP
    omp_set_num_threads(32);
    // print number of active omp threads
    std::cout << "Number of active threads: " << omp_get_max_threads() << std::endl;
    # endif

    # pragma omp parallel for
    for (int i = 0; i < n; i++) {
        arr[i] = distrib(gen);
    }
    // allocate memory on the devide
    kernel_wrapper(arr);

    time_t start, end;
    start = clock();
    int min_value_cpu = findMin(arr);
    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "Time taken CPU: " << time_taken << std::endl;
    printf("CPU computed minimum value is %d\n", min_value_cpu);
    return 0;
}