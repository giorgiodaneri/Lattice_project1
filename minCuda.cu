#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <iostream>
#include <ctime>
// extern "C++"
// {
// #include "../../include/CGSolver.hpp"
// #include "../../include/CGSolverCuda.hpp"
// }

#define BLOCK_SIZE 1024

// write a kernel that computes the minimum of an array of integers
__global__ void findMinFixpointKernel(int *arr, int *n, int *min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // every threads processes tis own portion of the array based on its id
    // ensure that the index is within the bounds of the array
    if (idx < *n) {
        if(arr[idx] < *min) {
            *min = arr[idx];
        }
    }
}

__global__ void findMinKernel(int *arr, int *n, int *min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // every threads processes tis own portion of the array based on its id
    if (idx < *n) {
        // compute the minimum using a reduction pattern
        *min = arr[idx];
    }
}

int kernel_wrapper(std::vector<int> &arr)
{
    int *d_arr;
    int *d_min;
    int *d_size;
    int min_value = 1000;
    int prev_min_value = 1001;
    int iters = 0;
    int size = arr.size();
    // allocate memory on the device
    cudaError_t err;
    err = cudaMalloc((void **)&d_arr, arr.size() * sizeof(int));
    // get last error and check if it is not a success

    cudaMalloc((void **)&d_min, sizeof(int));
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error1: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc((void **)&d_size, sizeof(int));
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error2: %s\n", cudaGetErrorString(err));
    }
    // initialize all the device variables
    cudaMemcpy(d_arr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error3: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_min, &min_value, sizeof(int), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error4: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        printf("Error5: %s\n", cudaGetErrorString(err));
    }

    // measure time
    time_t start, end;
    start = clock();

    // loop until min_value converges to a fixpoint
    while(min_value < prev_min_value) {
        // update the previous value
        prev_min_value = min_value;
        // call the kernel
        findMinFixpointKernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(d_arr, d_size, d_min);
        if(cudaGetLastError() != cudaSuccess) {
            printf("Kernel Error: %s\n", cudaGetErrorString(err));
        }
        // copy the result back to the host
        cudaMemcpy(&min_value, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        if(cudaGetLastError() != cudaSuccess) {
            printf("Memcpy Error: %s\n", cudaGetErrorString(err));
        }
        iters++;
    }

    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "Time taken Cuda: " << time_taken << std::endl;

    // copy the result back to the host
    cudaMemcpy(&min_value, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_min);
    cudaFree(d_size);
    std::cout << "Number of iterations: " << iters << std::endl;
    return min_value;
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

int main() {
    int n = 1000000;
    // generate array of random integers of size n
    // Initialize a random number generator
    int min = 1;
    int max = 1000;
    // initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(min, max);

    std::vector<int> arr(n);
    for (int i = 0; i < n-1; i++) {
        arr[i] = distrib(gen);
    }
    // add artificial minimum 
    arr[n-1] = -21;
    // allocate memory on the devide
    int min_value = kernel_wrapper(arr);
    printf("Cuda computed minimum value is %d\n", min_value);

    time_t start, end;
    start = clock();
    int min_value_cpu = findMin(arr);
    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "Time taken CPU: " << time_taken << std::endl;
    printf("CPU computed minimum value is %d\n", min_value_cpu);
    return 0;
}