#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
// extern "C++"
// {
// #include "../../include/CGSolver.hpp"
// #include "../../include/CGSolverCuda.hpp"
// }


void kernel_wrapper(std::vector<int> arr)
{
    int *d_arr;
    cudaMalloc(&d_arr, arr.size() * sizeof(int));
    cudaMemcpy(d_arr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    // CGSolverCuda solver;
    // solver.solve(d_arr, arr.size());
    cudaFree(d_arr);
}