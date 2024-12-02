#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stack>   

// Node
struct Node {
    int  depth = 0;
    // positions  configuration (permutation)
    std::vector<int> positions; 

    Node(int  N): positions (N) {
        for (int i = 0; i < N; ++i) {
            if(i >= depth) {
                positions[i] = -1;
            }
            else {
                positions[i] = 0;
            }
        }
    }
    Node(const Node&) = default;
    Node(Node&&) = default;
    Node() = default;
};

struct removedVal {
    int value;
    int depth;
    int var;

    __host__ __device__ removedVal() = default;
    __host__ __device__ removedVal(int val, int d, int v): value(val), depth(d), var(v) {}
};

// kernel to exclude values from the domain of the variables
__global__ void excludeValuesKernel(char* domains, int* offsets, removedVal* excludedValues, int* excludedCount,
                                    int* d_depth, int* d_newPos, const std::pair<int, int>* constraints, int numConstraints, int numVariables) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = *d_depth;
    int newPos = *d_newPos;
    if (idx >= numConstraints) return;

    int var1 = constraints[idx].first;
    int var2 = constraints[idx].second;

    // Shared memory for atomic writes to the excludedValues array
    __shared__ int localCount;

    if (threadIdx.x == 0) localCount = 0;
    __syncthreads();

    if (var1 < depth && var2 < depth) return;

    if ((var1 == depth && var2 > depth)) {
        int affectedVar = var2;

        // Check if value is in the domain of the affected variable
        if (offsets[affectedVar] > newPos && domains[offsets[affectedVar] + newPos] == 1) {
            // Exclude the value
            domains[offsets[affectedVar] + newPos] = false;

            // Add to excludedValues (atomic operation)
            int pos = atomicAdd(&localCount, 1);
            excludedValues[pos+*excludedCount] = removedVal(newPos, depth, affectedVar);
        }
    } else if ((var2 == depth && var1 > depth))
    {
        int affectedVar = var1;

        // Check if value is in the domain of the affected variable
        if (offsets[affectedVar] > newPos && domains[offsets[affectedVar] + newPos] == 1) {
            // Exclude the value
            domains[offsets[affectedVar] + newPos] = false;

            // Add to excludedValues (atomic operation)
            int pos = atomicAdd(&localCount, 1);
            excludedValues[pos+*excludedCount] = removedVal(newPos, depth, affectedVar);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(excludedCount, localCount);
}


bool isValid(const std::vector<int>& positions, const std::vector<std::pair<int, int>>& constraints, int depth, int newPos) {
    // check constraints only for variables that have been placed so far
    for (auto& constraint : constraints) {
        int var1 = constraint.first;
        int var2 = constraint.second;
        
        // skip constraints that involve variables beyond the current depth+1
        // since we are checking whether this configuration for the node at depth+1 is valid
        // before creating it
        if (var2 > (depth+1) || var1 > (depth+1)) continue;

        // check if this constraint is violated
        if ((positions[var2] == newPos && var1 == (depth+1)) || (positions[var1] == newPos && var2 == (depth+1))) {
                return false;
        }
    }
    return true;
}

// function to exclude all values that are incompatible with the latest variable assignment
// and store them in a stack to reinsert them in the domain once backtracking reached depth-1
void excludeValues(std::vector<std::vector<char>>& domains, std::vector<removedVal>& excludedValues, int depth, int newPos, const std::vector<std::pair<int, int>>& constraints) {
    for (auto& constraint : constraints) {
        int var1 = constraint.first;
        int var2 = constraint.second;

        if(var1 < depth && var2 < depth) {
            continue;
        }
        
        if(var1 == depth && var2 > depth)
        {
            // only add value to the stack if it is inside the domain of the other variable
           // and if it has not been already excluded 
            if(domains[var2].size() > newPos && domains[var2][newPos]) {
                // pair is (value, depth), third element is the depth of the variable that has been assigned the value
                removedVal val(newPos, depth, var2);
                excludedValues.push_back(val);
                // check if the domain of var2 is greater than newPos
                domains[var2][newPos] = false;
            }
        }
        else if(var2 == depth && var1 > depth)
        {
            if(domains[var1].size() > newPos && domains[var1][newPos]) {
                // create a new removedVal struct and push it to the array of structs
                removedVal val(newPos, depth, var1);
                excludedValues.push_back(val);
                // check if the domain of var1 is greater than newPos
                domains[var1][newPos] = false;
            }
        }
    }
}

void reinsertValues(std::vector<std::vector<char>>& domains, std::vector<removedVal>& excludedValues, int depth) {
    while(!excludedValues.empty() && excludedValues.back().depth == depth) {
        int value = excludedValues.back().value;
        int var = excludedValues.back().var;
        domains[var][value] = true;
        excludedValues.pop_back();
        std::cout << "Reinserted value: " << value << " of variable " << var << " at depth " << depth << std::endl;
        if(depth == 0)
        {
            std::cout << "Excluded value depth: " << excludedValues.back().depth << std::endl;
        }
    }
}

void generateAndBranch(const Node& parent, const std::vector<std::pair<int, int>>& constraints, 
    const std::vector<int>& upperBounds, std::vector<removedVal>& excludedValues, std::vector<std::vector<char>>& domains, int& numSolutions, int numVariables) {
    // reached a leaf node, all constraints are satisfied
    if(parent.depth == (numVariables-1)) {
        numSolutions++;
        // std::cout << "Configuration: ";
        //     for(int j = 0; j < numVariables; j++) {
        //         std::cout << parent.positions[j] << " ";
        //     }
        // std::cout << std::endl;
    }
    else {
        // generate a new node for each possible position of the current variable
        // compatibly with its upper bound => do not generate nodes that violate it
        // iterate over the values in the domain of the current variable 
        for(int i = 0; i < domains[parent.depth+1].size(); i++) {
            // if(domains[parent.depth+1][i])
            if(domains[parent.depth+1][i] && isValid(parent.positions, constraints, parent.depth, i)) {
                Node child(parent);
                // place the previous variable at the valid position that has been computed
                // increase depth and prepare for calculation of the current node possible positions
                child.depth++;
                child.positions[child.depth] = i;
                excludeValues(domains, excludedValues, child.depth, i, constraints);
                // check if top element of the stack has the same depth as the current node
                // if so, reinsert the value in the domain
                generateAndBranch(child, constraints, upperBounds, excludedValues, domains, numSolutions, numVariables);
                reinsertValues(domains, excludedValues, child.depth);
            } 
        }
    }
}

void checkCudaError(cudaError_t err, const char* file, int line) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << file << "' in line " << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void kernel_wrapper(std::vector<std::vector<char>>& domains, const std::vector<std::pair<int, int>>& constraints, const std::vector<int>& upperBounds, int& numSolutions)
{
    int numVariables = domains.size();
    int numConstraints = constraints.size();
    cudaError_t err;
    // use an array of structs to store the excluded values
    // the value will be reinserted in the domain once backtracking reached depth-1
    // since the corresponding value of the assigned variable, which violated the constraint, will be changed
    // due to visiting another branch in the tree
    std::vector<removedVal> excludedValues;
    int excludedCount = -1;


    // Flatten domains and prepare offsets
    std::vector<char> flatDomains;
    std::vector<int> offsets;
    int currentOffset = 0;

    for (const auto& domain : domains) {
        offsets.push_back(currentOffset);
        flatDomains.insert(flatDomains.end(), domain.begin(), domain.end());
        currentOffset += domain.size();
    }
    offsets.push_back(currentOffset);

    // Allocate device memory
    char* d_domains;
    int* d_offsets;
    int* d_depth;
    int* d_newPos;
    removedVal* d_excludedValues;
    int* d_excludedCount;
    std::pair<int, int>* d_constraints;

    err = cudaMalloc(&d_domains, sizeof(char) * flatDomains.size()); checkCudaError(err, __FILE__, __LINE__);    
    err = cudaMalloc(&d_offsets, sizeof(int) * offsets.size()); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_depth, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_newPos, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_excludedValues, sizeof(removedVal) * flatDomains.size()); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_excludedCount, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_constraints, sizeof(std::pair<int, int>) * constraints.size()); checkCudaError(err, __FILE__, __LINE__);
    // Copy data to device
        err = cudaMemcpy(d_domains, flatDomains.data(), sizeof(char) * flatDomains.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * offsets.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_constraints, constraints.data(), sizeof(std::pair<int, int>) * constraints.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemset(d_excludedCount, 0, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);
  
    // unroll first iteration of the recursion, since the dummy root node is already created
    // and all the nodes that correspond to the first variable need not increase depth
    // since they are found at the first level of the tree
    for(int i = 0; i <= upperBounds[0]; i++) {
        // do not check constraints for the first variable, since all the others have not been initialized yet
        Node child(numVariables);
        // place the previous variable at the valid position that has been computed
        // increase depth and prepare for calculation of the current node possible positions
        child.positions[child.depth] = i;
        err = cudaMemcpy(d_depth, &child.depth, sizeof(int), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_newPos, &i, sizeof(int), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_excludedValues, excludedValues.data(), sizeof(removedVal) * excludedValues.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
        // excludeValues(domains, excludedValues, child.depth, i, constraints);
        excludeValuesKernel<<<64, 8>>>(d_domains, d_offsets, d_excludedValues, d_excludedCount, d_depth, d_newPos, d_constraints, numConstraints, numVariables);
        cudaDeviceSynchronize();
        // Copy updated domains and excluded values back to the host
        err = cudaMemcpy(flatDomains.data(), d_domains, sizeof(char) * flatDomains.size(), cudaMemcpyDeviceToHost); checkCudaError(err, __FILE__, __LINE__);
        // Get the count of excluded values
        err = cudaMemcpy(&excludedCount, d_excludedCount, sizeof(int), cudaMemcpyDeviceToHost); checkCudaError(err, __FILE__, __LINE__);
        // Resize the host excludedValues vector and copy data
        std::vector<removedVal> excludedValues(excludedCount);
        err = cudaMemcpy(excludedValues.data(), d_excludedValues, sizeof(removedVal) * excludedCount, cudaMemcpyDeviceToHost); checkCudaError(err, __FILE__, __LINE__);
        // print current iter
        std::cout << "Iter: " << i << std::endl;
        // Reconstruct domains from flatDomains
        for (int i = 0; i < numVariables; i++) {
            domains[i].resize(offsets[i + 1] - offsets[i]);  // Ensure the domain is correctly sized
            for (int j = offsets[i]; j < offsets[i + 1]; j++) {
                domains[i][j - offsets[i]] = flatDomains[j];
            }
        }

        // print domains 
        // for (int i = 0; i < domains.size(); i++) {
        //     for (int j = 0; j < domains[i].size(); j++) {
        //         std::cout << (int)domains[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Print excluded values for debugging
        // for (const auto& val : excludedValues) {
        //     std::cout << "Excluded value: " << val.value
        //             << ", Depth: " << val.depth
        //             << ", Variable: " << val.var 
        //             << " at iter: " << i << std::endl;
        // }

        generateAndBranch(child, constraints, upperBounds, excludedValues, domains, numSolutions, numVariables);
        reinsertValues(domains, excludedValues, child.depth);
    }

    // print excluded values
    while(!excludedValues.empty()) {
        std::cout << "Excluded value: " << excludedValues.back().value << " of variable " << excludedValues.back().var << " from variable " << excludedValues.back().depth << std::endl;
        excludedValues.pop_back();
    }

    std::cout << "Domains after kernel call: " << std::endl;
    for (int i = 0; i < domains.size(); i++) {
        for (int j = 0; j < domains[i].size(); j++) {
            std::cout << (int)domains[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // free the memory
    cudaFree(d_domains);
    cudaFree(d_excludedValues);
    cudaFree(d_constraints);
    cudaFree(d_excludedCount);
    cudaFree(d_offsets);
    cudaFree(d_depth);
    cudaFree(d_newPos);
}