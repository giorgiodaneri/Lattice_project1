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
__global__ void excludeValuesKernel(char* domains, int* domainSizes, removedVal* excludedValues, int* excludedCount,
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
        if (domainSizes[affectedVar] > newPos && domains[affectedVar * numVariables + newPos] == 1) {
            // Exclude the value
            domains[affectedVar * numVariables + newPos] = false;

            // Add to excludedValues (atomic operation)
            int pos = atomicAdd(&localCount, 1);
            excludedValues[pos] = removedVal(newPos, depth, affectedVar);
        }
    } else if ((var2 == depth && var1 > depth))
    {
        int affectedVar = var1;

        // Check if value is in the domain of the affected variable
        if (domainSizes[affectedVar] > newPos && domains[affectedVar * numVariables + newPos] == 1) {
            // Exclude the value
            domains[affectedVar * numVariables + newPos] = false;

            // Add to excludedValues (atomic operation)
            int pos = atomicAdd(&localCount, 1);
            excludedValues[pos] = removedVal(newPos, depth, affectedVar);
        }
    }
    __syncthreads();

    // if (threadIdx.x == 0) atomicAdd(excludedCount, localCount);
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
void excludeValues(std::vector<std::vector<bool>>& domains, std::vector<removedVal>& excludedValues, int depth, int newPos, const std::vector<std::pair<int, int>>& constraints) {
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

void reinsertValues(std::vector<std::vector<bool>>& domains, std::vector<removedVal>& excludedValues, int depth) {
    while(!excludedValues.empty() && excludedValues.back().depth == depth) {
        int value = excludedValues.back().value;
        int var = excludedValues.back().var;
        domains[var][value] = true;
        excludedValues.pop_back();
    }
}

void generateAndBranch(const Node& parent, const std::vector<std::pair<int, int>>& constraints, 
    const std::vector<int>& upperBounds, std::vector<removedVal>& excludedValues, std::vector<std::vector<bool>>& domains, int& numSolutions, int numVariables) {
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

void checkCudaError(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void kernel_wrapper(std::vector<std::vector<bool>>& domains, const std::vector<std::pair<int, int>>& constraints, const std::vector<int>& upperBounds, int& numSolutions)
{
    int numVariables = domains.size();
    int numConstraints = constraints.size();
    int maxDomainSize = domains[0].size();
    for(int i = 1; i < numVariables; i++) {
        if(domains[i].size() > maxDomainSize) {
            maxDomainSize = domains[i].size();
        }
    }
    cudaError_t err;
    // use an array of structs to store the excluded values
    // the value will be reinserted in the domain once backtracking reached depth-1
    // since the corresponding value of the assigned variable, which violated the constraint, will be changed
    // due to visiting another branch in the tree
    std::vector<removedVal> excludedValues;

    // Flatten domains for device
    std::vector<char> flatDomains(numVariables * maxDomainSize, 0);
    std::vector<int> domainSizes(numVariables, 0);
    for(int i = 0; i < numVariables; i++) {
        domainSizes[i] = domains[i].size();
        for(int j = 0; j < domains[i].size(); j++) {
            flatDomains[i * maxDomainSize + j] = domains[i][j];
        }
    }

    // Allocate device memory
    char* d_domains;
    int* d_domainSizes;
    int* d_depth;
    int* d_newPos;
    removedVal* d_excludedValues;
    int* d_excludedCount;
    std::pair<int, int>* d_constraints;

    err = cudaMalloc(&d_domains, sizeof(char) * flatDomains.size());
    checkCudaError(err);
    err = cudaMalloc(&d_domainSizes, sizeof(int) * domainSizes.size());
    checkCudaError(err);
    err = cudaMalloc(&d_depth, sizeof(int));
    checkCudaError(err);
    err = cudaMalloc(&d_newPos, sizeof(int));
    checkCudaError(err);
    err = cudaMalloc(&d_excludedValues, sizeof(removedVal) * excludedValues.size());
    checkCudaError(err);
    err = cudaMalloc(&d_excludedCount, sizeof(int));
    checkCudaError(err);
    err = cudaMalloc(&d_constraints, sizeof(std::pair<int, int>) * constraints.size());
    checkCudaError(err);

    // Copy data to device
    err = cudaMemcpy(d_domains, flatDomains.data(), sizeof(char) * flatDomains.size(), cudaMemcpyHostToDevice);
    checkCudaError(err);
    err = cudaMemcpy(d_domainSizes, domainSizes.data(), sizeof(int) * domainSizes.size(), cudaMemcpyHostToDevice);
    checkCudaError(err);
    err = cudaMemcpy(d_constraints, constraints.data(), sizeof(std::pair<int, int>) * constraints.size(), cudaMemcpyHostToDevice);
    checkCudaError(err);
    err = cudaMemset(d_excludedCount, 0, sizeof(int));
    checkCudaError(err);

    // Reconstruct domains from flatDomains
    for (int i = 0; i < numVariables; i++) {
    // Resize domains[i] to match domainSizes[i]
    domains[i].resize(domainSizes[i]);

        for (int j = 0; j < domainSizes[i]; j++) {
            // Assign value from flatDomains to domains
            domains[i][j] = flatDomains[i * maxDomainSize + j];
        }
    }   
    // unroll first iteration of the recursion, since the dummy root node is already created
    // and all the nodes that correspond to the first variable need not increase depth
    // since they are found at the first level of the tree
    for(int i = 0; i <= upperBounds[0]; i++) {
        // do not check constraints for the first variable, since all the others have not been initialized yet
        Node child(numVariables);
        // place the previous variable at the valid position that has been computed
        // increase depth and prepare for calculation of the current node possible positions
        child.positions[child.depth] = i;
        cudaMemcpy(d_depth, &child.depth, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_newPos, &i, sizeof(int), cudaMemcpyHostToDevice);
        // excludeValues(domains, excludedValues, child.depth, i, constraints);
        excludeValuesKernel<<<64, 8>>>(d_domains, d_domainSizes, d_excludedValues, d_excludedCount, d_depth, d_newPos, d_constraints, numConstraints, numVariables);
        // copy domain back to host
        err = cudaMemcpy(flatDomains.data(), d_domains, sizeof(char) * flatDomains.size(), cudaMemcpyDeviceToHost);
        checkCudaError(err);
        // copy excluded values back to host
        err = cudaMemcpy(excludedValues.data(), d_excludedValues, sizeof(removedVal) * excludedValues.size(), cudaMemcpyDeviceToHost);
        checkCudaError(err);
        // copy flatDomains back to domains
        for (int k = 0; k < numVariables; k++) {
            for (int j = 0; j < domainSizes[k]; j++) {
                // Assign value from flatDomains to domains
                domains[k][j] = flatDomains[k * maxDomainSize + j];
            }
        }
        // excludeValues(domains, excludedValues, child.depth, i, constraints);
        // print excluded values
        while(!excludedValues.empty()) {
            removedVal val = excludedValues.back();
            std::cout << "Excluded value: " << val.value << " of variable " << val.depth << " from variable " << val.var << std::endl;
            excludedValues.pop_back();
        }

        generateAndBranch(child, constraints, upperBounds, excludedValues, domains, numSolutions, numVariables);
        reinsertValues(domains, excludedValues, child.depth);
    }

    // free the memory
    cudaFree(d_domains);
    cudaFree(d_excludedValues);
    cudaFree(d_constraints);
    cudaFree(d_domainSizes);
    cudaFree(d_excludedCount);
}