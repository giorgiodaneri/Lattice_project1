#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stack>  
#include <thread>

#define BLOCK_SIZE 16
#define SHARED_SIZE 128

struct Node {
    // vector of corresponding assigned values
    std::vector<int> assignedVals;
    // last variable that was assigned
    int branchedVar;

    Node( std::vector<int> vals, int var) : assignedVals(vals), branchedVar(var) {}
    Node(const Node& node) {
        assignedVals = node.assignedVals;
        branchedVar = node.branchedVar;
    }
    Node(Node&) = default;
    Node() {
        assignedVals = std::vector<int>();
        branchedVar = 0;
    }
    ~Node() = default;
};

__global__ void restrictDomainsKernel(
    const int* d_constraintsLeft, const int* d_constraintsRight, int* numConstraints,
    uint8_t* d_domains, const int* d_offsets, 
    const int* d_assignedVals, const int* d_branchVar) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if thread index exceeds number of constraints return
    if (tid >= *numConstraints) return; 
    // fetch the current constraint left and right variables
    int var1 = d_constraintsLeft[tid];
    int var2 = d_constraintsRight[tid];
    // check whether are already assigned
    bool isAssigned1 = var1 <= *d_branchVar;
    bool isAssigned2 = var2 <= *d_branchVar;
    // if both are assigned move to the next constraint, no point in checking this
    if (isAssigned1 && isAssigned2) return;
    // get the corresponding values, if any
    int assignedVal1 = isAssigned1 ? d_assignedVals[var1] : -1;
    int assignedVal2 = isAssigned2 ? d_assignedVals[var2] : -1;
    // compute domain start and end for both variables
    int start1 = d_offsets[var1];
    int end1 = d_offsets[var1 + 1];
    int start2 = d_offsets[var2];
    int end2 = d_offsets[var2 + 1];
    // if var1 is assigned and the value is in var2's domain, then remove it
    if (isAssigned1 && assignedVal1 >= 0 && start2 + assignedVal1 < end2) {
        d_domains[start2 + assignedVal1] = 0;
    }
    // analogous reasoning for var2
    if (isAssigned2 && assignedVal2 >= 0 && start1 + assignedVal2 < end1) {
        d_domains[start1 + assignedVal2] = 0;
    }
}

__global__ void restrictDomainsKernelShared(
    const int* d_constraintsLeft, const int* d_constraintsRight, int* numConstraints,
    uint8_t* d_domains, const int* d_offsets, 
    const int* d_assignedVals, const int* d_branchVar) 
{
    // use shared memory to store the constraints
    __shared__ int s_constraintsLeft[SHARED_SIZE];  
    __shared__ int s_constraintsRight[SHARED_SIZE]; 
    // global thread id and block thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blockTid = threadIdx.x;  
    // load constraints into shared memory, only for the current block
    if (blockTid < *numConstraints) {
        s_constraintsLeft[blockTid] = d_constraintsLeft[tid];
        s_constraintsRight[blockTid] = d_constraintsRight[tid];
    }

    // thread index exceeds number of constraints, exit
    if (tid >= *numConstraints) return;
    // fetch current constraint 
    int var1 = s_constraintsLeft[blockTid];
    int var2 = s_constraintsRight[blockTid];
    // same reasoning as the kernel using global memory
    bool isAssigned1 = var1 <= *d_branchVar;
    bool isAssigned2 = var2 <= *d_branchVar;
    if (isAssigned1 && isAssigned2) return;
    int assignedVal1 = isAssigned1 ? d_assignedVals[var1] : -1;
    int assignedVal2 = isAssigned2 ? d_assignedVals[var2] : -1;
    int start1 = d_offsets[var1];
    int end1 = d_offsets[var1 + 1];
    int start2 = d_offsets[var2];
    int end2 = d_offsets[var2 + 1];
    if (isAssigned1 && assignedVal1 >= 0 && start2 + assignedVal1 < end2) {
        d_domains[start2 + assignedVal1] = 0;
    }
    if (isAssigned2 && assignedVal2 >= 0 && start1 + assignedVal2 < end1) {
        d_domains[start1 + assignedVal2] = 0;
    }
}

// utility function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* file, int line) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << file << "' in line " << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// function to generate assignment of the varialbes, branch on them, find solutions and perform backtracking
void generate_and_branch(const std::vector<int>& constraintsLeft, const std::vector<int>& constraintsRight, int* d_constraintsLeft, int* d_constraintsRight, int* d_numConstraints, std::vector<std::vector<bool>> domains, uint8_t *d_domains, std::vector<uint8_t> flatDomains,
        std::vector<int> offsets, int* d_offsets, std::stack<Node>& nodes, int* d_assignedVals, int* d_branchVar, size_t &numSolutions, int n) {
    
    cudaError_t err;
    bool changed = false;
    std::vector<bool> singletons(n, 0);

    while(true)
    {
        singletons.clear();
        Node node = nodes.top();
        int currentOffset = 0;
        // declare a vector of bools to store the singleton domains
        // perform fixed point iteration to remove values from the domains
        if(node.branchedVar < n-1)
        {
            do {
                changed = false;
                // clear flatDomains
                flatDomains.clear();
                offsets.clear();
                currentOffset = 0;
                // flatten the domains using the offsets
                for (const auto& domain : domains) {
                    offsets.push_back(currentOffset);
                    flatDomains.insert(flatDomains.end(), domain.begin(), domain.end());
                    currentOffset += domain.size();
                }
                
                // copy the flatDomains to the device
                err = cudaMemcpyAsync(d_domains, flatDomains.data(), sizeof(uint8_t) * flatDomains.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
                err = cudaMemcpyAsync(d_assignedVals, node.assignedVals.data(), sizeof(int) * node.assignedVals.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
                err = cudaMemcpyAsync(d_branchVar, &node.branchedVar, sizeof(int), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
                // launch the kernel to restrict the domains
                restrictDomainsKernelShared<<<BLOCK_SIZE, BLOCK_SIZE>>>(d_constraintsLeft, d_constraintsRight, d_numConstraints, d_domains, d_offsets, d_assignedVals, d_branchVar);
                // copy the flatDomains back to the host
                err = cudaMemcpyAsync(flatDomains.data(), d_domains, sizeof(uint8_t) * flatDomains.size(), cudaMemcpyDeviceToHost); checkCudaError(err, __FILE__, __LINE__);
                // update the domains
                currentOffset = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < domains[i].size(); j++) {
                        domains[i][j] = flatDomains[currentOffset + j];
                    }
                    currentOffset += domains[i].size();
                }

                // find all the variables that have been forcefully assigned due to constraint application
                // and the relative values, so that we actually have a solution
                for(int i = node.branchedVar+1; i < n; i++) {
                    if(std::count(domains[i].begin(), domains[i].end(), 1) == 1) {
                        if(singletons[i] == 0) {
                            singletons[i] = 1;
                            changed = true;
                            // domain of the variable has only one value, corresponds to an assignment
                            // push as many -1 values as (i-branchedVar-1) to the assignedVals vector
                            // meaning that the variables between the last branched one and the current one
                            // have not been assigned. BranchedVar stays the same
                            for(int j = node.assignedVals.size(); j < i; j++) {
                                node.assignedVals.push_back(-1);
                            }
                            // if(node.assignedVals[i] >= 0) continue;
                            // node.assignedVals[i] = std::find(domains[i].begin(), domains[i].end(), 1) - domains[i].begin();
                            node.assignedVals.push_back(std::find(domains[i].begin(), domains[i].end(), 1) - domains[i].begin());
                        }
                    }
                }
            } while(changed);
        }

        // reached a fixed point, i.e. no more values can be removed from the domains

        // need to check if the current configuration is a solution, otherwise force
        // an assignment and create the corresponding new node
        if(node.branchedVar == n-1) {
            // print the solution
            // std::cout << "Solution: ";
            // for(int i = 0; i < n; i++) {
            //     std::cout << node.assignedVals[i] << " ";
            // }
            // std::cout << std::endl;

            // need to perform backtracking to find the next solution
            // get reference to the parent node and branch on the next value of the current variable
            // if there are still values in its domain
            // check if the domain of the variable is empty, if it is need to perform backtracking
            if(std::count(domains[node.branchedVar].begin(), domains[node.branchedVar].end(), 1) == 0)
            {      
                int depth = 0;
                nodes.pop();
                node = nodes.top();
                nodes.pop();
                // backtrack until a variable with a non-empty domain is found
                while(std::count(domains[node.branchedVar].begin(), domains[node.branchedVar].end(), 1) == 0) {
                    // check if the stack is emtpy, then all solutions have been found
                    if(nodes.size() == 0) 
                    {   
                        std::cout << "All solutions found" << std::endl;
                        return;
                    }
                    node = nodes.top();
                    nodes.pop();
                    depth++;
                }

                // reset the domains of all variables greater than branchedVar of current node
                for(int i = node.branchedVar+1; i < n; i++) {
                    domains[i] = std::vector<bool>(domains[i].size(), 1);
                }

                // modify the current node with the next valid value in the domain of the branched variable
                int nextVal = std::find(domains[node.branchedVar].begin(), domains[node.branchedVar].end(), 1) - domains[node.branchedVar].begin();
                domains[node.branchedVar][nextVal] = 0;
                node.assignedVals[node.branchedVar] = nextVal;
                nodes.push(node);
            } 
            else {
                nodes.pop();
                // get range of valid values in the domain of the variable
                int start = node.assignedVals.back()+1;
                int maxValue = domains[node.branchedVar].size();
                // iterate over all the remaining non zero values of the current branchedVar 
                // all the configurations are solutions
                numSolutions++;

                for(int i = start; i < maxValue; ++i) 
                {   
                    // remove the value from the domain of the variable
                    if(domains[node.branchedVar][i] == 1) {
                        numSolutions++;
                        domains[node.branchedVar][i] = 0;
                    }
                }
                node.assignedVals[node.branchedVar] = maxValue-1;
                nodes.push(node);
            }
        }   
        else {
            // since a fixed point has been reached but we do not have a solution yet
            // check if the next variable to branch has already an assigned value
            if(node.assignedVals.size() > node.branchedVar+1) {
                // check if the domain of the next variable to be assigned is empty
                if(std::count(domains[node.branchedVar+1].begin(), domains[node.branchedVar+1].end(), 1) == 0) {
                    // since it is impossible to find a valid value for the current variable, the current
                    // branch does not yields a solution => perform backtracking and find the next solution
                    // also, clean assignedVals for all variables greater than the current one
                    if(node.branchedVar+1 < n-1)
                    {   
                        for(int i = node.branchedVar+1; i < n; i++) {
                            node.assignedVals.pop_back();
                        }
                    }
                    node.branchedVar = n-1; 
                    nodes.push(node);
                    continue;   
                }
                // if it has an assigned value, just push the node to the stack
                if(node.assignedVals[node.branchedVar+1] != -1) {
                    node.branchedVar++;
                    domains[node.branchedVar][node.assignedVals[node.branchedVar]] = 0;
                    nodes.push(node);
                    if(node.branchedVar == n-1) {
                        numSolutions++;
                        numSolutions++;
                    }
                    continue;
                }
            }

            // if it does not have an assigned value,
            // we need to branch on the next variable by artifically imposing an assignment
            // remove the value from the domain of the branch variable in the parent node
            // find the first non-zero value in the domain of branchedVar+1
            int newVal = std::find(domains[node.branchedVar+1].begin(), domains[node.branchedVar+1].end(), 1) - domains[node.branchedVar+1].begin();
            // set this value to 0 in the parent domain
            domains[node.branchedVar+1][newVal] = 0;
            std::vector<int> assignedVals = node.assignedVals;
            if(assignedVals.size() > node.branchedVar+1) {
                assignedVals[node.branchedVar+1] = newVal;
            }
            else {
                assignedVals.push_back(newVal);
            }
            // select next variable to branch on as the one after branchedVar
            Node newNode(assignedVals, node.branchedVar+1);
            // and push the new node to the stack
            nodes.push(newNode);
        }
    }
}

void kernel_wrapper(std::vector<std::vector<bool>>& domains, const std::vector<int>& constraintsLeft, const std::vector<int>& constraintsRight, std::vector<int>& upperBounds, size_t& numSolutions)
{
    int n = upperBounds.size();
    cudaError_t err;

    // flatten the domains and prepare offsets to access them on the device
    // use uint8_t to store the domains
    std::vector<uint8_t> flatDomains;
    std::vector<int> offsets;
    int currentOffset = 0;

    for (const auto& domain : domains) {
        offsets.push_back(currentOffset);
        flatDomains.insert(flatDomains.end(), domain.begin(), domain.end());
        currentOffset += domain.size();
    }
    offsets.push_back(currentOffset);

    // number of constraints
    int numConstraints = constraintsLeft.size();

    // device variables
    uint8_t* d_domains;
    int* d_offsets;
    int* d_assignedVals;
    int* d_branchVar;
    int* d_constraintsLeft;
    int* d_constraintsRight;
    int* d_numConstraints;
    // allocate memory on the device for all necessary data structures
    err = cudaMalloc(&d_domains, sizeof(uint8_t) * flatDomains.size()); checkCudaError(err, __FILE__, __LINE__);    
    err = cudaMalloc(&d_offsets, sizeof(int) * offsets.size()); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_assignedVals, sizeof(int) * (n+10)); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_branchVar, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_constraintsLeft, sizeof(int) * constraintsLeft.size()); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_constraintsRight, sizeof(int) * constraintsRight.size()); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_numConstraints, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);  
    // copy constraints only once, since they never change
    err = cudaMemcpyAsync(d_constraintsLeft, constraintsLeft.data(), sizeof(int) * constraintsLeft.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_constraintsRight, constraintsRight.data(), sizeof(int) * constraintsRight.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_offsets, offsets.data(), sizeof(int) * offsets.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_numConstraints, &numConstraints, sizeof(int), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    // stack of nodes of the currently explored branch
    std::stack<Node> nodes;
    // vector to store the assigned values
    std::vector<int> assignedVals;
    assignedVals.push_back(0);
    // create root node that contains the initial domains, the first variable that 
    // will be branched is 0, we will proceed in increasing order
    Node root(assignedVals, 0);
    // remove the value from the domain of the variable
    domains[0][assignedVals[0]] = 0;
    nodes.push(root);

    generate_and_branch(constraintsLeft, constraintsRight, d_constraintsLeft, d_constraintsRight, d_numConstraints, domains, d_domains, flatDomains, offsets, d_offsets, nodes, d_assignedVals, d_branchVar, numSolutions, n);

    // free the memory
    checkCudaError(cudaFree(d_domains), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_offsets), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_constraintsLeft), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_constraintsRight), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_numConstraints), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_assignedVals), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_branchVar), __FILE__, __LINE__);
}