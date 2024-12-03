#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stack>  
#include <thread>

#define BLOCK_SIZE 64

struct Node {
    // vector of assigned variables, which will be empty at the beginning
    std::vector<int> assignedVars;
    // vector of corresponding assigned values
    std::vector<int> assignedVals;
    int branchedVar;

    Node(std::vector<int> vars, std::vector<int> vals, int var) : assignedVars(vars), assignedVals(vals), branchedVar(var) {}
    Node(const Node& node) {
        assignedVars = node.assignedVars;
        assignedVals = node.assignedVals;
        branchedVar = node.branchedVar;
    }
    Node(Node&) = default;
    Node() {
        assignedVars = std::vector<int>();
        assignedVals = std::vector<int>();
        branchedVar = 0;
    }
    ~Node() = default;
};

__global__ void restrictDomainsKernel(
    const std::pair<int, int>* d_constraints, int* numConstraints,
    uint8_t* d_domains, const int* d_offsets, 
    const int* d_assignedVals, const int* d_branchVar) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= *numConstraints) return; // Exit if thread index exceeds number of constraints

    // Fetch the current constraint
    auto constraint = d_constraints[tid];
    int var1 = constraint.first;
    int var2 = constraint.second;

    // Check if var1 and var2 are assigned
    bool isAssigned1 = var1 <= *d_branchVar;
    bool isAssigned2 = var2 <= *d_branchVar;

    // If both variables are assigned, move to the next constraint
    if (isAssigned1 && isAssigned2) return;

    // Get assigned values if any, using pointer arithmetic
    int assignedVal1 = isAssigned1 ? (d_assignedVals[var1]) : -1;
    int assignedVal2 = isAssigned2 ? (d_assignedVals[var2])  : -1;

    // Compute domain start and end for var1 and var2
    int start1 = d_offsets[var1];
    int end1 = d_offsets[var1 + 1];
    int start2 = d_offsets[var2];
    int end2 = d_offsets[var2 + 1];

    // If var1 is assigned and the value is in var2's domain, remove it
    // do not bother checking whether it is already set to 0, set it to 0 anyways
    if (isAssigned1 && assignedVal1 >= 0 && start2 + assignedVal1 < end2) {
        d_domains[start2 + assignedVal1] = 0;
    }

    // If var2 is assigned and the value is in var1's domain, remove it
    if (isAssigned2 && assignedVal2 >= 0 && start1 + assignedVal2 < end1) {
        d_domains[start1 + assignedVal2] = 0;
    }
}

void checkCudaError(cudaError_t err, const char* file, int line) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << file << "' in line " << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

bool restrict_domains(const std::vector<std::pair<int, int>> &constraints, std::vector<std::vector<bool>> &domains, Node &node) 
{   
    bool changed = false;
    // iterate over the constraints
    for(auto& constraint : constraints) {
        int var1 = constraint.first;
        int var2 = constraint.second;
        bool isAssigned1 = false;
        bool isAssigned2 = false;
        // check if var1 and var2 are inside the assigned variables
        if(node.assignedVars.size() > var1) {
            isAssigned1 = true;
        }
        if(node.assignedVars.size() > var2) {
            isAssigned2 = true;
        }
        // if both variables are assigned, check next constraint since current one cannot be violated
        if(isAssigned1 && isAssigned2) continue;
        // if one of the variables is assigned, check if the constraint is violated
        // with respect to its current value
        if(isAssigned1 && domains[var2][node.assignedVals[var1]] == 1) {
            // remove the value from the domain of var2
            domains[var2][node.assignedVals[var1]] = 0;
            changed = true;
        }
        else if (isAssigned2 && domains[var1][node.assignedVals[var2]] == 1) {
            // remove the value from the domain of var1
            domains[var1][node.assignedVals[var2]] = 0;
            changed = true;
        }
    }
    return changed;
}

void generate_and_branch(const std::vector<std::pair<int, int>> &constraints, std::pair<int, int>* d_constraints, int* d_numConstraints, std::vector<std::vector<bool>> domains, uint8_t *d_domains, std::vector<uint8_t> flatDomains,
        std::vector<int> offsets, int* d_offsets, std::stack<Node>& nodes, int* d_assignedVals, int* d_branchVar, size_t &numSolutions, int n) {
    
    cudaError_t err;
    bool changed = false;
    while(true)
    {
        Node node = nodes.top();
        bool changed = false;
        int currentOffset = 0;
        // perform fixed point iteration to remove values from the domains
        do {
            // clear flatDomains
            flatDomains.clear();
            currentOffset = 0;
            // flatten the domains using the offsets
            for (const auto& domain : domains) {
                offsets.push_back(currentOffset);
                flatDomains.insert(flatDomains.end(), domain.begin(), domain.end());
                currentOffset += domain.size();
            }
            // copy the flatDomains to the device
            err = cudaMemcpy(d_domains, flatDomains.data(), sizeof(uint8_t) * flatDomains.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
            err = cudaMemcpy(d_assignedVals, node.assignedVals.data(), sizeof(int) * node.assignedVals.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
            err = cudaMemcpy(d_branchVar, &node.branchedVar, sizeof(int), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
            // launch the kernel to restrict the domains
            restrictDomainsKernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(d_constraints, d_numConstraints, d_domains, d_offsets, d_assignedVals, d_branchVar);
            // copy the flatDomains back to the host
            err = cudaMemcpy(flatDomains.data(), d_domains, sizeof(uint8_t) * flatDomains.size(), cudaMemcpyDeviceToHost); checkCudaError(err, __FILE__, __LINE__);
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
            for(int i = 0; i < n; i++) {
                if(std::count(domains[i].begin(), domains[i].end(), 1) == 1) {
                    // domain of the variable has only one value, corresponds to an assignment
                    // check if it is already assigned, do not add it again
                    if(std::find(node.assignedVars.begin(), node.assignedVars.end(), i) != node.assignedVars.end()) continue;
                    node.assignedVars.push_back(i);
                    node.assignedVals.push_back(std::find(domains[i].begin(), domains[i].end(), 1) - domains[i].begin());
                    changed = true;
                }
            }
        } while(changed);

        // print the domains of the variables
        // for(int i = 0; i < n; i++) {
        //     std::cout << "Domain of variable " << i << ": ";
        //     for(int j = 0; j < domains[i].size(); j++) {
        //         std::cout << domains[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // return;

        // reached a fixed point, i.e. no more values can be removed from the domains

        // need to check if the current configuration is a solution, otherwise force
        // an assignment and create the corresponding new node
        if(node.assignedVars.size() == n) {
            numSolutions++;

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
                // branch on the next value of the variable
                int var = node.branchedVar;
                int nextVal = std::find(domains[var].begin(), domains[var].end(), 1) - domains[var].begin();
                domains[var][nextVal] = 0;
                node.assignedVals[var] = nextVal; 
                // push again the updated node, since it has been previously popped
                nodes.push(node);
            }
        }   
        else {
            // since a fixed point has been reached but we do not have a solution, we 
            // need to branch on the next variable by artifically imposing an assignment
            // remove the value from the domain of the branch variable in the parent node
            // find the first non-zero value in the domain of branchedVar+1
            int newVal = std::find(domains[node.branchedVar+1].begin(), domains[node.branchedVar+1].end(), 1) - domains[node.branchedVar+1].begin();
            // set this value to 0 in the parent domain
            domains[node.branchedVar+1][newVal] = 0;
            // node.domains[node.branchedVar+1][node.assignedVals[node.branchedVar]] = 0;
            std::vector<int> assignedVars = node.assignedVars;
            std::vector<int> assignedVals = node.assignedVals;
            assignedVars.push_back(node.branchedVar+1);
            assignedVals.push_back(newVal);
            // select next variable to branch on as the one after branchedVar
            Node newNode(assignedVars, assignedVals, node.branchedVar+1);
            // and push the new node to the stack
            nodes.push(newNode);
        }
    }
}

void kernel_wrapper(std::vector<std::vector<bool>>& domains, const std::vector<std::pair<int, int>>& constraints, std::vector<int>& upperBounds, size_t& numSolutions)
{
    int n = upperBounds.size();
    cudaError_t err;

    // flatten the domains and prepare offsets to access them on the device
    // use uint8_t to store the domains, since bool is not supported by CUDA
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
    int numConstraints = constraints.size();

    uint8_t* d_domains;
    int* d_offsets;
    int* d_assignedVals;
    int* d_branchVar;
    std::pair<int, int>* d_constraints;
    int* d_numConstraints;
    // Allocate and copy flattened domains
    err = cudaMalloc(&d_domains, sizeof(uint8_t) * flatDomains.size()); checkCudaError(err, __FILE__, __LINE__);    
    err = cudaMalloc(&d_offsets, sizeof(int) * offsets.size()); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_assignedVals, sizeof(int) * n); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_branchVar, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_constraints, sizeof(std::pair<int, int>) * constraints.size()); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_numConstraints, sizeof(int)); checkCudaError(err, __FILE__, __LINE__);  
    // copy constraints only once, since they never change
    err = cudaMemcpy(d_constraints, constraints.data(), sizeof(std::pair<int, int>) * constraints.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * offsets.size(), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_numConstraints, &numConstraints, sizeof(int), cudaMemcpyHostToDevice); checkCudaError(err, __FILE__, __LINE__);
    // stack of nodes of the currently explored branch
    std::stack<Node> nodes;
    // create root node that contains the initial domains, the first variable that 
    // will be branched is 0
    // declare two vectors of size n to hold the assigned variables and values
    std::vector<int> assignedVars;
    std::vector<int> assignedVals;
    assignedVars.push_back(0);
    assignedVals.push_back(0);
    // remove the value from the domain of the variable
    Node root(assignedVars, assignedVals, 0);
    domains[0][assignedVals[0]] = 0;
    nodes.push(root);

    generate_and_branch(constraints, d_constraints, d_numConstraints, domains, d_domains, flatDomains, offsets, d_offsets, nodes, d_assignedVals, d_branchVar, numSolutions, n);

    // free the memory
    checkCudaError(cudaFree(d_domains), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_offsets), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_constraints), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_numConstraints), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_assignedVals), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_branchVar), __FILE__, __LINE__);
}