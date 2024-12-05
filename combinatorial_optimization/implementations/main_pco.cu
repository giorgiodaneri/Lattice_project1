#include "pcoCUDA.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stack>  
#include <set>
#include <utility>  
#include "parser.hpp"

void extractUniqueConstraints(const std::vector<int>& constraintsLeft,
                               const std::vector<int>& constraintsRight,
                               std::vector<int>& uniqueConstraintsLeft,
                               std::vector<int>& uniqueConstraintsRight) {
    // use a set to store unique normalized pairs
    std::set<std::pair<int, int>> uniquePairs;
    for (size_t i = 0; i < constraintsLeft.size(); ++i) {
        // store the smaller value first
        int left = constraintsLeft[i];
        int right = constraintsRight[i];
        std::pair<int, int> constraintPair = std::minmax(left, right);
        uniquePairs.insert(constraintPair);
    }
    // remove the unique vectors to store new values
    uniqueConstraintsLeft.clear();
    uniqueConstraintsRight.clear();
    // fill the unique constraints
    for (const auto& pair : uniquePairs) {
        uniqueConstraintsLeft.push_back(pair.first);
        uniqueConstraintsRight.push_back(pair.second);
    }
}

int main(int argc, char** argv) {
    int  n;
    std::vector<int> upperBounds;
    std::vector<std::pair<int, int>> constraints;
    // parse the input file
    Data parser = Data();
    if (!parser.read_input(argv[1])) {
        return 1;
    }
    // get problem parameters from the parser
    n = parser.get_n();
    upperBounds.resize(n);
    // store constraints in two separate vectors, one for the left hand side, one for the right hand side
    std::vector<int> constraintsLeft;
    std::vector<int> constraintsRight;
    // store the constraints
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (parser.get_C_at(i, j) == 1) {
                constraintsLeft.push_back(i);
                constraintsRight.push_back(j);
            }
        }
    }
    // remove duplicates, since they are presented as a symmetric matrix
    std::vector<int> uniqueConstraintsLeft;
    std::vector<int> uniqueConstraintsRight;
    extractUniqueConstraints(constraintsLeft, constraintsRight, uniqueConstraintsLeft, uniqueConstraintsRight);
    // upper bounds of the domains 
    for (size_t i = 0; i < n; ++i) {
        upperBounds[i] = parser.get_u_at(i);
    }
    // vector of the domains of the variables, which is a vector of vectors of booleans, each one of size U_i
    std::vector<std::vector<bool>> domains;
    for(int i = 0; i < n; i++) {
        std::vector<bool> domain(upperBounds[i]+1, 1);
        domains.push_back(domain);
    }

    // use size_t to handle big numbers
    size_t numSolutions = 0;

    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    // call the function to handle memory and data structures on the device
    kernel_wrapper(domains, uniqueConstraintsLeft, uniqueConstraintsRight, upperBounds, numSolutions);
    // stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    // print the number of solutions and execution time
    std::cout << "Number of solutions: " << numSolutions << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    return 0;
}