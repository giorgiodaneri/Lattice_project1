#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stack>    
#include "parser.hpp"

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

    removedVal() = default;
    removedVal(const removedVal&) = default;
    removedVal(removedVal&&) = default;
    removedVal(int val, int d, int v): value(val), depth(d), var(v) {}
};

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
                // std::cout << "Excluded value: " << newPos << " of variable " << var1 << " from variable " << var2 << std::endl;
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
    // while(!excludedValues.empty() && excludedValues.top().second == depth) {
    //     std::pair<int, int> value = excludedValues.top().first;
    //     int var = excludedValues.top().second;
    //     domains[var][value.first] = true;
    //     excludedValues.pop();
    // }
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

int main(int argc, char** argv) {
    // Problem parameters
    int  n;
    std::vector<int> upperBounds;
    std::vector<std::pair<int, int>> constraints;
    // parse the input file
    Data parser = Data();

    if (!parser.read_input(argv[1])) {
        return 1;
    }

    // Get the problem parameters
    n = parser.get_n();
    upperBounds.resize(n);

    // Get the constraints
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (parser.get_C_at(i, j) == 1) {
                constraints.push_back({i, j});
            }
        }
    }

    // Remove duplicates
    std::vector<std::pair<int, int>> uniqueConstraints;
    for (auto& constraint : constraints) {
        bool isDuplicate = false;
        for (auto& unique : uniqueConstraints) {
            if ((constraint.first == unique.second && constraint.second == unique.first) ||
                (constraint.first == unique.first && constraint.second == unique.second)) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            uniqueConstraints.push_back(constraint);
        }
    }

    // Get the upper bounds
    for (size_t i = 0; i < n; ++i) {
        upperBounds[i] = parser.get_u_at(i);
    }


    // number of solutions
    int numSolutions = 0;
    // vector of the domains of the variables, which is a vector of vectors of booleans, each one of size U_i
    std::vector<std::vector<bool>> domains;
    for(int i = 0; i < n; i++) {
        std::vector<bool> domain;
        for(int j = 0; j <= upperBounds[i]; j++) {
            domain.push_back(true);
        }
        domains.push_back(domain);
    }
    
    // stack to hold the values that have been excluded by the domain of the variables
    // the structure of each element is a pair (value, depth)
    // the value will be reinserted in the domain once backtracking reached depth-1
    // since the corresponding value of the assigned variable, which violated the constraint, will be changed
    // due to visiting another branch in the tree
    // std::stack<std::pair<std::pair<int, int>, int>> excludedValues;
    
    // use an array of structs to store the excluded values
    std::vector<removedVal> excludedValues;

    // start timer
    auto start = std::chrono::high_resolution_clock::now();

    // unroll first iteration of the recursion, since the dummy root node is already created
    // and all the nodes that correspond to the first variable need not increase depth
    // since they are found at the first level of the tree
    for(int i = 0; i <= upperBounds[0]; i++) {
        // do not check constraints for the first variable, since all the others have not been initialized yet
        Node child(n);
        // place the previous variable at the valid position that has been computed
        // increase depth and prepare for calculation of the current node possible positions
        child.positions[child.depth] = i;
        excludeValues(domains, excludedValues, child.depth, i, constraints);
        generateAndBranch(child, uniqueConstraints, upperBounds, excludedValues, domains, numSolutions, n);
        reinsertValues(domains, excludedValues, child.depth);
    }

    // stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    
    // print excluded values
    // while(!excludedValues.empty()) {
    //     std::pair<std::pair<int, int>, int> value = excludedValues.top();
    //     std::cout << "Excluded value: " << value.first.first << " of variable " << value.second << " from variable " << value.first.second << std::endl;
    //     excludedValues.pop();
    // }

    // print the domains
    // for(int i = 0; i < n; i++) {
    //     std::cout << "Domain of variable " << i << ": ";
    //     for(int j = 0; j <= upperBounds[i]; j++) {
    //         std::cout << domains[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // print the number of solutions
    std::cout << "Number of solutions: " << numSolutions << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    return 0;
}