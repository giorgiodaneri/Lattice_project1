#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <queue>
#include "parser.hpp"

// N-Queens node
struct Node {
    int  depth = 0;
    std::vector<int> positions; // positions  configuration (permutation)

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


void generateAndBranch(const Node& parent, const std::vector<std::pair<int, int>>& constraints, const std::vector<int>& upperBounds, int& numSolutions, int numVariables, std::queue<Node>& fifo) {
    // loop until there are nodes in the queue
    while(!fifo.empty()) {
        // get the first node in the queue
        Node parent = fifo.front();
        fifo.pop();
        // reached a leaf node, all constraints are satisfied
        if(parent.depth == (numVariables-1)) {
            numSolutions++;
            std::cout << "Configuration: ";
                for(int j = 0; j < numVariables; j++) {
                    std::cout << parent.positions[j] << " ";
                }
            std::cout << std::endl;
        }
        else {
            // generate a new node for each possible position of the current variable
            // compatibly with its upper bound => do not generate nodes that violate it 
            int len = 0;
            if(numVariables > upperBounds[parent.depth+1]) {
                len = upperBounds[parent.depth+1];
            }
            else {
                len = numVariables;
            }
            for(int i = 0; i < len; i++) {
                if(isValid(parent.positions, constraints, parent.depth, i)) {
                    Node child(parent);
                    // place the previous variable at the valid position that has been computed
                    // increase depth and prepare for calculation of the current node possible positions
                    child.depth++;
                    child.positions[child.depth] = i;
                    fifo.push(child);
                } 
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

    // remove all duplicate constraints by checking if the first element of the first constraints is equal to
    // the second element of the second constraint and vice versa
    for(auto& constraint : constraints) {
        for(auto& constraint2 : constraints) {
            if(constraint.first == constraint2.second && constraint.second == constraint2.first) {
                // remove constraint2
                
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

    // start timer
    auto start = std::chrono::high_resolution_clock::now();

    // number of solutions
    int numSolutions = 0;

    // declare a fifo queue
    std::queue<Node> fifo;

    // create the root node
    Node root(n);

    // unroll first iteration of the recursion, since the dummy root node is already created
    // and all the nodes that correspond to the first variable need not increase depth
    // since they are found at the first level of the tree
    int len = 0;
    if(n > upperBounds[0]) {
        len = upperBounds[0];
    }
    else {
        len = n;
    }
    for(int i = 0; i < len; i++) {
        // do not check constraints for the first variable, since all the others have not been initialized yet
        Node child(n);
        // place the previous variable at the valid position that has been computed
        // increase depth and prepare for calculation of the current node possible positions
        child.positions[child.depth] = i;
        fifo.push(child);
        generateAndBranch(child, constraints, upperBounds, numSolutions, n, fifo);
    }
    
    // stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    // print the number of solutions
    std::cout << "Number of solutions: " << numSolutions << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    return 0;
}