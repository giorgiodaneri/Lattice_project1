#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stack>  
#include <thread>
#include <set>
#include <utility>
#include "parser.hpp" 

struct Node {
    // vector of assigned variables, which will be empty at the beginning
    // vector of corresponding assigned values
    std::vector<int> assignedVals;
    int branchedVar;

    Node(std::vector<int> vals, int var) : assignedVals(vals), branchedVar(var) {}
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

bool restrict_domains(const std::vector<int>& constraintsLeft, const std::vector<int>& constraintsRight, std::vector<std::vector<bool>> &domains, Node &node) 
{   
    bool changed = false;

    // Iterate over the constraints using the two separate vectors
    for (size_t i = 0; i < constraintsLeft.size(); ++i) {
        int var1 = constraintsLeft[i];
        int var2 = constraintsRight[i];
        bool isAssigned1 = false;
        bool isAssigned2 = false;

        // Check if var1 and var2 are inside the assigned variables
        if (node.branchedVar + 1 > var1) {
            isAssigned1 = true;
        }
        if (node.branchedVar + 1 > var2) {
            isAssigned2 = true;
        }

        // If both variables are assigned, skip the current constraint
        if (isAssigned1 && isAssigned2) continue;

        // If one variable is assigned, check if the constraint is violated
        if (isAssigned1 && domains[var2][node.assignedVals[var1]] == 1) {
            // Remove the value from the domain of var2
            domains[var2][node.assignedVals[var1]] = 0;
            changed = true;
        } else if (isAssigned2 && domains[var1][node.assignedVals[var2]] == 1) {
            // Remove the value from the domain of var1
            domains[var1][node.assignedVals[var2]] = 0;
            changed = true;
        }
    }
    return changed;
}

void generate_and_branch(const std::vector<int>& constraintsLeft, const std::vector<int>& constraintsRight, std::vector<std::vector<bool>> domains, std::stack<Node>& nodes, size_t &numSolutions, int n) {
    bool loop = true;
    bool changed = false;
    while(loop)
    {
        Node node = nodes.top();
        bool changed = false;

        // perform fixed point iteration to remove values from the domains
        do {
            changed = restrict_domains(constraintsLeft, constraintsRight, domains, node);
        
            // find all the variables that have been forcefully assigned due to constraint application
            // and the relative values, so that we actually have a solution
            // only check the domains of the variables that are > branchedVar, since the others are already assigned
            for(int i = node.branchedVar+1; i < n; i++) {
                if(std::count(domains[i].begin(), domains[i].end(), 1) == 1) {
                    // domain of the variable has only one value, corresponds to an assignment
                    node.branchedVar = i;
                    node.assignedVals.push_back(std::find(domains[i].begin(), domains[i].end(), 1) - domains[i].begin());
                    changed = true;
                }
            }
        } while(changed);

        // reached a fixed point, i.e. no more values can be removed from the domains

        // need to check if the current configuration is a solution, otherwise force
        // an assignment and create the corresponding new node
        if(node.branchedVar == n-1) {
            // numSolutions++;

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
                numSolutions++;
                nodes.pop();
                // iterate over all the remaining non zero values of the current branchedVar 
                // all the configurations are solutions
                int start = node.assignedVals.back()+1;
                int maxValue = domains[node.branchedVar].size();
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
            // since a fixed point has been reached but we do not have a solution, we 
            // need to branch on the next variable by artifically imposing an assignment
            // remove the value from the domain of the branch variable in the parent node
            // find the first non-zero value in the domain of branchedVar+1
            int newVal = std::find(domains[node.branchedVar+1].begin(), domains[node.branchedVar+1].end(), 1) - domains[node.branchedVar+1].begin();
            // set this value to 0 in the parent domain
            domains[node.branchedVar+1][newVal] = 0;
            std::vector<int> assignedVals = node.assignedVals;
            assignedVals.push_back(newVal);
            // select next variable to branch on as the one after branchedVar
            Node newNode(assignedVals, node.branchedVar+1);
            // and push the new node to the stack
            nodes.push(newNode);
        }
    }
}

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
    // parse the input file
    Data parser = Data();
    if (!parser.read_input(argv[1])) {
        return 1;
    }
    // get the problem parameters
    n = parser.get_n();
    upperBounds.resize(n);
    // store constraints in two separate vectors, one for the left hand side, one for the right hand side
    std::vector<int> constraintsLeft;
    std::vector<int> constraintsRight;
    // get the constraints
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (parser.get_C_at(i, j) == 1) {
                constraintsLeft.push_back(i);
                constraintsRight.push_back(j);
            }
        }
    }
    // remove duplicates from the constraints
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

    size_t numSolutions = 0;
    // stack of nodes of the currently explored branch
    std::stack<Node> nodes;
    // declare two vectors of size n to hold the assigned values, first one is 0
    std::vector<int> assignedVals;
    assignedVals.push_back(0);
    // create root node that contains the initial domains, the first variable that will be branched is 0
    Node root(assignedVals, 0);
    // remove the value from the domain of the variable
    domains[0][assignedVals[0]] = 0;
    nodes.push(root);

    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    generate_and_branch(uniqueConstraintsLeft, uniqueConstraintsRight, domains, nodes, numSolutions, n);
    // stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    // print the number of solutions and exeuction time
    std::cout << "Number of solutions: " << numSolutions << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    return 0;
}