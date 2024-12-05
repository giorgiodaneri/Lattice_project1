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

// function to restrict the domains based on the constraints
bool restrict_domains(const std::vector<int>& constraintsLeft, const std::vector<int>& constraintsRight, std::vector<std::vector<bool>> &domains, Node &node) 
{   
    bool changed = false;
    // iterate over the constraints
    for (size_t i = 0; i < constraintsLeft.size(); ++i) {
        // get the left and right hand side of the constraint
        int var1 = constraintsLeft[i];
        int var2 = constraintsRight[i];
        bool isAssigned1 = false;
        bool isAssigned2 = false;
        // check whether the two variables are already assigned 
        if (node.branchedVar + 1 > var1) {
            isAssigned1 = true;
        }
        if (node.branchedVar + 1 > var2) {
            isAssigned2 = true;
        }
        // if both variables are assigned, skip the current constraint, no point in checking it
        if (isAssigned1 && isAssigned2) continue;
        // if the first variable is assigned, remove all incompatible values from the domain of the second variable
        if (isAssigned1 && node.assignedVals[var1] >= 0 && domains[var2][node.assignedVals[var1]] == 1) {
            domains[var2][node.assignedVals[var1]] = 0;
            // domain has changed
            changed = true;
        } 
        // analogous reasoning for the second variable
        else if (isAssigned2 && node.assignedVals[var2] >= 0 && domains[var1][node.assignedVals[var2]] == 1) {
            domains[var1][node.assignedVals[var2]] = 0;
            changed = true;
        }
    }
    return changed;
}

void generate_and_branch(const std::vector<int>& constraintsLeft, const std::vector<int>& constraintsRight, std::vector<std::vector<bool>> domains, std::stack<Node>& nodes, size_t &numSolutions, int n) {
    bool loop = true;
    bool changed = false;
    // declare a vector of bools to store the singleton domains
    std::vector<bool> singletons(n, 0);
    while(loop)
    {
        singletons.clear();
        Node node = nodes.top();

        // perform fixed point iteration to remove values from the domains
        // if all variables are already assigned, it means that backtracking needs to be done
        if(node.branchedVar < n-1)
        {
            do {
                changed = false;
                changed = restrict_domains(constraintsLeft, constraintsRight, domains, node);
                // find all the variables that have been forcefully assigned due to constraint application
                // and the relative values, so that we actually have a solution
                // only check the domains of the variables that are > branchedVar, since the others are already assigned
                if(changed)
                {   
                    changed = false;
                    // check if singleton domains has changed
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
                                if(node.assignedVals.size() >= 8)
                                {
                                    std::cout << "ERROR AFTER PUSHBACK Assigned values: ";
                                    for(int i = 0; i < node.assignedVals.size(); i++) {
                                        std::cout << node.assignedVals[i] << " ";
                                    }
                                    std::cout << std::endl;
                                    return;
                                }
                            }
                            
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
            // check if there exist singleton domains, i.e. the size of assignedVals is bigger than the current variable index
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
    // declare vector of size n to hold the assigned values, first one is 0
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