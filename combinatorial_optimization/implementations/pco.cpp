#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stack>  
#include <thread>
#include "parser.hpp" 

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

bool restrict_domains(const std::vector<std::pair<int, int>> &constraints, std::vector<std::vector<bool>> &domains, Node &node, int n) 
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

void generate_and_branch(const std::vector<std::pair<int, int>> &constraints, std::vector<std::vector<bool>> domains, std::stack<Node>& nodes, size_t &numSolutions, int n) {
    bool loop = true;
    bool changed = false;
    while(loop)
    {
        Node node = nodes.top();
        bool changed = false;

        // perform fixed point iteration to remove values from the domains
        // do {
        // } while(changed);
        changed = restrict_domains(constraints, domains, node, n);

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
                // // branch on the next value of the variable
                int var = node.branchedVar;
                int nextVal = std::find(domains[var].begin(), domains[var].end(), 1) - domains[var].begin();
                domains[var][nextVal] = 0;
                node.assignedVals[var] = nextVal; 
                // push again the updated node, since it has been previously popped
                nodes.push(node);

                // iterate over all the remaining non zero values of the current branchedVar 
                // all the configurations are solutions
                // find the first non zero value in the domain
                // int start = std::find(node.domains[node.branchedVar].begin(), node.domains[node.branchedVar].end(), 1) - node.domains[node.branchedVar].begin();
                // for(int i = start; i <= node.domains[node.branchedVar].size(); ++i) 
                // {
                //     // remove the value from the domain of the variable
                //     if(node.domains[node.branchedVar][i] == 1) {
                //         numSolutions++;
                //         node.domains[node.branchedVar][i] = 0;
                //     }
                // }
                // node.assignedVals[node.branchedVar] = node.domains[node.branchedVar].size()-1;
                // nodes.push(node);
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

    // vector of the domains of the variables, which is a vector of vectors of booleans, each one of size U_i
    std::vector<std::vector<bool>> domains;
    for(int i = 0; i < n; i++) {
        std::vector<bool> domain(upperBounds[i]+1, 1);
        domains.push_back(domain);
    }


    // number of solutions
    size_t numSolutions = 0;
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

    // start timer
    auto start = std::chrono::high_resolution_clock::now();

    generate_and_branch(uniqueConstraints, domains, nodes, numSolutions, n);
 
    // stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    // print the number of solutions
    std::cout << "Number of solutions: " << numSolutions << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    return 0;
}