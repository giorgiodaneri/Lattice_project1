/*
 * Author: Guillaume HELBECQUE (Université du Luxembourg)
 * Date: 10/10/2024
 *
 * Description:
 * This program solves the N-Queens problem by counting all possible configurations
 * and checking how many of them are valid (i.e., queens do not threaten each other).
 * The program will explore all possible permutations of queen placements on the board.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include "parser.hpp"

// N-Queens node
struct Node {
  std::vector<int> board; // board configuration (permutation)

  Node(size_t N): board(N) {
    for (int i = 0; i < N; i++) {
      board[i] = i;
    }
  }
  Node(const Node&) = default;
  Node(Node&&) = default;
  Node() = default;
};

// check if placing a queen is safe (i.e., check if all the queens already placed share
// a same diagonal)
bool isSafe(const std::vector<int>& board, const int row, const int col)
{
  for (int i = 0; i < row; ++i) {
    if (board[i] == col - row + i || board[i] == col + row - i) {
      return false;
    }
  }

  return true;
}

// evaluate a given configuration and check if it is valid
bool isValidSolution(const std::vector<int>& board) {
  int N = board.size();
  for (int row = 0; row < N; row++) {
    if (!isSafe(board, row, board[row])) {
      return false;  // If any queen threatens another, it's not valid
    }
  }
  return true;
}

int main(int argc, char** argv) {
  // helper
  if (argc != 2) {
    std::cout << "usage: " << argv[0] << " <input-file> " << std::endl;
    exit(1);
  }

  // Problem parameters
    size_t n;
    std::vector<int> upperBounds;
    std::vector<std::pair<int, int>> constraints;
    Data parser = Data();

    if (!parser.read_input(argv[1])) {
        return 1;
    }

    // Get the problem parameters
    n = parser.get_n();
    upperBounds = std::vector<int>(n, n - 1);

    // Get the constraints

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (parser.get_C_at(i, j) == 1) {
            constraints.push_back({i, j});
            }
        }
    }

    // print the constraints
    for (size_t i = 0; i < constraints.size(); ++i) {
        std::cout << "Constraint " << i << ": " << constraints[i].first << " != " << constraints[i].second << "\n";
    }

    // Get the upper bounds
    for (size_t i = 0; i < n; ++i) {
        upperBounds[i] = parser.get_u_at(i);
    }

  // initialization of the board configuration (permutation of positions)
  Node root(N);

  // statistics to check correctness (number of nodes explored and number of solutions found)
  size_t totalConfigs = 0;
  size_t validSolutions = 0;

  // begin generating all permutations of the queens' positions
  auto start = std::chrono::steady_clock::now();

  // Generate all permutations of the positions of N queens
  do {
    totalConfigs++;  // Count this configuration
    if (isValidSolution(root.board)) {
      validSolutions++;  // If it's a valid configuration, increment validSolutions
    }
  } while (std::next_permutation(root.board.begin(), root.board.end()));  // Get next permutation

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // outputs
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
  std::cout << "Total configurations: " << totalConfigs << std::endl;
  std::cout << "Total valid solutions: " << validSolutions << std::endl;

  return 0;
}
