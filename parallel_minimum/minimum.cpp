#include <iostream>
#include <vector>
#include <random>
// #include "min.cu"
using namespace std;

// declare extern function to call the kernel
extern void kernel_wrapper(std::vector<int> arr);

// declare a function that takes an array as input, finds the minimum and returns it

int findMin(std::vector<int> arr) {
    int min = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

int main() {
    int n = 10000;
    // generate array of random integers of size n
    // Initialize a random number generator
    int min = 1;
    int max = 1000;
    // initialize random number generator
    random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(min, max);

    std::vector<int> arr(n);
    for (int i = 0; i < n-1; i++) {
        arr[i] = distrib(gen);
    }
    // add artificial minimum 
    arr[n-1] = -1;
    // find the minimum of the array
    std::cout << "Minimum: " << findMin(arr) << std::endl;
    return 0;
}