# Parallel minimum
This is a simple program that finds the minimum of a list of numbers in parallel. Its purpose is to compare the performance of a standard pessimistic approach using a reduction pattern with a fixed point iteration, which follows an optimistic approach free of locks and atomic operations.

## Compile and run the code
To build the code, I set up a Makefile to make it as easy as possible. To compile the code use the ```make``` command. The program can then be run with ```./mainCuda num_elements```, where ```num_elements``` is the number of elements in the list. The program will generate a random vector of numbers and output the minimum value found by both the serial, reduction and fixed point iteration methods, as well as the execution times for each version.