# Parallel Combinatorial Optimization
The problem consists in finding all the possible assignments of n variables that satisfy a set of constraints, where the assigned values are within the domains of each variable.

## How to compile and run the code
To build the code, I set up a Makefile to make it as easy as possible. To compile the serial version, use the ```make serial``` command. To compile the parallel version, first make sure your environment is set up with a CUDA toolkit version >= 11.1. On the Iris cluster, you can simply load the modules containing the NVIDIA drivers and CUDA libraries with the command ```ml toolchain/intelcuda```, then use the ```make``` command. 
To execute the both the serial and parallel version, the user needs to provide the input file as a command line argument, e.g. ```./main pco_5.txt```. The input files containing the number of variables, their domains and the constraints can be generated with the script ```generate_instance.py```. The usage is as follows:
```python3 generate_instance.py num_variables```. The user can also modify the range of the domains to adapt the test case to their needs. Some tast cases are already provided in the repository. 

### Example - Input data format
```
N
3
U
0;4
1;36
2;89
C
0,0;0
0,1;0
0,2;1
1,0;0
1,1;0
1,2;1
2,0;1
2,1;1
2,2;0
```
