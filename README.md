# Parallel Programming for Lattice Theory - Project 1
This project was carried out as part of the course "Parallel Programming for Lattice Theory" at the University of Luxembourg. It consists of two exercises that concern the application of lattice theory to parallel programming. The first exercise consists in implementing an algorithm to find the minimum of an array using different approaches. The core concept is the comparison between a more traditional approach, which uses a parallel reduction pattern, and a lattice-based implementation based on a fixed point iteration. The former makes use of the pessimistic approach, which introduces locks, atomic operations and critical sections to avoid data races, no matter how rare they are. The latter employs an optimistic approach that is free of data synchronization constructs and opts for additional work in order to reach a coherent and stable result, hence the term fixed point iteration.
The second exercise consists in implementing an algorithm for a combinatorial optimization problem, which consists in finding all possible solutions given n variables, the relative domains and a set of constraints. Again we compare a standard implementation with one based on lattice theory and the fixed point model of computation. 
The user can find more information in the subfolders on how to compile and run the code, as well as a brief explanation of the algorithm and implementation choices in the project report.