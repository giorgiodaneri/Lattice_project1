# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags specific to the V100 GPU
# add compilation flag for openmp
CXXFLAGS = -O3 -funroll-loops -march=native -fopenmp
NVCCFLAGS = -dlto -arch=sm_70 -O3 -use_fast_math -Xptxas -O3 -Xlinker -O3 -Xcompiler -O3 -Xcompiler -fopenmp

# Targets
TARGET = mainCuda
CUDA_OBJ = minCuda.o

# Make rules
all: $(TARGET)

$(TARGET): $(CUDA_OBJ)
	$(NVCC) -o $@ $^ $(LIBS) $(NVCCFLAGS)

main.o: minimum.cpp
	$(CXX) -c $< $(CXXFLAGS)

minCuda.o: minCuda.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f $(TARGET) $(CUDA_OBJ)