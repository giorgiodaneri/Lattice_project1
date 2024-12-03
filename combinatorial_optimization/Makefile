# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -dlto -arch=sm_70 -O3 -use_fast_math -Xptxas -O3 -Xlinker -O3 -Xcompiler -O3

# Source files
SRC = implementations/pco.cpp
SRC_CUDA = implementations/main_pco.cu
# Header files are usually not directly compiled but included in source files
HEADERS_CUDA = implementations/pcoCUDA.cuh

# output binary
OUT = main
OUT_CUDA = mainCuda

# Default rule
all: $(OUT_CUDA)

# Linking rule
$(OUT_CUDA): $(SRC_CUDA) $(HEADERS_CUDA)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_CUDA)

serial: $(SRC)
	g++ -O3 -march=native -o $(OUT) $(SRC)

# Clean rule
clean:
	rm -f $(OUT_CUDA)
	rm -f $(OUT)
