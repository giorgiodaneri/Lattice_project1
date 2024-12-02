# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -dlto -arch=sm_70 -O3 -use_fast_math -Xptxas -O3 -Xlinker -O3 -Xcompiler -O3

# Source files
SRC = implementations/main_pco.cu
# Header files are usually not directly compiled but included in source files
HEADERS = implementations/pcoCUDA.cuh

# Output binary
OUT = mainCuda

# Default rule
all: $(OUT)

# Linking rule
$(OUT): $(SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC)

# Clean rule
clean:
	rm -f $(OUT)
