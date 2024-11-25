NVCCFLAGS = -dlto -arch=sm_70 -O3 -use_fast_math -Xptxas -O3 -Xlinker -O3 -Xcompiler -O3

# Source file
SRC = CombinatorialOpt.cu

# Output binary
OUT = mainCuda

# Compiler
NVCC = nvcc

# Compile
all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(NVCC_FLAGS) --expt-relaxed-constexpr -o $(OUT) $(SRC)

# Clean
clean:
	rm -f $(OUT)