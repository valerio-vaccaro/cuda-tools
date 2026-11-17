CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++

CXXFLAGS  ?= -O2 -Wall -Wextra -std=c++17
# CUDA 11.x rejects gcc >= 12 by default; the kernel file uses only C headers, so the override is safe.
# sm_50 binary plus PTX, which newer GPUs JIT-compile at startup. For best speed on a
# specific card override it, e.g. make CUDA_ARCH="-arch=sm_86".
CUDA_ARCH ?= -gencode arch=compute_50,code=[sm_50,compute_50] -Wno-deprecated-gpu-targets
NVCCFLAGS ?= -O3 -allow-unsupported-compiler $(CUDA_ARCH)
LDFLAGS   += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS    += -lcudart -lpthread

OBJS = build/main.o build/sha256.o build/stratum.o build/sha256d_kernel.o

cudaminer: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

build/%.o: src/%.cpp $(wildcard src/*.h) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.cu src/gpu.h | build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build:
	mkdir -p build

test: cudaminer
	./cudaminer --selftest

clean:
	rm -rf build cudaminer

.PHONY: clean test
