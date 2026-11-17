#include "gpu_search.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

const uint32_t BLOCK = 256;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct DeviceEmit {
    Hit* hits;
    uint32_t* count;
    uint32_t maxHits;
    __device__ void operator()(uint64_t f, uint64_t y, uint32_t variant, uint32_t flags) {
        uint32_t idx = atomicAdd(count, 1u);
        if (idx < maxHits) {
            hits[idx].f = f;
            hits[idx].y = y;
            hits[idx].variant = variant;
            hits[idx].flags = flags;
        }
    }
};

__global__ void table_kernel(Seed seed, uint64_t yBase, uint32_t* yGroups, uint64_t* table) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y[4];
    uint64_t t[VARIANTS];
    build_y_group(seed, yBase + j, y, t);
    for (int i = 0; i < 4; i++) yGroups[4 * j + i] = y[i];
    for (uint32_t v = 0; v < VARIANTS; v++) table[j * VARIANTS + v] = t[v];
}

__global__ void search_kernel(Seed seed, uint64_t fBase, uint64_t yBase, const uint32_t* yGroups,
                              const uint64_t* table, uint32_t iters, bool both,
                              Hit* hits, uint32_t* count, uint32_t maxHits) {
    DeviceEmit emit{hits, count, maxHits};
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    scan_f(seed, fBase + tid, yBase, yGroups, table, (uint32_t)(tid * iters), iters, both, emit);
}

}  // namespace

GpuSearch::GpuSearch(int device, uint32_t threads, uint32_t maxHits)
    : threads_((threads + BLOCK - 1) / BLOCK * BLOCK), maxHits_(maxHits) {
    check(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    name_ = prop.name;
    check(cudaMalloc(&dHits_, sizeof(Hit) * maxHits_), "cudaMalloc hits");
    check(cudaMalloc(&dCount_, sizeof(uint32_t)), "cudaMalloc count");
    check(cudaMalloc(&dY_, sizeof(uint32_t) * 4 * Y_GROUPS), "cudaMalloc y groups");
    check(cudaMalloc(&dTable_, sizeof(uint64_t) * VARIANTS * Y_GROUPS), "cudaMalloc table");
}

GpuSearch::~GpuSearch() {
    cudaFree(dHits_);
    cudaFree(dCount_);
    cudaFree(dY_);
    cudaFree(dTable_);
}

uint32_t GpuSearch::run(const Seed& seed, uint64_t epoch, uint32_t iters, bool both, std::vector<Hit>& out) {
    uint64_t fBase = epoch * threads_, yBase = epoch * Y_GROUPS;
    check(cudaMemset(dCount_, 0, sizeof(uint32_t)), "cudaMemset");
    table_kernel<<<Y_GROUPS / BLOCK, BLOCK>>>(seed, yBase, dY_, dTable_);
    check(cudaGetLastError(), "table kernel launch");
    search_kernel<<<threads_ / BLOCK, BLOCK>>>(seed, fBase, yBase, dY_, dTable_, iters, both, dHits_, dCount_, maxHits_);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "kernel run");

    uint32_t count = 0;
    check(cudaMemcpy(&count, dCount_, sizeof(count), cudaMemcpyDeviceToHost), "cudaMemcpy count");
    uint32_t n = count < maxHits_ ? count : maxHits_;
    if (n) {
        size_t old = out.size();
        out.resize(old + n);
        check(cudaMemcpy(out.data() + old, dHits_, sizeof(Hit) * n, cudaMemcpyDeviceToHost), "cudaMemcpy hits");
    }
    return count;
}
