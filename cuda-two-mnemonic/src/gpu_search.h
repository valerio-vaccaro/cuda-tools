// Host-side interface to the CUDA search (no CUDA headers needed to include).
#pragma once
#include <string>
#include <vector>
#include "candidate.h"

class GpuSearch {
public:
    // threads: GPU threads per launch (rounded up to a multiple of 256).
    // Each thread handles one fixed mnemonic F.
    GpuSearch(int device, uint32_t threads, uint32_t maxHits);
    ~GpuSearch();
    GpuSearch(const GpuSearch&) = delete;
    GpuSearch& operator=(const GpuSearch&) = delete;

    std::string deviceName() const { return name_; }
    uint32_t threads() const { return threads_; }

    // Launch number `epoch`: builds its Y table, then tests threads() fresh F
    // against `iters` Y groups each. Appends matches to `out` and returns how
    // many matched on the GPU (may exceed what fit in the hit buffer; the
    // surplus is dropped).
    uint32_t run(const Seed& seed, uint64_t epoch, uint32_t iters, bool both, std::vector<Hit>& out);

private:
    std::string name_;
    uint32_t threads_, maxHits_;
    Hit* dHits_ = nullptr;
    uint32_t* dCount_ = nullptr;
    uint32_t* dY_ = nullptr;
    uint64_t* dTable_ = nullptr;
};
