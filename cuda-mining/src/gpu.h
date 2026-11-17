// Host-side interface to the CUDA SHA-256d scanner (implemented in sha256d_kernel.cu).
#pragma once
#include <cstddef>
#include <cstdint>

#define GPU_MAX_RESULTS 256

// Selects the device and allocates buffers. On failure returns false; see gpu_last_error().
bool gpu_init(int device, char* name_out, size_t name_len);
void gpu_shutdown();
const char* gpu_last_error();

// Hashes `count` nonces starting at `start_nonce`.
//   midstate   : SHA-256 state after the first 64 header bytes
//   tail       : header bytes 64..75 as three big-endian words (merkle tail, ntime, nbits)
//   target_top : most significant 64 bits of the 256-bit share target
// Nonces whose hash has its top 64 bits <= target_top are written to `out`
// (candidates: the caller must verify the full 256-bit comparison on the CPU).
// Returns the number of candidates (capped at max_out), or -1 on a CUDA error.
int gpu_scan(const uint32_t midstate[8], const uint32_t tail[3], uint64_t target_top,
             uint32_t start_nonce, uint32_t count, uint32_t* out, int max_out);
