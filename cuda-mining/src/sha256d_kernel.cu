// CUDA double-SHA256 nonce scanner for Bitcoin block headers.
//
// The 80-byte header spans two SHA-256 blocks. The first block (64 bytes) does not
// contain the nonce, so its compression ("midstate") is done once on the CPU. Each GPU
// thread then does two compressions: the second header block, and the hash of the hash.
// Of those 128 rounds, the first 4 are hoisted to the host (the nonce is the 4th message
// word) and the last 3 are skipped for nonces that cannot meet the target.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "gpu.h"

#define THREADS_PER_BLOCK 1024 // measured fastest of 64..1024 on a Quadro M1000M

__device__ static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ static const uint32_t H0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

struct ScanParams {
    uint32_t midstate[8];
    uint32_t s4[8];  // state after round 3 of the first hash, minus the nonce word in a and e
    uint32_t w16, w17;     // schedule words that do not depend on the nonce
    uint32_t pre18, pre19; // nonce-independent parts of W18, W19
    uint32_t pre31, pre32; // nonce-independent parts of W31, W32
    uint32_t start_nonce;
    uint32_t count;
    uint32_t target_hi; // top 32 bits of the target
    uint32_t target_lo; // next 32 bits
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define BSIG0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define BSIG1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SSIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SSIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))
#define CH(e, f, g) ((g) ^ ((e) & ((f) ^ (g))))
#define MAJ(a, b, c) (((a) & (b)) | ((c) & ((a) | (b))))

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0x0000ff00u) | ((x << 8) & 0x00ff0000u) | (x << 24);
}

// Next word of the rolling 16-word message schedule (valid for round i >= 16).
__device__ __forceinline__ uint32_t sched(uint32_t w[16], int i) {
    uint32_t w15 = w[(i + 1) & 15], w2 = w[(i + 14) & 15];
    return w[i & 15] += SSIG0(w15) + w[(i + 9) & 15] + SSIG1(w2);
}

#define ROUND(k, wi)                                          \
    {                                                         \
        uint32_t t1 = h + BSIG1(e) + CH(e, f, g) + (k) + (wi); \
        uint32_t t2 = BSIG0(a) + MAJ(a, b, c);                \
        h = g; g = f; f = e; e = d + t1;                      \
        d = c; c = b; b = a; a = t1 + t2;                     \
    }

// results[0] is the hit counter, results[1..GPU_MAX_RESULTS] are the nonces.
__global__ void scan_kernel(ScanParams p, uint32_t* results) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.count) return;
    uint32_t nonce = p.start_nonce + idx;
    uint32_t w3 = bswap32(nonce); // nonce is little-endian in the header

    // ---- first hash: second header block. Rounds 0..2 and most of round 3 do not
    // depend on the nonce and were done on the host; so were parts of the schedule.
    uint32_t a = p.s4[0] + w3, b = p.s4[1], c = p.s4[2], d = p.s4[3];
    uint32_t e = p.s4[4] + w3, f = p.s4[5], g = p.s4[6], h = p.s4[7];
    uint32_t w[16];
    w[0] = p.w16; // slots 0..3 already hold W16..W19: rounds 4..15 never read them
    w[1] = p.w17;
    w[2] = p.pre18 + SSIG0(w3);
    w[3] = p.pre19 + w3;
    w[4] = 0x80000000u;
#pragma unroll
    for (int i = 5; i < 15; i++) w[i] = 0;
    w[15] = 640; // 80 bytes
#pragma unroll
    for (int i = 4; i < 64; i++) {
        uint32_t wi;
        if (i < 20) wi = w[i & 15];
        else if (i == 31) wi = w[15] = p.pre31 + w[8] + SSIG1(w[13]);
        else if (i == 32) wi = w[0] = p.pre32 + w[9] + SSIG1(w[14]);
        else wi = sched(w, i);
        ROUND(K[i], wi);
    }

    // ---- second hash: SHA-256 of the 32-byte first hash.
    w[0] = p.midstate[0] + a; w[1] = p.midstate[1] + b;
    w[2] = p.midstate[2] + c; w[3] = p.midstate[3] + d;
    w[4] = p.midstate[4] + e; w[5] = p.midstate[5] + f;
    w[6] = p.midstate[6] + g; w[7] = p.midstate[7] + h;
    w[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 256; // 32 bytes
    a = H0[0]; b = H0[1]; c = H0[2]; d = H0[3];
    e = H0[4]; f = H0[5]; g = H0[6]; h = H0[7];
#pragma unroll
    for (int i = 0; i < 60; i++) {
        uint32_t wi = i < 16 ? w[i] : sched(w, i);
        ROUND(K[i], wi);
    }

    // The hash is a little-endian 256-bit number whose top word is the byte-swapped last
    // state word, and that word is final once round 60 has produced its `e`. Nearly every
    // nonce is rejected here, three rounds early.
    uint32_t e60 = d + h + BSIG1(e) + CH(e, f, g) + K[60] + sched(w, 60);
    uint32_t hi = bswap32(H0[7] + e60);
    if (hi > p.target_hi) return;
    uint32_t e61 = c + g + BSIG1(e60) + CH(e60, e, f) + K[61] + sched(w, 61);
    if (hi == p.target_hi && bswap32(H0[6] + e61) > p.target_lo) return;

    uint32_t slot = atomicAdd(&results[0], 1u);
    if (slot < GPU_MAX_RESULTS) results[1 + slot] = nonce;
}

// Host side: everything in the first hash that is the same for every nonce of a header.
static void precompute(const uint32_t midstate[8], const uint32_t tail[3], ScanParams* p) {
    static const uint32_t K03[4] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5};
    uint32_t a = midstate[0], b = midstate[1], c = midstate[2], d = midstate[3];
    uint32_t e = midstate[4], f = midstate[5], g = midstate[6], h = midstate[7];
    for (int i = 0; i < 3; i++) ROUND(K03[i], tail[i]);
    uint32_t t1 = h + BSIG1(e) + CH(e, f, g) + K03[3]; // round 3 without the nonce word
    uint32_t t2 = BSIG0(a) + MAJ(a, b, c);
    p->s4[0] = t1 + t2; p->s4[1] = a; p->s4[2] = b; p->s4[3] = c;
    p->s4[4] = d + t1;  p->s4[5] = e; p->s4[6] = f; p->s4[7] = g;

    // Header words 4..15 of this block are fixed padding: 0x80000000, zeros, length 640.
    p->w16 = tail[0] + SSIG0(tail[1]);
    p->w17 = tail[1] + SSIG0(tail[2]) + SSIG1(640u);
    p->pre18 = tail[2] + SSIG1(p->w16);
    p->pre19 = SSIG0(0x80000000u) + SSIG1(p->w17);
    p->pre31 = 640u + SSIG0(p->w16);
    p->pre32 = p->w16 + SSIG0(p->w17);
}

static uint32_t* d_results = NULL;
static char g_err[256] = "";

static bool check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    snprintf(g_err, sizeof(g_err), "%s: %s", what, cudaGetErrorString(e));
    return false;
}

const char* gpu_last_error() { return g_err; }

bool gpu_init(int device, char* name_out, size_t name_len) {
    int n = 0;
    if (!check(cudaGetDeviceCount(&n), "cudaGetDeviceCount")) return false;
    if (device < 0 || device >= n) {
        snprintf(g_err, sizeof(g_err), "device %d not found (%d CUDA device(s) available)", device, n);
        return false;
    }
    if (!check(cudaSetDevice(device), "cudaSetDevice")) return false;
    // Sleep instead of spinning a CPU core while the kernel runs.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaDeviceProp prop;
    if (!check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) return false;
    if (name_out && name_len) snprintf(name_out, name_len, "%s", prop.name);
    return check(cudaMalloc(&d_results, (1 + GPU_MAX_RESULTS) * sizeof(uint32_t)), "cudaMalloc");
}

void gpu_shutdown() {
    if (d_results) cudaFree(d_results);
    d_results = NULL;
}

int gpu_scan(const uint32_t midstate[8], const uint32_t tail[3], uint64_t target_top,
             uint32_t start_nonce, uint32_t count, uint32_t* out, int max_out) {
    if (count == 0) return 0;
    ScanParams p;
    memcpy(p.midstate, midstate, sizeof(p.midstate));
    precompute(midstate, tail, &p);
    p.start_nonce = start_nonce;
    p.count = count;
    p.target_hi = (uint32_t)(target_top >> 32);
    p.target_lo = (uint32_t)target_top;

    if (!check(cudaMemset(d_results, 0, sizeof(uint32_t)), "cudaMemset")) return -1;
    uint32_t blocks = (uint32_t)(((uint64_t)count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    scan_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_results);
    if (!check(cudaGetLastError(), "kernel launch")) return -1;

    static uint32_t host[1 + GPU_MAX_RESULTS];
    if (!check(cudaMemcpy(host, d_results, sizeof(host), cudaMemcpyDeviceToHost), "cudaMemcpy")) return -1;
    int hits = host[0] > GPU_MAX_RESULTS ? GPU_MAX_RESULTS : (int)host[0];
    if (hits > max_out) hits = max_out;
    memcpy(out, host + 1, hits * sizeof(uint32_t));
    return hits;
}
