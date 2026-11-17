// Single-block SHA-256, usable from both host (g++) and device (nvcc) code.
// Every message hashed by this project is <= 55 bytes, so one compression
// from the IV is always enough.
//
// The 64 rounds are written out (generated) so the compiler keeps the whole
// state in registers and folds the constant padding words of each caller into
// the message schedule. They are split after round 6 so that hashes which
// differ only from message word 7 onwards can share their first 7 rounds.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#define SHA_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHA_s0(x) (SHA_ROTR(x, 7) ^ SHA_ROTR(x, 18) ^ ((x) >> 3))
#define SHA_s1(x) (SHA_ROTR(x, 17) ^ SHA_ROTR(x, 19) ^ ((x) >> 10))
#define SHA_RND(a, b, c, d, e, f, g, h, k, w)                                              \
    {                                                                                      \
        uint32_t t1 = h + (SHA_ROTR(e, 6) ^ SHA_ROTR(e, 11) ^ SHA_ROTR(e, 25)) +           \
                      (g ^ (e & (f ^ g))) + k + (w);                                       \
        d += t1;                                                                           \
        h = t1 + (SHA_ROTR(a, 2) ^ SHA_ROTR(a, 13) ^ SHA_ROTR(a, 22)) + ((a & b) | (c & (a | b))); \
    }

// Rounds 0..6 and 7..63, acting on locals a..h and w0..w15.
#define SHA_ROUNDS_0_6 \
    SHA_RND(a, b, c, d, e, f, g, h, 0x428a2f98u, w0); \
    SHA_RND(h, a, b, c, d, e, f, g, 0x71374491u, w1); \
    SHA_RND(g, h, a, b, c, d, e, f, 0xb5c0fbcfu, w2); \
    SHA_RND(f, g, h, a, b, c, d, e, 0xe9b5dba5u, w3); \
    SHA_RND(e, f, g, h, a, b, c, d, 0x3956c25bu, w4); \
    SHA_RND(d, e, f, g, h, a, b, c, 0x59f111f1u, w5); \
    SHA_RND(c, d, e, f, g, h, a, b, 0x923f82a4u, w6);

#define SHA_ROUNDS_7_63 \
    SHA_RND(b, c, d, e, f, g, h, a, 0xab1c5ed5u, w7); \
    SHA_RND(a, b, c, d, e, f, g, h, 0xd807aa98u, w8); \
    SHA_RND(h, a, b, c, d, e, f, g, 0x12835b01u, w9); \
    SHA_RND(g, h, a, b, c, d, e, f, 0x243185beu, w10); \
    SHA_RND(f, g, h, a, b, c, d, e, 0x550c7dc3u, w11); \
    SHA_RND(e, f, g, h, a, b, c, d, 0x72be5d74u, w12); \
    SHA_RND(d, e, f, g, h, a, b, c, 0x80deb1feu, w13); \
    SHA_RND(c, d, e, f, g, h, a, b, 0x9bdc06a7u, w14); \
    SHA_RND(b, c, d, e, f, g, h, a, 0xc19bf174u, w15); \
    SHA_RND(a, b, c, d, e, f, g, h, 0xe49b69c1u, (w0 += SHA_s1(w14) + w9 + SHA_s0(w1))); \
    SHA_RND(h, a, b, c, d, e, f, g, 0xefbe4786u, (w1 += SHA_s1(w15) + w10 + SHA_s0(w2))); \
    SHA_RND(g, h, a, b, c, d, e, f, 0x0fc19dc6u, (w2 += SHA_s1(w0) + w11 + SHA_s0(w3))); \
    SHA_RND(f, g, h, a, b, c, d, e, 0x240ca1ccu, (w3 += SHA_s1(w1) + w12 + SHA_s0(w4))); \
    SHA_RND(e, f, g, h, a, b, c, d, 0x2de92c6fu, (w4 += SHA_s1(w2) + w13 + SHA_s0(w5))); \
    SHA_RND(d, e, f, g, h, a, b, c, 0x4a7484aau, (w5 += SHA_s1(w3) + w14 + SHA_s0(w6))); \
    SHA_RND(c, d, e, f, g, h, a, b, 0x5cb0a9dcu, (w6 += SHA_s1(w4) + w15 + SHA_s0(w7))); \
    SHA_RND(b, c, d, e, f, g, h, a, 0x76f988dau, (w7 += SHA_s1(w5) + w0 + SHA_s0(w8))); \
    SHA_RND(a, b, c, d, e, f, g, h, 0x983e5152u, (w8 += SHA_s1(w6) + w1 + SHA_s0(w9))); \
    SHA_RND(h, a, b, c, d, e, f, g, 0xa831c66du, (w9 += SHA_s1(w7) + w2 + SHA_s0(w10))); \
    SHA_RND(g, h, a, b, c, d, e, f, 0xb00327c8u, (w10 += SHA_s1(w8) + w3 + SHA_s0(w11))); \
    SHA_RND(f, g, h, a, b, c, d, e, 0xbf597fc7u, (w11 += SHA_s1(w9) + w4 + SHA_s0(w12))); \
    SHA_RND(e, f, g, h, a, b, c, d, 0xc6e00bf3u, (w12 += SHA_s1(w10) + w5 + SHA_s0(w13))); \
    SHA_RND(d, e, f, g, h, a, b, c, 0xd5a79147u, (w13 += SHA_s1(w11) + w6 + SHA_s0(w14))); \
    SHA_RND(c, d, e, f, g, h, a, b, 0x06ca6351u, (w14 += SHA_s1(w12) + w7 + SHA_s0(w15))); \
    SHA_RND(b, c, d, e, f, g, h, a, 0x14292967u, (w15 += SHA_s1(w13) + w8 + SHA_s0(w0))); \
    SHA_RND(a, b, c, d, e, f, g, h, 0x27b70a85u, (w0 += SHA_s1(w14) + w9 + SHA_s0(w1))); \
    SHA_RND(h, a, b, c, d, e, f, g, 0x2e1b2138u, (w1 += SHA_s1(w15) + w10 + SHA_s0(w2))); \
    SHA_RND(g, h, a, b, c, d, e, f, 0x4d2c6dfcu, (w2 += SHA_s1(w0) + w11 + SHA_s0(w3))); \
    SHA_RND(f, g, h, a, b, c, d, e, 0x53380d13u, (w3 += SHA_s1(w1) + w12 + SHA_s0(w4))); \
    SHA_RND(e, f, g, h, a, b, c, d, 0x650a7354u, (w4 += SHA_s1(w2) + w13 + SHA_s0(w5))); \
    SHA_RND(d, e, f, g, h, a, b, c, 0x766a0abbu, (w5 += SHA_s1(w3) + w14 + SHA_s0(w6))); \
    SHA_RND(c, d, e, f, g, h, a, b, 0x81c2c92eu, (w6 += SHA_s1(w4) + w15 + SHA_s0(w7))); \
    SHA_RND(b, c, d, e, f, g, h, a, 0x92722c85u, (w7 += SHA_s1(w5) + w0 + SHA_s0(w8))); \
    SHA_RND(a, b, c, d, e, f, g, h, 0xa2bfe8a1u, (w8 += SHA_s1(w6) + w1 + SHA_s0(w9))); \
    SHA_RND(h, a, b, c, d, e, f, g, 0xa81a664bu, (w9 += SHA_s1(w7) + w2 + SHA_s0(w10))); \
    SHA_RND(g, h, a, b, c, d, e, f, 0xc24b8b70u, (w10 += SHA_s1(w8) + w3 + SHA_s0(w11))); \
    SHA_RND(f, g, h, a, b, c, d, e, 0xc76c51a3u, (w11 += SHA_s1(w9) + w4 + SHA_s0(w12))); \
    SHA_RND(e, f, g, h, a, b, c, d, 0xd192e819u, (w12 += SHA_s1(w10) + w5 + SHA_s0(w13))); \
    SHA_RND(d, e, f, g, h, a, b, c, 0xd6990624u, (w13 += SHA_s1(w11) + w6 + SHA_s0(w14))); \
    SHA_RND(c, d, e, f, g, h, a, b, 0xf40e3585u, (w14 += SHA_s1(w12) + w7 + SHA_s0(w15))); \
    SHA_RND(b, c, d, e, f, g, h, a, 0x106aa070u, (w15 += SHA_s1(w13) + w8 + SHA_s0(w0))); \
    SHA_RND(a, b, c, d, e, f, g, h, 0x19a4c116u, (w0 += SHA_s1(w14) + w9 + SHA_s0(w1))); \
    SHA_RND(h, a, b, c, d, e, f, g, 0x1e376c08u, (w1 += SHA_s1(w15) + w10 + SHA_s0(w2))); \
    SHA_RND(g, h, a, b, c, d, e, f, 0x2748774cu, (w2 += SHA_s1(w0) + w11 + SHA_s0(w3))); \
    SHA_RND(f, g, h, a, b, c, d, e, 0x34b0bcb5u, (w3 += SHA_s1(w1) + w12 + SHA_s0(w4))); \
    SHA_RND(e, f, g, h, a, b, c, d, 0x391c0cb3u, (w4 += SHA_s1(w2) + w13 + SHA_s0(w5))); \
    SHA_RND(d, e, f, g, h, a, b, c, 0x4ed8aa4au, (w5 += SHA_s1(w3) + w14 + SHA_s0(w6))); \
    SHA_RND(c, d, e, f, g, h, a, b, 0x5b9cca4fu, (w6 += SHA_s1(w4) + w15 + SHA_s0(w7))); \
    SHA_RND(b, c, d, e, f, g, h, a, 0x682e6ff3u, (w7 += SHA_s1(w5) + w0 + SHA_s0(w8))); \
    SHA_RND(a, b, c, d, e, f, g, h, 0x748f82eeu, (w8 += SHA_s1(w6) + w1 + SHA_s0(w9))); \
    SHA_RND(h, a, b, c, d, e, f, g, 0x78a5636fu, (w9 += SHA_s1(w7) + w2 + SHA_s0(w10))); \
    SHA_RND(g, h, a, b, c, d, e, f, 0x84c87814u, (w10 += SHA_s1(w8) + w3 + SHA_s0(w11))); \
    SHA_RND(f, g, h, a, b, c, d, e, 0x8cc70208u, (w11 += SHA_s1(w9) + w4 + SHA_s0(w12))); \
    SHA_RND(e, f, g, h, a, b, c, d, 0x90befffau, (w12 += SHA_s1(w10) + w5 + SHA_s0(w13))); \
    SHA_RND(d, e, f, g, h, a, b, c, 0xa4506cebu, (w13 += SHA_s1(w11) + w6 + SHA_s0(w14))); \
    SHA_RND(c, d, e, f, g, h, a, b, 0xbef9a3f7u, (w14 += SHA_s1(w12) + w7 + SHA_s0(w15))); \
    SHA_RND(b, c, d, e, f, g, h, a, 0xc67178f2u, (w15 += SHA_s1(w13) + w8 + SHA_s0(w0)));

#define SHA_IV \
    uint32_t a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au; \
    uint32_t e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;

// Compress one already-padded 64-byte block (16 big-endian words) from the IV.
HD inline void sha256_block(const uint32_t in[16], uint32_t out[8]) {
    uint32_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3], w4 = in[4], w5 = in[5], w6 = in[6], w7 = in[7];
    uint32_t w8 = in[8], w9 = in[9], w10 = in[10], w11 = in[11], w12 = in[12], w13 = in[13], w14 = in[14], w15 = in[15];
    SHA_IV
    SHA_ROUNDS_0_6
    SHA_ROUNDS_7_63
    out[0] = 0x6a09e667u + a; out[1] = 0xbb67ae85u + b; out[2] = 0x3c6ef372u + c; out[3] = 0xa54ff53au + d;
    out[4] = 0x510e527fu + e; out[5] = 0x9b05688cu + f; out[6] = 0x1f83d9abu + g; out[7] = 0x5be0cd19u + h;
}

// SHA-256 of a 32-byte message in two steps: the first 7 words once, then the
// 8th word as many times as needed.
struct ShaMid {
    uint32_t s[8];  // state after round 6
    uint32_t w[7];
};

HD inline void sha256_32_begin(const uint32_t in[7], ShaMid& m) {
    uint32_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3], w4 = in[4], w5 = in[5], w6 = in[6];
    SHA_IV
    SHA_ROUNDS_0_6
    m.s[0] = a; m.s[1] = b; m.s[2] = c; m.s[3] = d; m.s[4] = e; m.s[5] = f; m.s[6] = g; m.s[7] = h;
    m.w[0] = w0; m.w[1] = w1; m.w[2] = w2; m.w[3] = w3; m.w[4] = w4; m.w[5] = w5; m.w[6] = w6;
}

// Returns the first word of the digest.
HD inline uint32_t sha256_32_finish(const ShaMid& m, uint32_t w7) {
    uint32_t a = m.s[0], b = m.s[1], c = m.s[2], d = m.s[3], e = m.s[4], f = m.s[5], g = m.s[6], h = m.s[7];
    uint32_t w0 = m.w[0], w1 = m.w[1], w2 = m.w[2], w3 = m.w[3], w4 = m.w[4], w5 = m.w[5], w6 = m.w[6];
    uint32_t w8 = 0x80000000u, w9 = 0, w10 = 0, w11 = 0, w12 = 0, w13 = 0, w14 = 0, w15 = 256;
    SHA_ROUNDS_7_63
    (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)w15;
    return 0x6a09e667u + a;
}
