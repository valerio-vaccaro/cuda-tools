// vanity.cu -- GPU vanity search over BIP39 mnemonics.
//
// For every random mnemonic the first 20 receive addresses (m/purpose'/coin'/0'/0/0..19)
// of the chosen address kind/network are derived; the search stops as soon as one of them
// matches the requested characters (placed right AFTER the fixed prefix of the address
// type: "1", "3", "bc1q", "bc1p", "tb1q", ...).
//
// Everything (SHA-2, RIPEMD160, PBKDF2, BIP32, secp256k1, base58/bech32) is implemented
// in code shared by host and device, so every result found on the GPU is re-derived and
// verified on the CPU before being printed.
//
// Build: make        Usage: ./build/vanity --help

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <cuda_runtime.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
#define HD __host__ __device__
#define HDI __host__ __device__ __forceinline__

#include "tables.inc"

static const u32 K256_H[64] = K256_INIT;
__constant__ u32 K256_D[64] = K256_INIT;
static const u64 K512_H[80] = K512_INIT;
__constant__ u64 K512_D[80] = K512_INIT;
static const char WL_H[2048][8] = WORDLIST_INIT;
__constant__ char WL_D[2048][8] = WORDLIST_INIT;

// RIPEMD-160 tables
#define RMD_RL {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8, 3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13}
#define RMD_RR {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2, 15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11}
#define RMD_SL {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12, 11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6}
#define RMD_SR {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11, 9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11}
static const u8 RL_H[80] = RMD_RL, RR_H[80] = RMD_RR, SL_H[80] = RMD_SL, SR_H[80] = RMD_SR;
__constant__ u8 RL_D[80] = RMD_RL, RR_D[80] = RMD_RR, SL_D[80] = RMD_SL, SR_D[80] = RMD_SR;

#ifdef __CUDA_ARCH__
#define SEL(name) name##_D
#else
#define SEL(name) name##_H
#endif

// ------------------------------------------------------------------ SHA-256
HDI u32 rotr32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
HDI u32 rotl32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
HDI u32 ld32be(const u8* p) { return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3]; }
HDI void st32be(u8* p, u32 v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
HDI u64 ld64be(const u8* p) { return ((u64)ld32be(p) << 32) | ld32be(p + 4); }
HDI void st64be(u8* p, u64 v) { st32be(p, (u32)(v >> 32)); st32be(p + 4, (u32)v); }
HDI void bcopy(u8* d, const u8* s, int n) { for (int i = 0; i < n; i++) d[i] = s[i]; }
HDI void bzero(u8* d, int n) { for (int i = 0; i < n; i++) d[i] = 0; }

HD __noinline__ void sha256_compress(u32 s[8], const u8* blk) {
    u32 w[16];
    for (int i = 0; i < 16; i++) w[i] = ld32be(blk + 4 * i);
    u32 a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
    const u32* K = SEL(K256);
#pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 wi;
        if (i < 16) wi = w[i];
        else {
            u32 w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            wi = w[i & 15] + (rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >> 3)) + w[(i - 7) & 15] +
                 (rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >> 10));
            w[i & 15] = wi;
        }
        u32 t1 = h + (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + wi;
        u32 t2 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

HD __noinline__ void sha256(const u8* data, int n, u8 out[32]) {
    const u32 iv[8] = IV256_INIT;
    u32 s[8];
    for (int i = 0; i < 8; i++) s[i] = iv[i];
    int i = 0;
    for (; i + 64 <= n; i += 64) sha256_compress(s, data + i);
    u8 buf[128];
    int r = n - i;
    bzero(buf, 128);
    bcopy(buf, data + i, r);
    buf[r] = 0x80;
    int tot = (r < 56) ? 64 : 128;
    u64 bits = (u64)n * 8;
    st64be(buf + tot - 8, bits);
    sha256_compress(s, buf);
    if (tot == 128) sha256_compress(s, buf + 64);
    for (int j = 0; j < 8; j++) st32be(out + 4 * j, s[j]);
}

// ------------------------------------------------------------------ SHA-512
HDI u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

// forced-inline body: used directly in the PBKDF2 loop so that state and schedule live in registers
// and the constant message words are folded by the compiler
HDI void sha512_compress_i(u64 s[8], const u64 in[16]) {
    u64 w[16];
    for (int i = 0; i < 16; i++) w[i] = in[i];
    u64 a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#if defined(__CUDA_ARCH__) && defined(K512_LOCAL)
    const u64 K[80] = K512_INIT;
#else
    const u64* K = SEL(K512);
#endif
#pragma unroll
    for (int i = 0; i < 80; i++) {
        u64 wi;
        if (i < 16) wi = w[i];
        else {
            u64 w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            wi = w[i & 15] + (rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7)) + w[(i - 7) & 15] +
                 (rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6));
            w[i & 15] = wi;
        }
        u64 t1 = h + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) + (g ^ (e & (f ^ g))) + K[i] + wi;
        u64 t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) + ((a & b) | (c & (a | b)));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}
HD __noinline__ void sha512_compress(u64 s[8], const u64 in[16]) { sha512_compress_i(s, in); }

HD __noinline__ void sha512(const u8* data, int n, u8 out[64]) {
    const u64 iv[8] = IV512_INIT;
    u64 s[8];
    for (int i = 0; i < 8; i++) s[i] = iv[i];
    u64 w[16];
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        for (int j = 0; j < 16; j++) w[j] = ld64be(data + i + 8 * j);
        sha512_compress(s, w);
    }
    u8 buf[256];
    int r = n - i;
    bzero(buf, 256);
    bcopy(buf, data + i, r);
    buf[r] = 0x80;
    int tot = (r < 112) ? 128 : 256;
    st64be(buf + tot - 8, (u64)n * 8);
    for (int b = 0; b < tot; b += 128) {
        for (int j = 0; j < 16; j++) w[j] = ld64be(buf + b + 8 * j);
        sha512_compress(s, w);
    }
    for (int j = 0; j < 8; j++) st64be(out + 8 * j, s[j]);
}

// HMAC-SHA512 with a reusable key state (ipad/opad midstates)
struct hctx { u64 i[8], o[8]; };
HD __noinline__ void hmac512_init(hctx& h, const u8* key, int klen) {   // klen <= 128
    const u64 iv[8] = IV512_INIT;
    u8 k[128];
    bzero(k, 128);
    bcopy(k, key, klen);
    u64 w[16];
    for (int i = 0; i < 8; i++) h.i[i] = h.o[i] = iv[i];
    for (int i = 0; i < 16; i++) w[i] = ld64be(k + 8 * i) ^ 0x3636363636363636ULL;
    sha512_compress(h.i, w);
    for (int i = 0; i < 16; i++) w[i] = ld64be(k + 8 * i) ^ 0x5c5c5c5c5c5c5c5cULL;
    sha512_compress(h.o, w);
}
// message length <= 111 (single inner block): 2 compressions
HD __noinline__ void hmac512_final(const hctx& h, const u8* msg, int mlen, u8 out[64]) {
    u8 blk[128];
    bzero(blk, 128);
    bcopy(blk, msg, mlen);
    blk[mlen] = 0x80;
    st64be(blk + 120, (u64)(128 + mlen) * 8);
    u64 w[16], s[8];
    for (int i = 0; i < 16; i++) w[i] = ld64be(blk + 8 * i);
    for (int i = 0; i < 8; i++) s[i] = h.i[i];
    sha512_compress(s, w);
    for (int i = 0; i < 8; i++) { w[i] = s[i]; s[i] = h.o[i]; }
    w[8] = 0x8000000000000000ULL;
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = (128 + 64) * 8;
    sha512_compress(s, w);
    for (int i = 0; i < 8; i++) st64be(out + 8 * i, s[i]);
}

// PBKDF2-HMAC-SHA512, 2048 iterations, 64-byte output (one block). saltlen <= 100.
HD void pbkdf2_sha512_2048(const u8* pw, int pwlen, const u8* salt, int saltlen, u8 out[64]) {
    const u64 iv[8] = IV512_INIT;
    u8 k[128];
    bzero(k, 128);
    if (pwlen > 128) sha512(pw, pwlen, k); else bcopy(k, pw, pwlen);
    u64 ist[8], ost[8], w[16];
    for (int i = 0; i < 8; i++) ist[i] = ost[i] = iv[i];
    for (int i = 0; i < 16; i++) w[i] = ld64be(k + 8 * i) ^ 0x3636363636363636ULL;
    sha512_compress(ist, w);
    for (int i = 0; i < 16; i++) w[i] = ld64be(k + 8 * i) ^ 0x5c5c5c5c5c5c5c5cULL;
    sha512_compress(ost, w);

    u8 blk[128];
    bzero(blk, 128);
    bcopy(blk, salt, saltlen);
    blk[saltlen + 3] = 1;
    blk[saltlen + 4] = 0x80;
    st64be(blk + 120, (u64)(128 + saltlen + 4) * 8);
    for (int i = 0; i < 16; i++) w[i] = ld64be(blk + 8 * i);
    u64 s[8], U[8], T[8];
    for (int i = 0; i < 8; i++) { s[i] = ist[i]; T[i] = 0; }
    sha512_compress(s, w);
    w[8] = 0x8000000000000000ULL;
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = (128 + 64) * 8;
    for (int it = 0; it < 2048; it++) {
        // outer hash of the inner digest s
        for (int i = 0; i < 8; i++) { w[i] = s[i]; U[i] = ost[i]; }
        sha512_compress_i(U, w);
        for (int i = 0; i < 8; i++) T[i] ^= U[i];
        // next inner hash (the one computed in the last round is unused)
        for (int i = 0; i < 8; i++) { w[i] = U[i]; s[i] = ist[i]; }
        sha512_compress_i(s, w);
    }
    for (int i = 0; i < 8; i++) st64be(out + 8 * i, T[i]);
}

// ------------------------------------------------------------------ RIPEMD-160 (32-byte input)
HDI u32 rmd_f(int r, u32 x, u32 y, u32 z) {
    switch (r) {
        case 0: return x ^ y ^ z;
        case 1: return (x & y) | (~x & z);
        case 2: return (x | ~y) ^ z;
        case 3: return (x & z) | (y & ~z);
        default: return x ^ (y | ~z);
    }
}
HD __noinline__ void ripemd160_32(const u8 in[32], u8 out[20]) {
    u32 X[16];
    for (int i = 0; i < 8; i++) X[i] = in[4 * i] | ((u32)in[4 * i + 1] << 8) | ((u32)in[4 * i + 2] << 16) | ((u32)in[4 * i + 3] << 24);
    X[8] = 0x80;
    for (int i = 9; i < 14; i++) X[i] = 0;
    X[14] = 256; X[15] = 0;
    const u32 KL[5] = {0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu};
    const u32 KR[5] = {0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u};
    const u8 *RL = SEL(RL), *RR = SEL(RR), *SL = SEL(SL), *SR = SEL(SR);
    u32 h0 = 0x67452301u, h1 = 0xEFCDAB89u, h2 = 0x98BADCFEu, h3 = 0x10325476u, h4 = 0xC3D2E1F0u;
    u32 a = h0, b = h1, c = h2, d = h3, e = h4, a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;
    for (int j = 0; j < 80; j++) {
        int rnd = j >> 4;
        u32 t = rotl32(a + rmd_f(rnd, b, c, d) + X[RL[j]] + KL[rnd], SL[j]) + e;
        a = e; e = d; d = rotl32(c, 10); c = b; b = t;
        t = rotl32(a2 + rmd_f(4 - rnd, b2, c2, d2) + X[RR[j]] + KR[rnd], SR[j]) + e2;
        a2 = e2; e2 = d2; d2 = rotl32(c2, 10); c2 = b2; b2 = t;
    }
    u32 t = h1 + c + d2; h1 = h2 + d + e2; h2 = h3 + e + a2; h3 = h4 + a + b2; h4 = h0 + b + c2; h0 = t;
    u32 hh[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; i++) { out[4 * i] = hh[i]; out[4 * i + 1] = hh[i] >> 8; out[4 * i + 2] = hh[i] >> 16; out[4 * i + 3] = hh[i] >> 24; }
}
HD void hash160(const u8* data, int n, u8 out[20]) {
    u8 t[32];
    sha256(data, n, t);
    ripemd160_32(t, out);
}

// ------------------------------------------------------------------ secp256k1 field (p = 2^256 - 2^32 - 977)
struct fe { u32 v[8]; };

// r += 2^256 mod p (= 2^32 + 977); returns carry out
HDI u32 fe_add_c(fe& r) {
    u64 a = (u64)r.v[0] + 977; r.v[0] = (u32)a; a >>= 32;
    a += (u64)r.v[1] + 1; r.v[1] = (u32)a; a >>= 32;
    for (int i = 2; i < 8; i++) { a += r.v[i]; r.v[i] = (u32)a; a >>= 32; }
    return (u32)a;
}
HDI u32 fe_sub_c(fe& r) {
    u64 d = (u64)r.v[0] - 977; r.v[0] = (u32)d; u64 br = (d >> 63) & 1;
    d = (u64)r.v[1] - 1 - br; r.v[1] = (u32)d; br = (d >> 63) & 1;
    for (int i = 2; i < 8; i++) { d = (u64)r.v[i] - br; r.v[i] = (u32)d; br = (d >> 63) & 1; }
    return (u32)br;
}
HDI void fe_add(fe& r, const fe& a, const fe& b) {
    u64 acc = 0;
    for (int i = 0; i < 8; i++) { acc += (u64)a.v[i] + b.v[i]; r.v[i] = (u32)acc; acc >>= 32; }
    if (acc) acc = fe_add_c(r);
    if (acc) fe_add_c(r);
}
HDI void fe_sub(fe& r, const fe& a, const fe& b) {
    u64 br = 0;
    for (int i = 0; i < 8; i++) { u64 d = (u64)a.v[i] - b.v[i] - br; r.v[i] = (u32)d; br = (d >> 63) & 1; }
    if (br) br = fe_sub_c(r);
    if (br) fe_sub_c(r);
}
HDI void fe_mul(fe& r, const fe& a, const fe& b) {
    u32 t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 8; i++) {
        u64 c = 0;
        for (int j = 0; j < 8; j++) {
            u64 cur = (u64)a.v[i] * b.v[j] + t[i + j] + c;
            t[i + j] = (u32)cur; c = cur >> 32;
        }
        t[i + 8] = (u32)c;
    }
    u32 s[10];
    u64 acc = 0;
    for (int i = 0; i < 8; i++) { acc += (u64)t[8 + i] * 977 + t[i]; s[i] = (u32)acc; acc >>= 32; }
    s[8] = (u32)acc;
    acc = 0;
    for (int i = 0; i < 8; i++) { acc += (u64)s[i + 1] + t[8 + i]; s[i + 1] = (u32)acc; acc >>= 32; }
    s[9] = (u32)acc;
    u64 ov = ((u64)s[9] << 32) | s[8];
    u64 m = ov * 977;
    acc = (u64)s[0] + (m & 0xffffffffULL); r.v[0] = (u32)acc; acc >>= 32;
    acc += (u64)s[1] + (m >> 32) + (ov & 0xffffffffULL); r.v[1] = (u32)acc; acc >>= 32;
    acc += (u64)s[2] + (ov >> 32); r.v[2] = (u32)acc; acc >>= 32;
    for (int i = 3; i < 8; i++) { acc += s[i]; r.v[i] = (u32)acc; acc >>= 32; }
    if (acc) fe_add_c(r);
}
HDI void fe_sqr(fe& r, const fe& a) { fe_mul(r, a, a); }
HDI void fe_sqrn(fe& r, const fe& a, int n) { fe_sqr(r, a); for (int i = 1; i < n; i++) fe_sqr(r, r); }
HDI void fe_norm(fe& r) {   // reduce to canonical [0,p)
    fe t = r;
    if (fe_add_c(t) == 1) r = t;
}
HDI void fe_neg(fe& r, const fe& a) { fe z; for (int i = 0; i < 8; i++) z.v[i] = 0; fe_sub(r, z, a); }
HD __noinline__ void fe_inv(fe& r, const fe& a) {   // a^(p-2)
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    fe_sqr(x2, a); fe_mul(x2, x2, a);
    fe_sqr(x3, x2); fe_mul(x3, x3, a);
    fe_sqrn(x6, x3, 3); fe_mul(x6, x6, x3);
    fe_sqrn(x9, x6, 3); fe_mul(x9, x9, x3);
    fe_sqrn(x11, x9, 2); fe_mul(x11, x11, x2);
    fe_sqrn(x22, x11, 11); fe_mul(x22, x22, x11);
    fe_sqrn(x44, x22, 22); fe_mul(x44, x44, x22);
    fe_sqrn(x88, x44, 44); fe_mul(x88, x88, x44);
    fe_sqrn(x176, x88, 88); fe_mul(x176, x176, x88);
    fe_sqrn(x220, x176, 44); fe_mul(x220, x220, x44);
    fe_sqrn(x223, x220, 3); fe_mul(x223, x223, x3);
    fe_sqrn(t, x223, 23); fe_mul(t, t, x22);
    fe_sqrn(t, t, 5); fe_mul(t, t, a);
    fe_sqrn(t, t, 3); fe_mul(t, t, x2);
    fe_sqrn(t, t, 2); fe_mul(r, t, a);
}

// ------------------------------------------------------------------ scalars mod n
HD void sc_add(u32 r[8], const u32 a[8], const u32 b[8]) {
    const u32 N[8] = {0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    u32 s[8]; u64 acc = 0;
    for (int i = 0; i < 8; i++) { acc += (u64)a[i] + b[i]; s[i] = (u32)acc; acc >>= 32; }
    int ge = (int)acc;
    if (!ge) {
        ge = 1;
        for (int i = 7; i >= 0; i--) { if (s[i] > N[i]) break; if (s[i] < N[i]) { ge = 0; break; } }
    }
    if (ge) {
        u64 br = 0;
        for (int i = 0; i < 8; i++) { u64 d = (u64)s[i] - N[i] - br; s[i] = (u32)d; br = (d >> 63) & 1; }
    }
    for (int i = 0; i < 8; i++) r[i] = s[i];
}
HDI void be32_to_limbs(const u8* b, u32 l[8]) { for (int i = 0; i < 8; i++) l[7 - i] = ld32be(b + 4 * i); }
HDI void limbs_to_be32(const u32 l[8], u8* b) { for (int i = 0; i < 8; i++) st32be(b + 4 * i, l[7 - i]); }

// ------------------------------------------------------------------ secp256k1 group
struct jp { fe x, y, z; };

// fixed-base table: NWIN windows of WBITS bits, each with the multiples (1..2^WBITS-1)*2^(WBITS*i)*G,
// affine, 16 u32 per point.  WBITS=16 -> 16 point additions per scalar multiplication, 64 MB table.
#ifndef WBITS
#define WBITS 16
#endif
#define NWIN ((256 + WBITS - 1) / WBITS)
#define WMASK ((1u << WBITS) - 1)
#define TAB_WORDS ((size_t)NWIN * WMASK * 16)

HD __noinline__ void jp_mixed_add(jp& p, const fe& x2, const fe& y2) {   // p += (x2,y2); p != +-(x2,y2)
    fe z1z1, u2, s2, h, hh, i4, j, r, v, t;
    fe_sqr(z1z1, p.z);
    fe_mul(u2, x2, z1z1);
    fe_mul(s2, y2, p.z); fe_mul(s2, s2, z1z1);
    fe_sub(h, u2, p.x);
    fe_sqr(hh, h);
    fe_add(i4, hh, hh); fe_add(i4, i4, i4);
    fe_mul(j, h, i4);
    fe_sub(r, s2, p.y); fe_add(r, r, r);
    fe_mul(v, p.x, i4);
    fe x3, y3, z3;
    fe_sqr(x3, r); fe_sub(x3, x3, j); fe_sub(x3, x3, v); fe_sub(x3, x3, v);
    fe_sub(t, v, x3); fe_mul(y3, r, t);
    fe_mul(t, p.y, j); fe_add(t, t, t); fe_sub(y3, y3, t);
    fe_add(z3, p.z, h); fe_sqr(z3, z3); fe_sub(z3, z3, z1z1); fe_sub(z3, z3, hh);
    p.x = x3; p.y = y3; p.z = z3;
}
HD __noinline__ void jp_double(jp& p) {
    fe a, b, c, d, e, f, t, x3, y3, z3;
    fe_sqr(a, p.x); fe_sqr(b, p.y); fe_sqr(c, b);
    fe_add(t, p.x, b); fe_sqr(t, t); fe_sub(t, t, a); fe_sub(t, t, c); fe_add(d, t, t);
    fe_add(e, a, a); fe_add(e, e, a);
    fe_sqr(f, e);
    fe_sub(x3, f, d); fe_sub(x3, x3, d);
    fe_sub(t, d, x3); fe_mul(y3, e, t);
    fe c8; fe_add(c8, c, c); fe_add(c8, c8, c8); fe_add(c8, c8, c8);
    fe_sub(y3, y3, c8);
    fe_mul(z3, p.y, p.z); fe_add(z3, z3, z3);
    p.x = x3; p.y = y3; p.z = z3;
}
HD void jp_to_affine(const jp& p, fe& x, fe& y) {
    fe zi, zi2, zi3;
    fe_inv(zi, p.z);
    fe_sqr(zi2, zi); fe_mul(zi3, zi2, zi);
    fe_mul(x, p.x, zi2); fe_mul(y, p.y, zi3);
    fe_norm(x); fe_norm(y);
}
// Converts n Jacobian points to affine in place with a single field inversion (Montgomery's trick).
// x,y end up normalised, z is destroyed.  c: scratch space for n field elements.
HD __noinline__ void batch_affine(jp* p, int n, fe* c) {
    c[0] = p[0].z;
    for (int i = 1; i < n; i++) fe_mul(c[i], c[i - 1], p[i].z);
    fe inv;
    fe_inv(inv, c[n - 1]);
    for (int i = n - 1; i >= 0; i--) {
        fe zi, zi2;
        if (i) { fe_mul(zi, inv, c[i - 1]); fe_mul(inv, inv, p[i].z); }
        else zi = inv;
        fe_sqr(zi2, zi);
        fe_mul(p[i].x, p[i].x, zi2);
        fe_mul(zi2, zi2, zi);
        fe_mul(p[i].y, p[i].y, zi2);
        fe_norm(p[i].x); fe_norm(p[i].y);
    }
}
// k*G, k != 0
HD __noinline__ void scalar_mult_base(const u32 k[8], const u32* tab, jp& out) {
    bool first = true;
    for (int i = 0; i < NWIN; i++) {
        int pos = i * WBITS, sh = pos & 31;
        u32 d = k[pos >> 5] >> sh;
        if (sh + WBITS > 32 && (pos >> 5) < 7) d |= k[(pos >> 5) + 1] << (32 - sh);
        d &= WMASK;
        if (!d) continue;
        const u32* e = tab + ((size_t)i * WMASK + (d - 1)) * 16;
        fe ex, ey;
        for (int j = 0; j < 8; j++) { ex.v[j] = e[j]; ey.v[j] = e[8 + j]; }
        if (first) {
            out.x = ex; out.y = ey;
            for (int j = 0; j < 8; j++) out.z.v[j] = 0;
            out.z.v[0] = 1;
            first = false;
        } else jp_mixed_add(out, ex, ey);
    }
}
HD void ser_pub(const fe& x, const fe& y, u8 out[33]) {
    out[0] = 2 | (y.v[0] & 1);
    limbs_to_be32(x.v, out + 1);
}

// ------------------------------------------------------------------ config shared by host/device
enum { K_P2PKH = 0, K_P2SH = 1, K_P2WPKH = 2, K_P2TR = 3 };
#define ADDR_MAX 100
struct Cfg {
    int kind, nwords, purpose, coin;
    int fixedLen, pfxLen, sfxLen, icase, nomatch;
    int hrpLen; char hrp[8];
    int verP2pkh, verP2sh;
    int passLen; char pass[64];
    char pfx[48], sfx[48];
    u8 seed[32];
};

// ------------------------------------------------------------------ address encoding
HD __noinline__ int base58check(const u8 payload21[21], char* out) {
    u8 buf[25], h1[32], h2[32];
    bcopy(buf, payload21, 21);
    sha256(payload21, 21, h1);
    sha256(h1, 32, h2);
    bcopy(buf + 21, h2, 4);
    const char* A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    u8 dig[40]; int dl = 0;
    for (int i = 0; i < 25; i++) {
        u32 carry = buf[i];
        for (int j = 0; j < dl; j++) { carry += (u32)dig[j] << 8; dig[j] = carry % 58; carry /= 58; }
        while (carry) { dig[dl++] = carry % 58; carry /= 58; }
    }
    int n = 0;
    for (int i = 0; i < 25 && buf[i] == 0; i++) out[n++] = '1';
    for (int j = dl - 1; j >= 0; j--) out[n++] = A[dig[j]];
    out[n] = 0;
    return n;
}
HDI u32 bech_step(u32 chk, u32 v) {
    u32 b = chk >> 25;
    chk = ((chk & 0x1ffffff) << 5) ^ v;
    if (b & 1) chk ^= 0x3b6a57b2u;
    if (b & 2) chk ^= 0x26508e6du;
    if (b & 4) chk ^= 0x1ea119fau;
    if (b & 8) chk ^= 0x3d4233ddu;
    if (b & 16) chk ^= 0x2a1462b3u;
    return chk;
}
HD __noinline__ int bech32_encode(const char* hrp, int hrplen, int witver, const u8* prog, int plen, char* out) {
    const char* CS = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    u8 data[64]; int dn = 0;
    data[dn++] = witver;
    u32 acc = 0; int bits = 0;
    for (int i = 0; i < plen; i++) {
        acc = (acc << 8) | prog[i]; bits += 8;
        while (bits >= 5) { bits -= 5; data[dn++] = (acc >> bits) & 31; }
    }
    if (bits) data[dn++] = (acc << (5 - bits)) & 31;
    u32 chk = 1;
    for (int i = 0; i < hrplen; i++) chk = bech_step(chk, (u8)hrp[i] >> 5);
    chk = bech_step(chk, 0);
    for (int i = 0; i < hrplen; i++) chk = bech_step(chk, (u8)hrp[i] & 31);
    for (int i = 0; i < dn; i++) chk = bech_step(chk, data[i]);
    for (int i = 0; i < 6; i++) chk = bech_step(chk, 0);
    chk ^= (witver == 0) ? 1u : 0x2bc830a3u;
    int n = 0;
    for (int i = 0; i < hrplen; i++) out[n++] = hrp[i];
    out[n++] = '1';
    for (int i = 0; i < dn; i++) out[n++] = CS[data[i]];
    for (int i = 0; i < 6; i++) out[n++] = CS[(chk >> (5 * (5 - i))) & 31];
    out[n] = 0;
    return n;
}

// BIP86: p (affine, normalised internal key) -> Q = lift_x(x) + H_TapTweak(x)*G, left in Jacobian form
HD __noinline__ void tap_tweak(jp& p, const u32* tab) {
    fe x = p.x, ye = p.y;
    if (ye.v[0] & 1) { fe_neg(ye, p.y); fe_norm(ye); }
    u8 th[32], buf[96], t32[32];
    const u8 tag[8] = {'T', 'a', 'p', 'T', 'w', 'e', 'a', 'k'};
    sha256(tag, 8, th);
    bcopy(buf, th, 32); bcopy(buf + 32, th, 32);
    limbs_to_be32(x.v, buf + 64);
    sha256(buf, 96, t32);
    u32 tl[8]; be32_to_limbs(t32, tl);
    scalar_mult_base(tl, tab, p);
    jp_mixed_add(p, x, ye);
}
// address for the (affine, normalised) key (x,y): the public key, or the tweaked output key for P2TR
HD __noinline__ int build_addr(const Cfg& cfg, const fe& x, const fe& y, char* out) {
    u8 pub[33], h[20], pl[21];
    if (cfg.kind == K_P2TR) {
        u8 prog[32];
        limbs_to_be32(x.v, prog);
        return bech32_encode(cfg.hrp, cfg.hrpLen, 1, prog, 32, out);
    }
    ser_pub(x, y, pub);
    hash160(pub, 33, h);
    if (cfg.kind == K_P2WPKH) return bech32_encode(cfg.hrp, cfg.hrpLen, 0, h, 20, out);
    if (cfg.kind == K_P2SH) {
        u8 script[22]; script[0] = 0; script[1] = 20; bcopy(script + 2, h, 20);
        hash160(script, 22, h);
        pl[0] = cfg.verP2sh;
    } else pl[0] = cfg.verP2pkh;
    bcopy(pl + 1, h, 20);
    return base58check(pl, out);
}

HDI char lc(char c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
HD bool match_addr(const Cfg& c, const char* a, int alen) {
    if (c.nomatch) return false;
    if (alen < c.fixedLen + c.pfxLen + c.sfxLen) return false;
    for (int i = 0; i < c.pfxLen; i++) {
        char x = a[c.fixedLen + i];
        if (c.icase) x = lc(x);
        if (x != c.pfx[i]) return false;
    }
    for (int j = 0; j < c.sfxLen; j++) {
        char x = a[alen - c.sfxLen + j];
        if (c.icase) x = lc(x);
        if (x != c.sfx[j]) return false;
    }
    return true;
}

// ------------------------------------------------------------------ BIP39 / BIP32
HD int build_mnemonic(const u8* ent, int nwords, char* out) {
    int eb = nwords * 4 / 3;
    u8 bits[33], hh[32];
    bcopy(bits, ent, eb);
    sha256(ent, eb, hh);
    bits[eb] = hh[0];
    int n = 0;
    for (int w = 0; w < nwords; w++) {
        int idx = 0;
        for (int b = 0; b < 11; b++) { int pos = w * 11 + b; idx = (idx << 1) | ((bits[pos >> 3] >> (7 - (pos & 7))) & 1); }
#ifdef __CUDA_ARCH__
        const char* wd = WL_D[idx];
#else
        const char* wd = WL_H[idx];
#endif
        if (w) out[n++] = ' ';
        for (int i = 0; i < 8 && wd[i]; i++) out[n++] = wd[i];
    }
    out[n] = 0;
    return n;
}

// child = (IL + k) mod n, chain = IR
HD void ckd_finish(u8 k[32], u8 c[32], const u8 I[64]) {
    u32 a[8], b[8], r[8];
    be32_to_limbs(I, a); be32_to_limbs(k, b);
    sc_add(r, a, b);
    limbs_to_be32(r, k);
    bcopy(c, I + 32, 32);
}
HD void ckd_hard(u8 k[32], u8 c[32], u32 idx) {
    u8 data[37], I[64];
    hctx h;
    data[0] = 0; bcopy(data + 1, k, 32); st32be(data + 33, idx | 0x80000000u);
    hmac512_init(h, c, 32);
    hmac512_final(h, data, 37, I);
    ckd_finish(k, c, I);
}
HD __noinline__ void pub_of(const u8 k[32], const u32* tab, fe& x, fe& y) {
    u32 kl[8]; be32_to_limbs(k, kl);
    jp p;
    scalar_mult_base(kl, tab, p);
    jp_to_affine(p, x, y);
}

#define NADDR 20
// Derives the 20 addresses for the mnemonic made from `ent`.
// store != NULL : all 20 addresses are written (store[i], NUL-terminated) and the first match index is returned.
// store == NULL : returns at the first matching address.  Returns -1 when nothing matches.
HD int derive_all(const Cfg& cfg, const u8* ent, const u32* tab, char (*store)[ADDR_MAX]) {
    char mn[260];
    int mnl = build_mnemonic(ent, cfg.nwords, mn);
    u8 salt[80];
    const u8 pre[8] = {'m', 'n', 'e', 'm', 'o', 'n', 'i', 'c'};
    bcopy(salt, pre, 8);
    for (int i = 0; i < cfg.passLen; i++) salt[8 + i] = (u8)cfg.pass[i];
    u8 seed[64], I[64], k[32], c[32], data[37];
    pbkdf2_sha512_2048((const u8*)mn, mnl, salt, 8 + cfg.passLen, seed);
    const u8 bs[12] = {'B', 'i', 't', 'c', 'o', 'i', 'n', ' ', 's', 'e', 'e', 'd'};
    hctx h;
    hmac512_init(h, bs, 12);
    hmac512_final(h, seed, 64, I);
    bcopy(k, I, 32); bcopy(c, I + 32, 32);
    ckd_hard(k, c, cfg.purpose);
    ckd_hard(k, c, cfg.coin);
    ckd_hard(k, c, 0);
    fe px, py;
    pub_of(k, tab, px, py); ser_pub(px, py, data);
    st32be(data + 33, 0);                   // external chain
    hmac512_init(h, c, 32);
    hmac512_final(h, data, 37, I);
    ckd_finish(k, c, I);
    pub_of(k, tab, px, py); ser_pub(px, py, data);

    // the 20 children share the HMAC key (chain code) and one field inversion
    jp P[NADDR]; fe scratch[NADDR];
    u32 kl[8];
    be32_to_limbs(k, kl);
    hmac512_init(h, c, 32);
    for (int i = 0; i < NADDR; i++) {
        u32 il[8], ki[8];
        st32be(data + 33, i);
        hmac512_final(h, data, 37, I);
        be32_to_limbs(I, il);
        sc_add(ki, il, kl);
        scalar_mult_base(ki, tab, P[i]);
    }
    batch_affine(P, NADDR, scratch);
    if (cfg.kind == K_P2TR) {
        for (int i = 0; i < NADDR; i++) tap_tweak(P[i], tab);
        batch_affine(P, NADDR, scratch);
    }
    int first = -1;
    for (int i = 0; i < NADDR; i++) {
        char tmp[ADDR_MAX];
        char* dst = store ? store[i] : tmp;
        int len = build_addr(cfg, P[i].x, P[i].y, dst);
        if (first < 0 && match_addr(cfg, dst, len)) {
            first = i;
            if (!store) return i;
        }
    }
    return first;
}

// ------------------------------------------------------------------ GPU kernels
struct Result { int found; int index; u8 entropy[32]; char addr[ADDR_MAX]; };
__device__ Result d_res;

HD void make_entropy(const Cfg& cfg, u64 ctr, u8 ent[32]) {
    u8 m[40];
    bcopy(m, cfg.seed, 32);
    for (int i = 0; i < 8; i++) m[32 + i] = (u8)(ctr >> (8 * i));
    sha256(m, 40, ent);
}

__global__ void search_kernel(Cfg cfg, const u32* d_tab, u64 base, int iters) {
    u64 total = (u64)gridDim.x * blockDim.x;
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = 0; it < iters; it++) {
        if (*(volatile int*)&d_res.found) return;
        u8 ent[32];
        make_entropy(cfg, base + (u64)it * total + tid, ent);
        int idx = derive_all(cfg, ent, d_tab, nullptr);
        if (idx >= 0) {
            if (atomicCAS(&d_res.found, 0, 1) == 0) {
                for (int i = 0; i < 32; i++) d_res.entropy[i] = ent[i];
                d_res.index = idx;
                __threadfence();
                d_res.found = 2;
            }
            return;
        }
    }
}

__global__ void selftest_kernel(Cfg cfg, const u32* d_tab, const u8* ent, char* out) {
    if (threadIdx.x || blockIdx.x) return;
    char (*st)[ADDR_MAX] = (char (*)[ADDR_MAX])out;
    derive_all(cfg, ent, d_tab, st);
}

// ====================================================================== host side
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
static std::vector<u32> h_tabv;
static u32* h_tab;
static u32* d_tab;

static void hex_to_fe(const char* hex, fe& r) {   // 64 hex chars, big endian
    for (int i = 0; i < 8; i++) { char s[9]; memcpy(s, hex + 8 * (7 - i), 8); s[8] = 0; r.v[i] = (u32)strtoul(s, nullptr, 16); }
}
static void build_table() {
    h_tabv.resize(TAB_WORDS);
    h_tab = h_tabv.data();
    fe bx, by;
    hex_to_fe("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", bx);
    hex_to_fe("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", by);
    std::vector<jp> pts(WMASK);
    std::vector<fe> scratch(WMASK);
    for (int i = 0; i < NWIN; i++) {   // base = 2^(WBITS*i) * G
        pts[0].x = bx; pts[0].y = by;
        for (int j = 0; j < 8; j++) pts[0].z.v[j] = 0;
        pts[0].z.v[0] = 1;
        pts[1] = pts[0]; jp_double(pts[1]);
        for (u32 m = 2; m < WMASK; m++) { pts[m] = pts[m - 1]; jp_mixed_add(pts[m], bx, by); }
        jp nb = pts[WMASK - 1];
        jp_mixed_add(nb, bx, by);       // 2^WBITS * base
        batch_affine(pts.data(), WMASK, scratch.data());
        u32* e = h_tab + (size_t)i * WMASK * 16;
        for (u32 m = 0; m < WMASK; m++, e += 16)
            for (int j = 0; j < 8; j++) { e[j] = pts[m].x.v[j]; e[8 + j] = pts[m].y.v[j]; }
        jp_to_affine(nb, bx, by);
    }
}
static void upload_table() {
    CK(cudaMalloc(&d_tab, TAB_WORDS * sizeof(u32)));
    CK(cudaMemcpy(d_tab, h_tab, TAB_WORDS * sizeof(u32), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------ probability / feasibility (pure C++)
struct Big { u32 w[8]; };
static Big big0() { Big b; memset(&b, 0, sizeof b); return b; }
static void big_mul_small(Big& a, u32 m) { u64 c = 0; for (int i = 0; i < 8; i++) { c += (u64)a.w[i] * m; a.w[i] = (u32)c; c >>= 32; } }
static void big_add_small(Big& a, u32 m) { u64 c = m; for (int i = 0; i < 8; i++) { c += a.w[i]; a.w[i] = (u32)c; c >>= 32; } }
static Big big_add(const Big& a, const Big& b) { Big r; u64 c = 0; for (int i = 0; i < 8; i++) { c += (u64)a.w[i] + b.w[i]; r.w[i] = (u32)c; c >>= 32; } return r; }
static Big big_sub(const Big& a, const Big& b) { Big r; u64 br = 0; for (int i = 0; i < 8; i++) { u64 d = (u64)a.w[i] - b.w[i] - br; r.w[i] = (u32)d; br = (d >> 63) & 1; } return r; }
static int big_cmp(const Big& a, const Big& b) { for (int i = 7; i >= 0; i--) { if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1; } return 0; }
static Big big_pow2(int bit) { Big b = big0(); b.w[bit / 32] = 1u << (bit % 32); return b; }
static Big big_mul(const Big& a, const Big& b) {   // b small-ish (fits), schoolbook truncated to 256 bits
    Big r = big0();
    for (int i = 0; i < 8; i++) { u64 c = 0; for (int j = 0; i + j < 8; j++) { c += (u64)a.w[i] * b.w[j] + r.w[i + j]; r.w[i + j] = (u32)c; c >>= 32; } }
    return r;
}
static long double big_ld(const Big& a) { long double r = 0; for (int i = 7; i >= 0; i--) r = r * 4294967296.0L + a.w[i]; return r; }

static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const char* BECH = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// probability that a uniformly random 24-byte tail h (hash160 + checksum) yields a base58 body whose
// first |D| digits are D.  version byte ver (0 => body is what follows the leading '1').
static long double b58_head_prob(int ver, const std::vector<int>& D) {
    Big pw[40]; pw[0] = big0(); pw[0].w[0] = 1;
    for (int i = 1; i < 40; i++) { pw[i] = pw[i - 1]; big_mul_small(pw[i], 58); }
    long double total = 0;
    int k = 0;
    std::vector<int> rest = D;
    Big Rlo, Rhi;
    if (ver == 0) {
        while (k < (int)D.size() && D[k] == 0) k++;
        rest.assign(D.begin() + k, D.end());
        if (k > 23) return 0;
        if (rest.empty()) return powl(2.0L, -8 * k);
        Rlo = big_pow2(8 * (23 - k));
        Rhi = big_sub(big_pow2(8 * (24 - k)), [] { Big o = big0(); o.w[0] = 1; return o; }());
    } else {
        Rlo = big0(); Rlo.w[6] = ver;
        Big hi = big0(); hi.w[6] = ver + 1;
        Big one = big0(); one.w[0] = 1;
        Rhi = big_sub(hi, one);
    }
    Big val = big0();
    for (int d : rest) { big_mul_small(val, 58); big_add_small(val, d); }
    int L = (int)rest.size();
    for (int N = L; N <= 35; N++) {
        Big lo = big_mul(val, pw[N - L]);
        Big hi = big_sub(big_add(lo, pw[N - L]), [] { Big o = big0(); o.w[0] = 1; return o; }());
        Big a = big_cmp(lo, Rlo) > 0 ? lo : Rlo;
        Big b = big_cmp(hi, Rhi) < 0 ? hi : Rhi;
        if (big_cmp(a, b) > 0) continue;
        Big n = big_sub(b, a); big_add_small(n, 1);
        total += ldexpl(big_ld(n), -192);
    }
    return total;
}
static int b58_index(char c) { const char* p = strchr(B58, c); return (p && c) ? (int)(p - B58) : -1; }
static int b58_count(char c, bool icase) {   // number of alphabet symbols matching c
    int n = 0;
    for (const char* p = B58; *p; p++) if (*p == c || (icase && lc(*p) == lc(c))) n++;
    return n;
}
// probability that the body (after the fixed 1st char) starts with pattern
static long double b58_prefix_prob(int ver, const std::string& pat, bool icase) {
    size_t headN = icase ? std::min<size_t>(3, pat.size()) : pat.size();
    long double tail = 1;
    for (size_t i = headN; i < pat.size(); i++) tail *= (long double)b58_count(pat[i], icase) / 58.0L;
    // case variants of the head
    std::vector<std::string> variants{pat.substr(0, headN)};
    if (icase) {
        for (size_t i = 0; i < headN; i++) {
            std::vector<std::string> nv;
            for (auto& s : variants) {
                nv.push_back(s);
                char o = isupper((unsigned char)s[i]) ? tolower(s[i]) : toupper(s[i]);
                if (o != s[i]) { std::string t = s; t[i] = o; nv.push_back(t); }
            }
            variants = nv;
        }
    }
    long double p = 0;
    for (auto& v : variants) {
        std::vector<int> D;
        bool ok = true;
        for (char c : v) { int ix = b58_index(c); if (ix < 0) { ok = false; break; } D.push_back(ix); }
        if (!ok) continue;
        if (ver == 0) p += b58_head_prob(0, D);
        else for (int c = 1; c < 58; c++) {   // first char of the address is variable (e.g. m/n)
            std::vector<int> E{c}; E.insert(E.end(), D.begin(), D.end());
            p += b58_head_prob(ver, E);
        }
    }
    return p * tail;
}

// ------------------------------------------------------------------ CLI
struct Opts {
    std::string kind = "p2wpkh", net = "mainnet", prefix, suffix, pass;
    bool icase = false, estimateOnly = false, selftest = false, noBench = false, color = true;
    int words = 12, blocks = 0, threads = 128, iters = 0, device = 0;
    double rate = 0;
};
static void usage() {
    puts("cuda-vanity: GPU search of BIP39 mnemonics whose first 20 addresses contain a vanity pattern\n\n"
         "  --kind     p2pkh | p2sh | p2wpkh | p2tr      (BIP44 / BIP49 / BIP84 / BIP86)   [p2wpkh]\n"
         "  --network  mainnet | testnet | signet | regtest                                [mainnet]\n"
         "  --prefix   chars required right AFTER the fixed prefix (1 / 3 / bc1q / bc1p / tb1q ...)\n"
         "  --suffix   chars required at the end of the address\n"
         "  --ignore-case   case-insensitive match (base58 only; bech32 is always lowercase)\n"
         "  --words    12 | 24                                                             [12]\n"
         "  --passphrase STR   BIP39 passphrase (<= 63 chars)\n"
         "  --estimate         check feasibility, benchmark the GPU, print time estimate, exit\n"
         "  --rate N           use N mnemonics/s instead of benchmarking (no GPU needed with --estimate)\n"
         "  --no-bench         skip the start-up benchmark\n"
         "  --no-color         disable colours (also honoured: NO_COLOR; auto-off when not a terminal)\n"
         "  --blocks N --threads N --iters N --device N   launch tuning\n"
         "  --selftest         verify the implementation against known BIP vectors (CPU and GPU)\n\n"
         "Example: ./build/vanity --kind p2wpkh --network mainnet --prefix qq -> bc1qqq...\n");
}
static std::string lower(std::string s) { for (auto& c : s) c = lc(c); return s; }

struct Net { const char* hrp; int p2pkh, p2sh, coin; };
static Net get_net(const std::string& n, bool& ok) {
    ok = true;
    if (n == "mainnet" || n == "main") return {"bc", 0x00, 0x05, 0};
    if (n == "testnet" || n == "test" || n == "signet") return {"tb", 0x6f, 0xc4, 1};
    if (n == "regtest") return {"bcrt", 0x6f, 0xc4, 1};
    ok = false; return {"", 0, 0, 0};
}

static const char *C_R = "", *C_G = "", *C_Y = "", *C_C = "", *C_B = "", *C_D = "", *C_0 = "";
static void init_colors(bool on) {
    if (!on) return;
    C_R = "\033[1;31m"; C_G = "\033[1;32m"; C_Y = "\033[33m"; C_C = "\033[36m";
    C_B = "\033[1m"; C_D = "\033[2m"; C_0 = "\033[0m";
}
// prints an address with the matched prefix/suffix characters highlighted
static void print_addr_hl(const char* a, const Cfg& c) {
    int n = (int)strlen(a);
    for (int i = 0; i < n; i++) {
        bool hl = (i >= c.fixedLen && i < c.fixedLen + c.pfxLen) || (c.sfxLen && i >= n - c.sfxLen);
        if (hl) printf("%s%c%s", C_G, a[i], C_0); else putchar(a[i]);
    }
}

static std::string human(double s) {
    char b[64];
    if (!std::isfinite(s)) return "never";
    if (s < 1) snprintf(b, sizeof b, "%.2f s", s);
    else if (s < 120) snprintf(b, sizeof b, "%.1f s", s);
    else if (s < 7200) snprintf(b, sizeof b, "%.1f min", s / 60);
    else if (s < 172800) snprintf(b, sizeof b, "%.1f hours", s / 3600);
    else if (s < 86400.0 * 730) snprintf(b, sizeof b, "%.1f days", s / 86400);
    else if (s < 86400.0 * 365 * 1e6) snprintf(b, sizeof b, "%.1f years", s / (86400.0 * 365));
    else snprintf(b, sizeof b, "%.2e years", s / (86400.0 * 365));
    return b;
}

static void print_estimate(long double pAddr, double rate) {
    long double q = 1 - powl(1 - pAddr, 20);   // success probability per mnemonic
    if (q <= 0) q = 20 * pAddr;
    long double expMn = 1 / q;
    printf("  P(one address matches)  : %s%.3Le%s  (1 in %.3Le)\n", C_Y, pAddr, C_0, 1 / pAddr);
    printf("  P(one mnemonic wins)    : %.3Le  (20 addresses each) -> ~%.3Le mnemonics on average\n", q, expMn);
    if (rate > 0) {
        printf("  speed                   : %s%.0f mnemonics/s%s (%.0f addresses/s)\n", C_C, rate, C_0, rate * 20);
        double lq = (double)log1pl(-q);
        auto t = [&](double pct) { return (double)(logl(1 - pct) / lq) / rate; };
        printf("  expected time (mean)    : %s%s%s\n", C_B, human((double)expMn / rate).c_str(), C_0);
        printf("  50%% done within         : %s\n", human(t(0.5)).c_str());
        printf("  90%% done within         : %s\n", human(t(0.9)).c_str());
        printf("  99%% done within         : %s\n", human(t(0.99)).c_str());
    }
}


static const char* KIND_NAMES[] = {"p2pkh", "p2sh", "p2wpkh", "p2tr"};

static void fill_cfg_base(Cfg& cfg, const Opts& o, int kind, const Net& net) {
    memset(&cfg, 0, sizeof cfg);
    cfg.kind = kind;
    cfg.nwords = o.words;
    static const int purposes[] = {44, 49, 84, 86};
    cfg.purpose = purposes[kind];
    cfg.coin = net.coin;
    cfg.hrpLen = (int)strlen(net.hrp);
    memcpy(cfg.hrp, net.hrp, cfg.hrpLen);
    cfg.verP2pkh = net.p2pkh; cfg.verP2sh = net.p2sh;
    cfg.passLen = (int)o.pass.size();
    memcpy(cfg.pass, o.pass.data(), cfg.passLen);
    cfg.fixedLen = (kind >= K_P2WPKH) ? cfg.hrpLen + 2 : 1;
}

static int run_selftest() {
    build_table();
    int bad = 0;
    // derivation of the BIP39 test mnemonic (all-zero entropy): "abandon x11 about"
    struct V { int kind; const char* a0; const char* a1; } vec[] = {
        {K_P2PKH, "1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA", "1Ak8PffB2meyfYnbXZR9EGfLfFZVpzJvQP"},
        {K_P2SH, "37VucYSaXLCAsxYyAPfbSi9eh4iEcbShgf", ""},
        {K_P2WPKH, "bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu", "bc1qnjg0jd8228aq7egyzacy8cys3knf9xvrerkf9g"},
        {K_P2TR, "bc1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr", "bc1p4qhjn9zdvkux4e44uhx8tc55attvtyu358kutcqkudyccelu0was9fqzwh"}};
    Opts o; bool ok; Net net = get_net("mainnet", ok);
    u8 ent[32] = {0};
    char mn[260]; build_mnemonic(ent, 12, mn);
    printf("mnemonic: %s\n", mn);
    u8* d_ent; char* d_out;
    CK(cudaMalloc(&d_ent, 32)); CK(cudaMalloc(&d_out, 20 * ADDR_MAX));
    CK(cudaMemcpy(d_ent, ent, 32, cudaMemcpyHostToDevice));
    upload_table();
    for (auto& v : vec) {
        Cfg cfg; fill_cfg_base(cfg, o, v.kind, net); cfg.nomatch = 1;
        static char st[20][ADDR_MAX];
        derive_all(cfg, ent, h_tab, st);
        char gout[20 * ADDR_MAX];
        selftest_kernel<<<1, 1>>>(cfg, d_tab, d_ent, d_out);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(gout, d_out, sizeof gout, cudaMemcpyDeviceToHost));
        bool gpuOk = true;
        for (int i = 0; i < 20; i++) if (strcmp(st[i], gout + i * ADDR_MAX)) gpuOk = false;
        bool c0 = !*v.a0 || !strcmp(st[0], v.a0), c1 = !*v.a1 || !strcmp(st[1], v.a1);
        auto tag = [](bool b) { return std::string(b ? C_G : C_R) + (b ? "OK " : "BAD") + C_0; };
        printf("%-7s idx0 %s  idx1 %s  GPU==CPU(20 addrs) %s\n          %s\n          %s\n", KIND_NAMES[v.kind],
               tag(c0).c_str(), tag(c1).c_str(), tag(gpuOk).c_str(), st[0], st[1]);
        if (!c0 || !c1 || !gpuOk) bad++;
    }
    printf("%s%s%s\n", bad ? C_R : C_G, bad ? "SELFTEST FAILED" : "SELFTEST PASSED", C_0);
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--kind") o.kind = lower(next());
        else if (a == "--network") o.net = lower(next());
        else if (a == "--prefix") o.prefix = next();
        else if (a == "--suffix") o.suffix = next();
        else if (a == "--ignore-case") o.icase = true;
        else if (a == "--words") o.words = atoi(next().c_str());
        else if (a == "--passphrase") o.pass = next();
        else if (a == "--estimate") o.estimateOnly = true;
        else if (a == "--rate") o.rate = atof(next().c_str());
        else if (a == "--no-bench") o.noBench = true;
        else if (a == "--no-color") o.color = false;
        else if (a == "--blocks") o.blocks = atoi(next().c_str());
        else if (a == "--threads") o.threads = atoi(next().c_str());
        else if (a == "--iters") o.iters = atoi(next().c_str());
        else if (a == "--device") o.device = atoi(next().c_str());
        else if (a == "--selftest") o.selftest = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
    init_colors(o.color && isatty(1) && !getenv("NO_COLOR"));
    if (o.selftest) { CK(cudaSetDevice(o.device)); return run_selftest(); }

    // ---- kind / network
    int kind = -1;
    const std::string& k = o.kind;
    if (k == "p2pkh" || k == "legacy" || k == "bip44") kind = K_P2PKH;
    else if (k == "p2sh" || k == "p2sh-p2wpkh" || k == "nested" || k == "bip49") kind = K_P2SH;
    else if (k == "p2wpkh" || k == "segwit" || k == "bech32" || k == "bip84") kind = K_P2WPKH;
    else if (k == "p2tr" || k == "taproot" || k == "bip86") kind = K_P2TR;
    if (kind < 0) { fprintf(stderr, "unknown --kind '%s'\n", o.kind.c_str()); return 2; }
    bool nok; Net net = get_net(o.net, nok);
    if (!nok) { fprintf(stderr, "unknown --network '%s'\n", o.net.c_str()); return 2; }
    if (o.words != 12 && o.words != 24) { fprintf(stderr, "--words must be 12 or 24\n"); return 2; }
    if (o.pass.size() > 63) { fprintf(stderr, "--passphrase too long (max 63)\n"); return 2; }
    bool bech = kind >= K_P2WPKH;

    // ---- pattern validation & feasibility
    std::string pfx = o.prefix, sfx = o.suffix;
    bool icase = o.icase;
    if (bech) {
        std::string lp = lower(pfx), ls = lower(sfx);
        if (lp != pfx || ls != sfx) puts("note: bech32 addresses are lowercase, pattern lowercased");
        pfx = lp; sfx = ls; icase = false;
        if (o.icase) puts("note: --ignore-case is irrelevant for bech32 (always lowercase)");
    } else if (icase) { pfx = lower(pfx); sfx = lower(sfx); }
    if (pfx.size() > 40 || sfx.size() > 40) { fprintf(stderr, "pattern too long\n"); return 2; }

    const char* ALPHA = bech ? BECH : B58;
    auto badchars = [&](const std::string& s) {
        std::string bad;
        for (char c : s) {
            bool in = false;
            for (const char* p = ALPHA; *p; p++) if (*p == c || (icase && lc(*p) == lc(c))) in = true;
            if (!in && bad.find(c) == std::string::npos) bad += c;
        }
        return bad;
    };
    std::string fixedDesc;
    if (bech) fixedDesc = std::string(net.hrp) + "1" + (kind == K_P2TR ? "p" : "q");
    else if (kind == K_P2PKH) fixedDesc = net.p2pkh == 0 ? "1" : "m or n";
    else fixedDesc = net.p2sh == 5 ? "3" : "2";
    printf("%skind=%s  network=%s  path=m/%d'/%d'/0'/0/0..19  words=%d%s%s\n", C_C, KIND_NAMES[kind], o.net.c_str(),
           kind == K_P2PKH ? 44 : kind == K_P2SH ? 49 : kind == K_P2WPKH ? 84 : 86, net.coin, o.words, o.pass.empty() ? "" : "  (passphrase set)", C_0);
    printf("fixed address prefix : %s   | your prefix: '%s'  suffix: '%s'%s\n", fixedDesc.c_str(), pfx.c_str(), sfx.c_str(), icase ? "  (ignore-case)" : "");

    std::string bp = badchars(pfx) + badchars(sfx);
    if (!bp.empty()) {
        printf("%sIMPOSSIBLE:%s character(s) '%s' can never appear in a %s address.\n", C_R, C_0, bp.c_str(),
               bech ? "bech32 (alphabet: qpzry9x8gf2tvdw0s3jn54khce6mua7l; missing: 1 b i o)" : "base58 (missing: 0 O I l)");
        if (bech) puts("            valid characters: qpzry9x8gf2tvdw0s3jn54khce6mua7l");
        return 3;
    }
    int bodyMax = bech ? (kind == K_P2TR ? 52 + 6 : 32 + 6) : 33;
    if ((int)(pfx.size() + sfx.size()) > bodyMax) {
        printf("%sIMPOSSIBLE:%s prefix+suffix (%zu chars) longer than the variable part of the address (%d chars).\n", C_R, C_0, pfx.size() + sfx.size(), bodyMax);
        return 3;
    }
    long double pAddr = 1;
    if (bech) {
        pAddr = powl(32.0L, -(long double)(pfx.size() + sfx.size()));
        if (kind == K_P2TR) {   // 256 bits = 52 chars, last data char only carries 1 bit (value 0 or 16 => 'q' or 's')
            int lastDataPos = 52;   // 1-based position among chars after "...1p"
            auto chk = [&](int pos, char c) { if (pos == lastDataPos) { if (c != 'q' && c != 's') { printf("%sIMPOSSIBLE:%s last taproot data char can only be 'q' or 's'\n", C_R, C_0); exit(3); } pAddr *= 16; } };
            for (size_t i = 0; i < pfx.size(); i++) chk((int)i + 1, pfx[i]);
            for (size_t j = 0; j < sfx.size(); j++) chk(58 - (int)sfx.size() + (int)j + 1, sfx[j]);
        }
    } else {
        if (!pfx.empty()) {
            long double pp = b58_prefix_prob(kind == K_P2PKH ? net.p2pkh : net.p2sh, pfx, icase);
            if (pp <= 0) {
                printf("%sIMPOSSIBLE:%s no %s address can have '%s' right after its fixed prefix (%s).\n", C_R, C_0, KIND_NAMES[kind], pfx.c_str(), fixedDesc.c_str());
                std::string ok;
                for (const char* p = B58; *p; p++) {
                    std::string s(1, *p);
                    if (b58_prefix_prob(kind == K_P2PKH ? net.p2pkh : net.p2sh, s, false) > 0) ok += *p;
                }
                printf("            characters possible right after the fixed prefix: %s%s%s\n", C_G, ok.c_str(), C_0);
                return 3;
            }
            pAddr *= pp;
        }
        for (char c : sfx) pAddr *= (long double)b58_count(c, icase) / 58.0L;
    }
    if (pfx.empty() && sfx.empty()) pAddr = 1;
    printf("feasibility: %sOK%s\n", C_G, C_0);

    // ---- CUDA init / benchmark
    Cfg cfg;
    fill_cfg_base(cfg, o, kind, net);
    cfg.pfxLen = (int)pfx.size(); cfg.sfxLen = (int)sfx.size(); cfg.icase = icase;
    memcpy(cfg.pfx, pfx.data(), pfx.size()); memcpy(cfg.sfx, sfx.data(), sfx.size());
    {
        FILE* f = fopen("/dev/urandom", "rb");
        if (!f || fread(cfg.seed, 1, 32, f) != 32) { fprintf(stderr, "cannot read /dev/urandom\n"); return 1; }
        fclose(f);
    }
    if (o.estimateOnly && o.rate > 0) { print_estimate(pAddr, o.rate); return 0; }

    CK(cudaSetDevice(o.device));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, o.device));
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    double tb0 = now_s();
    build_table();
    upload_table();
    printf("precomputed table: %.0f MB (%d-bit windows) built in %.1f s\n", TAB_WORDS * 4 / 1048576.0, WBITS, now_s() - tb0);
    // many more blocks than fit at once: keeps every SM full for the whole launch
    int blocks = o.blocks ? o.blocks : prop.multiProcessorCount * 64;
    int threads = o.threads;
    u64 perIter = (u64)blocks * threads;
    u64 ctr = 0;
    int iters = o.iters ? o.iters : 1;
    double rate = o.rate;

    auto launch = [&](int it) -> double {
        double t0 = now_s();
        search_kernel<<<blocks, threads>>>(cfg, d_tab, ctr, it);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        ctr += perIter * it;
        return now_s() - t0;
    };
    if (rate <= 0 && !o.noBench) {
        printf("benchmarking (%d blocks x %d threads)...\n", blocks, threads);
        Cfg bcfg = cfg; bcfg.nomatch = 1;
        Cfg save = cfg; cfg = bcfg;
        launch(1);                                  // warm-up
        double dt = launch(1);
        if (!o.iters) iters = std::max(1, std::min(1000, (int)(0.7 / std::max(dt, 1e-3))));
        double tot = 0; u64 n = 0;
        while (tot < 2.0) { double d = launch(iters); tot += d; n += perIter * iters; }
        rate = n / tot;
        cfg = save;
    } else if (!o.iters) iters = 1;
    printf("%sestimate:%s\n", C_B, C_0);
    print_estimate(pAddr, rate);
    if (o.estimateOnly) return 0;

    // ---- search
    Result zero; memset(&zero, 0, sizeof zero);
    CK(cudaMemcpyToSymbol(d_res, &zero, sizeof zero));
    printf("%ssearching... (Ctrl-C to abort)%s\n", C_B, C_0);
    double t0 = now_s(), lastp = t0; u64 done = 0;
    Result res;
    for (;;) {
        double dt = launch(iters);
        (void)dt;
        done += perIter * iters;
        CK(cudaMemcpyFromSymbol(&res, d_res, sizeof res));
        if (res.found) break;
        double t = now_s();
        if (t - lastp >= 5) {
            lastp = t;
            double r = done / (t - t0);
            long double q = 1 - powl(1 - pAddr, 20);
            double expSec = (double)(1 / q) / r;
            printf("%s  %llu mnemonics, %.0f/s, elapsed %s (avg total expected %s)%s\n", C_D, (unsigned long long)done, r, human(t - t0).c_str(), human(expSec).c_str(), C_0);
            fflush(stdout);
        }
    }
    double el = now_s() - t0;

    // ---- verify on the CPU and print
    static char st[20][ADDR_MAX];
    int mi = derive_all(cfg, res.entropy, h_tab, st);
    char mn[260]; build_mnemonic(res.entropy, cfg.nwords, mn);
    printf("\n%sFOUND%s after ~%llu mnemonics in %s\n", C_G, C_0, (unsigned long long)done, human(el).c_str());
    bool vok = (mi >= 0 && mi == res.index);
    printf("CPU verification: %s%s%s\n\n", vok ? C_G : C_R, vok ? "OK (CPU re-derived the same matching address)" : "MISMATCH (!)", C_0);
    printf("mnemonic  : %s%s%s\n", C_Y, mn, C_0);
    if (!o.pass.empty()) printf("passphrase: %s\n", o.pass.c_str());
    int purpose = cfg.purpose;
    for (int i = 0; i < 20; i++) {
        printf("  %sm/%d'/%d'/0'/0/%-2d%s  ", C_D, purpose, cfg.coin, i, C_0);
        if (i == mi) { print_addr_hl(st[i], cfg); printf("   %s<== match%s\n", C_G, C_0); }
        else printf("%s\n", st[i]);
    }
    printf("\n%sKEEP THIS MNEMONIC SECRET. Anyone with it controls these addresses.%s\n", C_R, C_0);
    return 0;
}
