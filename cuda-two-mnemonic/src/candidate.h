// Candidate derivation and the concatenation test. Shared by the CUDA kernel
// and the CPU fallback so both search exactly the same space.
//
// Bit layout
//   12-word mnemonic = 128 bits entropy + 4 bits checksum  (132 bits)
//   24-word mnemonic = 256 bits entropy + 8 bits checksum  (264 bits)
//
//   F || Y = entF(128) csF(4) entY(128) csY(4)
//            \_________ 256-bit entropy ___/\__ 8 bits __/
//                                  = entY[124..127] csY
//
// F||Y is a valid 24-word mnemonic iff
//   H = SHA256(entF | csF | entY[0..123])[0] == entY[124..127] << 4 | csY
//
// 1. The "final word" shortcut
//   Take a valid mnemonic F and 11 random words for Y (121 bits). The 12th
//   word of Y is entY[121..127] (7 bits) + csY (4 bits). H only reads the
//   first 3 of those 11 bits, so for each of their 8 values H *dictates* the
//   other 8: of the 2048 possible final words only 8 can make F||Y valid.
//   Each of them is a valid 12-word mnemonic on its own with probability 1/16
//   (its real checksum must equal the 4 bits H asked for).
//
// 2. Checksum table
//   Y's real checksum depends on Y only, so it is not hashed per candidate.
//   Each launch uses a shared set of Y prefixes (124 bits) and a table with
//   the checksum of each prefix for all 16 values of its last 4 bits. Every F
//   is tested against many prefixes with ONE hash each (H) plus a table
//   lookup; the 16 hashes per prefix are paid once for hundreds of F.
//
// 3. Shared rounds
//   The prefixes come in groups of 16 that differ only in the last message
//   word H reads, so the first 7 of the 64 SHA-256 rounds are computed once
//   per (F, group).
//
// Y||F ("both orders") costs one more hash on each hit, with 1/256 odds.
#pragma once
#include "sha256.h"

enum Mode : uint32_t { MODE_ANY = 0, MODE_AB = 1, MODE_BA = 2, MODE_BOTH = 3 };
constexpr uint32_t FLAG_FY = 1, FLAG_YF = 2;  // F||Y valid, Y||F valid

constexpr uint32_t Y_GROUPS = 4096;  // Y groups per launch (power of two)
constexpr uint32_t VARIANTS = 16;    // Y prefixes per group = candidates per (F, group)
constexpr uint32_t F_INDEX = 0xffffffffu, Y_INDEX = 0xfffffffeu;  // domain separation
constexpr uint32_t PENDING = 12;

struct Seed {
    uint32_t w[8];  // 256-bit search seed, big-endian words
};

// A hit names its two mnemonics: F number `f`, Y group number `y` (both unique
// over the whole run) and the variant inside the group.
struct Hit {
    uint64_t f, y;
    uint32_t variant;
    uint32_t flags;  // FLAG_YF is only looked for (and set) in "both" mode
};

struct Candidate {
    uint32_t f[4], y[4];  // 128-bit entropies, big-endian words
    uint32_t csF, csY;    // 4-bit BIP39 checksums
    uint32_t flags;       // FLAG_FY | FLAG_YF
};

// SHA256(seed || number || index): independent random bits for every input,
// as unpredictable as the seed.
HD inline void derive(const Seed& s, uint64_t number, uint32_t index, uint32_t h[8]) {
    uint32_t blk[16] = {s.w[0], s.w[1], s.w[2], s.w[3], s.w[4], s.w[5], s.w[6], s.w[7],
                        (uint32_t)(number >> 32), (uint32_t)number, index,
                        0x80000000u, 0, 0, 0, 352};
    sha256_block(blk, h);
}

// BIP39 checksum of a 128-bit entropy: top 4 bits of SHA256(entropy).
HD inline uint32_t checksum4(const uint32_t e[4]) {
    uint32_t blk[16] = {e[0], e[1], e[2], e[3], 0x80000000u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128};
    uint32_t h[8];
    sha256_block(blk, h);
    return h[0] >> 28;
}

// Checksum byte of the 24-word mnemonic X||Y. Does not read the low 4 bits of y[3].
HD inline uint32_t concat_checksum(const uint32_t x[4], uint32_t csX, const uint32_t y[4]) {
    uint32_t blk[16] = {x[0], x[1], x[2], x[3],
                        (csX << 28) | (y[0] >> 4),
                        (y[0] << 28) | (y[1] >> 4),
                        (y[1] << 28) | (y[2] >> 4),
                        (y[2] << 28) | (y[3] >> 4),
                        0x80000000u, 0, 0, 0, 0, 0, 0, 256};
    uint32_t h[8];
    sha256_block(blk, h);
    return h[0] >> 24;
}

// Is X||Y (two valid 12-word mnemonics) a valid 24-word mnemonic?
HD inline bool concat_valid(const uint32_t x[4], uint32_t csX, const uint32_t y[4], uint32_t csY) {
    return concat_checksum(x, csX, y) == (((y[3] & 0xf) << 4) | csY);
}

HD inline void derive_f(const Seed& s, uint64_t number, uint32_t f[4]) {
    uint32_t h[8];
    derive(s, number, F_INDEX, h);
    for (int i = 0; i < 4; i++) f[i] = h[i];
}

// A Y group: 120 random bits; the low byte of y[3] is variant << 4 | last 4 bits.
HD inline void derive_y(const Seed& s, uint64_t number, uint32_t y[4]) {
    uint32_t h[8];
    derive(s, number, Y_INDEX, h);
    for (int i = 0; i < 4; i++) y[i] = h[i];
    y[3] &= ~0xffu;
}

// Table entry of one Y group: for each variant, 16 checksums (4 bits each),
// one per value of the last 4 entropy bits.
HD inline void build_y_group(const Seed& s, uint64_t number, uint32_t y[4], uint64_t table[VARIANTS]) {
    derive_y(s, number, y);
    uint32_t e[4] = {y[0], y[1], y[2], 0};
    for (uint32_t v = 0; v < VARIANTS; v++) {
        uint64_t t = 0;
        for (uint32_t nib = 0; nib < 16; nib++) {
            e[3] = y[3] | (v << 4) | nib;
            t |= (uint64_t)checksum4(e) << (4 * nib);
        }
        table[v] = t;
    }
}

// Tests one F against `iters` Y groups (16 candidates each) of the launch
// whose first Y group number is yBase, starting at group `first`. Calls
// emit(f, y, variant, flags) for each match. Runs unchanged on GPU and CPU.
template <class Emit>
HD inline void scan_f(const Seed& s, uint64_t fNumber, uint64_t yBase, const uint32_t* yGroups,
                      const uint64_t* table, uint32_t first, uint32_t iters, bool both, Emit& emit) {
    uint32_t f[4];
    derive_f(s, fNumber, f);
    uint32_t csF = checksum4(f);

    uint32_t pending[PENDING], np = 0;  // hits waiting for the Y||F test: group << 12 | variant << 8 | need
    auto reverse = [&](uint32_t p) {
        uint32_t j = p >> 12, v = (p >> 8) & 0xf, need = p & 0xff;
        const uint32_t* g = yGroups + 4 * j;
        uint32_t y[4] = {g[0], g[1], g[2], g[3] | (v << 4) | (need >> 4)};
        if (concat_valid(y, need & 0xf, f, csF)) emit(fNumber, yBase + j, v, FLAG_FY | FLAG_YF);
    };

    for (uint32_t it = 0; it < iters; it++) {
        uint32_t j = (first + it) & (Y_GROUPS - 1);
        const uint32_t* g = yGroups + 4 * j;
        uint32_t head[7] = {f[0], f[1], f[2], f[3],
                            (csF << 28) | (g[0] >> 4), (g[0] << 28) | (g[1] >> 4), (g[1] << 28) | (g[2] >> 4)};
        ShaMid mid;
        sha256_32_begin(head, mid);
        uint32_t w7 = (g[2] << 28) | (g[3] >> 4);

        for (uint32_t v = 0; v < VARIANTS; v++) {
            uint32_t need = sha256_32_finish(mid, w7 | v) >> 24;  // what F||Y demands
            uint32_t cs = (uint32_t)(table[j * VARIANTS + v] >> (4 * (need >> 4))) & 0xf;
            if (cs != (need & 0xf)) continue;  // Y itself not valid
            uint32_t p = (j << 12) | (v << 8) | need;
            if (!both) emit(fNumber, yBase + j, v, FLAG_FY);
            else if (np < PENDING) pending[np++] = p;
            else reverse(p);
        }
        // The Y||F tests are batched: on a GPU, doing them inline would make
        // every thread of a warp wait whenever any one of them has a hit.
        if (both && ((it & 3) == 3 || it + 1 == iters)) {
            for (uint32_t i = 0; i < np; i++) reverse(pending[i]);
            np = 0;
        }
    }
}

// Rebuilds the pair behind a hit from scratch, recomputing every checksum and
// both concatenation tests without table or shortcut.
HD inline void rebuild(const Seed& s, uint64_t fNumber, uint64_t yNumber, uint32_t variant, Candidate& c) {
    derive_f(s, fNumber, c.f);
    c.csF = checksum4(c.f);
    derive_y(s, yNumber, c.y);
    c.y[3] |= variant << 4;
    c.y[3] |= concat_checksum(c.f, c.csF, c.y) >> 4;
    c.csY = checksum4(c.y);
    c.flags = (concat_valid(c.f, c.csF, c.y, c.csY) ? FLAG_FY : 0u) |
              (concat_valid(c.y, c.csY, c.f, c.csF) ? FLAG_YF : 0u);
}
