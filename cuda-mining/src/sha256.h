// Minimal CPU SHA-256 used for coinbase/merkle hashing, midstate and share verification.
#pragma once
#include <cstddef>
#include <cstdint>

extern const uint32_t SHA256_INIT[8];

// One compression round over a 64-byte block (state is updated in place).
void sha256_transform(uint32_t state[8], const uint8_t block[64]);
void sha256(const uint8_t* data, size_t len, uint8_t out[32]);
void sha256d(const uint8_t* data, size_t len, uint8_t out[32]);
