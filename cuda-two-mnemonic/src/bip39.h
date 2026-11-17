// Host-side BIP39 helpers. These work on 11-bit word indices and are
// deliberately independent from the bit tricks in candidate.h, so they can be
// used to double-check every hit the search reports.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bip39 {

// SHA-256 of a message of at most 55 bytes.
void sha256_short(const uint8_t* msg, size_t len, uint8_t out[32]);

// Entropy (16 or 32 bytes) -> word indices (12 or 24), checksum appended.
std::vector<uint16_t> entropy_to_indices(const uint8_t* ent, size_t len);

// Checks the checksum of a 12- or 24-word mnemonic given as indices.
bool indices_valid(const std::vector<uint16_t>& idx);

std::string to_words(const std::vector<uint16_t>& idx);

// Parses a space separated mnemonic; returns false on an unknown word.
bool from_words(const std::string& text, std::vector<uint16_t>& idx);

}  // namespace bip39
