// two-mnemonic: search for two valid 12-word BIP39 mnemonics A and B whose
// concatenation (A+B, B+A, or both) is itself a valid 24-word mnemonic.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "bip39.h"
#include "candidate.h"
#include "gpu_search.h"

namespace {

struct Options {
    uint32_t mode = MODE_ANY;
    uint64_t count = 5;
    uint32_t threads = 1u << 14;
    uint32_t iters = 64;
    int device = 0;
    bool cpu = false;
    bool selftest = false;
    bool haveSeed = false;
    Seed seed{};
    std::vector<std::string> check;
};

void usage() {
    puts("Usage: two-mnemonic [options]\n"
         "\n"
         "Finds pairs of valid 12-word BIP39 mnemonics A, B such that A+B and/or B+A\n"
         "is a valid 24-word mnemonic.\n"
         "\n"
         "  --mode any|ab|ba|both  which concatenation must be valid (default: any)\n"
         "                           any  = A+B, and says when B+A is valid too\n"
         "                           ab   = A+B\n"
         "                           ba   = B+A\n"
         "                           both = A+B and B+A (about 256x rarer)\n"
         "  --count N              number of pairs to print (default: 5)\n"
         "  --seed HEX64           256-bit search seed; makes the run reproducible\n"
         "                         (default: random, from /dev/urandom)\n"
         "  --threads N            GPU threads per launch, one first-half mnemonic\n"
         "                         each (default: 16384)\n"
         "  --iters N              per thread: N x 2 sets of 11 words for the second\n"
         "                         half, 8 final words tried for each (default: 64)\n"
         "  --device N             CUDA device index (default: 0)\n"
         "  --cpu                  search on the CPU instead of the GPU\n"
         "  --check \"A\" \"B\"        test two given 12-word mnemonics and exit\n"
         "  --selftest             run known-answer tests and a GPU-vs-CPU comparison\n"
         "  -h, --help             this text");
}

bool parse_seed(const std::string& hex, Seed& s) {
    if (hex.size() != 64) return false;
    for (int i = 0; i < 8; i++) {
        char* end = nullptr;
        std::string part = hex.substr(8 * i, 8);
        s.w[i] = (uint32_t)strtoul(part.c_str(), &end, 16);
        if (*end) return false;
    }
    return true;
}

Seed random_seed() {
    Seed s;
    std::ifstream f("/dev/urandom", std::ios::binary);
    if (!f.read((char*)s.w, sizeof(s.w))) throw std::runtime_error("cannot read /dev/urandom");
    return s;
}

Options parse(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(); exit(0); }
        else if (a == "--mode") {
            std::string m = need(i);
            if (m == "any") o.mode = MODE_ANY;
            else if (m == "ab") o.mode = MODE_AB;
            else if (m == "ba") o.mode = MODE_BA;
            else if (m == "both") o.mode = MODE_BOTH;
            else throw std::runtime_error("unknown mode: " + m);
        }
        else if (a == "--count") o.count = strtoull(need(i).c_str(), nullptr, 10);
        else if (a == "--threads") o.threads = (uint32_t)strtoul(need(i).c_str(), nullptr, 10);
        else if (a == "--iters") o.iters = (uint32_t)strtoul(need(i).c_str(), nullptr, 10);
        else if (a == "--device") o.device = atoi(need(i).c_str());
        else if (a == "--cpu") o.cpu = true;
        else if (a == "--selftest") o.selftest = true;
        else if (a == "--seed") {
            if (!parse_seed(need(i), o.seed)) throw std::runtime_error("--seed needs 64 hex digits");
            o.haveSeed = true;
        }
        else if (a == "--check") { o.check.push_back(need(i)); o.check.push_back(need(i)); }
        else throw std::runtime_error("unknown option: " + a + " (try --help)");
    }
    if (!o.count || !o.threads || !o.iters) throw std::runtime_error("--count, --threads and --iters must be > 0");
    if (o.iters > (1u << 24)) throw std::runtime_error("--iters is too large (max 16777216)");
    return o;
}

struct HostEmit {
    std::vector<Hit>& out;
    void operator()(uint64_t f, uint64_t y, uint32_t variant, uint32_t flags) { out.push_back({f, y, variant, flags}); }
};

template <class Fn>
void parallel_for(uint64_t n, Fn fn) {  // fn(thread, lo, hi)
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; t++) pool.emplace_back([=] { fn(t, n * t / nt, n * (t + 1) / nt); });
    for (auto& th : pool) th.join();
}

// The Y groups and checksum table of one launch.
struct YTable {
    std::vector<uint32_t> y = std::vector<uint32_t>(4 * Y_GROUPS);
    std::vector<uint64_t> table = std::vector<uint64_t>(VARIANTS * Y_GROUPS);
    YTable(const Seed& seed, uint64_t epoch) {
        parallel_for(Y_GROUPS, [&](unsigned, uint64_t lo, uint64_t hi) {
            for (uint64_t j = lo; j < hi; j++) build_y_group(seed, epoch * Y_GROUPS + j, &y[4 * j], &table[VARIANTS * j]);
        });
    }
};

// CPU version of GpuSearch::run with `nf` mnemonics F per launch.
void cpu_run(const Seed& seed, uint64_t epoch, uint64_t nf, uint32_t iters, bool both, std::vector<Hit>& out) {
    YTable t(seed, epoch);
    std::vector<std::vector<Hit>> found(std::max(1u, std::thread::hardware_concurrency()));
    parallel_for(nf, [&](unsigned th, uint64_t lo, uint64_t hi) {
        HostEmit emit{found[th]};
        for (uint64_t i = lo; i < hi; i++)
            scan_f(seed, epoch * nf + i, epoch * Y_GROUPS, t.y.data(), t.table.data(), (uint32_t)(i * iters), iters, both, emit);
    });
    for (auto& v : found) out.insert(out.end(), v.begin(), v.end());
}

std::vector<uint16_t> to_indices(const uint32_t w[4]) {
    uint8_t e[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) e[4 * i + j] = (uint8_t)(w[i] >> (24 - 8 * j));
    return bip39::entropy_to_indices(e, 16);
}

std::vector<uint16_t> concat(const std::vector<uint16_t>& x, const std::vector<uint16_t>& y) {
    std::vector<uint16_t> r(x);
    r.insert(r.end(), y.begin(), y.end());
    return r;
}

// Word-level (independent) evaluation of a pair: FLAG_FY | FLAG_YF, or
// 0xffffffff if one of the two is not a valid 12-word mnemonic.
uint32_t word_level_flags(const std::vector<uint16_t>& f, const std::vector<uint16_t>& y) {
    if (!bip39::indices_valid(f) || !bip39::indices_valid(y)) return 0xffffffffu;
    return (bip39::indices_valid(concat(f, y)) ? FLAG_FY : 0u) | (bip39::indices_valid(concat(y, f)) ? FLAG_YF : 0u);
}

// Rebuilds the pair for `hit`, re-validates everything through the independent
// word-index code path, and prints it. Returns false if the check disagrees
// with what the search reported.
bool report(const Seed& seed, const Hit& hit, uint64_t n, uint32_t mode) {
    Candidate c;
    rebuild(seed, hit.f, hit.y, hit.variant, c);
    auto f = to_indices(c.f), y = to_indices(c.y);
    uint32_t flags = word_level_flags(f, y);
    // Outside "both" mode the search does not look at Y+F, so it may hold on top.
    if (flags == 0xffffffffu || c.flags != flags || !(flags & FLAG_FY) || (hit.flags & ~flags)) {
        fprintf(stderr, "ERROR: verification mismatch at F %llu Y %llu/%u (search flags %u, check flags %u)\n",
                (unsigned long long)hit.f, (unsigned long long)hit.y, hit.variant, hit.flags, flags);
        return false;
    }

    // The search always finds F+Y; in "ba" mode name them so that B+A is the valid order.
    bool swap = mode == MODE_BA;
    const auto& a = swap ? y : f;
    const auto& b = swap ? f : y;
    bool ab = swap ? (flags & FLAG_YF) != 0 : true, ba = swap ? true : (flags & FLAG_YF) != 0;
    printf("#%llu  [%s]\n", (unsigned long long)n, ab && ba ? "A+B and B+A" : ab ? "A+B" : "B+A");
    printf("  A  : %s\n", bip39::to_words(a).c_str());
    printf("  B  : %s\n", bip39::to_words(b).c_str());
    if (ab) printf("  A+B: %s\n", bip39::to_words(concat(a, b)).c_str());
    if (ba) printf("  B+A: %s\n", bip39::to_words(concat(b, a)).c_str());
    printf("\n");
    return true;
}

int check_pair(const std::string& sa, const std::string& sb) {
    std::vector<uint16_t> ia, ib;
    if (!bip39::from_words(sa, ia) || !bip39::from_words(sb, ib)) {
        fprintf(stderr, "unknown word in mnemonic\n");
        return 2;
    }
    bool va = bip39::indices_valid(ia) && ia.size() == 12, vb = bip39::indices_valid(ib) && ib.size() == 12;
    bool ab = bip39::indices_valid(concat(ia, ib)), ba = bip39::indices_valid(concat(ib, ia));
    printf("A valid 12-word mnemonic  : %s\n", va ? "yes" : "NO");
    printf("B valid 12-word mnemonic  : %s\n", vb ? "yes" : "NO");
    printf("A+B valid 24-word mnemonic: %s\n", ab ? "yes" : "no");
    printf("B+A valid 24-word mnemonic: %s\n", ba ? "yes" : "no");
    return va && vb && (ab || ba) ? 0 : 1;
}

bool hit_order(const Hit& a, const Hit& b) {
    return a.f != b.f ? a.f < b.f : a.y != b.y ? a.y < b.y : a.variant < b.variant;
}

int selftest(const Options& o) {
    int fails = 0;
    auto expect = [&](bool ok, const char* what) {
        printf("%-58s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) fails++;
    };

    uint8_t h[32];
    bip39::sha256_short((const uint8_t*)"abc", 3, h);
    static const uint8_t abc[4] = {0xba, 0x78, 0x16, 0xbf};
    static const uint8_t abcEnd[4] = {0xf2, 0x00, 0x15, 0xad};
    expect(!memcmp(h, abc, 4) && !memcmp(h + 28, abcEnd, 4), "SHA-256(\"abc\")");

    uint8_t zero[32] = {0}, ff[16];
    memset(ff, 0xff, sizeof ff);
    expect(bip39::to_words(bip39::entropy_to_indices(zero, 16)) ==
               "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
           "BIP39 vector: 128 zero bits");
    expect(bip39::to_words(bip39::entropy_to_indices(ff, 16)) == "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
           "BIP39 vector: 128 one bits");
    auto art = bip39::entropy_to_indices(zero, 32);
    expect(art.size() == 24 && bip39::to_words(art).substr(8 * 23) == "art" && bip39::indices_valid(art),
           "BIP39 vector: 256 zero bits");
    art[23] ^= 1;
    expect(!bip39::indices_valid(art), "bad checksum is rejected");

    // The search must find exactly the final words that a brute force over
    // all 2048 of them finds, using only the word-level implementation.
    Seed seed{{1, 2, 3, 4, 5, 6, 7, 8}};
    {
        const uint64_t epoch = 3, nf = 128;
        const uint32_t iters = 2;
        YTable t(seed, epoch);
        bool same = true;
        uint64_t sets = 0, total = 0;
        for (uint64_t i = 0; i < nf && same; i++) {
            std::vector<Hit> hits;
            HostEmit emit{hits};
            uint64_t fn = epoch * nf + i;
            scan_f(seed, fn, epoch * Y_GROUPS, t.y.data(), t.table.data(), (uint32_t)(i * iters), iters, false, emit);
            for (uint32_t it = 0; it < iters; it++) {
                uint64_t yn = epoch * Y_GROUPS + ((i * iters + it) & (Y_GROUPS - 1));
                // Bit 3 of the variant belongs to the 11th word: two sets of 11 words per group.
                for (uint32_t hi = 0; hi < 2; hi++, sets++) {
                    std::vector<uint16_t> fast, brute;
                    Candidate c;
                    for (const Hit& hit : hits)
                        if (hit.y == yn && hit.variant >> 3 == hi) {
                            rebuild(seed, hit.f, hit.y, hit.variant, c);
                            if (!(c.flags & FLAG_FY) || hit.flags != FLAG_FY) same = false;
                            fast.push_back(to_indices(c.y)[11]);
                        }
                    rebuild(seed, fn, yn, hi << 3, c);
                    auto f = to_indices(c.f), y = to_indices(c.y);
                    for (uint16_t w = 0; w < 2048; w++) {
                        y[11] = w;
                        uint32_t flags = word_level_flags(f, y);
                        if (flags != 0xffffffffu && (flags & FLAG_FY)) brute.push_back(w);
                    }
                    std::sort(fast.begin(), fast.end());
                    if (fast != brute) same = false;
                    total += brute.size();
                }
            }
        }
        expect(same, "search == brute force over all 2048 final words");
        printf("  %llu sets of 11 words: %llu working final words (expect ~%llu)\n", (unsigned long long)sets,
               (unsigned long long)total, (unsigned long long)(sets / 2));
    }

    if (!o.cpu) {
        GpuSearch gpu(o.device, 1u << 12, 1u << 16);
        for (bool both : {false, true}) {
            std::vector<Hit> g, c;
            uint64_t epoch = 0x1234567;
            uint32_t iters = both ? 77 : 7;
            uint32_t n = gpu.run(seed, epoch, iters, both, g);
            cpu_run(seed, epoch, gpu.threads(), iters, both, c);
            std::sort(g.begin(), g.end(), hit_order);
            std::sort(c.begin(), c.end(), hit_order);
            bool same = g.size() == c.size() && n == c.size() && !g.empty();
            for (size_t i = 0; same && i < g.size(); i++)
                same = g[i].f == c[i].f && g[i].y == c[i].y && g[i].variant == c[i].variant && g[i].flags == c[i].flags;
            for (size_t i = 0; same && i < c.size(); i += both ? 1 : 16) {  // word-level check
                Candidate cand;
                rebuild(seed, c[i].f, c[i].y, c[i].variant, cand);
                uint32_t flags = word_level_flags(to_indices(cand.f), to_indices(cand.y));
                same = flags != 0xffffffffu && (flags & c[i].flags) == c[i].flags;
            }
            char msg[96];
            snprintf(msg, sizeof msg, "GPU == CPU, %s (%zu hits in %llu candidates)", both ? "both orders" : "one order",
                     c.size(), (unsigned long long)gpu.threads() * iters * VARIANTS);
            expect(same, msg);
        }
    }
    printf(fails ? "\n%d test(s) FAILED\n" : "\nall tests passed\n", fails);
    return fails ? 1 : 0;
}

int search(Options& o) {
    if (!o.haveSeed) o.seed = random_seed();
    bool both = o.mode == MODE_BOTH;

    std::unique_ptr<GpuSearch> gpu;
    uint64_t nf;
    if (o.cpu) {
        nf = 4096;
        fprintf(stderr, "device : CPU, %u threads\n", std::max(1u, std::thread::hardware_concurrency()));
    } else {
        gpu.reset(new GpuSearch(o.device, o.threads, 1u << 16));
        nf = gpu->threads();
        fprintf(stderr, "device : %s, %llu candidates per launch\n", gpu->deviceName().c_str(),
                (unsigned long long)nf * o.iters * VARIANTS);
    }
    static const char* names[] = {"A+B (B+A reported when it also holds)", "A+B", "B+A", "A+B and B+A"};
    fprintf(stderr, "target : %llu pair(s) where %s is a valid 24-word mnemonic\n\n", (unsigned long long)o.count, names[o.mode]);

    auto t0 = std::chrono::steady_clock::now();
    uint64_t epoch = 0, printed = 0, matched = 0;
    bool ok = true;
    while (printed < o.count && ok) {
        std::vector<Hit> hits;
        if (gpu) matched += gpu->run(o.seed, epoch, o.iters, both, hits);
        else { cpu_run(o.seed, epoch, nf, o.iters, both, hits); matched += hits.size(); }
        epoch++;
        std::sort(hits.begin(), hits.end(), hit_order);
        // Printed pairs never share a mnemonic nor a Y group (Y of the same
        // group have most of their words in common). F and Y numbers are
        // never reused by later launches.
        std::unordered_set<uint64_t> usedF, usedY;
        for (size_t i = 0; i < hits.size() && printed < o.count && ok; i++)
            if (!usedF.count(hits[i].f) && !usedY.count(hits[i].y)) {
                usedF.insert(hits[i].f);
                usedY.insert(hits[i].y);
                ok = report(o.seed, hits[i], ++printed, o.mode);
            }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double cands = (double)epoch * nf * o.iters * VARIANTS;
    fprintf(stderr, "tested %.0f final-word candidates in %.2f s (%.1f M/s), %llu matched (1 in %.0f)\n",
            cands, secs, cands / secs / 1e6, (unsigned long long)matched, matched ? cands / matched : 0.0);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        if (!o.check.empty()) return check_pair(o.check[0], o.check[1]);
        if (o.selftest) return selftest(o);
        return search(o);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
