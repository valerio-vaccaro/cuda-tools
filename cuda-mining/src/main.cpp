// cudaminer - a small Bitcoin (SHA-256d) Stratum v1 pool miner for NVIDIA GPUs.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu.h"
#include "json.h"
#include "sha256.h"
#include "stratum.h"

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------- options

struct Options {
    std::string address;
    std::string worker = "cudaminer";
    std::string host = "solo.homeminingitalia.org";
    int port = 3340;
    std::string password = "x";
    int device = 0;
    int batch_bits = 0; // 0 = auto-tune
    double suggest_diff = 0.1;
    bool benchmark = false;
    bool selftest = false;
};

static void usage(const char* prog) {
    printf(
        "Usage: %s -a <bitcoin-address> [options]\n"
        "\n"
        "  -a, --address ADDR      Bitcoin address to mine to (required)\n"
        "  -w, --worker NAME       worker name appended to the address (default: cudaminer)\n"
        "  -o, --pool HOST[:PORT]  pool host, stratum+tcp:// prefix allowed (default: solo.homeminingitalia.org)\n"
        "  -p, --port PORT         pool port (default: 3340)\n"
        "      --password PASS     pool password (default: x)\n"
        "  -d, --device N          CUDA device index (default: 0)\n"
        "  -b, --batch-bits N      hash 2^N nonces per GPU launch, 16..32 (default: auto)\n"
        "  -s, --suggest-diff D    share difficulty to suggest to the pool, 0 = don't (default: 0.1)\n"
        "      --benchmark         measure the GPU hashrate and exit (no network)\n"
        "      --selftest          verify the GPU kernel against the genesis block and exit\n"
        "  -h, --help              show this help\n",
        prog);
}

static bool parse_int(const char* s, int& out) {
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (end == s || *end) return false;
    out = (int)v;
    return true;
}

static bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto is = [&](const char* s, const char* l) { return a == l || (s && a == s); };
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                return nullptr;
            }
            return argv[++i];
        };
        const char* v = nullptr;
        if (is("-h", "--help")) { usage(argv[0]); exit(0); }
        else if (is(nullptr, "--benchmark")) o.benchmark = true;
        else if (is(nullptr, "--selftest")) o.selftest = true;
        else if (is("-a", "--address")) { if (!(v = next())) return false; o.address = v; }
        else if (is("-w", "--worker")) { if (!(v = next())) return false; o.worker = v; }
        else if (is(nullptr, "--password")) { if (!(v = next())) return false; o.password = v; }
        else if (is("-o", "--pool")) {
            if (!(v = next())) return false;
            std::string h = v;
            size_t scheme = h.find("://");
            if (scheme != std::string::npos) h = h.substr(scheme + 3);
            size_t colon = h.rfind(':');
            if (colon != std::string::npos) {
                if (!parse_int(h.c_str() + colon + 1, o.port)) { fprintf(stderr, "bad pool port in %s\n", v); return false; }
                h = h.substr(0, colon);
            }
            o.host = h;
        }
        else if (is("-p", "--port")) { if (!(v = next()) || !parse_int(v, o.port)) { fprintf(stderr, "bad port\n"); return false; } }
        else if (is("-d", "--device")) { if (!(v = next()) || !parse_int(v, o.device)) { fprintf(stderr, "bad device\n"); return false; } }
        else if (is("-b", "--batch-bits")) {
            if (!(v = next()) || !parse_int(v, o.batch_bits) || o.batch_bits < 16 || o.batch_bits > 32) {
                fprintf(stderr, "batch-bits must be between 16 and 32\n");
                return false;
            }
        }
        else if (is("-s", "--suggest-diff")) {
            if (!(v = next())) return false;
            char* end = nullptr;
            o.suggest_diff = strtod(v, &end);
            if (end == v || *end || o.suggest_diff < 0) { fprintf(stderr, "bad suggest-diff\n"); return false; }
        }
        else { fprintf(stderr, "unknown option: %s\n", a.c_str()); return false; }
    }
    if (o.port <= 0 || o.port > 65535) { fprintf(stderr, "bad port\n"); return false; }
    return true;
}

// ---------------------------------------------------------------- helpers

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

static void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf(const char* fmt, ...) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    time_t t = time(nullptr);
    tm tmv;
    localtime_r(&t, &tmv);
    printf("[%02d:%02d:%02d] ", tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static bool hex2bin(const std::string& hex, std::vector<uint8_t>& out) {
    if (hex.size() % 2) return false;
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < hex.size(); i += 2) {
        int h = nib(hex[i]), l = nib(hex[i + 1]);
        if (h < 0 || l < 0) return false;
        out.push_back((uint8_t)(h << 4 | l));
    }
    return true;
}

static std::string bin2hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}

// Block hashes are displayed byte-reversed.
static std::string hash_hex(const uint8_t h[32]) {
    uint8_t r[32];
    for (int i = 0; i < 32; i++) r[i] = h[31 - i];
    return bin2hex(r, 32);
}

static inline uint32_t be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static inline uint32_t le32(const uint8_t* p) { return (uint32_t)p[3] << 24 | (uint32_t)p[2] << 16 | (uint32_t)p[1] << 8 | p[0]; }

// 256-bit targets are held as 8 little-endian words, t[7] being the most significant.
struct Target { uint32_t w[8]; };

static Target target_from_diff(double diff) {
    Target t;
    if (!(diff > 0)) diff = 1;
    int k;
    for (k = 6; k > 0 && diff > 1.0; k--) diff /= 4294967296.0;
    double m = 4294901760.0 / diff; // 0xFFFF0000 / diff
    if (m >= 18446744073709551615.0) {
        memset(t.w, 0xff, sizeof(t.w));
    } else {
        memset(t.w, 0, sizeof(t.w));
        uint64_t mi = (uint64_t)m;
        t.w[k] = (uint32_t)mi;
        t.w[k + 1] = (uint32_t)(mi >> 32);
    }
    return t;
}

static Target target_from_nbits(uint32_t nbits) {
    uint8_t b[32] = {0};
    int exp = (int)(nbits >> 24);
    uint32_t mant = nbits & 0x007fffff;
    for (int i = 0; i < 3; i++) {
        int pos = exp - 3 + i;
        if (pos >= 0 && pos < 32) b[pos] = (uint8_t)(mant >> (8 * i));
    }
    Target t;
    for (int i = 0; i < 8; i++) t.w[i] = le32(b + 4 * i);
    return t;
}

static bool hash_meets(const uint8_t hash[32], const Target& t) {
    for (int i = 7; i >= 0; i--) {
        uint32_t h = le32(hash + 4 * i);
        if (h < t.w[i]) return true;
        if (h > t.w[i]) return false;
    }
    return true;
}

static double hash_difficulty(const uint8_t hash[32]) {
    double v = 0;
    for (int i = 31; i >= 0; i--) v = v * 256.0 + hash[i];
    if (v == 0) return 1e300;
    return 65535.0 * 26959946667150639794667015087019630673637144422540572481103610249216.0 /* 2^224 */ / 65536.0 / v;
}

// Splits an 80-byte header into what the GPU needs.
static void prepare_work(const uint8_t header[80], uint32_t midstate[8], uint32_t tail[3]) {
    memcpy(midstate, SHA256_INIT, 8 * sizeof(uint32_t));
    sha256_transform(midstate, header);
    for (int i = 0; i < 3; i++) tail[i] = be32(header + 64 + 4 * i);
}

static void hash_header(uint8_t header[80], uint32_t nonce, uint8_t hash[32]) {
    header[76] = (uint8_t)nonce;
    header[77] = (uint8_t)(nonce >> 8);
    header[78] = (uint8_t)(nonce >> 16);
    header[79] = (uint8_t)(nonce >> 24);
    sha256d(header, 80, hash);
}

static std::string fmt_rate(double hps) {
    char buf[64];
    if (hps >= 1e9) snprintf(buf, sizeof(buf), "%.2f GH/s", hps / 1e9);
    else if (hps >= 1e6) snprintf(buf, sizeof(buf), "%.2f MH/s", hps / 1e6);
    else snprintf(buf, sizeof(buf), "%.2f kH/s", hps / 1e3);
    return buf;
}

// ---------------------------------------------------------------- stratum state

struct Job {
    std::string id, prevhash, coinb1, coinb2, version, nbits, ntime;
    std::vector<std::string> merkle;
};

struct Shared {
    std::mutex mu;
    std::condition_variable cv;
    Job job;
    bool have_job = false;
    std::string extranonce1;
    int en2size = 4;
    double difficulty = 1.0;
    std::atomic<uint64_t> gen{0}; // bumped whenever the work to do changes

    std::atomic<uint64_t> accepted{0}, rejected{0};
    std::atomic<int> next_id{10};
};

// Builds the 80-byte header (nonce zeroed) for a job and extranonce2.
static bool build_header(const Job& j, const std::string& en1, const std::string& en2, uint8_t header[80]) {
    std::vector<uint8_t> coinbase, ver, prev, ntime, nbits;
    if (!hex2bin(j.coinb1 + en1 + en2 + j.coinb2, coinbase)) return false;
    if (!hex2bin(j.version, ver) || ver.size() != 4) return false;
    if (!hex2bin(j.prevhash, prev) || prev.size() != 32) return false;
    if (!hex2bin(j.ntime, ntime) || ntime.size() != 4) return false;
    if (!hex2bin(j.nbits, nbits) || nbits.size() != 4) return false;

    uint8_t root[64];
    sha256d(coinbase.data(), coinbase.size(), root);
    for (const std::string& br : j.merkle) {
        std::vector<uint8_t> b;
        if (!hex2bin(br, b) || b.size() != 32) return false;
        memcpy(root + 32, b.data(), 32);
        sha256d(root, 64, root);
    }

    // version, ntime and nbits arrive as big-endian hex numbers; the header stores them
    // little-endian. prevhash arrives with each 4-byte word swapped.
    for (int i = 0; i < 4; i++) header[i] = ver[3 - i];
    for (int i = 0; i < 32; i++) header[4 + i] = prev[(i & ~3) + 3 - (i & 3)];
    memcpy(header + 36, root, 32);
    for (int i = 0; i < 4; i++) header[68 + i] = ntime[3 - i];
    for (int i = 0; i < 4; i++) header[72 + i] = nbits[3 - i];
    memset(header + 76, 0, 4);
    return true;
}

// ---------------------------------------------------------------- mining thread

static void miner_loop(Shared& sh, StratumConn& conn, const Options& opt, const std::string& user) {
    uint64_t batch = opt.batch_bits ? (1ull << opt.batch_bits) : (1ull << 20);
    uint64_t seen_gen = 0;
    uint64_t en2_counter = 0;
    uint64_t hashes = 0;
    double best = 0;
    auto last_report = Clock::now();
    std::vector<uint32_t> hits(GPU_MAX_RESULTS);

    while (!g_stop) {
        Job job;
        std::string en1;
        int en2size;
        double diff;
        uint64_t gen;
        {
            std::unique_lock<std::mutex> lk(sh.mu);
            sh.cv.wait_for(lk, std::chrono::milliseconds(500), [&] { return g_stop || sh.have_job; });
            if (g_stop) break;
            if (!sh.have_job) { last_report = Clock::now(); hashes = 0; continue; }
            job = sh.job;
            en1 = sh.extranonce1;
            en2size = sh.en2size;
            diff = sh.difficulty;
            gen = sh.gen;
        }
        if (gen != seen_gen) { seen_gen = gen; en2_counter = 0; }

        // extranonce2: big-endian counter, bumped each time the 32-bit nonce space is exhausted.
        std::vector<uint8_t> en2(en2size > 0 ? en2size : 0, 0);
        for (int i = 0; i < (int)en2.size() && i < 8; i++) en2[en2.size() - 1 - i] = (uint8_t)(en2_counter >> (8 * i));
        en2_counter++;
        std::string en2hex = bin2hex(en2.data(), en2.size());

        uint8_t header[80];
        if (!build_header(job, en1, en2hex, header)) {
            logf("malformed job %s from pool, waiting for the next one", job.id.c_str());
            std::unique_lock<std::mutex> lk(sh.mu);
            sh.cv.wait_for(lk, std::chrono::seconds(5), [&] { return g_stop || sh.gen != gen; });
            continue;
        }
        uint32_t midstate[8], tail[3];
        prepare_work(header, midstate, tail);
        Target share_target = target_from_diff(diff);
        Target net_target = target_from_nbits(le32(header + 72));
        uint64_t target_top = (uint64_t)share_target.w[7] << 32 | share_target.w[6];

        for (uint64_t pos = 0; pos < (1ull << 32) && !g_stop && sh.gen == gen;) {
            uint64_t n = std::min<uint64_t>(batch, (1ull << 32) - pos);
            auto t0 = Clock::now();
            // count == 2^32 does not fit in 32 bits; split such a batch in two.
            if (n == (1ull << 32)) n >>= 1;
            int found = gpu_scan(midstate, tail, target_top, (uint32_t)pos, (uint32_t)n, hits.data(), GPU_MAX_RESULTS);
            if (found < 0) {
                logf("GPU error: %s", gpu_last_error());
                g_stop = true;
                break;
            }
            double dt = std::chrono::duration<double>(Clock::now() - t0).count();
            pos += n;
            hashes += n;
            if (!opt.batch_bits && dt < 0.1 && batch < (1ull << 28)) batch <<= 1;

            for (int i = 0; i < found && sh.gen == gen; i++) {
                uint8_t hash[32];
                hash_header(header, hits[i], hash);
                if (!hash_meets(hash, share_target)) continue; // GPU only compares the top 64 bits
                double d = hash_difficulty(hash);
                if (d > best) best = d;
                bool block = hash_meets(hash, net_target);
                char line[512];
                snprintf(line, sizeof(line),
                         "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%08x\"]}",
                         sh.next_id++, Json::escape(user).c_str(), Json::escape(job.id).c_str(), en2hex.c_str(),
                         job.ntime.c_str(), hits[i]);
                if (block) logf("*** BLOCK FOUND *** hash %s", hash_hex(hash).c_str());
                logf("share found: nonce %08x diff %.4g (target %.4g), submitting", hits[i], d, diff);
                if (!conn.send_line(line)) logf("submit failed: not connected");
            }

            auto now = Clock::now();
            double since = std::chrono::duration<double>(now - last_report).count();
            if (since >= 10.0) {
                logf("%s | shares %llu accepted, %llu rejected | best diff %.4g | pool diff %.4g | job %s",
                     fmt_rate(hashes / since).c_str(), (unsigned long long)sh.accepted.load(),
                     (unsigned long long)sh.rejected.load(), best, diff, job.id.c_str());
                last_report = now;
                hashes = 0;
            }
        }
    }
}

// ---------------------------------------------------------------- pool messages

enum class Handled { Ok, Reconnect, Fatal };

static Handled handle_message(const Json& m, Shared& sh, StratumConn& conn, const Options& opt,
                              const std::string& user) {
    const Json& method = m["method"];
    const Json& params = m["params"];
    if (method.is_str()) {
        if (method.str == "mining.notify") {
            if (!params.is_arr() || params.arr.size() < 8) return Handled::Ok;
            Job j;
            j.id = params[0].str;
            j.prevhash = params[1].str;
            j.coinb1 = params[2].str;
            j.coinb2 = params[3].str;
            for (const Json& b : params[4].arr) j.merkle.push_back(b.str);
            j.version = params[5].str;
            j.nbits = params[6].str;
            j.ntime = params[7].str;
            bool first;
            {
                std::lock_guard<std::mutex> lk(sh.mu);
                first = !sh.have_job;
                sh.job = j;
                sh.have_job = true;
                sh.gen++;
            }
            sh.cv.notify_all();
            if (first || params[8].is_true())
                logf("new job %s (nbits %s, ntime %s%s)", j.id.c_str(), j.nbits.c_str(), j.ntime.c_str(),
                     params[8].is_true() ? ", new block" : "");
        } else if (method.str == "mining.set_difficulty") {
            if (params[0].is_num() && params[0].num > 0) {
                std::lock_guard<std::mutex> lk(sh.mu);
                sh.difficulty = params[0].num;
                logf("pool set share difficulty to %g", params[0].num);
            }
        } else if (method.str == "mining.set_extranonce") {
            if (params[0].is_str() && params[1].is_num()) {
                std::lock_guard<std::mutex> lk(sh.mu);
                sh.extranonce1 = params[0].str;
                sh.en2size = (int)params[1].num;
            }
        } else if (method.str == "client.reconnect") {
            logf("pool requested a reconnect");
            return Handled::Reconnect;
        } else if (method.str == "client.show_message" && params[0].is_str()) {
            logf("pool message: %s", params[0].str.c_str());
        }
        return Handled::Ok;
    }

    if (!m["id"].is_num()) return Handled::Ok;
    int id = (int)m["id"].num;
    const Json& result = m["result"];
    const Json& error = m["error"];
    std::string errtxt = error.is_arr() && error[1].is_str() ? error[1].str
                         : error.is_str() ? error.str
                         : error["message"].is_str() ? error["message"].str : "unknown error";
    if (id == 1) { // mining.subscribe
        if (!result.is_arr() || !result[1].is_str() || !result[2].is_num()) {
            logf("subscribe failed: %s", errtxt.c_str());
            return Handled::Reconnect;
        }
        {
            std::lock_guard<std::mutex> lk(sh.mu);
            sh.extranonce1 = result[1].str;
            sh.en2size = (int)result[2].num;
            if (sh.en2size < 0 || sh.en2size > 32) return Handled::Reconnect;
        }
        logf("subscribed: extranonce1 %s, extranonce2 size %d", result[1].str.c_str(), (int)result[2].num);
        char buf[512];
        snprintf(buf, sizeof(buf), "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}",
                 Json::escape(user).c_str(), Json::escape(opt.password).c_str());
        conn.send_line(buf);
        if (opt.suggest_diff > 0) {
            snprintf(buf, sizeof(buf), "{\"id\":3,\"method\":\"mining.suggest_difficulty\",\"params\":[%g]}", opt.suggest_diff);
            conn.send_line(buf);
        }
    } else if (id == 2) { // mining.authorize
        if (!result.is_true()) {
            logf("authorization failed for %s: %s", user.c_str(), errtxt.c_str());
            return Handled::Fatal;
        }
        logf("authorized as %s", user.c_str());
    } else if (id >= 10) { // mining.submit
        if (result.is_true()) {
            sh.accepted++;
            logf("share accepted (%llu total)", (unsigned long long)sh.accepted.load());
        } else {
            sh.rejected++;
            logf("share rejected: %s", errtxt.c_str());
        }
    }
    return Handled::Ok;
}

// ---------------------------------------------------------------- self test / benchmark

// Bitcoin genesis block header, nonce 2083236893.
static const char* GENESIS_HEADER =
    "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e"
    "67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d00000000";
static const uint32_t GENESIS_NONCE = 2083236893u;
static const char* GENESIS_HASH = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";

static int run_selftest() {
    std::vector<uint8_t> hdr;
    hex2bin(GENESIS_HEADER, hdr);
    uint8_t hash[32];
    hash_header(hdr.data(), GENESIS_NONCE, hash);
    if (hash_hex(hash) != GENESIS_HASH) {
        printf("FAIL: CPU SHA-256d of the genesis header is %s\n", hash_hex(hash).c_str());
        return 1;
    }
    printf("CPU hash of genesis block OK\n");

    uint32_t midstate[8], tail[3];
    prepare_work(hdr.data(), midstate, tail);
    Target t = target_from_diff(1.0);
    uint64_t top = (uint64_t)t.w[7] << 32 | t.w[6];
    uint32_t out[GPU_MAX_RESULTS];
    const uint32_t count = 1u << 22, start = GENESIS_NONCE - (count / 2) - 123;
    int found = gpu_scan(midstate, tail, top, start, count, out, GPU_MAX_RESULTS);
    if (found < 0) { printf("FAIL: GPU error: %s\n", gpu_last_error()); return 1; }
    bool ok = false;
    for (int i = 0; i < found; i++) ok |= out[i] == GENESIS_NONCE;
    if (!ok || found != 1) {
        printf("FAIL: GPU scanned %u nonces and returned %d candidate(s), genesis nonce %s\n", count, found,
               ok ? "among them" : "NOT found");
        return 1;
    }
    printf("GPU found the genesis nonce %08x in a scan of %u nonces OK\n", GENESIS_NONCE, count);

    // Low-difficulty cross-check: every GPU candidate must verify on the CPU, and the
    // count must match a brute-force CPU scan of the same range.
    Target easy = target_from_diff(1.0 / 65536);
    top = (uint64_t)easy.w[7] << 32 | easy.w[6];
    const uint32_t n2 = 1u << 20;
    found = gpu_scan(midstate, tail, top, 0xfffffff0u - n2, n2, out, GPU_MAX_RESULTS);
    int cpu = 0;
    for (uint32_t i = 0; i < n2; i++) {
        hash_header(hdr.data(), 0xfffffff0u - n2 + i, hash);
        cpu += hash_meets(hash, easy);
    }
    int verified = 0;
    for (int i = 0; i < found; i++) {
        hash_header(hdr.data(), out[i], hash);
        verified += hash_meets(hash, easy);
    }
    if (found != cpu || verified != cpu || cpu == 0) {
        printf("FAIL: low-difficulty scan mismatch: GPU %d candidates (%d valid), CPU %d\n", found, verified, cpu);
        return 1;
    }
    printf("GPU and CPU agree on %d low-difficulty shares in %u nonces OK\n", cpu, n2);
    printf("selftest passed\n");
    return 0;
}

static int run_benchmark(const Options& opt) {
    std::vector<uint8_t> hdr;
    hex2bin(GENESIS_HEADER, hdr);
    uint32_t midstate[8], tail[3], out[GPU_MAX_RESULTS];
    prepare_work(hdr.data(), midstate, tail);
    uint64_t batch = opt.batch_bits ? (1ull << opt.batch_bits) : (1ull << 20);
    if (batch > (1ull << 31)) batch = 1ull << 31;
    uint64_t total = 0;
    uint32_t pos = 0;
    auto start = Clock::now();
    double elapsed = 0;
    while (elapsed < 10.0 && !g_stop) {
        auto t0 = Clock::now();
        if (gpu_scan(midstate, tail, 0, pos, (uint32_t)batch, out, GPU_MAX_RESULTS) < 0) {
            printf("GPU error: %s\n", gpu_last_error());
            return 1;
        }
        auto t1 = Clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        pos += (uint32_t)batch;
        if (!opt.batch_bits && dt < 0.1 && batch < (1ull << 28)) {
            batch <<= 1; // still tuning: don't count warm-up batches
            start = t1;
            total = 0;
            continue;
        }
        total += batch;
        elapsed = std::chrono::duration<double>(t1 - start).count();
    }
    if (elapsed > 0) printf("benchmark: %s (batch 2^%d nonces, %.1f s)\n", fmt_rate(total / elapsed).c_str(),
                            (int)__builtin_ctzll(batch), elapsed);
    return 0;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        fprintf(stderr, "try %s --help\n", argv[0]);
        return 2;
    }
    if (!opt.benchmark && !opt.selftest && opt.address.empty()) {
        usage(argv[0]);
        fprintf(stderr, "\nerror: a Bitcoin address is required (-a)\n");
        return 2;
    }

    struct sigaction sa {};
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    char gpu_name[256] = "";
    if (!gpu_init(opt.device, gpu_name, sizeof(gpu_name))) {
        fprintf(stderr, "CUDA init failed: %s\n", gpu_last_error());
        return 1;
    }
    logf("using CUDA device %d: %s", opt.device, gpu_name);

    if (opt.selftest || opt.benchmark) {
        int rc = opt.selftest ? run_selftest() : 0;
        if (rc == 0 && opt.benchmark) rc = run_benchmark(opt);
        gpu_shutdown();
        return rc;
    }

    std::string user = opt.worker.empty() ? opt.address : opt.address + "." + opt.worker;
    Shared sh;
    StratumConn conn;
    std::thread miner(miner_loop, std::ref(sh), std::ref(conn), std::cref(opt), std::cref(user));

    int rc = 0;
    int backoff = 1;
    while (!g_stop) {
        std::string err;
        logf("connecting to %s:%d", opt.host.c_str(), opt.port);
        if (!conn.connect(opt.host, opt.port, err)) {
            logf("connection failed: %s, retrying in %d s", err.c_str(), backoff);
            for (int i = 0; i < backoff * 10 && !g_stop; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            backoff = std::min(backoff * 2, 60);
            continue;
        }
        conn.send_line("{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"cudaminer/1.0\"]}");

        auto last_rx = Clock::now();
        bool fatal = false;
        while (!g_stop) {
            std::string line;
            int r = conn.read_line(line, 1000);
            if (r < 0) { logf("connection lost"); break; }
            if (r == 0) {
                if (Clock::now() - last_rx > std::chrono::minutes(5)) { logf("pool silent for 5 minutes, reconnecting"); break; }
                continue;
            }
            last_rx = Clock::now();
            backoff = 1;
            Json m;
            if (!Json::parse(line, m)) continue;
            Handled h = handle_message(m, sh, conn, opt, user);
            if (h == Handled::Fatal) { fatal = true; break; }
            if (h == Handled::Reconnect) break;
        }

        { // stop mining stale work while disconnected
            std::lock_guard<std::mutex> lk(sh.mu);
            sh.have_job = false;
            sh.gen++;
        }
        conn.close();
        if (fatal) { rc = 1; break; }
        for (int i = 0; i < backoff * 10 && !g_stop; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        backoff = std::min(backoff * 2, 60);
    }

    g_stop = true;
    sh.cv.notify_all();
    miner.join();
    gpu_shutdown();
    logf("stopped: %llu shares accepted, %llu rejected", (unsigned long long)sh.accepted.load(),
         (unsigned long long)sh.rejected.load());
    return rc;
}
