# cuda-tools

A collection of experimental CUDA projects for NVIDIA GPUs. Each project has
its own source and build configuration; run build commands from the project
directory shown below.

## Projects

### `cuda-mining` — Bitcoin Stratum miner

A C++/CUDA SHA-256d solo/pool miner speaking Stratum v1. It defaults to the
[Home Mining Italia solo pool](https://solo.homeminingitalia.org/#/), at
`stratum+tcp://solo.homeminingitalia.org:3340`. This is an educational lottery
miner; a GPU has no realistic chance of finding a block.

Requirements: CUDA Toolkit (`nvcc`), `g++`, `make`, and an NVIDIA GPU with a
compatible driver.

```sh
cd cuda-mining
make
make test  # check the GPU kernel against the genesis block
./cudaminer -a <your-bitcoin-address>
```

Example with a worker name and explicit pool:

```sh
./cudaminer -a bc1q... -w rig1 -o stratum+tcp://solo.homeminingitalia.org:3340
```

| Option | Meaning | Default |
| --- | --- | --- |
| `-a, --address ADDR` | Bitcoin address to mine to (required) | — |
| `-w, --worker NAME` | Worker name, sent as `ADDR.NAME` | `cudaminer` |
| `-o, --pool HOST[:PORT]` | Pool host | `solo.homeminingitalia.org` |
| `-p, --port PORT` | Pool port | `3340` |
| `--password PASS` | Pool password | `x` |
| `-d, --device N` | CUDA device index | `0` |
| `-b, --batch-bits N` | Nonces per GPU launch (`2^N`, where N is 16–32) | auto |
| `-s, --suggest-diff D` | Suggested share difficulty; 0 disables it | `0.1` |
| `--benchmark` | Measure hashrate and exit | — |
| `--selftest` | Verify the kernel and exit | — |

`CUDA_HOME` defaults to `/usr/local/cuda`. `CUDA_ARCH` defaults to an sm_50
binary plus PTX; set it for a specific GPU, for example
`make CUDA_ARCH="-arch=sm_86"`. Stop mining with Ctrl-C. Pool statistics are
shown on the pool website.

### `cuda-vanity` — BIP39 address prefix search

Searches for BIP39 mnemonics whose derived Bitcoin receive address matches
chosen characters after its address-type prefix. For each mnemonic, it checks
the first 20 receive addresses (`m/purpose'/coin'/0'/0/0..19`) for the selected
address type and network. GPU results are re-derived and verified on the CPU
before printing.

Requirements: CUDA Toolkit (`nvcc`), a compatible NVIDIA GPU and driver, Python
3 for regenerating lookup tables, and a host compiler supported by CUDA.

```sh
cd cuda-vanity
make
./build/vanity --help
```

The default build targets the native GPU. Set `ARCH` to choose an architecture
(for example, `make ARCH=sm_86`), `HOSTCC` to select the host compiler, or
`REGS` to adjust the register limit. Run `make clean` to remove generated build
files.

### `cuda-two-mnemonic` — BIP39 mnemonic pair search

Searches for two valid 12-word BIP39 mnemonics A and B where `A+B`, `B+A`, or
both are also valid 24-word mnemonics. A shortcut uses the checksum bits in
the final word to avoid hashing most candidate pairs. The host independently
validates each result.

Requirements: CMake 3.24 or newer, a C++14 compiler, the CUDA Toolkit, and a
compatible NVIDIA GPU. The program also supports CPU-only search.

```sh
cd cuda-two-mnemonic
cmake -S . -B build
cmake --build build -j
./build/two-mnemonic --selftest
```

Examples:

```sh
./build/two-mnemonic                       # find 5 pairs; A+B is valid
./build/two-mnemonic --mode both --count 3 # require both concatenations
./build/two-mnemonic --mode ba             # require B+A
./build/two-mnemonic --cpu                 # search on CPU
./build/two-mnemonic --check "A words..." "B words..."
./build/two-mnemonic --help
```

Mnemonics go to stdout; progress and statistics go to stderr. Without `--seed`,
the program reads a random seed from `/dev/urandom` and does not print it. A
supplied `--seed` makes results reproducible to anyone who knows that seed.
Treat printed phrases carefully before using them for funds: finding a
concatenation that is a valid 24-word phrase does not add security.

## Layout

Each project keeps its implementation under `src/`. BIP39 word lists live in
`data/`; generated and compiled files go under each project's `build/`
directory. `cuda-vanity/tools/` contains its table generator.
