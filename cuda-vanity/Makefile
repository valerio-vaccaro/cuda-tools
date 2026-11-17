# ARCH: GPU architecture, e.g. make ARCH=sm_86 (default: native GPU)
# HOSTCC: host compiler supported by your CUDA version (CUDA 11.8 needs gcc <= 11)
ARCH ?= native
# REGS: registers per GPU thread; fewer registers can increase occupancy.
REGS ?= 80
HOSTCC ?= $(shell command -v g++-11 2>/dev/null || echo g++)

TARGET = build/vanity
TABLES = build/tables.inc

$(TARGET): src/vanity.cu $(TABLES)
	nvcc -O3 -std=c++14 -arch=$(ARCH) -maxrregcount=$(REGS) -ccbin $(HOSTCC) $(DEFS) -Ibuild -o $@ $<

$(TABLES): tools/gen_tables.py data/english.txt
	python3 tools/gen_tables.py

clean:
	rm -f $(TARGET) $(TABLES)

.PHONY: clean
