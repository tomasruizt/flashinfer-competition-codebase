# Env var overrides:
#   N=3 make bench-fla          # limit to 3 workloads
#   NUM_WORKLOADS=3 make modal-fla
# TRITON_PRINT_AUTOTUNING is always on (logs go to logs/fib-bench/)

.PHONY: bench-fla bench-pt bench-tma bench-fi bench-cuda bench-cuda-v4 modal-fla modal-pt modal-tma modal-fi modal-cuda modal-get-logs modal-clear-logs bench-fla-all bench-tma-all clean-empty-logs proton-fla proton-example clean-triton-cache document-speedups ncu-fla ncu-fi ncu-cuda ncu-cuda-v4 ncu-export-fla ncu-export-fi ncu-export-cuda ncu-export-cuda-v4 nvbench nvbench-fla nvbench-fi nvbench-cuda nvbench-cuda-v4 nvbench-all nvbench-modal-fla nvbench-modal-fi nvbench-modal-cuda-all nvbench-modal-all

export TRITON_PRINT_AUTOTUNING=1
N ?= 0
ALGO ?= fla-recurrent
NCU := $(shell which ncu)
PYTHON := $(shell which python)
SUDO=
NCU_RESULTS_DIR := profiles/ncu
NCU_TXT_DIR := profiles/ncu-txt
TRITON_PROFILE_ENV := TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=profiles/ttgir_dump

bench-fla:
	python -m scripts.run_local --algo=fla-recurrent -n $(N)

bench-pt:
	python -m scripts.run_local --algo=pt-reference -n $(N)

bench-tma:
	python -m scripts.run_local --algo=fla-tma -n $(N)

bench-fi:
	python -m scripts.run_local --algo=fi-baseline -n $(N)

bench-cuda:
	python -m scripts.run_local --algo=cuda-v1 -n $(N)

bench-cuda-v4:
	python -m scripts.run_local --algo=cuda-v4 -n $(N)

modal-fla:
	ALGO=fla-recurrent modal run -m scripts.run_modal

modal-tma:
	ALGO=fla-tma modal run -m scripts.run_modal

modal-pt:
	ALGO=pt-reference modal run -m scripts.run_modal

modal-fi:
	ALGO=fi-baseline modal run -m scripts.run_modal

modal-cuda:
	ALGO=cuda-v1 modal run -m scripts.run_modal

modal-cuda-v4:
	ALGO=cuda-v4 modal run -m scripts.run_modal


COMMENT ?=

bench-fla-all:
	mkdir -p logs
	$(MAKE) bench-fla 2>&1 | tee logs/bench-fla-local.log
	$(MAKE) modal-fla 2>&1 | tee logs/bench-fla-modal.log

bench-tma-all:
	mkdir -p logs
	$(MAKE) bench-tma 2>&1 | tee logs/bench-tma-local.log
	$(MAKE) modal-tma 2>&1 | tee logs/bench-tma-modal.log

document-speedups:
	python -m scripts.log_speedups --algo=$(ALGO) "$(COMMENT)"

modal-get-logs:
	mkdir -p logs/fib-bench-modal
	modal volume get flashinfer-trace logs/ logs/fib-bench-modal/ --force

modal-clear-logs:
	modal volume rm -r flashinfer-trace logs/

proton-fla:
	$(TRITON_PROFILE_ENV) python -m scripts.profile_proton
	$(TRITON_PROFILE_ENV) python -m scripts.profile_proton --op-measure
	@echo "\n=== Scope-level breakdown (normalized cycles) ==="
	# script -q wraps in a pseudo-TTY so proton-viewer keeps colors through tee
	script -q -c "proton-viewer -m normalized_cycles profiles/gdn_decode.hatchet" /dev/null | tee profiles/gdn_decode_scopes.txt
	# Had to disable the lines below because it was incompatible with CUDA 13
	# python -m scripts.profile_proton --pcsampling --iters 10
	# @echo "\n=== Line-by-line breakdown (PC sampling %) ==="
	# script -q -c "proton-viewer -m num_samples/% profiles/gdn_decode_lines.hatchet -i profile" /dev/null | tee profiles/gdn_decode_lines.txt

clean-empty-logs:
	find logs/fib-bench logs/fib-bench-modal -empty -name '*.log' -delete 2>/dev/null; true

clean-triton-cache:
	rm -rf ~/.triton/cache

ncu-fla:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name fused_recurrent_gated_delta_rule_fwd_kernel \
		--launch-skip 3 --launch-count 1 \
		-fo $(NCU_RESULTS_DIR)/gdn-decode-fla \
		$(PYTHON) -m scripts.profile_ncu --algo=fla-recurrent

ncu-fi:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name regex:kernel_cutlass_gdn_decode \
		--launch-skip 3 --launch-count 1 \
		-fo $(NCU_RESULTS_DIR)/gdn-decode-fi \
		$(PYTHON) -m scripts.profile_ncu --algo=fi-baseline

ncu-cuda:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name gdn_decode_kernel \
		--launch-skip 3 --launch-count 1 \
		-fo $(NCU_RESULTS_DIR)/gdn-decode-cuda \
		$(PYTHON) -m scripts.profile_ncu --algo=cuda-v1

ncu-cuda-v4:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name gdn_decode_v4_kernel \
		--launch-skip 3 --launch-count 1 \
		-fo $(NCU_RESULTS_DIR)/gdn-decode-cuda-v4 \
		$(PYTHON) -m scripts.profile_ncu --algo=cuda-v4

ncu-export-fla:
	mkdir -p $(NCU_TXT_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name fused_recurrent_gated_delta_rule_fwd_kernel \
		--launch-skip 3 --launch-count 1 \
		$(PYTHON) -m scripts.profile_ncu --algo=fla-recurrent > $(NCU_TXT_DIR)/gdn-decode-fla.txt

ncu-export-fi:
	mkdir -p $(NCU_TXT_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name regex:kernel_cutlass_gdn_decode \
		--launch-skip 3 --launch-count 1 \
		$(PYTHON) -m scripts.profile_ncu --algo=fi-baseline > $(NCU_TXT_DIR)/gdn-decode-fi.txt

ncu-export-cuda:
	mkdir -p $(NCU_TXT_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name gdn_decode_kernel \
		--launch-skip 3 --launch-count 1 \
		$(PYTHON) -m scripts.profile_ncu --algo=cuda-v1 > $(NCU_TXT_DIR)/gdn-decode-cuda.txt

ncu-export-cuda-v4:
	mkdir -p $(NCU_TXT_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name gdn_decode_v4_kernel \
		--launch-skip 3 --launch-count 1 \
		$(PYTHON) -m scripts.profile_ncu --algo=cuda-v4 > $(NCU_TXT_DIR)/gdn-decode-cuda-v4.txt

nvbench:
	python -m scripts.bench_nvbench --algo=$(ALGO)

nvbench-cuda:
	python -m scripts.bench_nvbench --algo=cuda-v1

nvbench-cuda-v4:
	python -m scripts.bench_nvbench --algo=cuda-v4



nvbench-fla:
	python -m scripts.bench_nvbench --algo=fla-recurrent

nvbench-fi:
	python -m scripts.bench_nvbench --algo=fi-baseline

nvbench-all:
	python -m scripts.bench_nvbench --algo=all

nvbench-modal:
	ALGO=$(ALGO) modal run -m scripts.bench_nvbench_modal

nvbench-modal-fla:
	ALGO=fla-recurrent modal run -m scripts.bench_nvbench_modal

nvbench-modal-fi:
	ALGO=fi-baseline modal run -m scripts.bench_nvbench_modal

nvbench-modal-cuda-all:
	ALGO=cuda-all modal run -m scripts.bench_nvbench_modal

nvbench-modal-all:
	ALGO=all modal run -m scripts.bench_nvbench_modal

proton-example:
	cd timeline && TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=ttgir_dump python example_dsl.py


# Generate all results
bench-gen-all-results:
	mkdir -p final-nums/fi-bench/
	$(MAKE) bench-fla > final-nums/fi-bench/fla-recurrent.txt
	$(MAKE) bench-tma > final-nums/fi-bench/fla-tma.txt
	$(MAKE) bench-pt > final-nums/fi-bench/pt-reference.txt
	$(MAKE) bench-fi > final-nums/fi-bench/fi-baseline.txt
	$(MAKE) bench-cuda > final-nums/fi-bench/cuda-v1.txt
	$(MAKE) bench-cuda-v4 > final-nums/fi-bench/cuda-v4.txt

modal-gen-all-results:
	mkdir -p final-nums/fi-bench-modal/
	$(MAKE) modal-fla > final-nums/fi-bench-modal/fla-recurrent.txt
	$(MAKE) modal-tma > final-nums/fi-bench-modal/fla-tma.txt
	$(MAKE) modal-pt > final-nums/fi-bench-modal/pt-reference.txt
	$(MAKE) modal-fi > final-nums/fi-bench-modal/fi-baseline.txt
	$(MAKE) modal-cuda > final-nums/fi-bench-modal/cuda-v1.txt
	$(MAKE) modal-cuda-v4 > final-nums/fi-bench-modal/cuda-v4.txt

nvbench-gen-all-results:
	mkdir -p final-nums/nvbench/
	$(MAKE) ALGO=all nvbench > final-nums/nvbench/all.txt

nvbench-modal-gen-all-results:
	mkdir -p final-nums/nvbench-modal/
	$(MAKE) ALGO=all nvbench-modal > final-nums/nvbench-modal/all.txt
