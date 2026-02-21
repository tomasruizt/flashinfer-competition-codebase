# Env var overrides:
#   N=3 make bench-fla          # limit to 3 workloads
#   NUM_WORKLOADS=3 make modal-fla
# TRITON_PRINT_AUTOTUNING is always on (logs go to logs/fib-bench/)

.PHONY: bench-fla bench-pt bench-tma modal-fla modal-pt modal-tma modal-get-logs modal-clear-logs bench-fla-all bench-tma-all clean-empty-logs proton-fla proton-example clean-triton-cache document-speedups ncu-fla ncu-fla ncu-export-fla

export TRITON_PRINT_AUTOTUNING=1
N ?= 0
ALGO ?= fla-recurrent
NCU := $(shell which ncu)
PYTHON := $(shell which python)
SUDO=
NCU_RESULTS_DIR := profiles/ncu

bench-fla:
	python scripts/run_local.py --algo=fla-recurrent -n $(N)

bench-pt:
	python scripts/run_local.py --algo=pt-reference -n $(N)

bench-tma:
	python scripts/run_local.py --algo=fla-tma -n $(N)

bench-fi:
	python scripts/run_local.py --algo=fi-baseline -n $(N)

modal-fla:
	ALGO=fla-recurrent modal run scripts/run_modal.py

modal-tma:
	ALGO=fla-tma modal run scripts/run_modal.py

modal-pt:
	ALGO=pt-reference modal run scripts/run_modal.py

modal-fi:
	ALGO=fi-baseline modal run scripts/run_modal.py

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
	python scripts/log_speedups.py --algo=$(ALGO) "$(COMMENT)"

modal-get-logs:
	mkdir -p logs/fib-bench-modal
	modal volume get flashinfer-trace logs/ logs/fib-bench-modal/ --force

modal-clear-logs:
	modal volume rm -r flashinfer-trace logs/

proton-fla:
	python scripts/profile_proton.py
	python scripts/profile_proton.py --op-measure
	@echo "\n=== Scope-level breakdown (normalized cycles) ==="
	# script -q wraps in a pseudo-TTY so proton-viewer keeps colors through tee
	script -q -c "proton-viewer -m normalized_cycles profiles/gdn_decode.hatchet" /dev/null | tee profiles/gdn_decode_scopes.txt
	python scripts/profile_proton.py --pcsampling --iters 10
	@echo "\n=== Line-by-line breakdown (PC sampling %) ==="
	script -q -c "proton-viewer -m num_samples/% profiles/gdn_decode_lines.hatchet -i profile" /dev/null | tee profiles/gdn_decode_lines.txt

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
		$(PYTHON) scripts/profile_ncu.py --algo=fla-recurrent

ncu-fi:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name regex:kernel_cutlass_gdn_decode \
		--launch-skip 3 --launch-count 1 \
		-fo $(NCU_RESULTS_DIR)/gdn-decode-fi \
		$(PYTHON) scripts/profile_ncu.py --algo=fi-baseline

ncu-export-fi:
	mkdir -p $(NCU_TXT_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name regex:kernel_cutlass_gdn_decode \
		--launch-skip 3 --launch-count 1 \
		$(PYTHON) scripts/profile_ncu.py --algo=fi-baseline > $(NCU_TXT_DIR)/gdn-decode-fi.txt

NCU_TXT_DIR := profiles/ncu-txt

ncu-export-fla:
	mkdir -p $(NCU_TXT_DIR)
	$(SUDO) $(NCU) --set full \
		--import-source yes \
		--kernel-name fused_recurrent_gated_delta_rule_fwd_kernel \
		--launch-skip 3 --launch-count 1 \
		$(PYTHON) scripts/profile_ncu.py --algo=fla-recurrent > $(NCU_TXT_DIR)/gdn-decode-fla.txt

proton-example:
	cd timeline && TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=ttgir_dump python example_dsl.py
