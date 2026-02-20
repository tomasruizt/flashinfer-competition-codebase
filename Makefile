# Env var overrides:
#   N=3 make bench-fla          # limit to 3 workloads
#   NUM_WORKLOADS=3 make modal-fla
# TRITON_PRINT_AUTOTUNING is always on (logs go to logs/fib-bench/)

.PHONY: bench-fla bench-pt modal-fla modal-pt modal-logs bench-fla-all proton-fla proton-example clean-triton-cache document-speedups

export TRITON_PRINT_AUTOTUNING=1
N ?= 0

bench-fla:
	python scripts/run_local.py --algo=fla-recurrent -n $(N)

bench-pt:
	python scripts/run_local.py --algo=pt-reference -n $(N)

modal-fla:
	ALGO=fla-recurrent modal run scripts/run_modal.py

modal-pt:
	ALGO=pt-reference modal run scripts/run_modal.py

COMMENT ?=

bench-fla-all:
	mkdir -p logs
	$(MAKE) bench-fla 2>&1 | tee logs/bench-local.log
	$(MAKE) modal-fla 2>&1 | tee logs/bench-modal.log

document-speedups:
	python scripts/log_speedups.py "$(COMMENT)"

modal-logs:
	mkdir -p logs/fib-bench-modal
	modal volume get flashinfer-trace logs/ logs/fib-bench-modal/

proton-fla:
	python scripts/profile_proton.py
	python scripts/profile_proton.py --op-measure
	@echo "\n=== Scope-level breakdown (normalized cycles) ==="
	# script -q wraps in a pseudo-TTY so proton-viewer keeps colors through tee
	script -q -c "proton-viewer -m normalized_cycles profiles/gdn_decode.hatchet" /dev/null | tee profiles/gdn_decode_scopes.txt
	python scripts/profile_proton.py --pcsampling --iters 10
	@echo "\n=== Line-by-line breakdown (PC sampling %) ==="
	script -q -c "proton-viewer -m num_samples/% profiles/gdn_decode_lines.hatchet -i profile" /dev/null | tee profiles/gdn_decode_lines.txt

clean-triton-cache:
	rm -rf ~/.triton/cache

proton-example:
	cd timeline && TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=ttgir_dump python example_dsl.py
