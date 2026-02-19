.PHONY: bench-fla bench-pt modal-fla modal-pt proton-example proton-trace proton-cycles

bench-fla:
	python scripts/run_local.py --algo=fla-recurrent

bench-pt:
	python scripts/run_local.py --algo=pt-reference

modal-fla:
	ALGO=fla-recurrent modal run scripts/run_modal.py

modal-pt:
	ALGO=pt-reference modal run scripts/run_modal.py

proton-fla:
	python scripts/profile_proton.py
	python scripts/profile_proton.py --op-measure
	proton-viewer -m normalized_cycles profiles/gdn_decode.hatchet

proton-example:
	cd timeline && TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=ttgir_dump python example_dsl.py