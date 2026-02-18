.PHONY: bench-fla bench-pt modal-fla modal-pt

bench-fla:
	python scripts/run_local.py --algo=fla-recurrent

bench-pt:
	python scripts/run_local.py --algo=pt-reference

modal-fla:
	ALGO=fla-recurrent modal run scripts/run_modal.py

modal-pt:
	ALGO=pt-reference modal run scripts/run_modal.py
