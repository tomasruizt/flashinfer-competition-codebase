.PHONY: bench-fla bench-pt

bench-fla:
	python scripts/run_local.py --algo=fla-recurrent

bench-pt:
	python scripts/run_local.py --algo=pt-reference
