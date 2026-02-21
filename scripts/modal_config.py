"""
Shared Modal configuration, re-exported from run_modal.py.

Only used by scripts that mount scripts/ into the Modal image
(e.g. bench_nvbench_modal.py). Scripts that don't mount scripts/
(e.g. run_modal.py) define their config inline.
"""

from run_modal import ALGO_ENTRY_POINTS, TRACE_SET_PATH, image, trace_volume

__all__ = ["ALGO_ENTRY_POINTS", "TRACE_SET_PATH", "image", "trace_volume"]
