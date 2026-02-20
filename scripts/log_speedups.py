"""Parse bench log files and append a row to findings/speedups.csv."""

import argparse
import csv
import os
import re
import subprocess
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "findings" / "speedups.csv"

# Algo name -> log file prefix (matches Makefile bench-*-all targets)
ALGO_LOG_PREFIX = {
    "fla-recurrent": "bench-fla",
    "fla-tma": "bench-tma",
    "pt-reference": "bench-pt",
}


def parse_speedups(log_path: Path) -> list[float]:
    """Extract speedup values from a benchmark log file."""
    if not log_path.exists():
        return []
    pattern = re.compile(r"(\d+\.\d+)x speedup")
    return [float(m.group(1)) for m in pattern.finditer(log_path.read_text())]


def get_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Log speedups from bench logs to CSV")
    parser.add_argument("comment", nargs="?", default="")
    parser.add_argument(
        "--algo",
        default=os.getenv("ALGO", "fla-recurrent"),
        choices=ALGO_LOG_PREFIX.keys(),
    )
    args = parser.parse_args()

    prefix = ALGO_LOG_PREFIX[args.algo]
    local_log = PROJECT_ROOT / "logs" / f"{prefix}-local.log"
    modal_log = PROJECT_ROOT / "logs" / f"{prefix}-modal.log"

    local_speedups = parse_speedups(local_log)
    modal_speedups = parse_speedups(modal_log)

    if not local_speedups and not modal_speedups:
        print(f"No speedup data found in logs for {args.algo}.")
        print(f"  Checked: {local_log}")
        print(f"  Checked: {modal_log}")
        return

    row = {
        "date": date.today().isoformat(),
        "commit": get_commit_short(),
        "algo": args.algo,
        "speedup_local_min": f"{min(local_speedups):.1f}" if local_speedups else "",
        "speedup_local_max": f"{max(local_speedups):.1f}" if local_speedups else "",
        "speedup_modal_min": f"{min(modal_speedups):.1f}" if modal_speedups else "",
        "speedup_modal_max": f"{max(modal_speedups):.1f}" if modal_speedups else "",
        "comment": args.comment,
    }

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

    print(f"Appended to {CSV_PATH}:")
    print(",".join(row.values()))


if __name__ == "__main__":
    main()
