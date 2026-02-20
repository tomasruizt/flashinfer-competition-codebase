"""Parse bench log files and append a row to findings/speedups.csv."""

import csv
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "findings" / "speedups.csv"


def parse_speedups(log_path: Path) -> list[float]:
    """Extract speedup values from a benchmark log file."""
    if not log_path.exists():
        return []
    pattern = re.compile(r"(\d+\.\d+)x speedup")
    return [float(m.group(1)) for m in pattern.finditer(log_path.read_text())]


def get_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    return result.stdout.strip()


def main():
    comment = sys.argv[1] if len(sys.argv) > 1 else ""

    local_log = PROJECT_ROOT / "logs" / "bench-local.log"
    modal_log = PROJECT_ROOT / "logs" / "bench-modal.log"

    local_speedups = parse_speedups(local_log)
    modal_speedups = parse_speedups(modal_log)

    if not local_speedups and not modal_speedups:
        print("No speedup data found in logs.")
        return

    row = {
        "date": date.today().isoformat(),
        "commit": get_commit_short(),
        "algo": "fla-recurrent",
        "speedup_local_min": f"{min(local_speedups):.1f}" if local_speedups else "",
        "speedup_local_max": f"{max(local_speedups):.1f}" if local_speedups else "",
        "speedup_modal_min": f"{min(modal_speedups):.1f}" if modal_speedups else "",
        "speedup_modal_max": f"{max(modal_speedups):.1f}" if modal_speedups else "",
        "comment": comment,
    }

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

    print(f"Appended to {CSV_PATH}:")
    print(",".join(row.values()))


if __name__ == "__main__":
    main()
