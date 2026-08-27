import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FIELDNAMES = [
    "path", "experiment_id", "source", "n_beads", "correct", "accuracy", "method",
    "n_predicted_good", "n_predicted_bad", "tp", "tn", "fp", "fn",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-dir", required=True, help="directory containing result_0.csv, result_1.csv, ...")
    parser.add_argument("--output-dir", "-o", required=True, help="directory for method CSVs")
    parser.add_argument("--n-runs", type=int, default=1, help="number of replicate runs using different random seed (default: 1)")
    args = parser.parse_args()

    array_dir = os.path.abspath(args.array_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.n_runs == 1:
        if not os.path.isdir(array_dir):
            print(f"not a directory: {array_dir}", file=sys.stderr)
            return 1
        result_files = sorted(Path(array_dir).glob("result_*.csv"))
    else:
        result_files = []
        for run_num in range(args.n_runs):
            run_dir = Path(array_dir) / "runs" / str(run_num)
            if run_dir.is_dir():
                result_files.extend(sorted(run_dir.glob("result_*.csv")))

    if not result_files:
        print(f"no result_*.csv in {array_dir}", file=sys.stderr)
        return 1

    replicates = {}
    for path in result_files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row.get("method", "").strip()
                if not method:
                    continue
                run = (method, row.get("path", ""), row.get("experiment_id", ""))
                if run not in replicates:
                    replicates[run] = []
                row_clean = {k: row.get(k, "") for k in FIELDNAMES}
                replicates[run].append(row_clean)

    by_method = {}
    for key, rows in replicates.items():
        method = key[0]
        merged = dict(rows[0])
        acc_sum = 0.0
        correct_sum = 0.0
        for r in rows:
            acc_sum += float(r.get("accuracy", 0) or 0)
            correct_sum += float(r.get("correct", 0) or 0)
        merged["accuracy"] = f"{acc_sum / len(rows):.6f}"
        merged["correct"] = str(round(correct_sum / len(rows)))
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(merged)

    for method, rows in by_method.items():
        out_path = os.path.join(output_dir, f"{method}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
