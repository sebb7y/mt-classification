import argparse
import csv
import json
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="path to eval_manifest.csv")
    parser.add_argument("--output-dir", "-o", default="eval_synthetic_real", help="directory to write split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--by-source", action="store_true", help="train on majority source, test on others")
    parser.add_argument("--max-experiments", type=int, default=None, help="cap number of labelled experiments")
    parser.add_argument("--sources", type=str, default=None, help="comma-separated source names to include")
    parser.add_argument("--test-all", action="store_true", help="all in test only (rules-only eval)")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    if not os.path.isfile(manifest_path):
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with_labels = [r for r in rows if r.get("has_labels", "").strip().lower() in ("true", "1", "yes")]
    if with_labels and "has_both_classes" in (with_labels[0] or {}):
        usable = [r for r in with_labels if r.get("has_both_classes", "").strip().lower() in ("true", "1", "yes")]
        if len(usable) < len(with_labels):
            print(f"excluded {len(with_labels) - len(usable)} one-class experiment(s), {len(usable)} with both classes", file=sys.stderr)
        with_labels = usable

    if args.sources:
        allowed = {s.strip() for s in args.sources.split(",") if s.strip()}
        with_labels = [r for r in with_labels if (r.get("source", "").strip() or "") in allowed]
        if not with_labels:
            print("no experiments with labels in sources:", allowed, file=sys.stderr)
            return 1

    if args.max_experiments is not None:
        with_labels = with_labels[: args.max_experiments]

    if args.test_all:
        train_rows = []
        test_rows = with_labels
        print(f"test-all: 0 train, {len(test_rows)} test (rules-only eval)")
    elif len(with_labels) < 2:
        print("need at least 2 experiments with labels", file=sys.stderr)
        return 1
    elif args.by_source:
        sources = [r.get("source", "").strip() or "unknown" for r in with_labels]
        majority_source = Counter(sources).most_common(1)[0][0]
        train_rows = [r for r in with_labels if (r.get("source", "").strip() or "unknown") == majority_source]
        test_rows = [r for r in with_labels if (r.get("source", "").strip() or "unknown") != majority_source]
        print(f"split by source: train '{majority_source}' ({len(train_rows)}), test ({len(test_rows)})")
    else:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(with_labels))
        rng.shuffle(idx)
        n_train = max(1, int(len(with_labels) * args.train_frac))
        train_idx = set(idx[:n_train])
        train_rows = [with_labels[i] for i in range(len(with_labels)) if i in train_idx]
        test_rows = [with_labels[i] for i in range(len(with_labels)) if i not in train_idx]
        print(f"split (seed={args.seed}): {len(train_rows)} train, {len(test_rows)} test")

    if not test_rows:
        print("need at least one test experiment", file=sys.stderr)
        return 1
    if not args.test_all and not train_rows:
        print("need at least one train experiment (or use --test-all for rules-only)", file=sys.stderr)
        return 1

    def row_minimal(r):
        return {"path": r.get("path", "").strip(), "experiment_id": r.get("experiment_id", "").strip(), "source": r.get("source", "").strip()}

    split_data = {
        "manifest": os.path.abspath(manifest_path),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "by_source": args.by_source,
        "test_all": args.test_all,
        "train_rows": [row_minimal(r) for r in train_rows],
        "test_rows": [row_minimal(r) for r in test_rows]
    }

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(output_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"wrote {split_path} ({len(train_rows)} train, {len(test_rows)} test)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
