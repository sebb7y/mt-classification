import csv
import json
import os


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def write_markdown_report(path, *, config, data_summary, transfer_rows, distance_rows, discriminator, figures=None):
    figures = figures or []
    best_real = None
    real_rows = [r for r in transfer_rows if r.get("eval_set") == "real_test"]
    if real_rows:
        best_real = max(real_rows, key=lambda r: r.get("balanced_accuracy") if r.get("balanced_accuracy") == r.get("balanced_accuracy") else -1)
    largest_gaps = sorted(distance_rows, key=lambda r: r.get("ks_stat") if r.get("ks_stat") == r.get("ks_stat") else -1, reverse=True)[:5]

    lines = []
    lines.append("# Synthetic Data Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if best_real:
        lines.append(
            f"Best real-test result: `{best_real.get('train_setup', '')}` / `{best_real['model']}` with "
            f"balanced accuracy `{best_real['balanced_accuracy']:.3f}` and accuracy `{best_real['accuracy']:.3f}`."
        )
    else:
        lines.append("No synthetic-trained model result was available for the real test set.")
    if discriminator and discriminator.get("accuracy") == discriminator.get("accuracy"):
        lines.append(
            f"Real-vs-synthetic discriminator accuracy: `{discriminator['accuracy']:.3f}` "
            f"(lower is better; high values mean synthetic remains easy to separate from real)."
        )
    lines.append("")
    lines.append("## Data")
    lines.append("")
    for key, value in data_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Transfer Performance")
    lines.append("")
    lines.append("| Train setup | Model | Eval set | Accuracy | Balanced acc | Precision good | Recall good | MCC | TP | TN | FP | FN |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in transfer_rows:
        lines.append(
            f"| {row.get('train_setup', '')} | {row.get('model', '')} | {row.get('eval_set', '')} | "
            f"{_fmt(row.get('accuracy'))} | {_fmt(row.get('balanced_accuracy'))} | "
            f"{_fmt(row.get('precision_good'))} | {_fmt(row.get('recall_good'))} | {_fmt(row.get('mcc'))} | "
            f"{row.get('tp', '')} | {row.get('tn', '')} | {row.get('fp', '')} | {row.get('fn', '')} |"
        )
    lines.append("")
    lines.append("## Largest Feature Gaps")
    lines.append("")
    lines.append("| Feature | KS stat | Real median | Synthetic median | Median delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in largest_gaps:
        lines.append(
            f"| {row.get('feature')} | {_fmt(row.get('ks_stat'))} | {_fmt(row.get('real_median'))} | "
            f"{_fmt(row.get('synthetic_median'))} | {_fmt(row.get('median_delta'))} |"
        )
    lines.append("")
    if figures:
        lines.append("## Figures")
        lines.append("")
        for figure in figures:
            label = figure.get("label", os.path.basename(figure.get("path", "")))
            rel_path = figure.get("relative_path") or figure.get("path")
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{label}]({rel_path})")
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Synthetic quality should be judged mainly by held-out real performance, not synthetic test accuracy.")
    lines.append("- A high real-vs-synthetic discriminator score means the generator is still leaving obvious statistical fingerprints.")
    lines.append("- The largest feature gaps are the first places to tune the generator or augmentation process.")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, default=str))
    lines.append("```")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(value):
    try:
        if value != value:
            return ""
        return f"{float(value):.3f}"
    except Exception:
        return ""
