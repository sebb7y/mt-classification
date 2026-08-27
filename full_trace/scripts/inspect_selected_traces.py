#!/usr/bin/env python3
"""Plot selected and optionally glitch-corrected magnetic-tweezers traces."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dict(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(data).__name__}")
    return data


def array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def selected_entry(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        raise ValueError("Selected entries must contain shifted trace/time and selected trace/time")
    return tuple(array(item) for item in value[:4])


def glitch_entry(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("Glitch entries must contain trace and time")
    return array(value[0]), array(value[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected_file", type=Path, help="path to traces_selected.pickle")
    parser.add_argument("--glitch-file", type=Path, help="optional path to traces_glitch.pickle")
    parser.add_argument("--keys", help="comma-separated bead keys, e.g. Bead_1_z,Bead_2_z")
    parser.add_argument("--n-beads", type=int, default=6, help="number of entries to plot by default")
    parser.add_argument("--output", type=Path, default=Path("selected_trace_inspection.png"))
    parser.add_argument("--show", action="store_true", help="also open the figure interactively")
    args = parser.parse_args()

    selected_path = args.selected_file.expanduser().resolve()
    if not selected_path.is_file():
        parser.error(f"Selected file does not exist: {selected_path}")
    selected = load_dict(selected_path)

    glitch_path = args.glitch_file.expanduser().resolve() if args.glitch_file else None
    if glitch_path is None and "selected" in selected_path.name:
        candidate = selected_path.with_name(selected_path.name.replace("selected", "glitch"))
        if candidate.is_file():
            glitch_path = candidate
    glitch = load_dict(glitch_path) if glitch_path else {}

    if args.keys:
        keys = [key.strip() for key in args.keys.split(",") if key.strip()]
        missing = [key for key in keys if key not in selected]
        if missing:
            parser.error(f"Keys not found: {', '.join(missing)}")
    else:
        keys = list(selected)[: args.n_beads]
    if not keys:
        parser.error("No bead entries found")

    ncols = 2 if glitch else 1
    fig, axes = plt.subplots(len(keys), ncols, squeeze=False, figsize=(13 if glitch else 8, 3.2 * len(keys)))
    for row, key in enumerate(keys):
        shifted, shifted_time, chosen, chosen_time = selected_entry(selected[key])
        ax = axes[row, 0]
        ax.plot(shifted_time, shifted, color="0.65", linewidth=0.8, label="shifted trace")
        ax.plot(chosen_time, chosen, color="tab:blue", linewidth=1.0, label="selected trace")
        if len(chosen_time):
            ax.axvspan(chosen_time[0], chosen_time[-1], color="tab:blue", alpha=0.10)
        ax.set_title(str(key))
        ax.set_ylabel("z")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize="small")

        if glitch:
            right = axes[row, 1]
            if key in glitch:
                corrected, corrected_time = glitch_entry(glitch[key])
                right.plot(corrected_time, corrected, color="tab:orange", linewidth=0.8, label="glitch-corrected")
                right.legend(loc="upper right", fontsize="small")
            else:
                right.text(0.5, 0.5, "bead not present", ha="center", va="center", transform=right.transAxes)
            right.set_title(f"{key} — glitch file")
            right.set_ylabel("z")
            right.grid(alpha=0.2)

        if row == len(keys) - 1:
            ax.set_xlabel("time")
            if glitch:
                axes[row, 1].set_xlabel("time")

    fig.suptitle(selected_path.name)
    fig.tight_layout()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Plotted {len(keys)} bead(s) from {selected_path}")
    if glitch_path:
        print(f"Compared with {glitch_path}")
    print(f"Saved figure to {output_path}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
