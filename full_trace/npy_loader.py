import os
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

def end_of_file(f):
    curpos = f.tell()
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(curpos, 0)
    return curpos == file_size

def load_traces_raw(path, pixel_units=False):
    with open(path, "rb") as s:
        header = np.load(s, allow_pickle=True)
        if header.dtype == object:
            header = header.item()
            axis_scale = header["axisScale"]
        else:
            axis_scale = header
        blocks = []
        while not end_of_file(s):
            try:
                blk = np.load(s)
                blocks.append(blk)
            except ValueError as e:
                if "reshape" in str(e).lower() or "size" in str(e):
                    if blocks:
                        break
                    raise
                raise
    if not blocks:
        raise ValueError(f"no blocks loaded from {path}")
    if pixel_units:
        return np.concatenate(blocks), axis_scale
    return np.concatenate(blocks) * np.array(axis_scale)[None, None]

def load_traces(path, pixel_units=False):
    return load_traces_raw(path, pixel_units)

def read_settings(exp_folder):
    config_path = os.path.join(exp_folder, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
    else:
        settings = {"framerate": 58, "pixel_size": 100, "refIndices": -1, "processing_zcorrection": 0.88}
    correction_factor = 0.88 if "zcorrection" not in settings else -1
    ref_val = settings.get("refIndices", -1)
    if isinstance(ref_val, (int, float)):
        reference_id = int(ref_val)
    elif isinstance(ref_val, list):
        reference_id = int(ref_val[-1])
    else:
        ref_indices = str(ref_val).strip().split(",")
        reference_id = ref_indices[-1].strip().strip("'").strip('"')
        reference_id = int(reference_id)
    framerate = settings.get("framerate", 58)
    framerate = float(framerate) if framerate is not None else 58.0
    return settings, reference_id, correction_factor, framerate

def load_npy_data(filepath, correction_factor=None, reference_id=None, framerate=None, already_processed=False):
    exp_folder = os.path.dirname(os.path.abspath(filepath))
    settings, ref_from_config, correction_from_config, framerate_from_config = read_settings(exp_folder)
    if framerate is None:
        framerate = framerate_from_config
    if correction_factor is None:
        correction_factor = correction_from_config
    if reference_id is None:
        reference_id = ref_from_config

    traces = load_traces(filepath)
    traces = traces / 1000
    reshaped_data = np.transpose(traces, (1, 2, 0))[:, 2, :]
    reshaped_data *= -correction_factor

    num_beads, num_frames = reshaped_data.shape
    if reference_id is None:
        reference_id = -1
    ref_int = int(reference_id)
    if ref_int < 0:
        ref_int = num_beads + ref_int
    if ref_int > reshaped_data.shape[0] - 1:
        ref_int = reshaped_data.shape[0] - 1
    if reference_id and 0 <= ref_int < num_beads:
        reshaped_data = reshaped_data - reshaped_data[ref_int, :]

    time_s = np.arange(reshaped_data.shape[1]) / framerate

    exp_z_data = reshaped_data.T
    exp_z = pd.DataFrame(exp_z_data)
    exp_z.columns = range(exp_z.shape[1])
    exp_time = pd.DataFrame(time_s, columns=[0])

    if traces.shape[2] >= 3:
        exp_x_data = np.transpose(traces, (1, 2, 0))[:, 0, :].T
        exp_y_data = np.transpose(traces, (1, 2, 0))[:, 1, :].T
    else:
        exp_x_data = np.full((num_frames, num_beads), np.nan)
        exp_y_data = np.full((num_frames, num_beads), np.nan)
    exp_x = pd.DataFrame(exp_x_data)
    exp_x.columns = range(exp_x.shape[1])
    exp_y = pd.DataFrame(exp_y_data)
    exp_y.columns = range(exp_y.shape[1])

    return exp_z, exp_time, exp_x, exp_y

def plot_random_cols_grid(x_df, y_dfs, n=3, seed=None, dot_size=10, colours=None, use_lines=True, max_points=300000, show=True, title=None):
    rng = np.random.default_rng(seed)
    x = x_df.iloc[:, 0].values
    if colours is None:
        colours = [None] * len(y_dfs)
    total = n * len(y_dfs)
    rows = (total + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), sharex=True, sharey=True)
    if title:
        fig.suptitle(title)
    axes = axes.flatten()
    if max_points and len(x) > max_points:
        sl = slice(0, max_points)
        x_plot = x[sl]
    else:
        sl = slice(None)
        x_plot = x
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min = min(np.nanquantile(df.values, 0.01) for df in y_dfs)
    y_max = max(np.nanquantile(df.values, 0.99) for df in y_dfs)
    y_pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) != 0 else 1.0
    x_lim = (x_min, x_max)
    y_lim = (y_min - y_pad, y_max + y_pad)
    plot_idx = 0
    for df, colour in zip(y_dfs, colours):
        n_cols = min(n, len(df.columns))
        if 0 in df.columns:
            remaining = [c for c in df.columns if c != 0]
            n_extra = min(n_cols - 1, len(remaining))
            chosen_cols = [0] + list(rng.choice(remaining, size=n_extra, replace=False)) if n_extra > 0 else [0]
        else:
            chosen_cols = list(rng.choice(df.columns, size=n_cols, replace=False))
        for col in chosen_cols:
            ax = axes[plot_idx]
            y_vals = df[col].values[sl] if sl != slice(None) else df[col].values
            if use_lines:
                ax.plot(x_plot, y_vals, color=colour or "b", alpha=0.8)
            else:
                ax.scatter(x_plot, y_vals, s=dot_size, c=colour, alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(col)
            ax.set_title(f"{col} vs time")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            plot_idx += 1
    for ax in axes[plot_idx:]:
        ax.axis("off")
    plt.tight_layout()
    if show:
        plt.show()

def mask_within_n_std(exp_z, exp_time, exp_x, exp_y, n_std=2, min_fraction=0.95):
    z = exp_z.values
    mean_z = np.nanmean(z)
    std_z = np.nanstd(z)
    if std_z == 0 or not np.isfinite(std_z):
        std_z = 1.0
    lo, hi = mean_z - n_std * std_z, mean_z + n_std * std_z
    in_range = (z >= lo) & (z <= hi)
    valid = np.mean(in_range, axis=1) >= min_fraction
    padded = np.concatenate([[False], valid, [False]])
    run_starts = np.where(~padded[:-1] & padded[1:])[0]
    run_ends = np.where(padded[:-1] & ~padded[1:])[0]
    if len(run_starts) == 0:
        return exp_z, exp_time, exp_x, exp_y, (0, len(valid))
    lengths = run_ends - run_starts
    longest = np.argmax(lengths)
    start, end = run_starts[longest], run_ends[longest]
    exp_z_s = exp_z.iloc[start:end].reset_index(drop=True)
    exp_time_s = exp_time.iloc[start:end].reset_index(drop=True)
    exp_x_s = exp_x.iloc[start:end].reset_index(drop=True)
    exp_y_s = exp_y.iloc[start:end].reset_index(drop=True)
    return exp_z_s, exp_time_s, exp_x_s, exp_y_s, (start, end)

def load_and_plot_npy(filepath, n=6, seed=42, dot_size=1, focus_within_2std=False):
    exp_z, exp_time, exp_x, exp_y = load_npy_data(filepath)
    if focus_within_2std:
        exp_z, exp_time, exp_x, exp_y, (start, end) = mask_within_n_std(exp_z, exp_time, exp_x, exp_y, n_std=2)
    plot_random_cols_grid(exp_time, [exp_z], n=n, seed=seed, dot_size=dot_size, colours=["b"], use_lines=True, show=False, title="Line plot")
    plot_random_cols_grid(exp_time, [exp_z], n=n, seed=seed, dot_size=dot_size, colours=["b"], use_lines=False, show=False, title="Scatter plot")
    plt.show()
    return exp_z, exp_time, exp_x, exp_y

if __name__ == "__main__":
    argv = sys.argv[1:]
    focus = "--focus" in argv or "--within-2std" in argv
    if focus:
        argv = [a for a in argv if a not in ("--focus", "--within-2std")]
    path = argv[0] if argv else None
    if path:
        path = os.path.abspath(path)
    if path and os.path.isfile(path):
        load_and_plot_npy(path, focus_within_2std=focus)
    else:
        print("usage: python -m force_extension.load.npy_loader [--focus] <path/to/traces.npy>")
        print("  --focus (or --within-2std): plot only longest section where >=95% of beads are within mean +/- 2*std")
        print("  or: from npy_loader import load_and_plot_npy; load_and_plot_npy('path/to/traces.npy', focus_within_2std=True)")
