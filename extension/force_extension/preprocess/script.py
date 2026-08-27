import os
import numpy as np
import pandas as pd

def parse_txt_script_to_unified(commands):
    if not commands:
        return []
    filtered = [c.strip() for c in commands if c.strip().startswith(("move", "idle"))]
    time_s = 0.0
    pos, speed, rot, rot_speed = 0.0, 1.0, 0.0, 0.0
    last_move = 0.0
    events = []
    for cmd in filtered:
        parts = cmd.split()
        if parts[0] == "idle":
            idle_val = int(parts[1])
            time_s += max(0, idle_val - last_move)
            last_move = 0.0
            continue
        if parts[0] == "move" and len(parts) >= 3:
            t_s = time_s
            dur_s = 0.0
            if parts[1] == "magpos":
                target_pos = float(parts[2])
                speed = float(parts[4]) if len(parts) > 4 and parts[3] == "speed" else 1.0
                dur_s = abs(target_pos - pos) / speed if speed > 0 else 0.0
                pos = target_pos
            elif parts[1] == "magrot":
                target_rot = float(parts[2])
                rot_speed = float(parts[4]) if len(parts) > 4 and parts[3] == "speed" else 1.0
                dur_s = abs(target_rot - rot) / rot_speed if rot_speed > 0 else 0.0
                rot = target_rot
            last_move = dur_s
            events.append({
                "time": t_s,
                "end_time": t_s + dur_s,
                "mag_pos": pos,
                "mag_speed": speed,
                "mag_rot": rot,
                "rot_speed": rot_speed
            })
            time_s = t_s + dur_s
    return events


def parse_npy_script_to_unified(filepath):
    if not filepath or not os.path.isfile(filepath):
        return []
    df = pd.read_csv(filepath, sep=r"\s+", header=None, encoding="utf-8")
    if df.empty or df.shape[1] < 5:
        return []
    intervals = df.iloc[:, 0].astype(float).values
    absolute = np.isclose(intervals[0], 0.0)
    events = []
    pos = 0.0
    t_cursor = 0.0
    for i in range(len(df)):
        seg_dur = max(float(intervals[i]), 0.0)
        ti = float(intervals[i]) if absolute else t_cursor
        target_pos = float(df.iloc[i, 1])
        speed = float(df.iloc[i, 2])
        dur_s = abs(target_pos - pos) / speed if speed > 0 else 0.0
        if not absolute and seg_dur > 0:
            dur_s = min(dur_s, seg_dur)
        events.append({
            "time": ti,
            "end_time": ti + dur_s,
            "mag_pos": target_pos,
            "mag_speed": speed,
            "mag_rot": float(df.iloc[i, 3]),
            "rot_speed": float(df.iloc[i, 4]),
        })
        pos = target_pos
        if not absolute:
            t_cursor += seg_dur
    return events

def get_mag_positions(unified_script, time_series):
    if not unified_script or time_series is None or len(time_series) == 0:
        return np.array([])
    time_series = np.asarray(time_series)
    out = np.empty(len(time_series))
    pos = 0.0
    j = 0
    for i, t in enumerate(time_series):
        while j + 1 < len(unified_script) and unified_script[j + 1]["time"] <= t:
            j += 1
            pos = unified_script[j - 1]["mag_pos"]
        ev = unified_script[j]
        t0, t1 = ev["time"], ev.get("end_time", ev["time"])
        if t < t0:
            out[i] = pos
            continue
        if t1 > t0 and t <= t1:
            # linear interpolation
            frac = (t - t0) / (t1 - t0)
            out[i] = pos + (ev["mag_pos"] - pos) * frac
        else:
            pos = ev["mag_pos"]
            out[i] = pos
    return out

def get_mag_rotations(unified_script, time_series):
    if not unified_script or time_series is None or len(time_series) == 0:
        return np.array([])
    time_series = np.asarray(time_series)
    out = np.empty(len(time_series))
    rot = 0.0
    j = 0
    
    for i, t in enumerate(time_series):
        while j + 1 < len(unified_script) and unified_script[j + 1]["time"] <= t:
            j += 1
            rot = unified_script[j - 1]["mag_rot"]
        ev = unified_script[j]
        t0, t1 = ev["time"], ev.get("end_time", ev["time"])
        if t < t0:
            out[i] = rot
            continue
        if t1 > t0 and t <= t1:
            frac = (t - t0) / (t1 - t0)
            out[i] = rot + (ev["mag_rot"] - rot) * frac
        else:
            rot = ev["mag_rot"]
            out[i] = rot
    return out
