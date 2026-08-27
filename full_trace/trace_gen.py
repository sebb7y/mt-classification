import numpy as np
import multiprocessing as mp
from functools import partial
import time
from scipy import interpolate
import pandas as pd

from slurm_helpers import worker_count

class StateSpec:
    def __init__(self, name, weight, generator):
        self.name = name
        self.weight = weight
        self.generator = generator

def sample_uni(low, high, rng):
    return rng.uniform(low, high)

def sample_exp(rate_range, rng):
    rate = sample_uni(rate_range[0], rate_range[1], rng)
    return rng.exponential(1.0 / rate)

def choose_state(states, rng):
    weights = np.array([s.weight for s in states], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(states), p=weights)
    return states[idx]

def pause_state(name, rate_range, weight):
    def generate(t0, remaining, dt, last_value, rng):
        duration = min(sample_exp(rate_range, rng), remaining)
        times = np.arange(t0, t0 + duration, dt)
        values = np.full_like(times, last_value)
        return times, values, last_value

    return StateSpec(name=name, weight=weight, generator=generate)

def power_law_pause_state(name, alpha, min_pause, max_pause, weight):
    def generate(t0, remaining, dt, last_value, rng):
        raw = rng.pareto(alpha) + 1.0  # +1 ensures duration isn't below min_pause
        duration = min(min_pause * raw, max_pause, remaining)

        times = np.arange(t0, t0 + duration, dt)
        values = np.full_like(times, last_value)
        return times, values, last_value

    return StateSpec(name=name, weight=weight, generator=generate)

def activity_state(name, slope_range, duration_range, weight, direction=1):
    def generate(t0, remaining, dt, last_value, rng):
        duration = min(sample_uni(duration_range[0], duration_range[1], rng), remaining)
        times = np.arange(t0, t0 + duration, dt)
        slope = sample_uni(slope_range[0], slope_range[1], rng) * direction
        deltas = slope * (times - t0)
        values = last_value + deltas
        new_last = values[-1] if len(values) else last_value
        return times, values, new_last  

    return StateSpec(name=name, weight=weight, generator=generate)

def jump_state(name, jump_range, weight, direction=-1):
    def generate(t0, remaining, dt, last_value, rng):
        magnitude = sample_uni(jump_range[0], jump_range[1], rng) * direction
        times = np.arange(t0, t0 + dt, dt)
        values = np.full_like(times, last_value + magnitude)
        return times, values, last_value + magnitude

    return StateSpec(name=name, weight=weight, generator=generate)

def high_noise_pause_state(name, rate_range, noise_std_range, weight):
    def generate(t0, remaining, dt, last_value, rng):
        duration = min(sample_exp(rate_range, rng), remaining)
        times = np.arange(t0, t0 + duration, dt)
        noise_std = sample_uni(noise_std_range[0], noise_std_range[1], rng)
        noise = rng.normal(0.0, noise_std, size=times.shape)
        values = last_value + noise
        return times, values, last_value

    return StateSpec(name=name, weight=weight, generator=generate)

def simulate_trace(total_time, dt, states, background_noise_std=None, seed=None):
    rng = np.random.default_rng(seed)
    t = 0.0
    last_value = 0.0
    time_segments = []
    value_segments = []

    while t < total_time:
        state = choose_state(states, rng)
        times, values, last_value = state.generator(t, total_time - t, dt, last_value, rng)
        if times.size == 0:
            break
        time_segments.append(times)
        value_segments.append(values)
        t = times[-1] + dt

    time_axis = np.concatenate(time_segments)
    clean = np.concatenate(value_segments)

    if background_noise_std is not None:
        noisy = clean + rng.normal(0.0, background_noise_std, size=clean.shape)
    else:
        noisy = clean.copy()

    return {"time": time_axis, "clean": clean, "with_noise": noisy}

def default_usable_states():
    return [
        pause_state("pause_fast", rate_range=(1.5, 4.0), weight=0.30),
        pause_state("pause_slow", rate_range=(0.5, 1.2), weight=0.22),
        power_law_pause_state("pause_power", alpha=2.0, min_pause=0.3, max_pause=3.5, weight=0.06),
        activity_state("activity_up_fast", slope_range=(40.0, 70.0), duration_range=(0.3, 2.0), weight=0.16, direction=1),
        activity_state("activity_up_slow", slope_range=(20.0, 40.0), duration_range=(1.0, 4.0), weight=0.13, direction=1),
        activity_state("activity_up_medium", slope_range=(30.0, 60.0), duration_range=(0.5, 3.0), weight=0.10, direction=1),
        jump_state("jump_back", jump_range=(50.0, 400.0), weight=0.015, direction=-1),
        jump_state("jump_forward", jump_range=(30.0, 200.0), weight=0.010, direction=1),
        high_noise_pause_state("high_noise", rate_range=(0.5, 1.5), noise_std_range=(15.0, 50.0), weight=0.010),
    ]

def select_states(states, include=None, exclude=None):
    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else set()
    filtered = []
    for s in states:
        if s.name in exclude_set:
            continue
        if include_set is not None and s.name not in include_set:
            continue
        filtered.append(s)
    return filtered

def intro_lost_bead(trace, loss_time=None, noise_range=None, seed=None):
    rng = np.random.default_rng(seed)
    time = trace["time"]
    clean = trace["clean"]
    noisy = trace["with_noise"].copy()
    
    if loss_time is None:
        loss_time = rng.uniform(time[0] + 0.2 * (time[-1] - time[0]), 
                                time[0] + 0.8 * (time[-1] - time[0]))
    
    loss_idx = np.searchsorted(time, loss_time)
    if loss_idx < len(time):
        trace_before_loss = noisy[:loss_idx]
        if len(trace_before_loss) > 0:
            min_val = np.nanmin(trace_before_loss)
            max_val = np.nanmax(trace_before_loss)
            trace_range = max_val - min_val
            
            if noise_range is None:
                margin = max(trace_range * 4.0, 2000.0) # default noise range
                noise_min = min_val - margin
                noise_max = max_val + margin
            else:
                noise_min, noise_max = noise_range
        else:
            noise_min, noise_max = (-2000.0, 2000.0) if noise_range is None else noise_range
        
        n_points_after = len(time) - loss_idx
        noisy[loss_idx:] = rng.uniform(noise_min, noise_max, size=n_points_after)
        clean[loss_idx:] = np.nan
    
    result = trace.copy()
    result["with_noise"] = noisy
    result["clean"] = clean
    return result

def intro_missing_regions(trace, n_regions=1, region_duration_range=(2.0, 10.0), seed=None):
    rng = np.random.default_rng(seed)
    time = trace["time"]
    clean = trace["clean"].copy()
    noisy = trace["with_noise"].copy()
    
    total_duration = time[-1] - time[0]
    dt = time[1] - time[0] if len(time) > 1 else 0.02
    
    for _ in range(n_regions):
        start_time = rng.uniform(time[0] + 0.1 * total_duration, 
                                 time[-1] - 0.1 * total_duration)
        duration = rng.uniform(region_duration_range[0], region_duration_range[1])
        end_time = min(start_time + duration, time[-1])
        
        start_idx = np.searchsorted(time, start_time)
        end_idx = np.searchsorted(time, end_time)
        
        clean[start_idx:end_idx] = np.nan
        noisy[start_idx:end_idx] = np.nan
    
    result = trace.copy()
    result["clean"] = clean
    result["with_noise"] = noisy
    return result

def intro_high_noise(trace, noise_multiplier=3.0, seed=None):
    rng = np.random.default_rng(seed)
    clean = trace["clean"]
    noisy = trace["with_noise"]
    
    current_noise = noisy - clean
    estimated_std = np.std(current_noise)
    additional_noise = rng.normal(0.0, estimated_std * (noise_multiplier - 1.0), size=clean.shape)
    new_noisy = clean + current_noise + additional_noise
    
    result = trace.copy()
    result["with_noise"] = new_noisy
    return result

def intro_inconsistent_noise(trace, n_transitions=2, noise_std_range=(5.0, 50.0), transition_duration_range=(0.5, 2.0), seed=None, decreasing_noise=False):
    rng = np.random.default_rng(seed)
    time = trace["time"]
    clean = trace["clean"]
    noisy = trace["with_noise"].copy()
    
    total_duration = time[-1] - time[0]
    dt = time[1] - time[0] if len(time) > 1 else 0.02
    
    current_noise = noisy - clean
    baseline_std = np.std(current_noise)
    
    if decreasing_noise:
        high_noise = max(noise_std_range[0], noise_std_range[1])
        low_noise = min(noise_std_range[0], noise_std_range[1])
        noise_profile = np.linspace(high_noise, low_noise, len(time))
        noise_range = abs(high_noise - low_noise)
        noise_profile += rng.normal(0.0, noise_range * 0.1, size=len(time))
        noise_profile = np.clip(noise_profile, low_noise, high_noise)
    else:
        noise_profile = np.full(len(time), baseline_std)
        
        for _ in range(n_transitions):
            transition_time = rng.uniform(time[0] + 0.1 * total_duration,
                                         time[-1] - 0.1 * total_duration)
            transition_duration = rng.uniform(transition_duration_range[0], 
                                             transition_duration_range[1])
            
            start_time = transition_time
            end_time = min(transition_time + transition_duration, time[-1])
            
            start_idx = np.searchsorted(time, start_time)
            end_idx = np.searchsorted(time, end_time)
            
            new_noise_std = rng.uniform(noise_std_range[0], noise_std_range[1])
            
            if start_idx < len(time):
                n_points = end_idx - start_idx
                if n_points > 1:
                    transition = np.linspace(noise_profile[start_idx], new_noise_std, n_points)
                    noise_profile[start_idx:end_idx] = transition
                else:
                    noise_profile[start_idx:end_idx] = new_noise_std
    
    new_noise = rng.normal(0.0, noise_profile, size=clean.shape)
    new_noisy = clean + new_noise
    
    result = trace.copy()
    result["with_noise"] = new_noisy
    return result

def intro_unobservable_activity(trace, noise_multiplier=5.0, drift_rate_range=(0.5, 2.0), seed=None):
    rng = np.random.default_rng(seed)
    time = trace["time"]
    clean = trace["clean"]
    noisy = trace["with_noise"]
    
    current_noise = noisy - clean
    estimated_std = np.std(current_noise)
    excessive_noise = rng.normal(0.0, estimated_std * noise_multiplier, size=clean.shape)
    
    drift_rate = rng.uniform(drift_rate_range[0], drift_rate_range[1])
    drift = drift_rate * (time - time[0])
    
    new_noisy = clean + drift + excessive_noise
    
    result = trace.copy()
    result["with_noise"] = new_noisy
    result["clean"] = clean + drift
    return result

def intro_compaction_noise(trace, noise_std_range=(20.0, 5.0), seed=None):
    return intro_inconsistent_noise(
        trace,
        n_transitions=0,
        noise_std_range=noise_std_range,
        transition_duration_range=(0.5, 2.0),
        seed=seed,
        decreasing_noise=True
    )

def gen_single_trace_with_params(args):
    idx, params = args
    seed = params.get("seed", idx)
    is_unusable = params.get("is_unusable", False)
    bad_feature_type = params.get("bad_feature_type", None)
    exclude_states = params.get("exclude_states", [])
    
    states = default_usable_states()
    if exclude_states:
        states = select_states(states, exclude=exclude_states)
    
    trace = simulate_trace(
        total_time=params["total_time"],
        dt=params["dt"],
        states=states,
        background_noise_std=params["background_noise_std"],
        seed=seed
    )
    
    if is_unusable and bad_feature_type is not None:
        rng = np.random.default_rng(seed)
        
        if bad_feature_type == "lost_bead":
            loss_time = rng.uniform(trace["time"][0] + 0.2 * (trace["time"][-1] - trace["time"][0]),
                                   trace["time"][0] + 0.8 * (trace["time"][-1] - trace["time"][0]))
            trace = intro_lost_bead(trace, loss_time=loss_time, seed=seed)
        
        elif bad_feature_type == "missing_regions":
            n_regions = rng.integers(1, 4)
            trace = intro_missing_regions(trace, n_regions=n_regions, 
                                            region_duration_range=(1.0, 8.0), seed=seed)
        
        elif bad_feature_type == "high_noise":
            multiplier = rng.uniform(2.5, 5.0)
            trace = intro_high_noise(trace, noise_multiplier=multiplier, seed=seed)
        
        elif bad_feature_type == "inconsistent_noise":
            n_transitions = rng.integers(2, 5)
            trace = intro_inconsistent_noise(trace, n_transitions=n_transitions,
                                                noise_std_range=(5.0, 50.0), seed=seed)
        
        elif bad_feature_type == "unobservable_activity":
            multiplier = rng.uniform(4.0, 7.0)
            drift_rate = rng.uniform(0.3, 2.5)
            trace = intro_unobservable_activity(trace, noise_multiplier=multiplier,
                                                   drift_rate_range=(drift_rate, drift_rate + 0.5), seed=seed)
        
        elif bad_feature_type == "multiple":
            features = ["lost_bead", "high_noise", "inconsistent_noise", "unobservable_activity"]
            n_features = rng.integers(1, 3)
            selected = rng.choice(features, size=n_features, replace=False)
            
            for feat in selected:
                if feat == "lost_bead":
                    loss_time = rng.uniform(trace["time"][0] + 0.2 * (trace["time"][-1] - trace["time"][0]),
                                           trace["time"][0] + 0.8 * (trace["time"][-1] - trace["time"][0]))
                    trace = intro_lost_bead(trace, loss_time=loss_time, seed=seed)
                elif feat == "high_noise":
                    trace = intro_high_noise(trace, noise_multiplier=rng.uniform(2.5, 4.0), seed=seed)
                elif feat == "inconsistent_noise":
                    trace = intro_inconsistent_noise(trace, n_transitions=rng.integers(2, 4),
                                                        noise_std_range=(5.0, 50.0), seed=seed)
                elif feat == "unobservable_activity":
                    trace = intro_unobservable_activity(trace, noise_multiplier=rng.uniform(4.0, 6.0),
                                                          drift_rate_range=(0.5, 2.0), seed=seed)
    
    return {
        "trace": trace["with_noise"],
        "time": trace["time"],
        "clean": trace.get("clean", trace["with_noise"]),
        "label": 0 if is_unusable else 1,
        "is_unusable": is_unusable,
        "bad_feature_type": bad_feature_type if is_unusable else None,
        "index": idx
    }

def clean_nan_trace(trace):
    time = trace["time"]
    noisy = trace["with_noise"].copy()
    clean = trace.get("clean", noisy.copy())
    
    if isinstance(clean, np.ndarray):
        clean = clean.copy()
    else:
        clean = noisy.copy()
    
    if np.isnan(noisy).any():
        valid_mask = ~np.isnan(noisy)
        if valid_mask.sum() > 1:
            valid_indices = np.where(valid_mask)[0]
            valid_values = noisy[valid_indices]
            valid_times = time[valid_indices]
            
            if len(valid_indices) > 1:
                interp_func = interpolate.interp1d(
                    valid_times, valid_values,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                noisy = interp_func(time)
            else:
                fill_value = valid_values[0] if len(valid_values) > 0 else 0.0
                noisy = np.where(valid_mask, noisy, fill_value)
        else:
            noisy = np.zeros_like(noisy)
    
    if isinstance(clean, np.ndarray) and np.isnan(clean).any():
        valid_mask = ~np.isnan(clean)
        if valid_mask.sum() > 1:
            valid_indices = np.where(valid_mask)[0]
            valid_values = clean[valid_indices]
            valid_times = time[valid_indices]
            
            if len(valid_indices) > 1:
                interp_func = interpolate.interp1d(
                    valid_times, valid_values,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                clean = interp_func(time)
            else:
                fill_value = valid_values[0] if len(valid_values) > 0 else 0.0
                clean = np.where(valid_mask, clean, fill_value)
        else:
            clean = noisy.copy()
    
    return {
        "time": time,
        "with_noise": noisy,
        "clean": clean
    }

def split_trace(trace, split_duration, overlap=0.0, seed=None):
    time = trace["time"]
    noisy = trace["with_noise"]
    clean = trace.get("clean", noisy)
    
    if len(time) == 0:
        return []
    
    total_duration = time[-1] - time[0]
    if total_duration < split_duration:
        cleaned = clean_nan_trace(trace)
        return [cleaned]
    
    step = split_duration - overlap
    n_splits = int(np.ceil((total_duration - split_duration) / step)) + 1
    
    splits = []
    for i in range(n_splits):
        start_time = time[0] + i * step
        end_time = start_time + split_duration
        
        if end_time > time[-1]:
            end_time = time[-1]
        
        start_idx = np.searchsorted(time, start_time)
        end_idx = np.searchsorted(time, end_time)
        
        if start_idx >= end_idx:
            continue
        
        split_time = time[start_idx:end_idx] - time[start_idx]
        split_noisy = noisy[start_idx:end_idx]
        if isinstance(clean, np.ndarray):
            split_clean = clean[start_idx:end_idx]
        else:
            split_clean = split_noisy.copy()
        
        split_trace_dict = {
            "time": split_time,
            "with_noise": split_noisy,
            "clean": split_clean
        }
        
        split_trace_dict = clean_nan_trace(split_trace_dict)
        
        if len(split_trace_dict["with_noise"]) > 0:
            if not np.all(np.isnan(split_trace_dict["with_noise"])):
                splits.append(split_trace_dict)
    
    return splits

def generate_training_dataset(n_traces=1000000, usable_ratio=0.5, total_time_range=(100.0, 300.0), dt_range=(0.01, 0.05), noise_std_range=(5.0, 20.0), n_workers=None, base_seed=42, chunk_size=10000, generation_method='split', split_trace_duration=None):
    n_workers = worker_count(n_workers)
    
    if generation_method == "split":
        if split_trace_duration is None:
            avg_big_duration = (total_time_range[0] + total_time_range[1]) / 2
            split_trace_duration = avg_big_duration / 100.0
        
        avg_big_duration = (total_time_range[0] + total_time_range[1]) / 2
        traces_per_big = int(avg_big_duration / split_trace_duration)
        n_big_traces = max(1, int(np.ceil(n_traces / traces_per_big)))
        
        rng = np.random.default_rng(base_seed)
        big_trace_params = []
        for i in range(n_big_traces):
            exclude_states = []
            if rng.random() < 0.3:
                exclude = rng.choice(["jump_back", "high_noise"], size=rng.integers(0, 2), replace=False)
                exclude_states = list(exclude)
            
            big_trace_params.append({
                "seed": base_seed + i,
                "is_unusable": False,
                "bad_feature_type": None,
                "total_time": rng.uniform(*total_time_range),
                "dt": rng.uniform(*dt_range),
                "exclude_states": exclude_states,
                "background_noise_std": rng.uniform(*noise_std_range)
            })
        
        all_big_traces = []
        n_chunks = (n_big_traces + chunk_size - 1) // chunk_size
        
        with mp.Pool(n_workers) as pool:
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_big_traces)
                chunk_params = [(i, big_trace_params[i]) for i in range(start_idx, end_idx)]
                
                chunk_results = pool.map(gen_single_trace_with_params, chunk_params)
                all_big_traces.extend(chunk_results)
        
        all_small_traces = []
        split_start_time = time.time()
        
        bad_feature_types = [
            "lost_bead", "missing_regions", "high_noise",
            "inconsistent_noise", "unobservable_activity", "multiple"
        ]
        bad_feature_probs = [0.2, 0.15, 0.2, 0.15, 0.2, 0.1]
        
        for big_idx, big_result in enumerate(all_big_traces):
            big_trace = {
                "time": big_result["time"],
                "with_noise": big_result["trace"],
                "clean": big_result.get("clean", big_result["trace"])
            }
            
            splits = split_trace(big_trace, split_trace_duration, overlap=0.0, seed=base_seed + big_idx)
            
            n_splits = len(splits)
            n_unusable_splits = int(n_splits * (1 - usable_ratio))
            
            # offset to not reuse seeds
            split_rng = np.random.default_rng(base_seed + big_idx + 10000)
            unusable_indices = set(split_rng.choice(n_splits, size=n_unusable_splits, replace=False))
            
            for split_idx, split_trace_dict in enumerate(splits):
                is_unusable = split_idx in unusable_indices
                bad_feature_type = None
                
                if is_unusable:
                    bad_feature_type = split_rng.choice(bad_feature_types, p=bad_feature_probs)
                    split_seed = base_seed + big_idx * 1000 + split_idx
                    
                    if bad_feature_type == "lost_bead":
                        loss_time = split_rng.uniform(
                            split_trace_dict["time"][0] + 0.2 * (split_trace_dict["time"][-1] - split_trace_dict["time"][0]),
                            split_trace_dict["time"][0] + 0.8 * (split_trace_dict["time"][-1] - split_trace_dict["time"][0])
                        )
                        split_trace_dict = intro_lost_bead(split_trace_dict, loss_time=loss_time, seed=split_seed)
                    elif bad_feature_type == "missing_regions":
                        n_regions = split_rng.integers(1, 4)
                        split_trace_dict = intro_missing_regions(
                            split_trace_dict, n_regions=n_regions,
                            region_duration_range=(0.1, 0.5), seed=split_seed
                        )
                    elif bad_feature_type == "high_noise":
                        multiplier = split_rng.uniform(2.5, 5.0)
                        split_trace_dict = intro_high_noise(split_trace_dict, noise_multiplier=multiplier, seed=split_seed)
                    elif bad_feature_type == "inconsistent_noise":
                        n_transitions = split_rng.integers(2, 5)
                        split_trace_dict = intro_inconsistent_noise(
                            split_trace_dict, n_transitions=n_transitions,
                            noise_std_range=(5.0, 50.0), seed=split_seed
                        )
                    elif bad_feature_type == "unobservable_activity":
                        multiplier = split_rng.uniform(4.0, 7.0)
                        drift_rate = split_rng.uniform(0.3, 2.5)
                        split_trace_dict = intro_unobservable_activity(
                            split_trace_dict, noise_multiplier=multiplier,
                            drift_rate_range=(drift_rate, drift_rate + 0.5), seed=split_seed
                        )
                    elif bad_feature_type == "multiple":
                        features = ["lost_bead", "high_noise", "inconsistent_noise", "unobservable_activity"]
                        n_features = split_rng.integers(1, 3)
                        selected = split_rng.choice(features, size=n_features, replace=False)
                        for feat in selected:
                            if feat == "lost_bead":
                                loss_time = split_rng.uniform(
                                    split_trace_dict["time"][0] + 0.2 * (split_trace_dict["time"][-1] - split_trace_dict["time"][0]),
                                    split_trace_dict["time"][0] + 0.8 * (split_trace_dict["time"][-1] - split_trace_dict["time"][0])
                                )
                                split_trace_dict = intro_lost_bead(split_trace_dict, loss_time=loss_time, seed=split_seed)
                            elif feat == "high_noise":
                                split_trace_dict = intro_high_noise(split_trace_dict, noise_multiplier=split_rng.uniform(2.5, 4.0), seed=split_seed)
                            elif feat == "inconsistent_noise":
                                split_trace_dict = intro_inconsistent_noise(
                                    split_trace_dict, n_transitions=split_rng.integers(2, 4),
                                    noise_std_range=(5.0, 50.0), seed=split_seed
                                )
                            elif feat == "unobservable_activity":
                                split_trace_dict = intro_unobservable_activity(
                                    split_trace_dict, noise_multiplier=split_rng.uniform(4.0, 6.0),
                                    drift_rate_range=(0.5, 2.0), seed=split_seed
                                )
                
                all_small_traces.append({
                    "trace": split_trace_dict["with_noise"],
                    "time": split_trace_dict["time"],
                    "clean": split_trace_dict.get("clean", split_trace_dict["with_noise"]),
                    "label": 0 if is_unusable else 1,
                    "is_unusable": is_unusable,
                    "bad_feature_type": bad_feature_type if is_unusable else None,
                    "index": len(all_small_traces)
                })
        
        if len(all_small_traces) > n_traces:
            rng = np.random.default_rng(base_seed + 99999)
            indices = rng.choice(len(all_small_traces), size=n_traces, replace=False)
            all_small_traces = [all_small_traces[i] for i in indices]
        
        traces = [r["trace"] for r in all_small_traces]
        times = [r["time"] for r in all_small_traces]
        labels = np.array([r["label"] for r in all_small_traces], dtype=np.int32)
        metadata = [{k: v for k, v in r.items() if k not in ["trace", "time", "label"]} 
                    for r in all_small_traces]
        
        n_usable_actual = int(np.sum(labels == 1))
        n_unusable_actual = int(np.sum(labels == 0))
        
        return {
            "traces": traces,
            "times": times,
            "labels": labels,
            "metadata": metadata,
            "n_traces": len(traces),
            "n_usable": n_usable_actual,
            "n_unusable": n_unusable_actual
        }
    
    else:
        n_usable = int(n_traces * usable_ratio)
        n_unusable = n_traces - n_usable
        
        bad_feature_types = [
            "lost_bead",
            "missing_regions", 
            "high_noise",
            "inconsistent_noise",
            "unobservable_activity",
            "multiple"
        ]
        bad_feature_probs = [0.2, 0.15, 0.2, 0.15, 0.2, 0.1]
        
        rng = np.random.default_rng(base_seed)
        all_params = []
        
        for i in range(n_usable):
            exclude_states = []
            if rng.random() < 0.3:
                exclude = rng.choice(["jump_back", "high_noise"], size=rng.integers(0, 2), replace=False)
                exclude_states = list(exclude)
            
            all_params.append({
                "seed": base_seed + i,
                "is_unusable": False,
                "bad_feature_type": None,
                "total_time": rng.uniform(*total_time_range),
                "dt": rng.uniform(*dt_range),
                "exclude_states": exclude_states,
                "background_noise_std": rng.uniform(*noise_std_range)
            })
        
        for i in range(n_unusable):
            exclude_states = []
            if rng.random() < 0.3:
                exclude = rng.choice(["jump_back", "high_noise"], size=rng.integers(0, 2), replace=False)
                exclude_states = list(exclude)
            
            bad_feature_type = rng.choice(bad_feature_types, p=bad_feature_probs)
            
            all_params.append({
                "seed": base_seed + n_usable + i,
                "is_unusable": True,
                "bad_feature_type": bad_feature_type,
                "total_time": rng.uniform(*total_time_range),
                "dt": rng.uniform(*dt_range),
                "exclude_states": exclude_states,
                "background_noise_std": rng.uniform(*noise_std_range)
            })
        
        rng.shuffle(all_params)
        
        all_results = []
        n_chunks = (n_traces + chunk_size - 1) // chunk_size
        
        with mp.Pool(n_workers) as pool:
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_traces)
                chunk_params = [(i, all_params[i]) for i in range(start_idx, end_idx)]
                
                chunk_results = pool.map(gen_single_trace_with_params, chunk_params)
                all_results.extend(chunk_results)
        
        traces = [r["trace"] for r in all_results]
        times = [r["time"] for r in all_results]
        labels = np.array([r["label"] for r in all_results], dtype=np.int32)
        metadata = [{k: v for k, v in r.items() if k not in ["trace", "time", "label"]} 
                    for r in all_results]
        
        n_usable_actual = int(np.sum(labels == 1))
        n_unusable_actual = int(np.sum(labels == 0))
        
        return {
            "traces": traces,
            "times": times,
            "labels": labels,
            "metadata": metadata,
            "n_traces": len(traces),
            "n_usable": n_usable_actual,
            "n_unusable": n_unusable_actual
        }

def save_dataset(dataset, filepath, format='npz'):
    import time
    
    if format == "npz":
        start_time = time.time()
        np.savez_compressed(
            filepath,
            labels=dataset["labels"],
            metadata=dataset["metadata"],
            n_traces=dataset["n_traces"]
        )
        print(f"warning: variable-length traces not saved in NPZ format")
        print(f"  Use format='npz_padded' for fixed-size arrays, or save traces separately.")
    
    elif format == "npz_padded":
        n_traces = len(dataset["traces"])
        start_time = time.time()
        max_len = max(len(t) for t in dataset["traces"])
        median_len = int(np.median([len(t) for t in dataset["traces"]]))
        
        padded_traces_median = []
        padded_times_median = []
        padded_traces_max = []
        padded_times_max = []
        
        n_nan_traces = 0
        n_nan_points = 0
        
        for i, (trace, time_arr) in enumerate(zip(dataset["traces"], dataset["times"])):
            has_nan = np.isnan(trace).any()
            if has_nan:
                n_nan_traces += 1
                n_nan_points += np.isnan(trace).sum()
                valid_mask = ~np.isnan(trace)
                if valid_mask.sum() > 1:
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = trace[valid_indices]
                    valid_times = time_arr[valid_indices]
                    
                    if len(valid_indices) > 1:
                        interp_func = interpolate.interp1d(
                            valid_times, valid_values, 
                            kind='linear', 
                            bounds_error=False, 
                            fill_value='extrapolate'
                        )
                        trace_interp = interp_func(time_arr)
                        trace = trace_interp
                    else:
                        trace = np.where(valid_mask, trace, np.nan)
                        trace_series = pd.Series(trace)
                        trace_series = trace_series.ffill().bfill()
                        trace = trace_series.values
                else:
                    trace_series = pd.Series(trace)
                    trace_series = trace_series.ffill().bfill()
                    trace = trace_series.values
            
            if len(trace) > median_len:
                trace_median = trace[:median_len]
                time_median = time_arr[:median_len]
            else:
                trace_median = np.pad(trace, (0, median_len - len(trace)), mode='edge')
                time_median = np.pad(time_arr, (0, median_len - len(time_arr)), mode='edge')
            
            if np.isnan(trace_median).any():
                trace_series = pd.Series(trace_median)
                trace_series = trace_series.ffill().bfill()
                trace_median = trace_series.values
            
            padded_traces_median.append(trace_median)
            padded_times_median.append(time_median)
            
            trace_max = np.pad(trace, (0, max_len - len(trace)), mode='constant', constant_values=np.nan)
            time_max = np.pad(time_arr, (0, max_len - len(time_arr)), mode='constant', constant_values=np.nan)
            padded_traces_max.append(trace_max)
            padded_times_max.append(time_max)
        
        padded_traces_median = np.array(padded_traces_median)
        padded_times_median = np.array(padded_times_median)
        padded_traces_max = np.array(padded_traces_max)
        padded_times_max = np.array(padded_times_max)
        
        if np.isnan(padded_traces_median).any():
            for i in range(len(padded_traces_median)):
                if np.isnan(padded_traces_median[i]).any():
                    pd_series = pd.Series(padded_traces_median[i])
                    pd_series = pd_series.ffill().bfill()
                    padded_traces_median[i] = pd_series.values
        
        # add channel dim for sktime
        traces_sktime = padded_traces_median[:, np.newaxis, :]
        
        if np.isnan(traces_sktime).any():
            for i in range(len(traces_sktime)):
                trace_2d = traces_sktime[i, 0, :]
                if np.isnan(trace_2d).any():
                    pd_series = pd.Series(trace_2d)
                    pd_series = pd_series.ffill().bfill()
                    if pd_series.isna().any():
                        pd_series = pd_series.fillna(0.0)
                    traces_sktime[i, 0, :] = pd_series.values
            
            if np.isnan(traces_sktime).any():
                traces_sktime = np.nan_to_num(traces_sktime, nan=0.0)
        
        np.savez_compressed(
            filepath,
            traces=padded_traces_max,
            times=padded_times_max,
            traces_sktime=traces_sktime,
            traces_sktime_2d=padded_traces_median,
            times_sktime=padded_times_median,
            labels=dataset["labels"],
            n_traces=dataset["n_traces"],
            median_length=median_len,
            max_length=max_len,
            format_version="2.0"
        )
        print(f"saved {filepath} (sktime shape {traces_sktime.shape}, median len {median_len})")
    
    else:
        raise ValueError(f"unknown format: {format}")

def load_dataset(filepath, format='npz_padded'):
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"dataset file not found: {filepath}")
    
    if format == "npz_padded":
        data = np.load(filepath, allow_pickle=True)
        
        format_version = data.get('format_version', '1.0')
        has_sktime_format = 'traces_sktime' in data or 'traces_sktime_2d' in data
        
        labels = data['labels']
        n_traces = int(data['n_traces'])
        
        metadata = None
        if 'metadata' in data:
            metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
        
        traces_padded = data['traces']
        times_padded = data['times']
        
        traces = []
        times = []
        for i in range(n_traces):
            trace = traces_padded[i]
            time = times_padded[i]
            
            trace_valid = ~np.isnan(trace)
            time_valid = ~np.isnan(time)
            
            valid_len = min(np.sum(trace_valid), np.sum(time_valid))
            
            traces.append(trace[:valid_len])
            times.append(time[:valid_len])
        
        dataset = {
            "traces": traces,
            "times": times,
            "labels": labels,
            "metadata": metadata if metadata is not None else [{}] * n_traces,
            "n_traces": n_traces,
            "n_usable": int(np.sum(labels == 1)),
            "n_unusable": int(np.sum(labels == 0))
        }
        
        if has_sktime_format:
            if 'traces_sktime' in data:
                dataset["traces_sktime"] = data['traces_sktime']
            elif 'traces_sktime_2d' in data:
                dataset["traces_sktime"] = data['traces_sktime_2d'][:, np.newaxis, :]
            
            if 'times_sktime' in data:
                dataset["times_sktime"] = data['times_sktime']
            
            if 'median_length' in data:
                dataset["median_length"] = int(data['median_length'])
            if 'max_length' in data:
                dataset["max_length"] = int(data['max_length'])
        
        return dataset
    
    else:
        raise ValueError(f"loading format '{format}' not implemented use 'npz_padded'")
