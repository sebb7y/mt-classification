import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import signal
import multiprocessing as mp
from functools import partial
import time
import pandas as pd
import os
import glob
import json
import pickle
import sys
import argparse
from trace_gen import generate_training_dataset, save_dataset, load_dataset

from slurm_helpers import worker_count as get_n_workers, PathManager, get_slurm_job_id

from sktime.transformations.panel.rocket import MiniRocket, Rocket
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import BOSSEnsemble, WEASEL
from sktime.classification.ensemble import BaggingClassifier, WeightedEnsembleClassifier

SKTIME_CLASSIFIERS = {
    'tsf': TimeSeriesForestClassifier,
    'boss': BOSSEnsemble,
    'weasel': WEASEL,
    'bagging': BaggingClassifier,
    'weighted_ensemble': WeightedEnsembleClassifier
}
SKTIME_TRANSFORMERS = {
    'minirocket': MiniRocket,
    'rocket': Rocket
}

from sktime.classification.hybrid import HIVECOTEV2, HIVECOTEV1
SKTIME_CLASSIFIERS['hivecote'] = HIVECOTEV2
SKTIME_CLASSIFIERS['hivecotev2'] = HIVECOTEV2
SKTIME_CLASSIFIERS['hivecotev1'] = HIVECOTEV1

def generate_random_kernels(n_kernels=10000, kernel_length_range=(7, 9), seed=None):
    rng = np.random.default_rng(seed)
    kernels = []

    for _ in range(n_kernels):
        length = rng.integers(kernel_length_range[0], kernel_length_range[1] + 1)
        weights = rng.normal(0, 1, size=length)
        weights = weights / np.linalg.norm(weights)
        bias = rng.uniform(-1, 1)
        dilation = rng.integers(1, 3)
        
        if dilation > 1:
            dilated_weights = np.zeros(length * dilation)
            # why dilate?

            dilated_weights[::dilation] = weights
            weights = dilated_weights

        kernels.append((weights, bias))

    return kernels

def convolve_kernel(trace, kernel):
    weights, bias = kernel
    
    if len(trace) < len(weights):
        return np.array([np.dot(trace, weights[:len(trace)]) + bias])
    
    result = signal.convolve(trace, weights, mode='same')
    return result + bias

def extract_features(trace, kernels):
    features = []
    
    for kernel in kernels:
        convolved = convolve_kernel(trace, kernel)
        max_val = np.max(convolved)
        ppv = np.mean(convolved > 0)
        features.extend([max_val, ppv])
    
    return np.array(features)

def extract_features_single(args):
    idx, trace, kernels = args
    features = []
    for kernel in kernels:
        convolved = convolve_kernel(trace, kernel)
        features.extend([np.max(convolved), np.mean(convolved > 0)])
    return (idx, np.array(features))

def transform_dataset(traces, kernels, n_workers=None):
    n_traces = len(traces)
    n_kernels = len(kernels)
    
    n_workers = get_n_workers(n_workers)

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            args = [(i, trace, kernels) for i, trace in enumerate(traces)]
            
            features_list = [None] * n_traces
            completed = 0
            for idx, result in pool.imap(extract_features_single, args, chunksize=10):
                features_list[idx] = result
                completed += 1
        
        features = np.array(features_list)
    else:
        features = np.zeros((n_traces, n_kernels * 2))
        for i, trace in enumerate(traces):
            for j, kernel in enumerate(kernels):
                convolved = convolve_kernel(trace, kernel)
                features[i, j*2] = np.max(convolved)
                features[i, j*2 + 1] = np.mean(convolved > 0)
    
    return features

def convert_to_sktime_format(traces, use_numpy3d=True):
    n_traces = len(traces)
    
    if isinstance(traces, np.ndarray) and traces.ndim == 2:
        if np.isnan(traces).any():
            for i in range(len(traces)):
                if np.isnan(traces[i]).any():
                    trace_series = pd.Series(traces[i])
                    trace_series = trace_series.ffill().bfill()
                    traces[i] = trace_series.values
        
        if use_numpy3d:
            traces_3d = traces[:, np.newaxis, :]
            return traces_3d
        else:
            return convert_to_multiindex(traces)
    
    lengths = [len(t) for t in traces]
    if len(set(lengths)) == 1:
        traces_2d = np.array(traces)
        if use_numpy3d:
            traces_3d = traces_2d[:, np.newaxis, :]
            return traces_3d
        else:
            return convert_to_multiindex(traces_2d)
    
    return convert_to_multiindex(traces)

def convert_to_multiindex(traces):
    n_traces = len(traces)
    
    if isinstance(traces, np.ndarray) and traces.ndim == 2:
        n_timepoints = traces.shape[1]
        total_points = n_traces * n_timepoints
        
        instance_indices = np.repeat(np.arange(n_traces), n_timepoints)
        time_indices = np.tile(np.arange(n_timepoints), n_traces)
        values = traces.flatten()
        
        index = pd.MultiIndex.from_arrays([instance_indices, time_indices], names=['instance', 'time'])
        return pd.DataFrame({'value': values}, index=index)
    
    else:
        inst_idx = []
        time_idx = []
        vals = []
        for i, trace in enumerate(traces):
            n = len(trace)
            inst_idx.append(np.full(n, i, dtype=np.int32))
            time_idx.append(np.arange(n, dtype=np.int32))
            vals.append(np.asarray(trace, dtype=np.float64))
        
        index = pd.MultiIndex.from_arrays(
            [np.concatenate(inst_idx), np.concatenate(time_idx)],
            names=['instance', 'time']
        )
        return pd.DataFrame({'value': np.concatenate(vals)}, index=index)

def transform_dataset_sktime(traces, transformer, fit=True, use_numpy3d=True):
    traces_sktime = convert_to_sktime_format(traces, use_numpy3d=use_numpy3d)
    
    if fit:
        transformer.fit(traces_sktime)
    
    start_time = time.time()
    features = transformer.transform(traces_sktime)
    
    if hasattr(features, 'values'):
        features = features.values
    elif not isinstance(features, np.ndarray):
        features = np.array(features)
    
    return features

def create_model(model_type, model_params=None, seed=42):
    if model_params is None:
        model_params = {}
    
    if 'random_state' not in model_params:
        model_params['random_state'] = seed
    
    model_type_lower = model_type.lower()
    
    if model_type_lower == 'tsf':
        return SKTIME_CLASSIFIERS['tsf'](**model_params)
    
    elif model_type_lower == 'boss':
        if 'feature_selection' not in model_params:
            model_params['feature_selection'] = 'chi2'
        if 'use_boss_distance' not in model_params:
            model_params['use_boss_distance'] = False
        if 'n_jobs' not in model_params:
            model_params['n_jobs'] = get_n_workers()
        return SKTIME_CLASSIFIERS['boss'](**model_params)
    
    elif model_type_lower == 'weasel':
        if 'feature_selection' not in model_params:
            model_params['feature_selection'] = 'chi2'
        if 'n_jobs' not in model_params:
            n_cpus = get_n_workers()
            model_params['n_jobs'] = min(8, n_cpus)
        return SKTIME_CLASSIFIERS['weasel'](**model_params)
    
    elif model_type_lower == 'minirocket':
        if 'num_kernels' not in model_params:
            model_params['num_kernels'] = 10000
        return SKTIME_TRANSFORMERS['minirocket'](**model_params)
    
    elif model_type_lower == 'rocket':
        if 'num_kernels' not in model_params:
            model_params['num_kernels'] = 10000
        return SKTIME_TRANSFORMERS['rocket'](**model_params)
    
    elif model_type_lower == 'bagging':
        if 'estimator' not in model_params and 'base_estimator' not in model_params:
            model_params['estimator'] = SKTIME_CLASSIFIERS['tsf'](random_state=seed)
        elif 'base_estimator' in model_params:
            model_params['estimator'] = model_params.pop('base_estimator')
        if 'n_estimators' not in model_params:
            model_params['n_estimators'] = 5
        if 'n_jobs' in model_params:
            model_params.pop('n_jobs')
        return SKTIME_CLASSIFIERS['bagging'](**model_params)
    
    elif model_type_lower == 'weighted_ensemble':
        if 'classifiers' not in model_params and 'base_estimators' not in model_params and 'estimators' not in model_params:
            model_params['classifiers'] = [
                SKTIME_CLASSIFIERS['tsf'](random_state=seed),
                SKTIME_CLASSIFIERS['boss'](random_state=seed),
            ]
        elif 'base_estimators' in model_params:
            model_params['classifiers'] = model_params.pop('base_estimators')
        elif 'estimators' in model_params:
            model_params['classifiers'] = model_params.pop('estimators')
        return SKTIME_CLASSIFIERS['weighted_ensemble'](**model_params)
    
    elif model_type_lower in ['hivecote', 'hivecotev2']:
        if 'hivecote' not in SKTIME_CLASSIFIERS and 'hivecotev2' not in SKTIME_CLASSIFIERS:
            raise ImportError("hivecotev2 not available")
        return SKTIME_CLASSIFIERS.get('hivecotev2', SKTIME_CLASSIFIERS.get('hivecote'))(**model_params)
    
    elif model_type_lower == 'hivecotev1':
        if 'hivecotev1' not in SKTIME_CLASSIFIERS:
            raise ImportError("hivecotev1 not available")
        return SKTIME_CLASSIFIERS['hivecotev1'](**model_params)
    
    elif model_type_lower == 'custom':
        return None
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(SKTIME_CLASSIFIERS.keys()) + list(SKTIME_TRANSFORMERS.keys()) + ['custom']}"
        )

def train_model(model_type='minirocket', n_traces=100000, n_kernels=10000, test_ratio=0.2, use_existing_dataset=None, use_existing_train_dataset=None, use_existing_test_dataset=None, save_dataset_path=None, auto_cache=True, seed=42, model_params=None, track_runtime=True, force_regenerate=False, allow_larger_dataset=False, path_manager=None):
    model_type_lower = model_type.lower()
    
    runtime_timings = {}
    if track_runtime:
        overall_start = time.time()
        runtime_timings["overall_start"] = overall_start

    training_stats = None

    if path_manager is None:
        path_manager = PathManager()
        path_manager.ensure_dirs()
    
    pre_split_loaded = False
    if use_existing_train_dataset and use_existing_test_dataset and os.path.exists(use_existing_train_dataset) and os.path.exists(use_existing_test_dataset):
        dataset_train = load_dataset(use_existing_train_dataset, format="npz_padded")
        dataset_test = load_dataset(use_existing_test_dataset, format="npz_padded")
        X_train_sktime = np.asarray(dataset_train["traces_sktime"])
        y_train = np.asarray(dataset_train["labels"], dtype=np.intp)
        X_test_sktime = np.asarray(dataset_test["traces_sktime"])
        y_test = np.asarray(dataset_test["labels"], dtype=np.intp)
        if X_train_sktime.shape[2] != X_test_sktime.shape[2]:
            raise ValueError(
                f"Train and test trace length must match: {X_train_sktime.shape[2]} vs {X_test_sktime.shape[2]}"
            )
        target_length = X_train_sktime.shape[2]
        _mean, _std = np.nanmean(X_train_sktime), np.nanstd(X_train_sktime)
        training_stats = {"mean": float(_mean), "std": float(_std) if _std > 0 else 1.0}
        for name, X in [("train", X_train_sktime), ("test", X_test_sktime)]:
            if np.isnan(X).any():
                X_2d = X[:, 0, :].copy()
                for i in range(len(X_2d)):
                    if np.isnan(X_2d[i]).any():
                        s = pd.Series(X_2d[i]).ffill().bfill()
                        X_2d[i] = s.values
                X[:, 0, :] = X_2d
        pre_split_loaded = True
    
    dataset = None
    dataset_path = use_existing_dataset
    
    if dataset_path is None:
        cache_dir = path_manager.dataset_cache_dir(use_scratch=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        exact_path = cache_dir / f"dataset_n{n_traces}_seed{seed}.npz"
        
        if force_regenerate:
            dataset_path = exact_path
        elif exact_path.exists():
            dataset_path = str(exact_path)
        elif auto_cache:
            dataset_found = path_manager.find_dataset(f"dataset_n{n_traces}_seed{seed}.npz", prefer_scratch=True)
            if dataset_found:
                dataset_path = str(dataset_found)
            else:
                try:
                    scratch_cache = path_manager.dataset_cache_dir(use_scratch=True)
                    home_cache = path_manager.dataset_cache_dir(use_scratch=False)
                    
                    pattern = f"dataset_n*_seed{seed}.npz"
                    matching_files = list(scratch_cache.glob(pattern)) + list(home_cache.glob(pattern))
                    matching_files = [str(f) for f in matching_files]
                    
                    if matching_files:
                        compatible_files = []
                        for f in matching_files:
                            try:
                                n_traces_in_file = int(f.split('_n')[1].split('_seed')[0])
                                if n_traces_in_file >= n_traces:
                                    compatible_files.append((f, n_traces_in_file))
                            except (ValueError, IndexError):
                                continue
                        
                        if compatible_files:
                            exact_matches = [f for f, n in compatible_files if n == n_traces]
                            if exact_matches:
                                dataset_path = exact_matches[0]
                            elif allow_larger_dataset:
                                compatible_files.sort(key=lambda x: x[1])
                                dataset_path = compatible_files[0][0]
                            else:
                                dataset_path = str(exact_path)
                        else:
                            dataset_path = str(exact_path)
                    else:
                        dataset_path = str(exact_path)
                except Exception as e:
                    dataset_path = str(exact_path)
        else:
            dataset_path = None
    
    if not pre_split_loaded and dataset_path and os.path.exists(dataset_path):
        try:
            dataset = load_dataset(dataset_path, format="npz_padded")
            if dataset['n_traces'] > n_traces and not allow_larger_dataset:
                pass
        except Exception as e:
            dataset = None
    
    if not pre_split_loaded and dataset is None:
        dataset = generate_training_dataset(
            n_traces=n_traces,
            usable_ratio=0.5,
            total_time_range=(100.0, 300.0),
            dt_range=(0.01, 0.05),
            noise_std_range=(5.0, 20.0),
            base_seed=seed
        )
        
        if save_dataset_path:
            save_path = save_dataset_path
        elif auto_cache:
            if dataset_path is None:
                cache_dir = path_manager.dataset_cache_dir(use_scratch=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
                dataset_path = str(cache_dir / f"dataset_n{n_traces}_seed{seed}.npz")
            save_path = dataset_path
        else:
            save_path = None
        
        if save_path:
            save_dataset(dataset, save_path, format="npz_padded")
    
    if not pre_split_loaded:
        labels = dataset["labels"]
    
    has_sktime_format = pre_split_loaded or (dataset is not None and "traces_sktime" in dataset)
    
    if has_sktime_format:
        if not pre_split_loaded:
            traces_sktime = dataset["traces_sktime"]
            target_length = dataset.get("median_length", traces_sktime.shape[2]) # time dim
            _mean, _std = np.nanmean(traces_sktime), np.nanstd(traces_sktime)
            training_stats = {"mean": float(_mean), "std": float(_std) if _std > 0 else 1.0}

            if np.isnan(traces_sktime).any():
                traces_sktime_2d = traces_sktime[:, 0, :]
                for i in range(len(traces_sktime_2d)):
                    if np.isnan(traces_sktime_2d[i]).any():
                        trace_series = pd.Series(traces_sktime_2d[i])
                        trace_series = trace_series.ffill().bfill()
                        traces_sktime_2d[i] = trace_series.values
                traces_sktime = traces_sktime_2d[:, np.newaxis, :]
        
        if not pre_split_loaded:
            if use_existing_train_dataset and use_existing_test_dataset:
                dataset_train = load_dataset(use_existing_train_dataset, format="npz_padded")
                dataset_test = load_dataset(use_existing_test_dataset, format="npz_padded")
                X_train_sktime = np.asarray(dataset_train["traces_sktime"])
                y_train = np.asarray(dataset_train["labels"], dtype=np.intp)
                X_test_sktime = np.asarray(dataset_test["traces_sktime"])
                y_test = np.asarray(dataset_test["labels"], dtype=np.intp)
                if X_train_sktime.shape[2] != X_test_sktime.shape[2]:
                    raise ValueError(
                        f"Train and test trace length must match: {X_train_sktime.shape[2]} vs {X_test_sktime.shape[2]}"
                    )
                target_length = X_train_sktime.shape[2]
                _mean, _std = np.nanmean(X_train_sktime), np.nanstd(X_train_sktime)
                training_stats = {"mean": float(_mean), "std": float(_std) if _std > 0 else 1.0}
                for name, X in [("train", X_train_sktime), ("test", X_test_sktime)]:
                    if np.isnan(X).any():
                        X_2d = X[:, 0, :].copy()
                        for i in range(len(X_2d)):
                            if np.isnan(X_2d[i]).any():
                                s = pd.Series(X_2d[i]).ffill().bfill()
                                X_2d[i] = s.values
                        X[:, 0, :] = X_2d
            else:
                n_test = int(len(traces_sktime) * test_ratio)
                indices = np.arange(len(traces_sktime))
                np.random.default_rng(seed).shuffle(indices)
                test_indices = indices[:n_test]
                train_indices = indices[n_test:]
                X_train_sktime = traces_sktime[train_indices]
                X_test_sktime = traces_sktime[test_indices]
                y_train = labels[train_indices]
                y_test = labels[test_indices]
        
    else:
        traces = dataset["traces"]
        
        lengths = [len(t) for t in traces]
        target_length = int(np.median(lengths))
        
        processed_traces = []
        for trace in traces:
            if len(trace) > target_length:
                processed_traces.append(trace[:target_length])
            else:
                padded = np.pad(trace, (0, target_length - len(trace)), mode='edge')
                processed_traces.append(padded)
        
        traces = np.array(processed_traces)
        _mean, _std = np.nanmean(traces), np.nanstd(traces)
        training_stats = {"mean": float(_mean), "std": float(_std) if _std > 0 else 1.0}
        
        n_test = int(len(traces) * test_ratio)
        indices = np.arange(len(traces))
        np.random.default_rng(seed).shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = traces[train_indices]
        y_train = labels[train_indices]
        X_test = traces[test_indices]
        y_test = labels[test_indices]

    if model_type_lower in ['tsf', 'boss', 'weasel', 'bagging', 'weighted_ensemble',
                            'hivecote', 'hivecotev2', 'hivecotev1']:
        if not has_sktime_format:
            X_train_sktime = convert_to_sktime_format(X_train)
            X_test_sktime = convert_to_sktime_format(X_test)

        if model_params is None:
            model_params = {}

        model = create_model(model_type, model_params=model_params, seed=seed)
        train_start = time.time()
        model.fit(X_train_sktime, y_train)
        train_time = time.time() - train_start
        y_train_pred = model.predict(X_train_sktime)
        y_test_pred = model.predict(X_test_sktime)

        transformer = None
        scaler = None

    elif model_type_lower in ['minirocket', 'rocket']:
        if has_sktime_format:
            X_train = X_train_sktime[:, 0, :]
            X_test = X_test_sktime[:, 0, :]

        if model_params is None:
            model_params = {}
        model_params['num_kernels'] = n_kernels

        transformer = create_model(model_type, model_params=model_params, seed=seed)
        start_time = time.time()
        X_train_features = transform_dataset_sktime(X_train, transformer, fit=True)
        train_transform_time = time.time() - start_time
        start_time = time.time()
        X_test_features = transform_dataset_sktime(X_test, transformer, fit=False)
        test_transform_time = time.time() - start_time
        scaler = StandardScaler()
        X_train_features_scaled = scaler.fit_transform(X_train_features)
        X_test_features_scaled = scaler.transform(X_test_features)
        train_start = time.time()
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        model.fit(X_train_features_scaled, y_train)
        train_time = time.time() - train_start
        y_train_pred = model.predict(X_train_features_scaled)
        y_test_pred = model.predict(X_test_features_scaled)

    elif model_type_lower == 'custom':
        if has_sktime_format:
            X_train = X_train_sktime[:, 0, :]
            X_test = X_test_sktime[:, 0, :]

        kernels = generate_random_kernels(n_kernels=n_kernels, seed=seed)
        start_time = time.time()
        X_train_features = transform_dataset(X_train, kernels, n_workers=None)
        train_transform_time = time.time() - start_time
        start_time = time.time()
        X_test_features = transform_dataset(X_test, kernels, n_workers=None)
        test_transform_time = time.time() - start_time
        scaler = StandardScaler()
        X_train_features_scaled = scaler.fit_transform(X_train_features)
        X_test_features_scaled = scaler.transform(X_test_features)
        train_start = time.time()
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        model.fit(X_train_features_scaled, y_train)
        train_time = time.time() - train_start
        y_train_pred = model.predict(X_train_features_scaled)
        y_test_pred = model.predict(X_test_features_scaled)

        transformer = kernels

    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_predictions": y_train_pred,
        "test_predictions": y_test_pred,
        "y_train": y_train,
        "y_test": y_test,
        "classification_report": classification_report(y_test, y_test_pred),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred)
    }
    # if needed: print(f"  train accuracy: {train_acc:.4f}, test accuracy: {test_acc:.4f}")
    
    if track_runtime:
        overall_end = time.time()
        runtime_timings["overall_end"] = overall_end
        runtime_timings["overall_runtime"] = overall_end - overall_start
    
    return {
        "model": model,
        "scaler": scaler,
        "transformer": transformer,
        "results": results,
        "target_length": target_length,
        "model_type": model_type,
        "runtime_timings": runtime_timings if track_runtime else None,
        "training_stats": training_stats
    }


def save_training_outputs(train_result, config, runtime_info, output_dir=None, save_model=True, command_line=None, path_manager=None):
    if path_manager is None:
        path_manager = PathManager()
        path_manager.ensure_dirs()
    
    if output_dir is None:
        output_dir = str(path_manager.outdir(use_scratch=True))
    
    model_type = train_result.get("model_type", "unknown")
    run_dir = os.path.join(output_dir, model_type)
    os.makedirs(run_dir, exist_ok=True)
    
    config_to_save = config.copy()
    config_to_save["command_executed"] = command_line if command_line else "unknown"
    if train_result.get("target_length") is not None:
        config_to_save["target_length"] = int(train_result["target_length"])
    
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2, default=str)

    if train_result.get("training_stats"):
        stats_path = os.path.join(run_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(train_result["training_stats"], f, indent=2)
    
    runtime_path = os.path.join(run_dir, "runtime.json")
    with open(runtime_path, 'w') as f:
        json.dump(runtime_info, f, indent=2, default=str)
    
    
    results = train_result.get("results", {})
    confusion_matrix_data = results.get("confusion_matrix", [])
    if hasattr(confusion_matrix_data, "tolist"):
        confusion_matrix_data = confusion_matrix_data.tolist()

    results_to_save = {
        "train_accuracy": float(results.get("train_accuracy", 0)),
        "test_accuracy": float(results.get("test_accuracy", 0)),
        "classification_report": results.get("classification_report", ""),
        "confusion_matrix": confusion_matrix_data,
        "target_length": int(train_result.get("target_length", 0)),
        "model_type": train_result.get("model_type", "unknown")
    }
    
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    if "train_predictions" in results:
        np.save(os.path.join(run_dir, "train_predictions.npy"), results["train_predictions"])
        np.save(os.path.join(run_dir, "test_predictions.npy"), results["test_predictions"])
        np.save(os.path.join(run_dir, "y_train.npy"), results["y_train"])
        np.save(os.path.join(run_dir, "y_test.npy"), results["y_test"])
    
    if save_model:
        model = train_result.get("model")
        scaler = train_result.get("scaler")
        transformer = train_result.get("transformer")
        
        if model is not None:
            try:
                model_path = os.path.join(run_dir, "model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception:
                pass
        
        if scaler is not None:
            try:
                scaler_path = os.path.join(run_dir, "scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            except Exception:
                pass
        
        if transformer is not None:
            try:
                transformer_path = os.path.join(run_dir, "transformer.pkl")
                with open(transformer_path, 'wb') as f:
                    pickle.dump(transformer, f)
            except Exception:
                pass
    
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Training Run Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Type: {train_result.get('model_type', 'unknown')}\n")
        f.write("Command Executed:\n")
        f.write(f"  {command_line if command_line else 'unknown'}\n\n")
        f.write("Configuration:\n")
        for key, value in config_to_save.items():
            if key not in ['command_executed']:
                f.write(f"  {key}: {value}\n")
        f.write("\nRuntime Information:\n")
        for key, value in runtime_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nResults:\n")
        f.write(f"  Train Accuracy: {results.get('train_accuracy', 0):.4f}\n")
        f.write(f"  Test Accuracy: {results.get('test_accuracy', 0):.4f}\n")
        f.write(f"\nClassification Report:\n{results.get('classification_report', '')}\n")
        f.write(f"\nConfusion Matrix:\n{results.get('confusion_matrix', [])}\n")
    
    return run_dir

def train_forest_model(n_traces=100000, n_kernels=10000, test_ratio=0.2, use_existing_dataset=None, save_dataset_path=None, auto_cache=True, seed=42, use_sktime=False, sktime_variant='minirocket'):
    model_type = "minirocket" if (use_sktime and sktime_variant == "minirocket") else \
                 "rocket" if (use_sktime and sktime_variant == "rocket") else \
                 "custom"
    
    return train_model(
        model_type=model_type,
        n_traces=n_traces,
        n_kernels=n_kernels,
        test_ratio=test_ratio,
        use_existing_dataset=use_existing_dataset,
        save_dataset_path=save_dataset_path,
        auto_cache=auto_cache,
        seed=seed
    )

def main():
    parser = argparse.ArgumentParser(
        description="train time series classification models using sktime",
    )
    
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--tsf", "--timeseriesforest", action="store_const", 
                            dest="model_type", const="tsf",
                            help="timeseriesforestclassifier (tree-based)")
    model_group.add_argument("--boss", action="store_const", 
                            dest="model_type", const="boss",
                            help="bossensemble (bag-of-words)")
    model_group.add_argument("--weasel", action="store_const", 
                            dest="model_type", const="weasel",
                            help="WEASEL (bag-of-words)")
    model_group.add_argument("--rocket", action="store_const", 
                            dest="model_type", const="rocket",
                            help="ROCKET + RidgeClassifier")
    model_group.add_argument("--minirocket", "--sktime", "-s", action="store_const", 
                            dest="model_type", const="minirocket",
                            help="MiniROCKET + RidgeClassifier (default)")
    model_group.add_argument("--bagging", action="store_const", 
                            dest="model_type", const="bagging",
                            help="baggingclassifier (ensemble)")
    model_group.add_argument("--weighted-ensemble", "--ensemble", action="store_const", 
                            dest="model_type", const="weighted_ensemble",
                            help="weightedensembleclassifier (ensemble)")
    model_group.add_argument("--hivecote", "--hivecotev2", action="store_const", 
                            dest="model_type", const="hivecotev2",
                            help="hivecotev2 (hierarchical ensemble)")
    model_group.add_argument("--hivecotev1", action="store_const", 
                            dest="model_type", const="hivecotev1",
                            help="hivecotev1 (earlier version)")
    model_group.add_argument("--custom", action="store_const", 
                            dest="model_type", const="custom",
                            help="custom ROCKET implementation")
    
    parser.add_argument("--n-traces", type=int, default=50000,
                       help="number of traces to generate/use (default: 50000)")
    parser.add_argument("--n-kernels", type=int, default=5000,
                       help="number of kernels for ROCKET/MiniROCKET models (default: 5000)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="random seed (default: 42)")
    
    parser.add_argument("--dataset", "--use-existing-dataset", dest="use_existing_dataset",
                       type=str, default=None,
                       help="path to existing dataset file (overrides auto-cache)")
    parser.add_argument("--save-dataset", dest="save_dataset_path",
                       type=str, default=None,
                       help="path to save generated dataset")
    parser.add_argument("--no-cache", dest="auto_cache", action="store_false",
                       help="disable automatic dataset caching/loading")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="force regeneration of dataset even if cached version exists")
    parser.add_argument("--allow-larger-dataset", action="store_true",
                       help="allow using cached datasets with more traces than requested (default: exact match only)")
    parser.set_defaults(auto_cache=True, force_regenerate=False, allow_larger_dataset=False)
    
    parser.add_argument("--output-dir", type=str, default=None,
                       help="base directory for outputs (default: 'outputs')")
    parser.add_argument("--no-save", dest="save_outputs", action="store_false",
                       help="don't save outputs")
    parser.add_argument("--all-models", action="store_true",
                       help="train all available models and compare (like test_models.py)")
    parser.set_defaults(save_outputs=True, all_models=False)
    
    args = parser.parse_args()
    
    if args.all_models:
        from test_models import test_all_models
        test_all_models(n_traces=args.n_traces, n_kernels=args.n_kernels, seed=args.seed)
        return
    
    model_type = args.model_type if args.model_type else "minirocket"
    
    command_line = " ".join(sys.argv)

    path_manager = PathManager()
    path_manager.ensure_dirs()
    
    total_start_time = time.time()
    
    if path_manager.is_slurm:
        print(f"SLURM job id: {get_slurm_job_id()}")

    results = train_model(
        model_type=model_type,
        n_traces=args.n_traces,
        n_kernels=args.n_kernels,
        test_ratio=args.test_ratio,
        use_existing_dataset=args.use_existing_dataset,
        save_dataset_path=args.save_dataset_path,
        auto_cache=args.auto_cache,
        seed=args.seed,
        force_regenerate=args.force_regenerate,
        allow_larger_dataset=args.allow_larger_dataset,
        path_manager=path_manager
    )

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time

    print(f"test accuracy: {results['results']['test_accuracy']:.4f}, runtime: {total_runtime:.2f}s")

    saved_dir = None
    if args.save_outputs:
        config = {
            "model_type": model_type,
            "n_traces": args.n_traces,
            "n_kernels": args.n_kernels,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "use_existing_dataset": args.use_existing_dataset,
            "save_dataset_path": args.save_dataset_path,
            "auto_cache": args.auto_cache,
            "model_params": None
        }

        runtime_info = {
            "total_runtime_seconds": total_runtime,
            "total_runtime_formatted": f"{total_runtime:.2f}s"
        }

        if results.get("runtime_timings"):
            runtime_info["detailed_timings"] = results["runtime_timings"]

        saved_dir = save_training_outputs(
            train_result=results,
            config=config,
            runtime_info=runtime_info,
            output_dir=args.output_dir,
            save_model=True,
            command_line=command_line,
            path_manager=path_manager
        )

        print(f"outputs saved to {saved_dir}")

        if path_manager.is_slurm:
            saved_path = Path(saved_dir)
            try:
                scratch_base = Path(path_manager.scratch_base).resolve()
                if str(saved_path.resolve()).startswith(str(scratch_base)):
                    home_output = path_manager.copy_outputs_to_home(
                        saved_path,
                        job_id=get_slurm_job_id()
                    )
                    print(f"results copied to home space: {home_output}")
            except Exception as e:
                print(f"warning: failed to copy results to home space: {e}")
    else:
        print("outputs not saved (--no-save)")

if __name__ == "__main__":
    main()
