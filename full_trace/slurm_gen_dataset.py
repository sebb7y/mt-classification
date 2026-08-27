import sys
import argparse
import shutil
from trace_gen import generate_training_dataset, save_dataset
from slurm_helpers import PathManager, get_slurm_cpus

from estimate_dataset_size import estimate_dataset_size, format_size

def main():
    parser = argparse.ArgumentParser(description="generate synthetic dataset for SLURM")
    parser.add_argument("--n-traces", type=int, default=100000, help="number of traces")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--usable-ratio", type=float, default=0.5, help="ratio of usable traces")
    
    parser.add_argument("--total-time-min", type=float, default=100.0, help="min trace duration in seconds")
    parser.add_argument("--total-time-max", type=float, default=300.0, help="max trace duration in seconds")
    parser.add_argument("--dt-min", type=float, default=0.01, help="min time step in seconds")
    parser.add_argument("--dt-max", type=float, default=0.05, help="max time step in seconds")
    parser.add_argument("--generation-method", type=str, default="split", choices=["direct", "split"],
                       help="direct or split (default)")
    parser.add_argument("--split-trace-duration", type=float, default=None,
                       help="split trace duration in seconds")
    
    args = parser.parse_args()
    
    pm = PathManager()
    pm.ensure_dirs()
    
    n_cpus = get_slurm_cpus()
    
    dataset = generate_training_dataset(
        n_traces=args.n_traces,
        usable_ratio=args.usable_ratio,
        total_time_range=(args.total_time_min, args.total_time_max),
        dt_range=(args.dt_min, args.dt_max),
        noise_std_range=(5.0, 20.0),
        base_seed=args.seed,
        n_workers=n_cpus,
        generation_method=args.generation_method,
        split_trace_duration=args.split_trace_duration
    )
    
    if args.total_time_max != 300.0 or args.total_time_min != 100.0:
        dataset_name = f"dataset_n{args.n_traces}_t{args.total_time_min}-{args.total_time_max}s_seed{args.seed}.npz"
    else:
        dataset_name = f"dataset_n{args.n_traces}_seed{args.seed}.npz"
    scratch_path = pm.dataset_cache_dir(use_scratch=True) / dataset_name
    home_path = pm.dataset_cache_dir(use_scratch=False) / dataset_name
    
    save_dataset(dataset, str(scratch_path), format='npz_padded')
    home_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scratch_path, home_path)
    print(f'saved to {scratch_path} and {home_path}')
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
