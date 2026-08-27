import os
import shutil
import multiprocessing as mp
from pathlib import Path
import json

def get_slurm_cpus():
    cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if cpus:
        return int(cpus)
    
    cpus = os.environ.get('SLURM_JOB_CPUS_PER_NODE')
    if cpus:
        return int(cpus)
    
    cpus = os.environ.get('SLURM_CPUS_ON_NODE')
    if cpus:
        return int(cpus)
    
    return mp.cpu_count()

def is_slurm_job():
    return 'SLURM_JOB_ID' in os.environ

def get_slurm_job_id():
    return os.environ.get('SLURM_JOB_ID')

def homedir():
    if is_slurm_job():
        submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
        if submit_dir:
            return Path(submit_dir)
    
    return Path(os.path.expanduser('~'))

def scratchdir():
    if is_slurm_job():
        tmpdir = os.environ.get('TMPDIR')
        if tmpdir:
            return Path(tmpdir)
    
    home = homedir()
    return home / 'scratch'

def project_root():
    if is_slurm_job():
        submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
        if submit_dir:
            return Path(submit_dir)
    
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / 'train_models.py').exists() or (parent / 'trace_gen.py').exists():
            return parent
    
    return Path.cwd()

class PathManager:
    
    def __init__(self, home_base=None, scratch_base=None, project_name='tweezcat'):
        self.project_name = project_name
        self.home_base = home_base or homedir()
        self.scratch_base = scratch_base or scratchdir()
        self.is_slurm = is_slurm_job()
        
        self.home_data = self.home_base / project_name / "data"
        self.home_models = self.home_base / project_name / "models"
        self.home_outputs = self.home_base / project_name / "outputs"
        self.home_cache = self.home_base / project_name / "dataset_cache"
        
        self.scratch_data = self.scratch_base / project_name / "data"
        self.scratch_models = self.scratch_base / project_name / "models"
        self.scratch_outputs = self.scratch_base / project_name / "outputs"
        self.scratch_cache = self.scratch_base / project_name / "dataset_cache"
    
    def dataset_cache_dir(self, use_scratch=True):
        if self.is_slurm and use_scratch:
            return self.scratch_cache
        return self.home_cache
    
    def outdir(self, use_scratch=True):
        if self.is_slurm and use_scratch:
            return self.scratch_outputs
        return self.home_outputs
    
    def modeldir(self, use_scratch=False):
        if self.is_slurm and use_scratch:
            return self.scratch_models
        return self.home_models
    
    def ensure_dirs(self):
        dirs = [
            self.home_data, self.home_models, self.home_outputs, self.home_cache,
            self.scratch_data, self.scratch_models, self.scratch_outputs, self.scratch_cache
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def copy_dataset_to_scratch(self, dataset_name, force=False):
        home_path = self.home_cache / dataset_name
        scratch_path = self.scratch_cache / dataset_name
        
        if not home_path.exists():
            raise FileNotFoundError(f"dataset not found in home: {home_path}")
        
        if scratch_path.exists() and not force:
            return scratch_path
        
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(home_path, scratch_path)
        return scratch_path
    
    def copy_outputs_to_home(self, output_dir, job_id=None):
        if not output_dir.exists():
            raise FileNotFoundError(f"output directory not found: {output_dir}")
        
        if job_id:
            dest_name = f"{output_dir.name}_job{job_id}"
        else:
            dest_name = output_dir.name
        
        home_output = self.home_outputs / dest_name
        
        if home_output.exists():
            shutil.rmtree(home_output)
        
        shutil.copytree(output_dir, home_output)
        return home_output
    
    def find_dataset(self, dataset_name, prefer_scratch=True):
        scratch_path = self.scratch_cache / dataset_name
        home_path = self.home_cache / dataset_name
        
        if prefer_scratch and scratch_path.exists():
            return scratch_path
        if home_path.exists():
            return home_path
        if scratch_path.exists():
            return scratch_path
        
        return None
    
def worker_count(n_workers=None):
    if n_workers is not None:
        return n_workers
    
    return get_slurm_cpus()

def save_path_cfg(path_manager, filepath):
    config = {
        "project_name": path_manager.project_name,
        "home_base": str(path_manager.home_base),
        "scratch_base": str(path_manager.scratch_base),
        "is_slurm": path_manager.is_slurm,
        "job_id": get_slurm_job_id(),
        "home_cache": str(path_manager.home_cache),
        "scratch_cache": str(path_manager.scratch_cache),
        "home_outputs": str(path_manager.home_outputs),
        "scratch_outputs": str(path_manager.scratch_outputs),
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    pm = PathManager()
    pm.ensure_dirs()
    
    print(f"project: {pm.project_name}, SLURM: {pm.is_slurm}, job: {get_slurm_job_id()}, CPUs: {get_slurm_cpus()}")
    print(f"home: {pm.home_base}")
    print(f"scratch: {pm.scratch_base}")
