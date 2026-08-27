# MT Data Classification Framework


**ML framework for magnetic tweezers trace classification written as a part of my master's thesis. Currently reworking this project with AI assisted coding, focusing on improving the synthetic data generation algorithm to enable more expressive models and increase overall performance.**







### Modular evaluation for force extension

Example run
`python extension/scripts/run_eval.py data_directory --task force -o out_directory`

Run with `--help` to see available arguments

Old (unfixed) results in `old_results_force_ext`

### Modular evaluation for torque extension

Example run
`python extension/scripts/run_eval.py /path/to/data --task rotation -o out`

Run with `--help` to see available arguments

Old (unfixed) results in `old_results_torque_ext`

### Modular evaluation for full trace

Running on HPC cluster with working directory as `full_trace:`

`DATA_ROOT=/path/to/data ./scripts/run_eval_pipeline.sh`

Orchestration is a little more messy for the full trace runs

Old (unfixed) results in `old_results_full_trace`
