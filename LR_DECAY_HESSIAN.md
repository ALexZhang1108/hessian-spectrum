# Hessian Computation During LR Decay

The training scripts can now compute the Hessian spectrum automatically during the learning-rate (LR) decay phase.

## Usage

1. Set the `hessian_interval` variable to a positive integer either in your config file or on the command line. The value controls how often (in training iterations) to evaluate the Hessian once the LR starts to decay.
   - Example config snippet:
     ```python
     hessian_interval = 1000  # compute Hessian every 1000 steps after warmup
     ```
   - Command line override:
     ```bash
     python train_gpt2.py --hessian_interval=1000
     ```

2. Run training as usual. After the warmup period finishes and the LR begins to decay, the script will call the Hessian spectrum estimator every `hessian_interval` iterations on the master process. Results are stored under `files/` with filenames indicating the checkpoint iteration.

## Notes

- Setting `hessian_interval` to `0` (default) disables the automatic computation.
- The computation uses the existing `hessian_spectrum.Hessian` class and may take significant time; choose an interval that balances coverage and runtime.

