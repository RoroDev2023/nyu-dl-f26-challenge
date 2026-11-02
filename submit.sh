#!/bin/bash
# submit without the Jupyter session's Slurm env settings (they make cloud nodes time out)
env -u SLURM_EXPORT_ENV -u SLURM_GET_USER_ENV sbatch --export=ALL "$@"
