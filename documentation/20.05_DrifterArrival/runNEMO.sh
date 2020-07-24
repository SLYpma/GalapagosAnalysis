#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -J ReleaseDrifter_NEMO8

echo 'Executing program ...'

python drifterrun_fwd_nemo.py

echo 'Finished computation.'
