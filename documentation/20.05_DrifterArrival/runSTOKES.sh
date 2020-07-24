#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -J ReleaseDrifter_STOKES

echo 'Executing program ...'

python drifterrun_fwd_nemo_wstokes.py

echo 'Finished computation.'
