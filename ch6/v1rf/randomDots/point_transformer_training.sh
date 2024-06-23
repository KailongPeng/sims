#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=point_transformer
#SBATCH --partition=psych_day
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e

module load miniconda
conda --version
conda deactivate
conda activate torch
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"


cd /gpfs/milgram/project/turk-browne/projects/sandbox/simulation/sims/ch6/v1rf/randomDots
echo python3 -u /gpfs/milgram/project/turk-browne/projects/sandbox/simulation/sims/ch6/v1rf/randomDots/point_transformer_training.py
python3 -u /gpfs/milgram/project/turk-browne/projects/sandbox/simulation/sims/ch6/v1rf/randomDots/point_transformer_training.py

echo "done"

#"
#cd /gpfs/milgram/project/turk-browne/projects/sandbox/simulation/sims/ch6/v1rf/randomDots
#sbatch /gpfs/milgram/project/turk-browne/projects/sandbox/simulation/sims/ch6/v1rf/randomDots/point_transformer_training.sh
#27044136 27044137
#"