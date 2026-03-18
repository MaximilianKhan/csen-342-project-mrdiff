#!/bin/bash
#SBATCH --job-name=mrdiff_2scale
#SBATCH --partition=condo
#SBATCH --nodelist=dmlab01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-user=ktamil@scu.edu
#SBATCH --mail-type=END,FAIL

module purge
module load Anaconda3
source /WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh
conda activate /WAVE/projects/CSEN-342-Wi26/Group2/conda-envs/mrDiff
echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
cd /WAVE/projects/CSEN-342-Wi26/Group2/submission_twoscale

echo "[Exp31] Two-scale decomposition on $(hostname)"

echo "[Exp31] Two-scale decomposition single models"
python train_single.py

echo "[Exp31] Two-scale decomposition ensemble"
python train_ensemble.py
echo "[Exp31] Done."