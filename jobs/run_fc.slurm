#!/bin/bash
#SBATCH --job-name=parcellation_decomp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=ie-itk
#SBATCH --mail-user=herminea@stud.ntnu.no
#SBATCH --mail-type=ALL

# --- Environment setup ---
module purge
module load Anaconda3/2022.10
source activate neuro

# Move to project directory
cd /cluster/home/herminea/mental_health_project/test

# --- Run FC script ---
echo "[$(date)] Starting FC computation..."
python scripts/compute_fc.py
status_fc=$?
if [ $status_fc -ne 0 ]; then
  echo "FC computation failed with exit code $status_fc"
  exit $status_fc
fi

# --- Run FC combine script ---
echo "[$(date)] Starting FC combining..."
python scripts/combine_fc.py
status_fc=$?
if [ $status_fc -ne 0 ]; then
  echo "FC combining failed with exit code $status_fc"
  exit $status_fc
fi

echo "[$(date)] All processing completed successfully."
