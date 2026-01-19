#!/bin/bash
#SBATCH --job-name=parcellation_decomp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=ie-itk
#SBATCH --mail-user=herminea@stud.ntnu.no
#SBATCH --mail-type=ALL

# --- Environment setup ---
module purge
module load Anaconda3/2022.10
source activate neuro

# Move to project directory
cd /cluster/home/herminea/mental_health_project/test

# --- Run parcellation script ---
# echo "[$(date)] Starting parcellation..."
# python scripts/parcellate.py
# status_parc=$?
# if [ $status_parc -ne 0 ]; then
#   echo "Parcellation failed with exit code $status_parc"
#   exit $status_parc
# fi
# echo "[$(date)] Parcellation completed successfully."

# --- Run decomposition script ---
echo "[$(date)] Starting decomposition..."
python scripts/run_decomp.py
status_decomp=$?
if [ $status_decomp -ne 0 ]; then
  echo "Decomposition failed with exit code $status_decomp"
  exit $status_decomp
fi

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

