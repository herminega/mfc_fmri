import os
import subprocess

# === CONFIG ===
SUBJECT_FILE = "/cluster/home/herminea/mental_health_project/workspace/subjects_lists/subjects_hc.txt"
DATA_ROOT = "/cluster/home/herminea/mental_health_project/tcp_dataset/ds005237/"
DATA_ROOT_FMRI = os.path.join(DATA_ROOT, "fMRI_timeseries_clean_denoised_GSR_parcellated/")
RUN_COMMANDS = True  # True = run datalad get, False = dry-run

# === Load subject IDs ===
with open(SUBJECT_FILE, "r") as f:
    subjects = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(subjects)} subjects from {SUBJECT_FILE}")

# === Generate full paths to resting-state files ===
rest_files = []
for subj in subjects:
    for phase in ["restAP", "restPA"]:
        for run in ["01", "02"]:
            fname = f"task-{phase}_run-{run}_bold_Atlas_hp2000_clean_GSR_parcellated.h5"
            fullpath = os.path.join(DATA_ROOT_FMRI, subj, fname)
            rest_files.append(fullpath)

print(f"Total resting-state files to get: {len(rest_files)}\n")

# === Run or print datalad commands ===
for path in rest_files:
    if os.path.exists(path):
        print(f"Already exists: {path}")
        continue

    cmd = ["datalad", "get", "-d", DATA_ROOT, path]
    if RUN_COMMANDS:
        try:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to get: {path}")
    else:
        print("Would run:", " ".join(cmd))
