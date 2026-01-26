# utils/parcellation.py
import os
import numpy as np
import h5py
from nilearn.maskers import NiftiLabelsMasker
from tqdm import tqdm


def reduce_buckner7_to_2(ts):
    """Merge 7-network Buckner cerebellar signals into 2 regions."""
    anterior = ts[0:3, :].mean(axis=0)
    posterior = ts[3:7, :].mean(axis=0)
    return np.vstack([anterior, posterior])


def parcellate_run(bold_path, atlas_paths):
    """Parcellate one fMRIPrep run into combined ROI × T."""
    all_rois = []
    for atlas in atlas_paths:
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            standardize=False,
            detrend=False,
            resampling_target="data"
        )
        ts = masker.fit_transform(bold_path).T
        if "Buckner7" in atlas:
            ts = reduce_buckner7_to_2(ts)
        all_rois.append(ts)

    roi_data = np.vstack(all_rois)
    return roi_data


def load_subject_roi_timeseries(run_paths, atlas_paths):
    """Load, parcellate, and concatenate runs for one subject."""
    roi_runs = {}
    
    for bold_path in run_paths:
        print(f"[Parcellating] {os.path.basename(bold_path)}")
        roi_data = parcellate_run(bold_path, atlas_paths)
        roi_runs[bold_path] = roi_data

    print(f"→ {len(roi_runs)} runs processed.")
    return roi_runs


def save_roi_timeseries_h5(roi_dict, out_path, subj_id, atlas_list, tr, n_runs):
    """Save all runs (each separately) to HDF5 with metadata."""
    with h5py.File(out_path, "w") as f:
        for run_name, data in roi_dict.items():
            # Store each run in its own dataset
            f.create_dataset(os.path.basename(run_name).replace(".nii.gz", ""), data=data)
        f.attrs["subject"] = subj_id
        f.attrs["atlas"] = ", ".join([os.path.basename(a) for a in atlas_list])
        f.attrs["TR"] = tr
        f.attrs["n_runs"] = n_runs
    print(f"[IO] Saved: {out_path}")


