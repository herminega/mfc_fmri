# utils.py
import os, glob, json
import numpy as np
import h5py
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt


# -----------------------------
# I/O: subjects and BOLD files
# -----------------------------

def read_subjects(txt_path):
    with open(txt_path, "r") as f:
        return [s.strip() for s in f.read().splitlines() if s.strip()]

def index_bold_files(base_dir, subject_ids,
                     pattern="task-rest*_run-*_bold_Atlas_hp2000_clean_GSR_parcellated.h5"):
    """
    Return dict: subject_id -> list of run file paths (resting-state only).
    """
    idx = {}
    for sid in subject_ids:
        subj_dir = os.path.join(base_dir, sid)
        files = glob.glob(os.path.join(subj_dir, pattern))
        if files:
            idx[sid] = sorted(files)
    return idx

def load_bold_matrix(h5_file_path, dataset_key="dataset", detrend=True, zscore_rois=True):
    """Load (ROIs, T) from .h5 and standardize per ROI."""
    with h5py.File(h5_file_path, "r") as f:
        X = f[dataset_key][:]  # (R, T)
    if detrend:
        X = signal.detrend(X, axis=1)
    if zscore_rois:
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return X

def iter_subject_runs(bold_index, subjects, limit_per_group=None):
    """
    Yield (subj, run_name, X) for subjects that exist in bold_index.
    If limit_per_group is given, yields only the first N subjects (not runs).
    """
    count = 0
    for subj in subjects:
        if subj not in bold_index:
            continue
        for run_path in bold_index[subj]:
            run_name = os.path.basename(run_path).replace(".h5", "")
            X = load_bold_matrix(run_path)
            yield subj, run_name, X
        count += 1
        if limit_per_group and count >= limit_per_group:
            break

# -----------------------------
# Decomposition helpers
# -----------------------------

def mvmd_decompose(X, alpha=1500, K=8, tau=0.0, tol=1e-7):
    from pysdkit import MVMD
    mvmd = MVMD(alpha, K=K, tau=tau, tol=tol)
    imfs = mvmd.fit_transform(X.T)
    if imfs.shape[1] == X.shape[0]:
        imfs = np.transpose(imfs, (0, 2, 1))
    return imfs

def vlmd_run(X, num_modes=8, num_latents=32, alpha=800, reg_lambda=0.15, reg_rho=1.0, fs=1.25):
    from vlmd import vlmd
    return vlmd(
        X,
        num_modes=num_modes,
        num_latents=num_latents,
        alpha=alpha,
        reg_lambda=reg_lambda,
        reg_rho=reg_rho,
        sampling_rate=fs
    )


def vlmd_to_roi_imfs(latent_modes, latent_coefs):
    """
    latent_modes: (K, L, T) or (K, T, L)
    latent_coefs: (L, C) or (C, L)
    Returns imfs in (K, T, C).
    """
    lm = np.asarray(latent_modes)
    lc = np.asarray(latent_coefs)

    if lc.ndim != 2:
        raise ValueError(f"latent_coefs must be 2D, got {lc.ndim}D")
    L_from_coefs = lc.shape[0] if lc.shape[0] < lc.shape[1] else lc.shape[1]

    # make (K, L, T)
    if lm.ndim != 3:
        raise ValueError(f"latent_modes must be 3D, got shape {lm.shape}")
    if lm.shape[1] == L_from_coefs:
        pass  # (K, L, T)
    elif lm.shape[2] == L_from_coefs:
        lm = np.transpose(lm, (0, 2, 1))  # (K, T, L) -> (K, L, T)
    else:
        raise ValueError(
            f"Incompatible shapes: latent_modes {lm.shape}, latent_coefs {lc.shape} "
            f"(cannot find latent dimension L={L_from_coefs})"
        )

    K, L, T = lm.shape
    # project to ROI space → (K, C, T)
    if lc.shape[0] == L:  # (L, C)
        roi_modes = np.einsum('lc,klt->kct', lc, lm)
    elif lc.shape[1] == L:  # (C, L)
        roi_modes = np.einsum('cl,klt->kct', lc, lm)
    else:
        raise ValueError(f"latent_coefs shape {lc.shape} incompatible with L={L}")
    return np.transpose(roi_modes, (0, 2, 1))  # (K, T, C)

def sort_filter_by_freq(imfs, omegas, fs=1.25, fmin=0.005, fmax=0.3):
    """
    imfs: (K, T, R)
    omegas: (iters, K) or (K,) in rad/sample
    Return (imfs_kept, freqs_kept_hz)
    """
    omega_final = omegas[-1] if omegas.ndim == 2 else omegas  # (K,)
    freqs_hz = (omega_final / (2*np.pi)) * fs
    order = np.argsort(freqs_hz)
    freqs_sorted = freqs_hz[order]
    keep = (freqs_sorted >= fmin) & (freqs_sorted <= fmax)
    return imfs[order][keep], freqs_sorted[keep]

# -----------------------------
# FC & features
# -----------------------------

def compute_fc_per_mode(imfs, roi_count=None):
    """
    imfs: (K, T, R) or (K, R, T)
    Returns: (K, R, R) Fisher-z FC per mode.
    """
    K, A, B = imfs.shape
    FCs = []
    for k in range(K):
        arr = imfs[k]
        # ensure rows=ROIs, cols=Time
        if roi_count is None:
            # try to infer: time usually ~488; ROI likely > time
            Xk = arr.T if arr.shape[0] < arr.shape[1] else arr
        else:
            if arr.shape[0] == roi_count:
                Xk = arr
            elif arr.shape[1] == roi_count:
                Xk = arr.T
            else:
                raise ValueError(f"IMF shape {arr.shape} doesn't match roi_count={roi_count}")
        Xk = zscore(Xk, axis=1)
        fc = np.corrcoef(Xk)  # (R, R)
        fc_z = np.arctanh(np.clip(fc, -0.999999, 0.999999))
        FCs.append(fc_z.astype(np.float32))
    return np.stack(FCs, axis=0)

def fc_to_feature_vector(fc_matrices):
    """Stack upper triangles across modes → 1D feature vector."""
    iu = np.triu_indices(fc_matrices.shape[1], 1)
    return np.concatenate([fc[iu] for fc in fc_matrices], axis=0)

# -----------------------------
# Plotting
# -----------------------------

def plot_imfs_for_roi(imfs, roi_idx=100, subj_label=""):
    K, T, R = imfs.shape
    fig, axs = plt.subplots(K, 1, figsize=(12, 2*K), sharex=True)
    for k in range(K):
        axs[k].plot(imfs[k, :, roi_idx])
        axs[k].set_ylabel(f"IMF {k+1}")
    axs[0].set_title(f"IMFs for ROI {roi_idx} - {subj_label}")
    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

def plot_fc_matrices(fc_matrices, subj_label="", vmax=1.0, roi_labels=None, label_step=50):
    """
    fc_matrices: (K, R, R)
    """
    K, R, _ = fc_matrices.shape
    ncols = min(K, 4)
    fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 4), constrained_layout=True)
    if ncols == 1:
        axs = [axs]

    if roi_labels is None:
        roi_labels = [str(i) for i in range(R)]
    tick_locs = np.arange(0, R, label_step)
    tick_labels = [roi_labels[i] for i in tick_locs]

    for k in range(ncols):
        ax = axs[k]
        im = ax.imshow(fc_matrices[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_title(f"FC - IMF {k+1}")
        ax.set_xticks(tick_locs); ax.set_yticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
    cbar = fig.colorbar(im, ax=axs, shrink=0.7)
    fig.suptitle(f"Functional Connectivity (z-FC) - {subj_label}", fontsize=12)
    plt.show()

# -----------------------------
# Saving
# -----------------------------

def save_npz_json(out_dir, subj, run_name, method, data_dict, meta):
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{subj}_{run_name}_{method}"
    np.savez_compressed(os.path.join(out_dir, stem + ".npz"), **data_dict)
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
