# utils/signal_generator.py
import numpy as np

def generate_synthetic_bold(
    n_channels=434,
    n_timepoints=488,
    n_modes=5,
    fs=1.25,
    freq_range=(0.005, 0.25),
    noise_level=0.05,
    amplitude_mod=True,
    freq_mod=True,
    sparsity=0.6,
    random_state=None,
):
    """
    Generate synthetic multivariate BOLD-like signals using AM-FM latent components.
    
    Parameters
    ----------
    n_channels : int
        Number of ROIs/channels.
    n_timepoints : int
        Length of the time series.
    n_modes : int
        Number of latent oscillatory modes.
    fs : float
        Sampling frequency (Hz), e.g., 1/TR.
    freq_range : tuple
        Min and max frequency for modes (Hz).
    noise_level : float
        Standard deviation of additive Gaussian noise (relative to signal power).
    amplitude_mod : bool
        Whether to include slow amplitude modulation.
    freq_mod : bool
        Whether to include small frequency modulation.
    sparsity : float
        Fraction of zero weights in the ROI–mode mixing matrix.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray (n_channels × n_timepoints)
        Generated synthetic BOLD data.
    ground_truth : dict
        Contains 'freqs', 'amplitudes', 'mixing_matrix', and 'latent_modes'.
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(n_timepoints) / fs

    # Define central frequencies
    freqs = np.linspace(freq_range[0], freq_range[1], n_modes)
    latent_modes = np.zeros((n_modes, n_timepoints))

    # Generate latent AM–FM components
    for k, f0 in enumerate(freqs):
        amp = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.005, 0.02) * t) if amplitude_mod else 1.0
        freq = f0 + 0.005 * np.sin(2 * np.pi * rng.uniform(0.005, 0.015) * t) if freq_mod else f0
        phase = 2 * np.pi * np.cumsum(freq) / fs
        latent_modes[k] = amp * np.cos(phase + rng.uniform(0, 2 * np.pi))

    # Sparse mixing matrix (ROIs × modes)
    A = rng.normal(size=(n_channels, n_modes))
    mask = rng.random(A.shape) > sparsity
    A *= mask

    # Mix latent modes across channels
    X = A @ latent_modes

    # Normalize and add noise
    X /= np.std(X)
    X += noise_level * rng.normal(size=X.shape)

    ground_truth = {
        "freqs": freqs,
        "latent_modes": latent_modes,
        "mixing_matrix": A,
        "t": t,
        "fs": fs,
    }
    return X, ground_truth
