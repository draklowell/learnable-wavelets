import matplotlib.pyplot as plt
import numpy as np

from learnable_wavelets.tools import compute_wavelet


def plot_wavelet(filters, axes: tuple[plt.Axes, plt.Axes], J: int = 5):
    _, psi, t, w, Psi = compute_wavelet(filters, J)

    if len(axes) != 2:
        raise ValueError(
            "axes must contain exactly two matplotlib axes: (ax_time, ax_freq)"
        )

    ax_time, ax_freq = axes

    # Time-domain plot
    ax_time.plot(t, psi)
    ax_time.grid(True)
    ax_time.set_xlabel("t")
    ax_time.set_ylabel("psi(t)")
    ax_time.set_title("Time Response of Mother Wavelet")

    # Frequency-domain plot
    ax_freq.plot(w, np.abs(Psi))
    ax_freq.grid(True)
    ax_freq.set_xlabel("w")
    ax_freq.set_ylabel("|Psi(w)|")
    ax_freq.set_title("Frequency Response of Mother Wavelet (Positive Frequencies)")
