import matplotlib.pyplot as plt
import numpy as np

from learnable_wavelets import tools


def plot_wavelet(filters, axes: tuple[plt.Axes, plt.Axes], J: int = 5):
    _, psi, t, w, Psi = tools.compute_wavelet(filters, J)

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
    ax_freq.set_title("Frequency Response of Mother Wavelet")


def plot_reconstruction(x_rec, x, axes: tuple[plt.Axes, plt.Axes]):
    original = tools.change_range(x, -1, 1, 0, 255)
    original = original.numpy().astype(np.uint8)
    reconstruction = tools.change_range(x_rec, -1, 1, 0, 255)
    reconstruction = reconstruction.numpy().astype(np.uint8)

    ax_original, ax_reconstructed = axes

    ax_original.imshow(original, cmap="gray", vmin=0, vmax=255)
    ax_original.set_title("Original Image")
    ax_original.axis("off")

    ax_reconstructed.imshow(reconstruction, cmap="gray", vmin=0, vmax=255)
    ax_reconstructed.set_title("Reconstructed Image")
    ax_reconstructed.axis("off")
