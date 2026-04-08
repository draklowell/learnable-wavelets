import matplotlib.pyplot as plt
import numpy as np


def plot_wavelet(filters, axes: tuple[plt.Axes, plt.Axes], J: int = 5):
    # Low-pass
    h = np.asarray(filters[0], dtype=float).squeeze()
    # High-pass
    g = np.asarray(filters[1], dtype=float).squeeze()

    if h.ndim != 1 or g.ndim != 1:
        raise ValueError(
            f"h and g must be 1D. Got h.shape={h.shape}, g.shape={g.shape}"
        )

    if len(axes) != 2:
        raise ValueError(
            "axes must contain exactly two matplotlib axes: (ax_time, ax_freq)"
        )

    ax_time, ax_freq = axes

    # Build scaling function approximation
    phi = np.array([1.0])
    for _ in range(J):
        up = np.zeros(2 * len(phi), dtype=float)
        up[::2] = phi
        phi = np.sqrt(2) * np.convolve(up, h, mode="full")

    # Build mother wavelet approximation
    up_phi = np.zeros(2 * len(phi), dtype=float)
    up_phi[::2] = phi
    psi = np.sqrt(2) * np.convolve(up_phi, g, mode="full")

    dt = 1 / (2**J)
    t_psi = (np.arange(len(psi)) - len(psi) // 2) * dt

    # Time-domain plot
    ax_time.plot(t_psi, psi)
    ax_time.grid(True)
    ax_time.set_xlabel("t")
    ax_time.set_ylabel("psi(t)")
    ax_time.set_title("Time Response of Mother Wavelet")

    # Frequency-domain plot: positive frequencies only
    # rfft is best here because psi is real-valued and it preserves all nonnegative-frequency info
    Psi_pos = np.fft.rfft(psi)
    w_pos = 2 * np.pi * np.fft.rfftfreq(len(psi), d=dt)

    ax_freq.plot(w_pos, np.abs(Psi_pos))
    ax_freq.grid(True)
    ax_freq.set_xlabel("omega")
    ax_freq.set_ylabel("|Psi(omega)|")
    ax_freq.set_title("Frequency Response of Mother Wavelet (Positive Frequencies)")

    return phi, psi, t_psi, w_pos, Psi_pos
