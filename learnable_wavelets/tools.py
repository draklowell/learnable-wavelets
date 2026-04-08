import json

import numpy as np

try:
    from google.colab import _message
except ImportError:
    _message = None


def get_code(path: str = "notebook.ipynb") -> None:
    if _message is None:
        raise RuntimeError("This function can only be used in Google Colab.")

    resp = _message.blocking_request("get_ipynb", timeout_sec=5)["ipynb"]

    with open(path, "w") as f:
        json.dump(resp, f)


def compute_wavelet(filters, J: int = 5):
    # Low-pass
    h = np.asarray(filters[0], dtype=float).squeeze()
    # High-pass
    g = np.asarray(filters[1], dtype=float).squeeze()

    if h.ndim != 1 or g.ndim != 1:
        raise ValueError(
            f"h and g must be 1D. Got h.shape={h.shape}, g.shape={g.shape}"
        )

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
    t = (np.arange(len(psi)) - len(psi) // 2) * dt

    # Positive frequencies only
    # rfft is best here because psi is real-valued and it preserves all nonnegative-frequency info
    Psi = np.fft.rfft(psi)
    w = 2 * np.pi * np.fft.rfftfreq(len(psi), d=dt)

    return phi, psi, t, w, Psi


def change_range(x, old_min, old_max, new_min, new_max):
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
