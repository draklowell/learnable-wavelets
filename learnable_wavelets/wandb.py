import numpy as np
import wandb

from learnable_wavelets import tools
from learnable_wavelets.tools import compute_wavelet


def add_code_to_artifact(
    artifact: wandb.Artifact,
    name: str = "notebook.ipynb",
    path: str = "_notebook.ipynb",
) -> None:
    tools.get_code(path)

    artifact.add_file(path, name)


def get_wavelet(filters, J: int = 5):
    _, psi, t, w, Psi = compute_wavelet(filters, J)

    time = wandb.Table(
        data=np.stack([t, psi], axis=1),
        columns=["t", "psi(t)"],
    )
    frequency = wandb.Table(
        data=np.stack([w, np.abs(Psi)], axis=1),
        columns=["w", "|Psi(w)|"],
    )

    time_plot = wandb.plot.line(
        time, x="t", y="psi(t)", title=f"Time Response of Mother Wavelet (J={J})"
    )
    frequency_plot = wandb.plot.line(
        frequency,
        x="w",
        y="|Psi(w)|",
        title=f"Frequency Response of Mother Wavelet (J={J})",
    )
    return {
        "time": time_plot,
        "frequency": frequency_plot,
    }


def get_reconstruction(x_rec, x):
    # Expects [-1, 1] range

    original = (((x + 1) / 2) * 255).numpy().astype(np.uint8)
    reconstruction = (((x_rec + 1) / 2) * 255).numpy().astype(np.uint8)

    return {
        "original": wandb.Image(original, caption="Original Image", normalize=False),
        "reconstruction": wandb.Image(
            reconstruction, caption="Reconstructed Image", normalize=False
        ),
    }
