import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms

import wandb
from learnable_wavelets import wandb as lw_wandb
from learnable_wavelets.config import load_config
from learnable_wavelets.datasets.mixed import MixedImageVisionDataset
from learnable_wavelets.model.loss import mse_loss
from learnable_wavelets.module import WaveletModule
from learnable_wavelets.train import Train

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
VALIDATION_SIZE = 10000
DELTA = 1e-5
PATIENCE = 20
MAX_EPOCHS = 1500
PATCH_SIZE = 128

CONFIG_PATH = "model.yaml"
IS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if IS_CUDA else "cpu"
IMAGE_DTYPE = torch.float32
PARAMS_DTYPE = torch.float64
LOG_INTERVAL = 50
VAL_INTERVAL = 1000

MODEL_CONFIG = load_config(CONFIG_PATH)
CONFIG = {
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "patch_size": PATCH_SIZE,
    "delta": DELTA,
    "validation_size": VALIDATION_SIZE,
    "device": DEVICE,
    "image_dtype": str(IMAGE_DTYPE),
    "params_dtype": str(PARAMS_DTYPE),
    "model_config": MODEL_CONFIG,
    "log_interval": LOG_INTERVAL,
    "val_interval": VAL_INTERVAL,
}


wandb.init(
    project="wavelets-test",
    config=CONFIG,
)


class ImageOnlyDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


train_dataset = MixedImageVisionDataset(
    split="train",
    split_seed=42,
    include_liu4k=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop((wandb.config.patch_size, wandb.config.patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ConvertImageDtype(IMAGE_DTYPE),
            transforms.Grayscale(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]
    ),
)

train_loader = data.DataLoader(
    ImageOnlyDataset(train_dataset),
    batch_size=wandb.config.batch_size,
    shuffle=True,
    num_workers=2,
)


val_size = wandb.config.validation_size

val_dataset_full = MixedImageVisionDataset(
    split="valid",
    split_seed=42,
    include_liu4k=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((wandb.config.patch_size, wandb.config.patch_size)),
            transforms.ConvertImageDtype(IMAGE_DTYPE),
            transforms.Grayscale(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]
    ),
)


val_dataset = data.Subset(val_dataset_full, list(range(val_size)))

val_loader = data.DataLoader(
    ImageOnlyDataset(val_dataset),
    batch_size=val_size,
    shuffle=False,
    num_workers=1,
    download=True,
)


module = WaveletModule(MODEL_CONFIG)
module = module.to(PARAMS_DTYPE)


def log_train(epoch, step, loss):
    if step % wandb.config.log_interval != 0:
        return

    print(f"{epoch}/{step}: Train Loss = {loss:.6f}")
    wandb.log({"train.loss": loss, "epoch": epoch})

    if step % wandb.config.val_interval != 0:
        return

    train.validate()


def log_validation(epoch, step, x_rec, x):
    loss = mse_loss(x_rec, x).item()
    print(f"{epoch}/{step}: Validation Loss = {loss:.6f}")

    wandb.log({"val.loss": loss, "epoch": epoch})

    for wavelet, generator in module.wavelets.items():
        filters = generator().cpu().detach()
        wavelet_plots = lw_wandb.get_wavelet(filters)
        wandb.log({f"val.wavelet.{wavelet}": wavelet_plots})

        angles = next(generator.parameters())
        data = angles.cpu().detach().tolist()

        angle_table = wandb.Table(
            data=[[f"θ{i}", angle] for i, angle in enumerate(data)],
            columns=["angle", "value"],
        )
        wandb.log(
            {
                f"val.angles.{wavelet}": wandb.plot.bar(
                    angle_table,
                    label="angle",
                    value="value",
                    title=f"Angles for {wavelet}",
                )
            }
        )

    reconstruction = lw_wandb.get_reconstruction(
        x_rec[0].cpu().detach(),
        x[0].cpu().detach(),
    )

    wandb.log({"val.example": reconstruction})


train = Train(
    train_loader=train_loader,
    val_loader=val_loader,
    module=module,
    optimizer_factory=lambda params: torch.optim.Adam(
        params, lr=wandb.config.learning_rate
    ),
    device=wandb.config.device,
    delta=wandb.config.delta,
    patience=wandb.config.patience,
    max_epochs=wandb.config.max_epochs,
    log_train=log_train,
    log_validation=log_validation,
)

# Safe training
try:
    train.run()
except KeyboardInterrupt:
    print("Training interrupted. Saving artifacts...")
finally:
    # Save artifacts to WandB
    code_artifact = wandb.Artifact("notebook", type="notebook")
    lw_wandb.add_code_to_artifact(code_artifact)
    wandb.log_artifact(code_artifact)

    torch.save(module.state_dict(), "model.pt")
    model_artifact = wandb.Artifact("model", type="model")
    model_artifact.add_file("model.pt")
    wandb.log_artifact(model_artifact)

    config_artifact = wandb.Artifact("config", type="config")
    config_artifact.add_file(CONFIG_PATH)
    wandb.log_artifact(config_artifact)

    wandb.finish()
