import tempfile
from copy import deepcopy

import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import wandb
from learnable_wavelets import psnr_metric
from learnable_wavelets import wandb as lw_wandb
from learnable_wavelets.config import ModuleConfig, load_config
from learnable_wavelets.model.loss import mse_loss
from learnable_wavelets.module import WaveletModule
from learnable_wavelets.train import Train


class ImageOnlyDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


class Run:
    def __init__(
        self,
        run: wandb.Run,
        config: ModuleConfig,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
    ):
        self.run = run
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.module = WaveletModule(config).to(dtype=run.config.params_dtype)
        self.train = Train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            module=self.module,
            optimizer_factory=lambda params: torch.optim.Adam(
                params, lr=run.config.learning_rate
            ),
            device=run.config.device,
            delta=run.config.delta,
            patience=run.config.patience,
            max_epochs=run.config.max_epochs,
            log_train=self.log_train,
            log_validation=self.log_validation,
        )
        self.best_psnr = None

    def log(self, epoch, step, message):
        print(f"[{self.run.id}] Epoch {epoch} (Step {step}): {message}")

    def log_train(self, epoch, step, loss):
        if step % self.run.config.log_interval != 0:
            return

        self.log(epoch, step, f"Train: Loss = {loss:.6f}")
        self.run.log({"train": {"loss": loss}, "epoch": epoch})

        if step % self.run.config.val_interval != 0:
            return

        self.train.validate()

    def log_validation(self, epoch, step, x_rec, x):
        loss = mse_loss(x_rec, x).item()
        psnr = psnr_metric(x_rec, x).item()
        self.log(epoch, step, f"Validation: Loss = {loss:.6f}; PSNR = {psnr:.2f} dB")

        info = {"loss": loss, "psnr": psnr}
        if self.best_psnr is None or psnr > self.best_psnr:
            self.best_psnr = psnr
            info["best_psnr"] = psnr

        info["wavelets"] = {}
        for wavelet, generator in self.module.wavelets.items():
            wavelet_info = {}
            filters = generator().cpu().detach()
            wavelet_plots = lw_wandb.get_wavelet(filters)
            wavelet_info["plot"] = wavelet_plots

            angles = torch.rad2deg(next(generator.parameters()))
            data = angles.cpu().detach().tolist()

            angle_table = wandb.Table(
                data=[[f"θ{i}", angle] for i, angle in enumerate(data)],
                columns=["angle", "value"],
            )
            wavelet_info["angles"] = wandb.plot.bar(
                angle_table,
                label="angle",
                value="value",
                title=f"Angles for {wavelet}",
            )

            info["wavelets"][wavelet] = wavelet_info

        reconstruction = lw_wandb.get_reconstruction(
            x_rec[0].cpu().detach(),
            x[0].cpu().detach(),
        )
        info["example"] = reconstruction

        self.run.log({"val": info, "epoch": epoch})

    def save(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(self.module.state_dict(), tmp.name)
            model_artifact = wandb.Artifact("model", type="model")
            model_artifact.add_file(tmp.name)
            self.run.log_artifact(model_artifact)

    def start(self):
        try:
            self.train.run()
        finally:
            self.save()

    def get_best_psnr(self) -> float | None:
        return self.best_psnr


class Runner:
    def __init__(
        self,
        config: dict,
    ):
        self.project_name = config["project_name"]
        self.config = deepcopy(config)
        self.config.pop("project_name")
        self.train_loader, self.val_loader = self._create_dataloaders(self.config)

    def run(self, config: ModuleConfig) -> float | None:
        with wandb.init(
            project=self.project_name, config=self.config | {"tree": config}
        ) as run:
            run_obj = Run(run, config, self.train_loader, self.val_loader)

            try:
                run_obj.start()
            except KeyboardInterrupt:
                pass

            return run_obj.get_best_psnr()

    @staticmethod
    def _create_dataloaders(config: dict) -> tuple[data.DataLoader, data.DataLoader]:
        train_dataset = datasets.CelebA(
            root=config["dataset"],
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop((config["patch_size"], config["patch_size"])),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ConvertImageDtype(config["image_dtype"]),
                    transforms.Grayscale(),
                    transforms.Lambda(lambda x: x * 2 - 1),
                ]
            ),
        )

        train_loader = data.DataLoader(
            ImageOnlyDataset(train_dataset),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["loader_num_workers"],
            pin_memory=config["device"] == "cuda",
        )

        val_dataset_full = datasets.CelebA(
            root=config["dataset"],
            split="valid",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop((config["patch_size"], config["patch_size"])),
                    transforms.ConvertImageDtype(config["image_dtype"]),
                    transforms.Grayscale(),
                    transforms.Lambda(lambda x: x * 2 - 1),
                ]
            ),
        )

        val_size = min(config["val_size"], len(val_dataset_full))
        val_dataset = data.Subset(val_dataset_full, list(range(val_size)))

        val_loader = data.DataLoader(
            ImageOnlyDataset(val_dataset),
            batch_size=val_size,
            shuffle=False,
            num_workers=config["loader_num_workers"],
            pin_memory=config["device"] == "cuda",
        )

        return train_loader, val_loader
