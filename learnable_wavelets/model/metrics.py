import torch


def psnr_metric(x_rec, x, range_: tuple[float, float] = (-1.0, 1.0)):
    mse = torch.mean((x_rec - x) ** 2, dim=[1, 2])
    max_i = torch.tensor(range_[1] - range_[0], device=x.device, dtype=x.dtype)
    psnr = 20 * torch.log10(max_i) - 10 * torch.log10(mse)
    return torch.mean(psnr)
