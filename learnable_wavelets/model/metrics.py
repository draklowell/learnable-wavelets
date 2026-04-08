import torch


def psnr_metric(x_rec, x):
    mse = torch.mean((x_rec - x) ** 2, dim=[1, 2])
    max_ = torch.amax(x, dim=[1, 2]) ** 2
    psnr = 10 * torch.log10(max_ / mse)
    return torch.mean(psnr)
