import torch


def mse_loss(x_rec, x):
    return torch.mean((x_rec - x) ** 2)


def l1_sparsity_loss(x):
    return torch.mean(torch.abs(x))
