import torch


def psnr_metric(
    x_rec=None, x=None, loss=None, range_: tuple[float, float] = (-1.0, 1.0)
):
    if loss is not None:
        mse = loss
        device = "cpu"
        dtype = torch.float32
    else:
        mse = torch.mean((x_rec - x) ** 2, dim=[1, 2])
        device = x.device
        dtype = x.dtype

    max_i = torch.tensor(range_[1] - range_[0], device=device, dtype=dtype)
    psnr = 20 * torch.log10(max_i) - 10 * torch.log10(mse)
    return torch.mean(psnr)
