import numpy as np
import torch
from tqdm import tqdm

from src import schedule


def sample_images(
    model: torch.nn.Module,
    schedule: schedule.NoiseSchedule,
    n_channels: int,
    img_size: int,
    n_images=8,
    device="cpu",
):
    x = torch.randn((n_images, n_channels, img_size, img_size), device=device)
    T = schedule.get_timestamps()
    betas = schedule.get_betas()
    alphas_bar = schedule.get_alphas_bar()

    model.eval()
    for i in tqdm(range(T - 1, -1, -1)):
        with torch.no_grad():
            z = torch.randn_like(x, device=device)
            t = torch.full((n_images,), fill_value=i, dtype=torch.int, device=device)

            if i > 0:
                posterior_variance = betas[i]
            else:
                posterior_variance = 0.0

            noise_pred = model(x, t)
            x = (1 / np.sqrt(1 - betas[i])) * (
                x - (betas[i] / np.sqrt(1 - alphas_bar[i])) * noise_pred
            ) + np.sqrt(posterior_variance) * z

    return x
