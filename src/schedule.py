from abc import ABC, abstractmethod

import numpy as np
import torch


class NoiseSchedule(ABC):
    def __init__(self, timestamps: int):
        self.timestamps = timestamps

    def get_timestamps(self) -> int:
        return self.timestamps

    @abstractmethod
    def get_betas(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_alphas_bar(self) -> torch.Tensor:
        pass


class LinearSchedule(NoiseSchedule):
    def __init__(self, timestamps: int, beta_start=1e-4, beta_end=2e-2):
        super().__init__(timestamps)
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get_betas(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.timestamps + 1)

    def get_alphas_bar(self) -> torch.Tensor:
        alpha = 1 - self.get_betas()
        return torch.cumprod(alpha, dim=0)


class CosineSchedule(NoiseSchedule):
    # Improved schedule presented in https://arxiv.org/pdf/2102.09672
    def __init__(self, timestamps: int, offset=0.008):
        super().__init__(timestamps)
        self.offset = offset

    def f(self, t: int):
        return (
            torch.cos(
                0.5 * np.pi * (t / self.timestamps + self.offset) / (1 + self.offset)
            )
            ** 2
        )

    def get_betas(self) -> torch.Tensor:
        alphas_bar = self.get_alphas_bar()
        betas = torch.zeros(self.timestamps + 1)
        betas[1:] = 1 - alphas_bar[1:] / alphas_bar[:-1]
        return torch.clip(betas, 0.0, 0.999)

    def get_alphas_bar(self) -> torch.Tensor:
        t = torch.linspace(0, self.timestamps, self.timestamps + 1)
        f_0 = np.cos(0.5 * np.pi * self.offset / (1 + self.offset)) ** 2
        return self.f(t) / f_0
