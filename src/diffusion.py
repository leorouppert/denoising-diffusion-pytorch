import torch

from src import schedule


class Diffusion:
    def __init__(self, schedule: schedule.NoiseSchedule, device="cpu"):
        self.schedule = schedule
        self.device = device
    
    def get_schedule(self) -> schedule.NoiseSchedule:
        return self.schedule

    def q(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0, device=self.device)
        input_batch_size = x_0.size(0)
        alphas_bar_t = self.schedule.get_alphas_bar()[t].to(self.device)
        return torch.sqrt(alphas_bar_t).view(input_batch_size, 1, 1, 1) * x_0.to(
            self.device
        ) + torch.sqrt(1 - alphas_bar_t.view(input_batch_size, 1, 1, 1)) * noise, noise
