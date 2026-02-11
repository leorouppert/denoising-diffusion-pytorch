import torch
from torch.utils.data import DataLoader

from src import diffusion
from tools import sample


def train(
    model: torch.nn.Module,
    diffusion: diffusion.Diffusion,
    train_loader: DataLoader,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    lr_scheduler=None,
    use_mixed_precision=False,
    autocast_dtype=torch.float16,
    device="cpu",
):
    train_losses = []
    samples = []

    scaler = torch.GradScaler(
        enabled=(use_mixed_precision and autocast_dtype == torch.float16)
    )

    schedule = diffusion.get_schedule()
    T = schedule.get_timestamps()
    n_channels = None
    img_size = None

    device_type_str = device.type if isinstance(device, torch.device) else device

    for epoch in range(n_epochs):
        train_loss = 0

        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device=device)
            current_batch_size = batch.size(0)
            n_channels = batch.size(1)
            img_size = batch.size(2)
            t = torch.randint(
                low=1, high=T + 1, size=(current_batch_size,), device="cpu"
            )

            noisy_image, noise = diffusion.q(batch, t)

            optimizer.zero_grad()

            with torch.autocast(
                device_type=device_type_str,
                dtype=autocast_dtype,
                enabled=use_mixed_precision,
            ):
                output = model(noisy_image, t.to(device=device))
                loss = criterion(output, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * current_batch_size

        if lr_scheduler is not None:
            lr_scheduler.step()

        train_loss /= len(train_loader.sampler)
        train_losses.append(train_loss)

        print(f"Epoch: {epoch + 1} | Training Loss: {train_loss:.5f}")
        torch.save(model.state_dict(), "model.pt")

        if (epoch + 1) % max(1, (n_epochs // 10)) == 0 or epoch == n_epochs - 1:
            model.eval()
            print("Sampling...")
            samples.append(
                (
                    epoch + 1,
                    sample.sample_images(
                        model,
                        schedule,
                        n_channels=n_channels,
                        n_images=4,
                        img_size=img_size,
                        device=device,
                    ),
                )
            )
    torch.save(model.state_dict(), "final_model.pt")
    return train_losses, samples
