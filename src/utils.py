import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith((".jpg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, ""


def display_images(image_list, rows, cols, save=False):
    """Displays images in the list as a grid.

    Automatically determines if the images are BW or RGB.
    """
    if len(image_list) > rows * cols:
        print("Some images won't be displayed")

    _, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(image_list):
            img = image_list[i]

            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            if img.shape[0] == 3:
                ax.imshow(img.transpose(1, 2, 0))
            elif img.shape[0] == 1:
                ax.imshow(img.transpose(1, 2, 0), cmap="gray")
            elif img.shape[2] == 3:
                ax.imshow(img)
            elif img.shape[2] == 1:
                ax.imshow(img, cmap="gray")

            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout(pad=0.1)
    if save:
        plt.savefig("images.png", dpi=300)
    plt.show()
