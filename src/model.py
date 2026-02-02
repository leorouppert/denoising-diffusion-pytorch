import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sinusoidal_positional_encoding(t, d_model):
    batch_size = t.size(0)
    pe = torch.zeros(batch_size, d_model, device=t.device)
    position = t.unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=t.device).float()
        * -(math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, groups, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(p=dropout, inplace=True),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.embedding_dim = embedding_dim

    def forward(self, x, time):
        pos_emb = sinusoidal_positional_encoding(time, self.embedding_dim)
        pos_emb = pos_emb[:, : x.size(1), None, None]

        x = x + pos_emb

        residual = self.res_conv(x)

        x = self.block(x) + residual
        x = F.relu(x)

        return x


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj1(x)
        x = rearrange(x, "b L (C H K) -> K b H L C", K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout_prob
        )
        x = rearrange(x, "b H (h w) C -> b h w (C H)", h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, "b h w C -> b C h w")


class UNetBlock(nn.Module):
    def __init__(
        self,
        attention: bool,
        in_channels,
        out_channels,
        embedding_dim,
        groups,
        dropout=0.1,
    ):
        super().__init__()

        self.block1 = ConvResBlock(
            out_channels, out_channels, embedding_dim, groups, dropout
        )
        self.block2 = ConvResBlock(
            out_channels, out_channels, embedding_dim, groups, dropout
        )

        self.conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.attention = (
            Attention(out_channels, 8, dropout) if attention else nn.Identity()
        )

    def forward(self, x, time):
        x = self.conv(x)
        x = self.block1(x, time)
        x = self.attention(x)
        x = self.block2(x, time)

        return x


class UNetWithPosition(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=64,
        groups=64 // 2,
        embedding_dim=64 * 16,
    ):
        super().__init__()

        self.encoder1 = UNetBlock(
            False, in_channels, base_channels, embedding_dim, groups
        )
        self.encoder2 = UNetBlock(
            False, base_channels, base_channels * 2, embedding_dim, groups
        )
        self.encoder3 = UNetBlock(
            True, base_channels * 2, base_channels * 4, embedding_dim, groups
        )
        self.encoder4 = UNetBlock(
            False, base_channels * 4, base_channels * 8, embedding_dim, groups
        )

        self.bottleneck = UNetBlock(
            False, base_channels * 8, base_channels * 16, embedding_dim, groups
        )

        self.pool = nn.MaxPool2d(2)

        self.decoder4 = UNetBlock(
            False,
            base_channels * (8 + 8),
            base_channels * (8 + 8),
            embedding_dim,
            groups,
        )
        self.decoder3 = UNetBlock(
            True,
            base_channels * (8 + 4),
            base_channels * (8 + 4),
            embedding_dim,
            groups,
        )
        self.decoder2 = UNetBlock(
            False,
            base_channels * (6 + 2),
            base_channels * (6 + 2),
            embedding_dim,
            groups,
        )

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=4, stride=2, padding=1
        )
        self.up3 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=4, stride=2, padding=1
        )
        self.up2 = nn.ConvTranspose2d(
            base_channels * 12, base_channels * 6, kernel_size=4, stride=2, padding=1
        )
        self.up1 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * 5, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, time):
        # Encoder
        enc1 = self.encoder1(x, time)
        enc2 = self.encoder2(self.pool(enc1), time)
        enc3 = self.encoder3(self.pool(enc2), time)
        enc4 = self.encoder4(self.pool(enc3), time)

        # Bottleneck
        middle = self.bottleneck(self.pool(enc4), time)

        # Decoder
        dec4 = self.decoder4(torch.cat((self.up4(middle), enc4), dim=1), time)
        dec3 = self.decoder3(torch.cat((self.up3(dec4), enc3), dim=1), time)
        dec2 = self.decoder2(torch.cat((self.up2(dec3), enc2), dim=1), time)
        dec1 = torch.cat((self.up1(dec2), enc1), dim=1)

        out = self.final_conv(dec1)

        return out
