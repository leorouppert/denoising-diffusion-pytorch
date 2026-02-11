import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.embedding_dim = embedding_dim

        self.time_proj = nn.Linear(embedding_dim, out_channels)

    def forward(self, x, time_emb):
        time_emb = self.time_proj(F.silu(time_emb))
        time_emb = time_emb[:, :, None, None]

        residual = self.res_conv(x)

        x = self.block1(x)
        x = x + time_emb
        x = self.block2(x)

        x = x + residual
        x = F.silu(x)

        return x


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=C, num_heads=num_heads, dropout=dropout_prob, batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x, _ = self.mha(x, x, x, need_weights=False)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


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

    def forward(self, x, time_emb):
        x = self.conv(x)
        x = self.block1(x, time_emb)
        x = self.attention(x)
        x = self.block2(x, time_emb)

        return x


class UNetWithPosition(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=64,
        groups=64 // 2,
        embedding_dim=64 * 4,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

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
            base_channels * 8,
            embedding_dim,
            groups,
        )
        self.decoder3 = UNetBlock(
            True,
            base_channels * (4 + 4),
            base_channels * 4,
            embedding_dim,
            groups,
        )
        self.decoder2 = UNetBlock(
            False,
            base_channels * (2 + 2),
            base_channels * 2,
            embedding_dim,
            groups,
        )

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=4, stride=2, padding=1
        )
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, time):
        time_emb = sinusoidal_positional_encoding(time, self.embedding_dim)
        time_emb = self.time_mlp(time_emb)

        # Encoder
        enc1 = self.encoder1(x, time_emb)
        enc2 = self.encoder2(self.pool(enc1), time_emb)
        enc3 = self.encoder3(self.pool(enc2), time_emb)
        enc4 = self.encoder4(self.pool(enc3), time_emb)

        # Bottleneck
        middle = self.bottleneck(self.pool(enc4), time_emb)

        # Decoder
        dec4 = self.decoder4(torch.cat((self.up4(middle), enc4), dim=1), time_emb)
        dec3 = self.decoder3(torch.cat((self.up3(dec4), enc3), dim=1), time_emb)
        dec2 = self.decoder2(torch.cat((self.up2(dec3), enc2), dim=1), time_emb)
        dec1 = torch.cat((self.up1(dec2), enc1), dim=1)

        out = self.final_conv(dec1)

        return out
