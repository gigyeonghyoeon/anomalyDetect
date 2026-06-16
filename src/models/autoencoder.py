"""1-channel Convolutional AutoEncoder for chest X-ray."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoEncoder(nn.Module):
    """224x224 grayscale -> latent -> reconstruction."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 128, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        feat_size = image_size // 32
        self.feat_dim = 512 * feat_size * feat_size
        self.fc_enc = nn.Linear(self.feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.feat_dim)
        self.decoder = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat_size = self.image_size // 32
        h = self.fc_dec(z).view(-1, 512, feat_size, feat_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    @staticmethod
    def reconstruction_error(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        err = (x - recon).pow(2).flatten(1).mean(dim=1)
        return err
