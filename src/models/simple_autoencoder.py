import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvAutoencoder(nn.Module):
    """
    Small convolutional autoencoder for 224x224 RGB images.

    Encoder:
        224 -> 112 -> 56 -> 28 -> 14 latent spatial size
    Decoder:
        14 -> 28 -> 56 -> 112 -> 224

    Latent code is [B, latent_dim, 14, 14].
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 28 -> 14
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder (mirror)
        self.dec = nn.Sequential(
            # 14 -> 28
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 28 -> 56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 56 -> 112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 112 -> 224
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0,1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

