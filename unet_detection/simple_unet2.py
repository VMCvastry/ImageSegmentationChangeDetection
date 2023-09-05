import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet2(nn.Module):
    def __init__(self):
        super(SimpleUNet2, self).__init__()

        # Encoder (downsampling)
        self.enc_conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (upsampling)
        self.up_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(64, 32, 3, padding=1)

        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.enc_conv1(x))
        enc2 = F.relu(self.enc_conv2(self.pool(enc1)))

        # Decoder
        dec1 = torch.cat([self.up_conv2(enc2), enc1], dim=1)
        dec1 = F.relu(self.dec_conv1(dec1))

        # Final layer (no activation)
        out = self.final_conv(dec1)

        return out.squeeze(1)
