import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet3(nn.Module):
    def __init__(self):
        super(SimpleUNet3, self).__init__()

        # Encoder (downsampling)
        self.enc_conv1 = nn.Conv2d(8, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(256, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (upsampling)
        self.up_conv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec_conv3 = nn.Conv2d(512, 256, 3, padding=1)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)

        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.enc_conv1(x))
        enc2 = F.relu(self.enc_conv2(self.pool(enc1)))
        enc3 = F.relu(self.enc_conv3(self.pool(enc2)))
        enc4 = F.relu(self.enc_conv4(self.pool(enc3)))

        # Decoder
        dec3 = torch.cat([self.up_conv4(enc4), enc3], dim=1)
        dec3 = F.relu(self.dec_conv3(dec3))
        dec2 = torch.cat([self.up_conv3(dec3), enc2], dim=1)
        dec2 = F.relu(self.dec_conv2(dec2))
        dec1 = torch.cat([self.up_conv2(dec2), enc1], dim=1)
        dec1 = F.relu(self.dec_conv1(dec1))

        # Final layer (no activation)
        out = self.final_conv(dec1)

        return out.squeeze(1)
