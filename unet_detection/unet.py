from .blocks import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, reduction_factor=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64 // reduction_factor)
        self.down1 = Down(64 // reduction_factor, 128 // reduction_factor)
        self.down2 = Down(128 // reduction_factor, 256 // reduction_factor)
        self.down3 = Down(256 // reduction_factor, 512 // reduction_factor)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // reduction_factor, (1024 // reduction_factor) // factor)
        self.up1 = Up(
            1024 // reduction_factor, (512 // reduction_factor) // factor, bilinear
        )
        self.up2 = Up(
            512 // reduction_factor, (256 // reduction_factor) // factor, bilinear
        )
        self.up3 = Up(
            256 // reduction_factor, (128 // reduction_factor) // factor, bilinear
        )
        self.up4 = Up(128 // reduction_factor, (64 // reduction_factor), bilinear)
        self.outc = OutConv(64 // reduction_factor, n_classes)
        self.sigmoid = nn.Sigmoid()  # remove for classification

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # probs = self.sigmoid(logits)
        return logits.squeeze(1)  # added squeeze

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)
