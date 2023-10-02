from .ext_unet import EUNet
from .simple_unet import ChangeDetectionNet
from .simple_unet2 import SimpleUNet2
from .simple_unet3 import SimpleUNet3
from .unet import UNet


def get_model(net, net_reduction, binary_change_detection):
    if binary_change_detection:
        out_channels = 1
    else:
        out_channels = 43
    if net == "unet":
        model = UNet(8, out_channels, reduction_factor=net_reduction)
    elif net == "sunet":
        if not binary_change_detection:
            raise ValueError("sunet only supports binary change detection")
        model = ChangeDetectionNet()
    elif net == "sunet2":
        model = SimpleUNet2(out_channels=out_channels)
    elif net == "sunet3":
        if not binary_change_detection:
            raise ValueError("sunet3 only supports binary change detection")
        model = SimpleUNet3()
    elif net == "eunet":
        model = EUNet(in_channels=8, out_channels=out_channels)
    else:
        raise ValueError("Invalid net name")
    return model
