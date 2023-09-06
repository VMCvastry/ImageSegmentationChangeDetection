from .ext_unet import EUNet
from .simple_unet import ChangeDetectionNet
from .simple_unet2 import SimpleUNet2
from .simple_unet3 import SimpleUNet3
from .unet import UNet


def get_model(net, net_reduction):
    if net == "unet":
        model = UNet(8, 1, reduction_factor=net_reduction)
    elif net == "sunet":
        model = ChangeDetectionNet()
    elif net == "sunet2":
        model = SimpleUNet2()
    elif net == "sunet3":
        model = SimpleUNet3()
    elif net == "eunet":
        model = EUNet(in_channels=8, out_channels=1)
    else:
        raise ValueError("Invalid net name")
    return model
