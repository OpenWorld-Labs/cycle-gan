import torch
from torch import nn
import torch.nn.functional as F

from ..nn.resnet import UpBlock, DownBlock, SameBlock
from ..configs import ResNetConfig
from ..nn.augs import PairedRandomAffine

class Generator(nn.Module):
    def __init__(self, config : ResNetConfig):
        super().__init__()

        self.conv_in = nn.Conv2d(3, 64, 1, 1, 0, bias = False)
        self.blocks = nn.Sequential(
            DownBlock(64,128,1,10),
            DownBlock(128,256,1,10),
            SameBlock(256,256,1,10),
            SameBlock(256,256,1,10),
            SameBlock(256,256,1,10),
            SameBlock(256,256,1,10),
            SameBlock(256,256,1,10),
            SameBlock(256,256,1,10),
            SameBlock(256,256,1,10),
            UpBlock(256,128,1,10),
            UpBlock(128,64,1,10)
        )
        self.conv_out = nn.Conv2d(64,3,1,1,0,bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)

        return x
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_in = nn.Conv2d(3,64,1,1,0,bias=False)
        self.encoder = nn.Sequential(
            DownBlock(64,128,1,10),
            DownBlock(128,256,1,10),
            DownBlock(256,256,1,10),
            DownBlock(256,256,1,10)
        )
        self.conv_out = nn.Conv2d(256,1,1,1,0,bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.encoder(x)
        x = self.conv_out(x)
        x = x.mean([2,3])
        return x

if __name__ == "__main__":
    from ..configs import Config

    cfg_path = "configs/base.yml"
    cfg = Config.from_yaml(cfg_path).model

    model = Generator(cfg).cuda().bfloat16()
    x = torch.randn(1,3,256,256).cuda().bfloat16()

    with torch.no_grad():
        y = model(x)
        print(y.shape)