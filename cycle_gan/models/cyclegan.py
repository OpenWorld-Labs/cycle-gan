import torch
from torch import nn
import torch.nn.functional as F

from ..nn.resnet import UpBlock, DownBlock, SameBlock
from ..configs import ResNetConfig
from ..nn.augs import PairedRandomAffine

class Encoder(nn.Module):
    def __init__(self, config : ResNetConfig):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.conv_in = nn.Conv2d(3, ch_0, 1, 1, 0, bias = False)

        blocks = []
        ch = ch_0

        blocks_per_stage = config.encoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in blocks_per_stage[:-1]:
            next_ch = min(ch*2, ch_max)
            blocks.append(DownBlock(ch, next_ch, block_count, total_blocks))
            size = size // 2
            ch = next_ch

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, config : ResNetConfig):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        blocks = []
        ch = ch_max

        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in reversed(blocks_per_stage[:-1]):
            next_ch = max(ch//2, ch_0)
            blocks.append(UpBlock(ch, next_ch, block_count, total_blocks))
            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.conv_out = nn.Conv2d(ch_0, 3, 1, 1, 0, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.conv_out = nn.Conv2d(config.ch_max,config.ch_max,4,1,0,bias=False)
        self.head = nn.Linear(config.ch_max, 1, bias = False)
        self.aug = PairedRandomAffine()

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_out(x)
        x = x.flatten(1)
        x = self.head(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        blocks_per_stage = config.encoder_blocks_per_stage
        self.mid = SameBlock(config.ch_max, config.ch_max, blocks_per_stage[-1], len(blocks_per_stage))
        self.decoder = Decoder(config)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x)
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