"""R(2+1)D-34, IG65M + Kinetics, adapted to this competition's 4-channel input.

Architecture is torchvision's VideoResNet with BasicBlock, Conv2Plus1D, layers
[3,4,6,3] and R2Plus1dStem, matching moabitcoin/ig65m-pytorch. Two deviations from
torchvision defaults are required for the Caffe2-converted weights to load:
  - stem and downsample convs use the Caffe2 shapes,
  - BatchNorm3d uses eps=1e-3, momentum=0.9.

Input adaptation: the checkpoint's stem conv takes 3 channels (RGB). This dataset
gives 4 (Depth_Color 3 + IR 1). The 4th channel is initialised with the mean of the
three pretrained ones, which is what the public 0.716 notebook does, so IR starts as
a grey-world copy of the RGB filters rather than random noise.

RULES NOTE: this is a 63.7M-parameter backbone pretrained on 65M Instagram videos
plus Kinetics-400. The Small Model Track rules read "no large pretrained backbones".
Eligibility is an organizer question, not one this file answers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video.resnet import (
    BasicBlock, Conv2Plus1D, R2Plus1dStem, VideoResNet,
)

WEIGHTS_URL = ("https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/"
               "r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth")


def _r2plus1d_34(num_classes: int = 400) -> VideoResNet:
    m = VideoResNet(block=BasicBlock,
                    conv_makers=[Conv2Plus1D] * 4,
                    layers=[3, 4, 6, 3],
                    stem=R2Plus1dStem)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    # Caffe2 deviation, verified against the checkpoint's actual tensor shapes.
    # torchvision reuses the BLOCK's input midplanes for conv2:
    #     midplanes = (in*out*27) // (in*9 + 3*out)   with in = the block's inplanes
    # The Caffe2 model computes conv2's midplanes from (planes, planes) instead, so
    # only conv2 of each layer's FIRST block differs. conv1 already matches:
    #     layer2.0.conv1 230 OK / conv2 230 -> 288
    #     layer3.0.conv1 460 OK / conv2 460 -> 576
    #     layer4.0.conv1 921 OK / conv2 921 -> 1152
    m.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    m.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    m.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm3d):
            mod.eps, mod.momentum = 1e-3, 0.9
    return m


def _adapt_stem(conv: nn.Conv3d, channels: int) -> nn.Conv3d:
    """3-channel stem -> `channels`, extra channels seeded with the RGB mean."""
    new = nn.Conv3d(channels, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        if channels > 3:
            new.weight[:, 3:] = conv.weight.mean(1, keepdim=True).expand(
                -1, channels - 3, -1, -1, -1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class R2Plus1D34(nn.Module):
    """Backbone + a fresh 40-class head. Forward takes (B, T, C, H, W)."""

    def __init__(self, n_classes: int = 40, channels: int = 4,
                 pretrained: bool = True, drop: float = 0.3,
                 weights_path: str | None = None):
        super().__init__()
        net = _r2plus1d_34(400)
        if pretrained:
            sd = (torch.load(weights_path, map_location="cpu", weights_only=True)
                  if weights_path else
                  torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu",
                                                     progress=True))
            net.load_state_dict(sd, strict=True)   # strict: a partial load is the bug
            print(f"[r2p1d] loaded IG65M+Kinetics weights, "
                  f"{sum(p.numel() for p in net.parameters()):,} params")
        net.stem[0] = _adapt_stem(net.stem[0], channels)
        feats = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(feats, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) -> (B, C, T, H, W)
        return self.head(self.encoder(x.permute(0, 2, 1, 3, 4)))


if __name__ == "__main__":
    import sys
    m = R2Plus1D34(pretrained="--no-weights" not in sys.argv)
    n = sum(p.numel() for p in m.parameters())
    print(f"total params : {n:,}  ({n/1e6:.1f}M)")
    print(f"fp32         : {n*4/1e6:.1f} MB   int8: {n/1e6:.1f} MB   "
          f"int6: {n*6/8/1e6:.1f} MB   int5: {n*5/8/1e6:.1f} MB")
    x = torch.randn(2, 16, 4, 128, 128)
    with torch.no_grad():
        y = m(x)
    print(f"forward      : {tuple(x.shape)} -> {tuple(y.shape)}")
