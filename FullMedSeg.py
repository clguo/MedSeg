import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ------------------------------------
# StdConv: Standard Conv + GN + SiLU
# ------------------------------------
class StdConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, rate=0.2, dilation=1):
        super().__init__()

        # Standard Convolution instead of Depthwise Separable
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=k,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.gn = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x



# ------------------------------------
class FullMedSeg(nn.Module):
    def __init__(self,input_channel=3, num_classes=1, channel=32, rate=0.15, pretrained=True):
        super().__init__()

        # -------- DenseNet121 Encoder --------
        try:
            backbone = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            ).features
        except Exception:
            backbone = models.densenet121(pretrained=pretrained).features

        self.enc_conv0 = nn.Sequential(
            backbone.conv0,
            backbone.norm0,
            backbone.relu0
        )
        self.pool0 = backbone.pool0

        self.denseblock1 = backbone.denseblock1
        self.transition1 = backbone.transition1

        self.denseblock2 = backbone.denseblock2
        self.transition2 = backbone.transition2

        self.denseblock3 = backbone.denseblock3

        # -------- Decoder Standard Conv Blocks --------
        # All using StdConv instead of SepConv
        self.p1 = StdConv(64, channel, rate=rate)

        self.p2a = StdConv(256, channel * 4, rate=rate)
        self.p2b = StdConv(channel * 4, channel * 2, rate=rate)

        self.p3a = StdConv(512, channel * 8, rate=rate)
        self.p3b = StdConv(channel * 8, channel * 4, rate=rate)
        self.p3c = StdConv(channel * 4, channel * 2, rate=rate)

        self.p4a = StdConv(1024, channel * 16, rate=rate)
        self.p4b = StdConv(channel * 16, channel * 8, rate=rate)
        self.p4c = StdConv(channel * 8, channel * 4, rate=rate)
        self.p4d = StdConv(channel * 4, channel * 2, rate=rate)

        # -------- ASPP Style Convs (NO DILATION) --------
        # Using standard Conv instead of SepConv
        self.a4 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, dilation=1, padding=1, bias=False),
            nn.GroupNorm(8, channel),
            nn.SiLU()
        )

        self.a3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, dilation=1, padding=1, bias=False),
            nn.GroupNorm(8, channel),
            nn.SiLU()
        )

        self.a2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, dilation=1, padding=1, bias=False),
            nn.GroupNorm(8, channel),
            nn.SiLU()
        )

        self.a1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, dilation=1, padding=1, bias=False),
            nn.GroupNorm(8, channel),
            nn.SiLU()
        )

        # -------- Fuse Layer: Standard Conv --------
        self.fuse = nn.Sequential(
            nn.Conv2d(channel * 4, channel, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channel),
            nn.SiLU()
        )
        
        # -------- High-Res Branch (Level 0) --------
        self.input_block = StdConv(input_channel, channel, k=3, rate=rate)
        self.final_fuse = StdConv(channel + channel, channel, rate=rate)
        
        self.out = nn.Conv2d(channel, num_classes, kernel_size=1)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    def forward(self, x):
        # x: [B, 3, 256, 256]
        
        # -------- High-Res Branch --------
        f0 = self.input_block(x)
        # f0: [B, 32, 256, 256]

        # -------- Encoder --------
        f1 = self.enc_conv0(x)
        # [B, 64, 128, 128]

        x0 = self.pool0(f1)
        # [B, 64, 64, 64]

        f2 = self.denseblock1(x0)
        # [B, 256, 64, 64]

        x1 = self.transition1(f2)
        # [B, 128, 32, 32]

        f3 = self.denseblock2(x1)
        # [B, 512, 32, 32]

        x2 = self.transition2(f3)
        # [B, 256, 16, 16]

        f4 = self.denseblock3(x2)
        # [B, 1024, 16, 16]

        # -------- Decoder f1 --------
        f1d = self.p1(f1)
        f1d = F.interpolate(f1d, scale_factor=2, mode="bilinear", align_corners=False)

        # -------- Decoder f2 --------
        f2d = self.p2a(f2)
        f2d = F.interpolate(f2d, scale_factor=2, mode="bilinear", align_corners=False)
        f2d = self.p2b(f2d)
        f2d = F.interpolate(f2d, scale_factor=2, mode="bilinear", align_corners=False)

        # -------- Decoder f3 --------
        f3d = self.p3a(f3)
        f3d = F.interpolate(f3d, scale_factor=2, mode="bilinear", align_corners=False)
        f3d = self.p3b(f3d)
        f3d = F.interpolate(f3d, scale_factor=2, mode="bilinear", align_corners=False)
        f3d = self.p3c(f3d)
        f3d = F.interpolate(f3d, scale_factor=2, mode="bilinear", align_corners=False)

        # -------- Decoder f4 --------
        f4d = self.p4a(f4)
        f4d = F.interpolate(f4d, scale_factor=2, mode="bilinear", align_corners=False)
        f4d = self.p4b(f4d)
        f4d = F.interpolate(f4d, scale_factor=2, mode="bilinear", align_corners=False)
        f4d = self.p4c(f4d)
        f4d = F.interpolate(f4d, scale_factor=2, mode="bilinear", align_corners=False)
        f4d = self.p4d(f4d)
        f4d = F.interpolate(f4d, scale_factor=2, mode="bilinear", align_corners=False)


        f1a = self.a1(f1d)
        f2a = self.a2(f2d)
        f3a = self.a3(f3d)
        f4a = self.a4(f4d)

        # -------- fuse --------
        f = torch.cat([f1a, f2a, f3a, f4a], dim=1)
        f = self.fuse(f)
        # f: [B, 32, 256, 256]
        
        # -------- Fuse with High-Res Branch --------
        f = torch.cat([f, f0], dim=1)
        # f: [B, 64, 256, 256]
        f = self.final_fuse(f)
        # f: [B, 32, 256, 256]
        
        out = self.out(f)

        return out
