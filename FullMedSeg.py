import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ------------------------------------
# StdConv: Conv + GN + SiLU
# ------------------------------------
class StdConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.gn = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


# ------------------------------------
# FullMedSeg (clean version)
# ------------------------------------
class FullMedSeg(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, channel=32, pretrained=True):
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

        # -------- Decoder --------
        self.p1 = StdConv(64, channel)

        self.p2a = StdConv(256, channel * 4)
        self.p2b = StdConv(channel * 4, channel * 2)

        self.p3a = StdConv(512, channel * 8)
        self.p3b = StdConv(channel * 8, channel * 4)
        self.p3c = StdConv(channel * 4, channel * 2)

        self.p4a = StdConv(1024, channel * 16)
        self.p4b = StdConv(channel * 16, channel * 8)
        self.p4c = StdConv(channel * 8, channel * 4)
        self.p4d = StdConv(channel * 4, channel * 2)

        # -------- Alignment Convs --------
        self.a1 = StdConv(channel, channel)
        self.a2 = StdConv(channel * 2, channel)
        self.a3 = StdConv(channel * 2, channel)
        self.a4 = StdConv(channel * 2, channel)

        # -------- Fuse --------
        self.fuse = StdConv(channel * 4, channel)

        # -------- High-Res Branch --------
        self.input_block = StdConv(input_channel, channel)
        self.final_fuse = StdConv(channel * 2, channel)

        self.out = nn.Conv2d(channel, num_classes, kernel_size=1)

    # ------------------------------------
    def forward(self, x):
        # x: [B, 3, 256, 256]

        # -------- High-Res --------
        f0 = self.input_block(x)            # [B, 32, 256, 256]

        # -------- Encoder --------
        f1 = self.enc_conv0(x)              # [B, 64, 128, 128]
        x0 = self.pool0(f1)                 # [B, 64, 64, 64]

        f2 = self.denseblock1(x0)           # [B, 256, 64, 64]
        x1 = self.transition1(f2)           # [B, 128, 32, 32]

        f3 = self.denseblock2(x1)           # [B, 512, 32, 32]
        x2 = self.transition2(f3)           # [B, 256, 16, 16]

        f4 = self.denseblock3(x2)           # [B, 1024, 16, 16]

        # -------- Decoder --------
        f1d = F.interpolate(self.p1(f1), scale_factor=2, mode="bilinear", align_corners=False)

        f2d = self.p2b(
            F.interpolate(self.p2a(f2), scale_factor=2, mode="bilinear", align_corners=False)
        )
        f2d = F.interpolate(f2d, scale_factor=2, mode="bilinear", align_corners=False)

        f3d = self.p3c(
            F.interpolate(
                self.p3b(
                    F.interpolate(self.p3a(f3), scale_factor=2, mode="bilinear", align_corners=False)
                ),
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
        )
        f3d = F.interpolate(f3d, scale_factor=2, mode="bilinear", align_corners=False)

        f4d = self.p4d(
            F.interpolate(
                self.p4c(
                    F.interpolate(
                        self.p4b(
                            F.interpolate(self.p4a(f4), scale_factor=2, mode="bilinear", align_corners=False)
                        ),
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False
                    )
                ),
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
        )
        f4d = F.interpolate(f4d, scale_factor=2, mode="bilinear", align_corners=False)

        # -------- Align --------
        f1a = self.a1(f1d)
        f2a = self.a2(f2d)
        f3a = self.a3(f3d)
        f4a = self.a4(f4d)

        # -------- Fuse --------
        f = torch.cat([f1a, f2a, f3a, f4a], dim=1)   # [B, 128, 256, 256]
        f = self.fuse(f)                             # [B, 32, 256, 256]

        # -------- High-Res Fuse --------
        f = torch.cat([f, f0], dim=1)                # [B, 64, 256, 256]
        f = self.final_fuse(f)                       # [B, 32, 256, 256]

        return self.out(f)


# ------------------------------------
# Quick test
# ------------------------------------
if __name__ == "__main__":
    model = FullMedSeg(num_classes=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(y.shape)  # [2, 1, 256, 256]
