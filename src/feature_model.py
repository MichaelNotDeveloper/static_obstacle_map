import torch
from torch import nn
import torch.nn.functional as F

try:
    from .utils import make_norm
except ImportError:
    from utils import make_norm


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False
        )
        self.norm1 = make_norm(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = make_norm(out_ch)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)

        y = self.conv1(x)
        y = self.norm1(y)
        y = F.silu(y)

        y = self.conv2(y)
        y = self.norm2(y)

        y = y + residual
        y = F.silu(y)

        return y


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            make_norm(out_ch),
            nn.SiLU(),
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(out_ch, out_ch))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, num_blocks=2):
        super().__init__()

        blocks = [ResBlock(in_ch + skip_ch, out_ch)]
        for _ in range(num_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        x = self.blocks(x)
        return x


class ContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.local = ResBlock(channels, channels, dilation=1)
        self.mid = ResBlock(channels, channels, dilation=2)
        self.large = ResBlock(channels, channels, dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x1 = self.local(x)
        x2 = self.mid(x)
        x3 = self.large(x)

        y = torch.cat([x1, x2, x3], dim=1)
        y = self.fuse(y)

        return y


class RoadPixelFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feature_dim=64,
        base_ch=32,
        num_classes=None,
        normalize_features=True,
    ):
        super().__init__()

        self.normalize_features = normalize_features
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False),
            make_norm(base_ch),
            nn.SiLU(),
            ResBlock(base_ch, base_ch),
        )

        self.enc1 = DownBlock(base_ch, base_ch * 2, num_blocks=2)
        self.enc2 = DownBlock(base_ch * 2, base_ch * 4, num_blocks=2)
        self.enc3 = DownBlock(base_ch * 4, base_ch * 8, num_blocks=3)

        self.context = ContextBlock(base_ch * 8)

        self.dec2 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, num_blocks=2)
        self.dec1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, num_blocks=2)
        self.dec0 = UpBlock(base_ch * 2, base_ch, base_ch, num_blocks=2)

        self.feature_head = nn.Sequential(
            ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, feature_dim, kernel_size=1),
        )

        self.classifier = (
            nn.Conv2d(feature_dim, num_classes, kernel_size=1)
            if num_classes is not None
            else None
        )

    def forward(self, x):
        original_size = x.shape[-2:]

        x0 = self.stem(x)  # [B, C, H, W]
        x1 = self.enc1(x0)  # [B, 2C, H/2, W/2]
        x2 = self.enc2(x1)  # [B, 4C, H/4, W/4]
        x3 = self.enc3(x2)  # [B, 8C, H/8, W/8]

        y = self.context(x3)

        y = self.dec2(y, x2)
        y = self.dec1(y, x1)
        y = self.dec0(y, x0)

        features = self.feature_head(y)

        if features.shape[-2:] != original_size:
            features = F.interpolate(
                features,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        if self.normalize_features:
            features = F.normalize(features, dim=1)

        if self.classifier is None:
            return features

        logits = self.classifier(features)
        return features, logits


if __name__ == "__main__":
    model = RoadPixelFeatureNet(4, 32, 8).to(torch.device("mps"))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    from dataset import BaseDataset, DATA_DIR
    from bev_projection import DepthToBEVProjection
    from tqdm import tqdm

    dataset = BaseDataset(DATA_DIR / "autonomy_yandex_dataset_train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    proj = DepthToBEVProjection(depth_scale=30)
    images, depths, intrinsics, car2cams, static_grids = next(iter(dataloader))
    data = []
    for i in tqdm(range(dataset.num_cams)):
        img_proc = torch.cat([images[i], depths[i]], dim=1).to(torch.device("mps"))
        data.append(model(img_proc))
    res = proj(depths, intrinsics, car2cams, pixel_values=data, pixel_step=6)
