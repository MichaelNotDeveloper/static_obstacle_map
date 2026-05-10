import torch
from torch import nn
import torch.nn.functional as F

try:
    from .utils import make_norm
except ImportError:
    from utils import make_norm


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False
        )
        self.norm1 = make_norm(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False
        )
        self.norm2 = make_norm(out_ch)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            make_norm(out_ch),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[ResBlock(out_ch, out_ch) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.blocks(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        blocks = [ResBlock(in_ch + skip_ch, out_ch)]
        blocks.extend(ResBlock(out_ch, out_ch) for _ in range(num_blocks - 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, channels: int):
        super().__init__()

        self.local = ResBlock(channels, channels, dilation=1)
        self.mid = ResBlock(channels, channels, dilation=2)
        self.large = ResBlock(channels, channels, dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat(
            [
                self.local(x),
                self.mid(x),
                self.large(x),
            ],
            dim=1,
        )
        return self.fuse(y)


class BEVSegNet(nn.Module):
    def __init__(self, in_ch: int = 64, base_ch: int = 32):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            make_norm(base_ch),
            nn.SiLU(),
            ResBlock(base_ch, base_ch),
        )

        self.enc1 = DownBlock(base_ch, base_ch * 2, num_blocks=2)
        self.enc2 = DownBlock(base_ch * 2, base_ch * 4, num_blocks=2)
        self.enc3 = DownBlock(base_ch * 4, base_ch * 4, num_blocks=2)

        self.context = ContextBlock(base_ch * 4)
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 4, base_ch * 4),
            ResBlock(base_ch * 4, base_ch * 4, dilation=2),
        )

        self.dec2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 4, num_blocks=2)
        self.dec1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, num_blocks=2)
        self.dec0 = UpBlock(base_ch * 2, base_ch, base_ch, num_blocks=2)

        self.head = nn.Sequential(
            ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        x0 = self.stem(x)  # [B, 32, 188, 126]
        x1 = self.enc1(x0)  # [B, 64, 94, 63]
        x2 = self.enc2(x1)  # [B, 128, 47, 32]
        x3 = self.enc3(x2)  # [B, 64, 24, 16] for base_ch=16

        y = self.context(x3)
        y = self.bottleneck(y)

        y = self.dec2(y, x2)  # [B, 128, 47, 32]
        y = self.dec1(y, x1)  # [B, 64, 94, 63]
        y = self.dec0(y, x0)  # [B, 32, 188, 126]

        logits = self.head(y)  # [B, 1, 188, 126]

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


if __name__ == "__main__":
    from dataset import BaseDataset, DATA_DIR
    from bev_projection import DepthToBEVProjection
    from tqdm import tqdm
    from feature_model import RoadPixelFeatureNet
    from utils import binary_bce_loss_with_ignore

    dataset = BaseDataset(DATA_DIR / "autonomy_yandex_dataset_train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cpu")
    proj = DepthToBEVProjection(depth_scale=30).to(device)
    feature_model = RoadPixelFeatureNet(4, 32, 8).to(device)
    mapping_model = BEVSegNet(32).to(device)
    total_params = sum(p.numel() for p in mapping_model.parameters())
    print(f"Total Parameters Mapping: {total_params}")
    images, depths, intrinsics, car2cams, static_grids = next(iter(dataloader))
    data = []
    for i in tqdm(range(dataset.num_cams)):
        images[i] = images[i].to(device)
        depths[i] = depths[i].to(device)
        img_proc = torch.cat([images[i], depths[i]], dim=1)
        data.append(feature_model(img_proc))
    feat_map = proj(depths, intrinsics, car2cams, pixel_values=data, pixel_step=6)
    result = mapping_model(feat_map)
    loss = binary_bce_loss_with_ignore(result, static_grids, ignore_index=-1)
    loss.backward()
    for name, param in feature_model.named_parameters():
        if not param.requires_grad:
            continue
        print(param.grad.detach().norm().item())
