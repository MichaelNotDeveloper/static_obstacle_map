from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .dataset import (
        CAMERA_NAMES,
        INTRINSICS_NAMES,
        CAR2CAM_NAMES,
        CALIBRATION_IMAGE_SHAPE,
    )
except ImportError:
    try:
        from dataset import (
            CAMERA_NAMES,
            INTRINSICS_NAMES,
            CAR2CAM_NAMES,
            CALIBRATION_IMAGE_SHAPE,
        )
    except ImportError:
        from src.dataset import (
            CAMERA_NAMES,
            INTRINSICS_NAMES,
            CAR2CAM_NAMES,
            CALIBRATION_IMAGE_SHAPE,
        )


class DepthToBEVProjection(nn.Module):
    def __init__(
        self,
        bev_shape=(188, 126),
        meters_per_pixel=150.0 / 188.0,
        x_min=0.0,
        y_min=None,
        min_depth=0.1,
        max_depth=150.0,
        depth_scale=1.0,
        depth_bias=0.0,
        raw_depth_is_inverse=False,
        inverse_depth_eps=1e-6,
        extrinsics_are_car_to_cam=True,
        bev_y_sign=-1.0,
        normalize_features=True,
        calibration_image_shape=CALIBRATION_IMAGE_SHAPE,
    ):
        super().__init__()

        self.bev_shape = bev_shape
        self.meters_per_pixel = float(meters_per_pixel)
        self.x_min = float(x_min)
        self.y_min = (
            -bev_shape[1] * self.meters_per_pixel / 2.0
            if y_min is None
            else float(y_min)
        )
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.raw_depth_is_inverse = raw_depth_is_inverse
        self.inverse_depth_eps = float(inverse_depth_eps)
        self.extrinsics_are_car_to_cam = extrinsics_are_car_to_cam
        self.bev_y_sign = float(bev_y_sign)
        self.normalize_features = normalize_features
        self.calibration_image_shape = calibration_image_shape

        self.register_buffer(
            "log_depth_scale", torch.log(torch.tensor(float(depth_scale)))
        )
        self.register_buffer("depth_bias", torch.tensor(float(depth_bias)))

    def forward(
        self,
        depths,
        intrinsics,
        extrinsics,
        pixel_values=None,
        pixel_step=1,
        return_debug=False,
    ):
        depths = self._stack_camera_tensor(depths)  # [B, 1, H, W]
        intrinsics = self._stack_camera_tensor(intrinsics).to(
            device=depths.device, dtype=depths.dtype
        )  # []
        extrinsics = self._stack_camera_tensor(extrinsics).to(
            device=depths.device, dtype=depths.dtype
        )

        if pixel_values is None:
            pixel_values = torch.ones_like(depths)
        else:
            pixel_values = self._stack_camera_tensor(pixel_values).to(
                device=depths.device, dtype=depths.dtype
            )

        points_car, valid, values, depth = self.unproject_to_car(
            depths, intrinsics, extrinsics, pixel_values, pixel_step
        )
        bev, weights = self.splat_to_bev(points_car, valid, values)

        if self.normalize_features:
            bev = bev / weights.clamp_min(1e-6)

        if return_debug:
            points_bev = points_car.clone()
            points_bev[..., 1] = points_bev[..., 1] * self.bev_y_sign
            return {
                "bev": bev,
                "weights": weights,
                "points_car": points_car,
                "points_bev": points_bev,
                "valid": valid,
                "values": values,
                "depths": depth,
            }

        return bev

    def unproject_to_car(
        self, depths, intrinsics, extrinsics, pixel_values=None, pixel_step=1
    ):
        batch, num_cams, _, height, width = depths.shape
        depth = depths[:, :, :, ::pixel_step, ::pixel_step]

        values = (
            torch.ones_like(depth)
            if pixel_values is None
            else pixel_values[:, :, :, ::pixel_step, ::pixel_step]
        )

        depth = self.depth_to_meters(depth)
        depth = depth.clamp(self.min_depth, self.max_depth)

        pixels = self._make_pixel_grid(
            height, width, pixel_step, depth.device, depth.dtype
        )
        pixels = pixels.view(1, 1, *pixels.shape)

        intrinsic = self._scaled_intrinsics(intrinsics, height, width)
        fx = intrinsic[:, :, 0, 0].view(batch, num_cams, 1, 1)
        fy = intrinsic[:, :, 1, 1].view(batch, num_cams, 1, 1)
        cx = intrinsic[:, :, 0, 2].view(batch, num_cams, 1, 1)
        cy = intrinsic[:, :, 1, 2].view(batch, num_cams, 1, 1)

        u = pixels[..., 0]
        v = pixels[..., 1]
        z_cam = depth.squeeze(2)
        x_cam = (u - cx) / fx.clamp_min(1e-6) * z_cam
        y_cam = (v - cy) / fy.clamp_min(1e-6) * z_cam

        points_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)
        cam_to_car = (
            torch.linalg.inv(extrinsics)
            if self.extrinsics_are_car_to_cam
            else extrinsics
        )
        points_car = torch.matmul(
            points_cam.unsqueeze(-2),
            cam_to_car[:, :, None, None].transpose(-1, -2),
        ).squeeze(-2)[..., :3]

        valid = torch.isfinite(points_car).all(dim=-1)
        valid = valid & torch.isfinite(depth.squeeze(2))
        valid = valid & (depth.squeeze(2) > self.min_depth)

        return points_car, valid, values, depth

    def depth_to_meters(self, depth):
        if self.raw_depth_is_inverse:
            depth = 1.0 / depth.clamp_min(self.inverse_depth_eps)

        return depth * self.log_depth_scale.exp() + self.depth_bias

    def splat_to_bev(self, points_car, valid, values):
        batch, num_cams, height, width, _ = points_car.shape
        value_channels = values.shape[2]
        bev_h, bev_w = self.bev_shape

        rows = (points_car[..., 0] - self.x_min) / self.meters_per_pixel
        y_bev = points_car[..., 1] * self.bev_y_sign
        cols = (y_bev - self.y_min) / self.meters_per_pixel

        values = values.permute(0, 2, 1, 3, 4).reshape(batch, value_channels, -1)
        rows = rows.reshape(batch, -1)
        cols = cols.reshape(batch, -1)
        valid = valid.reshape(batch, -1)

        bev = values.new_zeros(batch, value_channels, bev_h * bev_w)
        weights = values.new_zeros(batch, 1, bev_h * bev_w)

        row0 = torch.floor(rows)
        col0 = torch.floor(cols)

        for d_row in (0, 1):
            for d_col in (0, 1):
                row = row0 + d_row
                col = col0 + d_col

                weight = (1.0 - (rows - row).abs()) * (1.0 - (cols - col).abs())
                inside = (
                    valid
                    & (row >= 0)
                    & (row < bev_h)
                    & (col >= 0)
                    & (col < bev_w)
                    & (weight > 0)
                )
                weight = weight * inside.to(weight.dtype)

                index = (row.long() * bev_w + col.long()).clamp(0, bev_h * bev_w - 1)
                bev.scatter_add_(
                    2,
                    index.unsqueeze(1).expand(-1, value_channels, -1),
                    values * weight.unsqueeze(1),
                )
                weights.scatter_add_(2, index.unsqueeze(1), weight.unsqueeze(1))

        bev = bev.view(batch, value_channels, bev_h, bev_w)
        weights = weights.view(batch, 1, bev_h, bev_w)

        return bev, weights

    def _scaled_intrinsics(self, intrinsics, height, width):
        intrinsic = intrinsics[..., :3, :3].clone()
        if self.calibration_image_shape is None:
            return intrinsic

        calib_h, calib_w = self.calibration_image_shape
        scale_x = width / calib_w
        scale_y = height / calib_h
        intrinsic[..., 0, 0] *= scale_x
        intrinsic[..., 0, 2] *= scale_x
        intrinsic[..., 1, 1] *= scale_y
        intrinsic[..., 1, 2] *= scale_y

        return intrinsic

    @staticmethod
    def _make_pixel_grid(height, width, step, device, dtype):
        ys = torch.arange(0, height, step, device=device, dtype=dtype)
        xs = torch.arange(0, width, step, device=device, dtype=dtype)
        v, u = torch.meshgrid(ys, xs, indexing="ij")

        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _stack_camera_tensor(value):
        if isinstance(value, (list, tuple)):
            value = [torch.as_tensor(sample) for sample in value]
            if value[0].ndim == 2:
                return torch.stack(value, dim=0).unsqueeze(0)
            if value[0].ndim == 3 and value[0].shape[-2:] not in (
                (3, 3),
                (3, 4),
                (4, 4),
            ):
                return torch.stack(value, dim=0).unsqueeze(0)

            return torch.stack(value, dim=1)

        value = torch.as_tensor(value)
        if value.ndim == 2:
            return value.unsqueeze(0).unsqueeze(0)
        if value.ndim == 3:
            if value.shape[0] in (1, 3):
                return value.unsqueeze(0).unsqueeze(0)

            return value.unsqueeze(0)
        if value.ndim == 4:
            if value.shape[-2:] in ((3, 3), (3, 4), (4, 4)):
                return value
            if value.shape[0] == len(CAMERA_NAMES) and value.shape[1] in (1, 3):
                return value.unsqueeze(0)

            return value.unsqueeze(1)

        return value


def plot_debug_batch(
    projector,
    batch,
    save_path,
    camera_ids=None,
    pixel_step=8,
    max_samples=4,
    max_scatter_points=25000,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def stack_cameras(value):
        value = [torch.as_tensor(sample) for sample in value]
        return torch.stack(value, dim=1)

    def resolve_ids(ids):
        if ids is None:
            return list(range(len(CAMERA_NAMES)))
        return [idx if isinstance(idx, int) else CAMERA_NAMES.index(idx) for idx in ids]

    def unnormalize(images):
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

    images, depths, intrinsics, car2cams, static_grids = batch
    images = unnormalize(stack_cameras(images))
    depths = stack_cameras(depths)
    intrinsics = stack_cameras(intrinsics).to(dtype=depths.dtype)
    car2cams = stack_cameras(car2cams).to(dtype=depths.dtype)
    gt_grids = stack_cameras(static_grids)[:, 0]

    camera_ids = resolve_ids(camera_ids)
    camera_names = [CAMERA_NAMES[idx] for idx in camera_ids]
    batch_size = min(images.shape[0], max_samples)

    with torch.no_grad():
        debug = projector(
            depths,
            intrinsics,
            car2cams,
            pixel_values=images,
            pixel_step=pixel_step,
            return_debug=True,
        )

    cols = max(3, len(camera_ids))
    rows_per_sample = 3
    fig, axes = plt.subplots(
        batch_size * rows_per_sample,
        cols,
        figsize=(5.2 * cols, 4.3 * batch_size * rows_per_sample),
    )
    axes = np.asarray(axes).reshape(batch_size * rows_per_sample, cols)

    for sample_idx in range(batch_size):
        row0 = sample_idx * rows_per_sample
        points = debug["points_bev"][sample_idx].detach().cpu()
        valid = debug["valid"][sample_idx].detach().cpu()
        values = debug["values"][sample_idx].detach().cpu()
        metric_depths = debug["depths"][sample_idx].detach().cpu()

        for local_idx, camera_id in enumerate(camera_ids):
            cam_points = points[camera_id][valid[camera_id]]
            colors = values[camera_id, :3].permute(1, 2, 0)[valid[camera_id]]
            if cam_points.shape[0] > max_scatter_points:
                ids = torch.linspace(
                    0, cam_points.shape[0] - 1, max_scatter_points
                ).long()
                cam_points = cam_points[ids]
                colors = colors[ids]

            axes[row0, 0].scatter(
                cam_points[:, 1],
                cam_points[:, 0],
                c=colors.clamp(0.0, 1.0),
                s=0.25,
                alpha=0.55,
                label=camera_names[local_idx].split("/")[-2],
            )

        axes[row0, 0].set_title(f"sample {sample_idx}: projected points")
        axes[row0, 0].set_xlabel("y, meters right")
        axes[row0, 0].set_ylabel("x, meters forward")
        axes[row0, 0].axis("equal")
        axes[row0, 0].grid(True)
        axes[row0, 0].legend(markerscale=12)

        axes[row0, 1].imshow(debug["weights"][sample_idx, 0].cpu() > 0, cmap="gray")
        axes[row0, 1].set_title("projection occupancy")
        axes[row0, 1].axis("off")

        axes[row0, 2].imshow(gt_grids[sample_idx, 0].cpu(), cmap="viridis")
        axes[row0, 2].set_title("GT occupancy")
        axes[row0, 2].axis("off")

        for col in range(3, cols):
            axes[row0, col].axis("off")

        for local_idx, camera_id in enumerate(camera_ids):
            image = images[sample_idx, camera_id].permute(1, 2, 0).cpu()
            axes[row0 + 1, local_idx].imshow(image)
            axes[row0 + 1, local_idx].set_title(camera_names[local_idx])
            axes[row0 + 1, local_idx].axis("off")

            depth_image = metric_depths[camera_id, 0].cpu()
            finite_depth = depth_image[torch.isfinite(depth_image)]
            vmin = float(finite_depth.quantile(0.02)) if finite_depth.numel() else 0.0
            vmax = float(finite_depth.quantile(0.98)) if finite_depth.numel() else 1.0
            axes[row0 + 2, local_idx].imshow(
                depth_image,
                cmap="magma_r",
                vmin=vmin,
                vmax=max(vmax, vmin + 1e-6),
            )
            axes[row0 + 2, local_idx].set_title(f"distance {vmin:.1f}-{vmax:.1f} m")
            axes[row0 + 2, local_idx].axis("off")

        for row in (row0 + 1, row0 + 2):
            for col in range(len(camera_ids), cols):
                axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    try:
        from .dataset import BaseDataset, collate_camera_batch
    except ImportError:
        try:
            from dataset import BaseDataset, collate_camera_batch
        except ImportError:
            from src.dataset import BaseDataset, collate_camera_batch

    data_dir = Path("autonomy_yandex_dataset_train")
    batch_size = 4
    debug_cameras = None
    # debug_cameras = [0, 1, 2, 3]
    # debug_cameras = [2, 3]
    # debug_cameras = ["/camera/inner/frontal/middle", "/side/left/forward"]
    pixel_step = 1
    depth_min = 0.0
    depth_max = 80.0
    depth_scale = 0.5
    depth_bias = 0.0
    raw_depth_is_inverse = False
    meters_per_pixel = 150.0 / 188.0
    x_min = 0.0
    y_min = None
    save_path = Path("bev_projection_debug.png")

    dataset = BaseDataset(data_dir, mode="train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_camera_batch,
    )
    projector = DepthToBEVProjection(
        meters_per_pixel=meters_per_pixel,
        x_min=x_min,
        y_min=y_min,
        min_depth=depth_min,
        max_depth=depth_max,
        depth_scale=depth_scale,
        depth_bias=depth_bias,
        raw_depth_is_inverse=raw_depth_is_inverse,
    )

    batch = next(iter(dataloader))
    plot_debug_batch(
        projector,
        batch,
        save_path,
        camera_ids=debug_cameras,
        pixel_step=pixel_step,
        max_samples=batch_size,
    )
    print(f"Saved debug figure to {save_path}")
