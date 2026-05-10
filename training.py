import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from src.bev_projection import DepthToBEVProjection
from src.dataset import BaseDataset, DATA_DIR, collate_camera_batch
from src.feature_model import RoadPixelFeatureNet
from src.logger import Logger
from src.obstacle_model import BEVSegNet
from src.utils import get_device, binary_bce_dice_loss_with_ignore, binary_seg_metrics


@contextmanager
def do_nothing():
    yield


def use_mixed_precision(args, device):
    return bool(args.mixed_precision and device.type == "cuda")


def feature_map_autocast(args, device):
    if use_mixed_precision(args, device):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return do_nothing()


def move_camera_list(items, device, dtype=None):
    moved = []
    for item in items:
        item = item.to(device=device, non_blocking=True)
        if dtype is not None:
            item = item.to(dtype=dtype)
        moved.append(item)
    return moved


def get_target(static_grids, device):
    target = static_grids[0] if isinstance(static_grids, (list, tuple)) else static_grids
    return target.to(device=device, non_blocking=True)


def unnormalize_images(images):
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def make_camera_color_features(images, camera_id, num_cams):
    rgb = unnormalize_images(images)
    features = rgb.new_zeros(rgb.shape[0], num_cams * 3, *rgb.shape[-2:])
    start = camera_id * 3
    features[:, start : start + 3] = rgb
    return features


def make_camera_feature_block(features, camera_id, num_cams):
    batch, channels, height, width = features.shape
    output = features.new_zeros(batch, num_cams * channels, height, width)
    start = camera_id * channels
    output[:, start : start + channels] = features
    return output


def predict_logits(
    batch, feature_model, projection_model, mapping_model, device, args, nograd=False
):
    use_grad = torch.no_grad if nograd else do_nothing

    with use_grad():
        images, depths, intrinsics, car2cams = batch[:4]
        depths = move_camera_list(depths, device)
        intrinsics = move_camera_list(intrinsics, device, dtype=torch.float32)
        car2cams = move_camera_list(car2cams, device, dtype=torch.float32)

        features = []
        for camera_id in range(args.num_cams):
            image = images[camera_id].to(device=device, non_blocking=True)
            depth = depths[camera_id]
            if args.use_feature_model:
                img_proc = torch.cat([image, depth], dim=1) if args.use_depth else image
                with feature_map_autocast(args, device):
                    feature = feature_model(img_proc)
                if args.camera_feature_channels:
                    feature = make_camera_feature_block(feature, camera_id, args.num_cams)
            else:
                feature = make_camera_color_features(image, camera_id, args.num_cams)
            features.append(feature.float())

        bev_features = projection_model(
            depths,
            intrinsics,
            car2cams,
            features,
            pixel_step=args.pixel_step,
            use_ground_plane=not args.use_depth,
            append_coverage=args.use_coverage_map,
            camera_separated_values=(
                (not args.use_feature_model) or args.camera_feature_channels
            ),
        )
        with feature_map_autocast(args, device):
            logits = mapping_model(bev_features)
        logits = logits.float()

    return logits


def predict(
    batch, feature_model, projection_model, mapping_model, device, args, nograd=False
):
    logits = predict_logits(
        batch,
        feature_model,
        projection_model,
        mapping_model,
        device,
        args,
        nograd=nograd,
    )
    target = get_target(batch[4], device)
    return logits, target


def train_batch(
    criterion,
    metrics,
    optimizer,
    scheduler,
    feature_model,
    projection_model,
    mapping_model,
    train_dataloader,
    logger,
    scaler,
    device,
    args,
    global_step,
    total_steps,
    warmup_steps,
):
    if feature_model is not None:
        feature_model.train()
    projection_model.train()
    mapping_model.train()

    progress = tqdm(train_dataloader, desc="train", leave=False)
    for batch in progress:
        global_step += 1
        set_warmup_cosine_lr(optimizer, args, global_step, total_steps, warmup_steps)
        optimizer.zero_grad(set_to_none=True)
        logits, target = predict(
            batch,
            feature_model,
            projection_model,
            mapping_model,
            device,
            args,
            nograd=False,
        )
        loss = criterion(
            logits,
            target,
            ignore_index=args.ignore_index,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )
        score = metrics(logits, target, ignore_index=args.ignore_index)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = grad_norm((feature_model, projection_model, mapping_model))
        scaler.step(optimizer)
        scaler.update()

        logger.log_train_loss(loss.detach().item())
        logger.log_train_score(score)
        progress.set_postfix(
            loss=f"{loss.detach().item():.5f}",
            iou=f"{score.get('iou', 0.0):.5f}",
            grad_norm=f"{norm:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return global_step


def val_batch(
    criterion,
    metrics,
    feature_model,
    projection_model,
    mapping_model,
    val_dataloader,
    logger,
    device,
    args,
) -> None:
    if feature_model is not None:
        feature_model.eval()
    projection_model.eval()
    mapping_model.eval()

    for batch_id, batch in enumerate(tqdm(val_dataloader, desc="val", leave=False)):
        logits, target = predict(
            batch,
            feature_model,
            projection_model,
            mapping_model,
            device,
            args,
            nograd=True,
        )
        loss = criterion(
            logits,
            target,
            ignore_index=args.ignore_index,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )
        score = metrics(logits, target, ignore_index=args.ignore_index)

        logger.log_val_loss(loss.detach().item())
        logger.log_val_score(score)

        if batch_id == 0:
            logger.add_sample(logits, target)


def make_dataloader(dataset, batch_size, shuffle, num_workers, device):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_camera_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def make_param_group(model, lr):
    if model is None:
        return None
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        return None
    return {"params": params, "lr": lr}


def set_warmup_cosine_lr(optimizer, args, step, total_steps, warmup_steps):
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)

    if warmup_steps > 0 and step <= warmup_steps:
        factor = step / float(warmup_steps)
    else:
        denom = max(total_steps - warmup_steps, 1)
        progress = (step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        base_lr = group.setdefault("base_lr", group["lr"])
        min_lr = min(float(args.min_lr), float(base_lr))
        group["lr"] = min_lr + (float(base_lr) - min_lr) * factor


def grad_norm(models):
    total = 0.0
    for model in models:
        if model is None:
            continue
        for param in model.parameters():
            if param.grad is None:
                continue
            norm = param.grad.detach().float().norm(2)
            total += norm.item() ** 2
    return total ** 0.5


def make_models(args, device):
    in_channels = 4 if args.use_depth else 3
    feature_model = (
        RoadPixelFeatureNet(
            in_channels, feature_dim=args.feature_dim, base_ch=args.base_ch_feat
        ).to(device)
        if args.use_feature_model
        else None
    )
    projection_model = DepthToBEVProjection(
        depth_scale=args.depth_scale,
        ground_z=args.ground_z,
    ).to(device)
    if args.use_feature_model:
        map_in_ch = args.feature_dim
        if args.camera_feature_channels:
            map_in_ch *= args.num_cams
    else:
        map_in_ch = args.num_cams * 3
    if args.use_coverage_map:
        map_in_ch += 1
    mapping_model = BEVSegNet(in_ch=map_in_ch, base_ch=args.base_ch_map).to(
        device
    )
    return feature_model, projection_model, mapping_model


def load_checkpoint(checkpoint_path, feature_model, projection_model, mapping_model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    models = checkpoint["models"]
    if feature_model is not None and "feature_model" in models:
        feature_model.load_state_dict(models["feature_model"])
    projection_model.load_state_dict(models["projection_model"])
    mapping_model.load_state_dict(models["mapping_model"])
    return checkpoint


def resolve_test_data_dir(args):
    if args.test_data_dir is not None:
        return Path(args.test_data_dir)

    data_dir = Path(args.data_dir)
    candidates = [
        data_dir / "test",
        data_dir / "test/autonomy_yandex_dataset_test",
        data_dir / "autonomy_yandex_dataset_test",
    ]
    for candidate in candidates:
        if (candidate / "info.csv").exists():
            return candidate
    return candidates[0]


def prediction_file_name(row, row_id):
    if "predicted_occupancy_grid" in row:
        return Path(str(row["predicted_occupancy_grid"]).replace(":", "_")).name
    if "gt_occupancy_grid" in row:
        return Path(str(row["gt_occupancy_grid"]).replace(":", "_")).name
    return f"{row_id}_grid.npy"


def run_test_predictions(args):
    checkpoint_path = Path(args.best_checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.log_dir) / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    device = get_device()
    test_data_dir = resolve_test_data_dir(args)
    output_dir = Path(args.submission_dir)
    predicted_dir = output_dir / "predicted_static_grids"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_data_dir / "info.csv", output_dir / "info.csv")

    dataset_test = BaseDataset(
        test_data_dir,
        mode="test",
        use_depth=args.use_depth,
        default_depth=args.default_depth,
    )
    test_dataloader = make_dataloader(
        dataset_test, args.batch_size, False, args.num_workers, device
    )
    feature_model, projection_model, mapping_model = make_models(args, device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        feature_model,
        projection_model,
        mapping_model,
    )

    if feature_model is not None:
        feature_model.eval()
    projection_model.eval()
    mapping_model.eval()

    saved = 0
    offset = 0
    for batch in tqdm(test_dataloader, desc="test", leave=False):
        logits = predict_logits(
            batch,
            feature_model,
            projection_model,
            mapping_model,
            device,
            args,
            nograd=True,
        )
        pred = (torch.sigmoid(logits) > args.pred_threshold).to(torch.int32)
        pred = pred[:, 0].detach().cpu().numpy()

        for sample_id in range(pred.shape[0]):
            row = dataset_test.info.iloc[offset + sample_id]
            file_name = prediction_file_name(row, offset + sample_id)
            np.save(predicted_dir / file_name, pred[sample_id])
            saved += 1

        offset += pred.shape[0]

    print(
        f"Saved {saved} test predictions to {predicted_dir} "
        f"from epoch {checkpoint.get('epoch')}, "
        f"best {checkpoint.get('monitor')}={checkpoint.get('best_metric')}"
    )


def run_training(args):
    device = get_device()
    data_dir = Path(args.data_dir)
    dataset_train = BaseDataset(
        data_dir / "train/autonomy_yandex_dataset_train",
        use_depth=args.use_depth,
        default_depth=args.default_depth,
        augment=args.augment,
    )
    dataset_val = BaseDataset(
        data_dir / "val/autonomy_yandex_dataset_val",
        use_depth=args.use_depth,
        default_depth=args.default_depth,
    )
    train_dataloader = make_dataloader(
        dataset_train, args.batch_size, True, args.num_workers, device
    )
    val_dataloader = make_dataloader(
        dataset_val, args.batch_size, False, args.num_workers, device
    )

    feature_model, projection_model, mapping_model = make_models(args, device)

    param_groups = [
        make_param_group(feature_model, args.feat_lr),
        make_param_group(projection_model, args.projection_lr),
        make_param_group(mapping_model, args.mapping_lr),
    ]
    optimizer = torch.optim.AdamW(
        [group for group in param_groups if group is not None],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_mixed_precision(args, device),
    )
    logger = Logger(
        {
            "feature_model": feature_model,
            "projection_model": projection_model,
            "mapping_model": mapping_model,
        },
        output_dir=args.log_dir,
        monitor="val_iou",
        mode="max",
    )
    total_steps = max(args.epochs * len(train_dataloader), 1)
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0:
        warmup_steps = args.warmup_epochs * len(train_dataloader)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        global_step = train_batch(
            binary_bce_dice_loss_with_ignore,
            binary_seg_metrics,
            optimizer,
            None,
            feature_model,
            projection_model,
            mapping_model,
            train_dataloader,
            logger,
            scaler,
            device,
            args,
            global_step,
            total_steps,
            warmup_steps,
        )
        val_batch(
            binary_bce_dice_loss_with_ignore,
            binary_seg_metrics,
            feature_model,
            projection_model,
            mapping_model,
            val_dataloader,
            logger,
            device,
            args,
        )
        logger.update_batch(epoch)


def main():
    parser = argparse.ArgumentParser(description="Train BEV obstacle map model.")
    parser.add_argument("--pixel_step", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_cams", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_ch_feat", type=int, default=16)
    parser.add_argument("--base_ch_map", type=int, default=16)
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--default_depth", type=float, default=30.0)
    parser.add_argument("--feat_lr", type=float, default=1e-4)
    parser.add_argument("--projection_lr", type=float, default=1e-4)
    parser.add_argument("--mapping_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--log_dir", type=str, default="runs/train")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--submission_dir", type=str, default="submission")
    parser.add_argument(
        "--best_checkpoint",
        type=str,
        default="runs/train/checkpoints/best.pt",
    )
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--use_coverage_map", dest="use_coverage_map", action="store_true")
    parser.add_argument("--no_coverage_map", dest="use_coverage_map", action="store_false")
    parser.add_argument(
        "--camera_feature_channels",
        dest="camera_feature_channels",
        action="store_true",
    )
    parser.add_argument(
        "--shared_camera_features",
        dest="camera_feature_channels",
        action="store_false",
    )
    parser.add_argument("--use_feature_model", dest="use_feature_model", action="store_true")
    parser.add_argument("--no_feature_model", dest="use_feature_model", action="store_false")
    parser.add_argument("--use_depth", dest="use_depth", action="store_true")
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--mixed_precision", dest="mixed_precision", action="store_true")
    parser.add_argument(
        "--no_mixed_precision", dest="mixed_precision", action="store_false"
    )
    parser.set_defaults(
        mixed_precision=True,
        use_depth=True,
        use_feature_model=True,
        use_coverage_map=True,
        camera_feature_channels=True,
        augment=True,
    )
    args = parser.parse_args()
    run_training(args)
    if not args.skip_test:
        run_test_predictions(args)


if __name__ == "__main__":
    main()
