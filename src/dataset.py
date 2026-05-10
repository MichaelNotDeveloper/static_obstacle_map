import pandas as pd
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
import functools
from torch.utils.data._utils.collate import default_collate

DATA_DIR = Path("/Users/meshaza/Desktop/projects/static_obstacle_map/")

CAMERA_NAMES = [
    "/camera/inner/frontal/middle",
    "/camera/inner/frontal/far",
    "/side/left/forward",
    "/side/right/forward",
]

INTRINSICS_NAMES = [
    "/camera/inner/frontal/middle/intrinsic_params",
    "/camera/inner/frontal/far/intrinsic_params",
    "/side/left/forward/intrinsic_params",
    "/side/right/forward/intrinsic_params",
]

CAR2CAM_NAMES = [
    "/camera/inner/frontal/middle/car_to_cam",
    "/camera/inner/frontal/far/car_to_cam",
    "/side/left/forward/car_to_cam",
    "/side/right/forward/car_to_cam",
]

GRIDS_NAMES = [
    "gt_occupancy_grid",
]

IMG_SHAPE = (256, 512)
CALIBRATION_IMAGE_SHAPE = (540, 1024)

class FastDepthAnythingMeters:
    def __init__(
        self,
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        device=None,
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, images, batch_size=8, return_numpy=False):
        single = isinstance(images, Image.Image)
        images = [images] if single else list(images)

        all_depths = []

        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            sizes = [(img.height, img.width) for img in batch]

            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            pred = outputs.predicted_depth  # [B, h, w], metric depth for metric model

            if len(set(sizes)) == 1:
                depth = F.interpolate(
                    pred.unsqueeze(1),
                    size=sizes[0],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)

                all_depths.extend(depth.cpu())
            else:
                for d, size in zip(pred, sizes):
                    d = F.interpolate(
                        d[None, None],
                        size=size,
                        mode="bicubic",
                        align_corners=False,
                    )[0, 0]
                    all_depths.append(d.cpu())

        if return_numpy:
            all_depths = [d.numpy() for d in all_depths]

        return all_depths[0] if single else all_depths


@functools.cache
def get_depth_model():
    return FastDepthAnythingMeters(
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    )


@functools.cache
def get_resize_transform():
    transform = v2.Compose(
        [
            v2.Resize(IMG_SHAPE, antialias=True),
        ]
    )
    return transform


@functools.cache
def get_transforms():
    transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


def collate_camera_batch(batch):
    def collate_camera_list(samples):
        return [default_collate(camera_samples) for camera_samples in zip(*samples)]

    images = collate_camera_list([sample[0] for sample in batch])
    depths = collate_camera_list([sample[1] for sample in batch])
    intrinsics = collate_camera_list([sample[2] for sample in batch])
    car2cams = collate_camera_list([sample[3] for sample in batch])

    if len(batch[0]) == 4:
        return images, depths, intrinsics, car2cams

    static_grids = [default_collate([sample[4] for sample in batch])]
    return images, depths, intrinsics, car2cams, static_grids


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path,
        mode: str = "train",
        use_depth: bool = True,
        default_depth: float = 30.0,
        depth=None,
    ):
        self.mode = mode
        self.data_dir = Path(data_dir)
        self.use_depth = use_depth if depth is None else depth
        self.default_depth = float(default_depth)
        self.resize_transform = get_resize_transform()
        self.transform = get_transforms()
        self.depth_model = get_depth_model() if self.use_depth else None
        self.info = pd.read_csv(self.data_dir / "info.csv", index_col=0)
        self.images_paths = []
        self.intrinsics_paths = []
        self.car2cam_paths = []
        self.num_cams = 4
        if self.mode != "test":
            self.static_grids_paths = []

        for _, row in self.info.iterrows():
            self.images_paths.append(
                [self.resolve_data_path(row[name]) for name in CAMERA_NAMES]
            )
            self.intrinsics_paths.append(
                [self.resolve_data_path(row[name]) for name in INTRINSICS_NAMES]
            )
            self.car2cam_paths.append(
                [self.resolve_data_path(row[name]) for name in CAR2CAM_NAMES]
            )
            if self.mode != "test":
                self.static_grids_paths.append(
                    [self.resolve_data_path(row[name]) for name in GRIDS_NAMES]
                )

    def resolve_data_path(self, value):
        path = Path(str(value).replace(":", "_"))
        if path.is_absolute():
            return path

        if path.parts and path.parts[0] == self.data_dir.name:
            return self.data_dir / Path(*path.parts[1:])

        return self.data_dir / path

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        images = [
            self.resize_transform(Image.open(img_path).convert("RGB"))
            for img_path in self.images_paths[idx]
        ]
        if self.use_depth:
            depth_outputs = self.depth_model(images, batch_size=len(images))
            depths = [
                depth.to(dtype=torch.float32).unsqueeze(0)
                for depth in depth_outputs
            ]
        else:
            height, width = images[0].height, images[0].width
            depths = [
                torch.full(
                    (1, height, width),
                    self.default_depth,
                    dtype=torch.float32,
                )
                for _ in images
            ]

        images = [self.transform(sample) for sample in images]
        intrinsics = [np.load(intr_path) for intr_path in self.intrinsics_paths[idx]]
        car2cams = [np.load(car2cam_path) for car2cam_path in self.car2cam_paths[idx]]

        if self.mode != "test":
            static_grids = [
                np.load(grid_path) for grid_path in self.static_grids_paths[idx]
            ][0]
            return images, depths, intrinsics, car2cams, static_grids

        return images, depths, intrinsics, car2cams


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = BaseDataset(DATA_DIR / "autonomy_yandex_dataset_train", "train")
    img_num = 10
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=img_num,
        collate_fn=collate_camera_batch,
    )
    fig, ax = plt.subplots(img_num, 8, figsize=(100, 140))
    for imgs, depths, _, _, static_grids in dataloader:
        for i in range(len(imgs)):
            for j in range(imgs[i].shape[0]):
                ax[j][2 * i].imshow(imgs[i][j].permute(1, 2, 0))
                ax[j][2 * i + 1].imshow(depths[i][j].permute(1, 2, 0).numpy())
                ax[j][2 * i].axis("off")
                ax[j][2 * i + 1].axis("off")
        break
    plt.title("cam images")
    plt.savefig("image_transforms.png", bbox_inches="tight", pad_inches=0)
