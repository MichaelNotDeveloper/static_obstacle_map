import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torchvision.transforms import v2
import numpy as np
from transformers import pipeline
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

IMG_SHAPE = (540, 1024)


@functools.cache
def get_transforms():
    transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize(IMG_SHAPE, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


def get_depth_transforms():
    transform = v2.Compose(
        [
            v2.Resize(IMG_SHAPE, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return transform


@functools.cache
def get_depth_model():
    return pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
    )


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, mode: str = "train"):
        self.mode = mode
        self.data_dir = data_dir
        self.transform = get_transforms()
        self.depth_transform = get_depth_transforms()
        self.depth_model = get_depth_model()
        self.info = pd.read_csv(data_dir / "info.csv", index_col=0)
        self.images_paths = []
        self.intrinsics_paths = []
        self.car2cam_paths = []
        self.num_cams = 4
        if self.mode != "test":
            self.static_grids_paths = []

        for _, row in self.info.iterrows():
            self.images_paths.append(
                [row[name].replace(":", "_") for name in CAMERA_NAMES]
            )
            self.intrinsics_paths.append(
                [row[name].replace(":", "_") for name in INTRINSICS_NAMES]
            )
            self.car2cam_paths.append(
                [row[name].replace(":", "_") for name in CAR2CAM_NAMES]
            )
            if self.mode != "test":
                self.static_grids_paths.append(
                    [row[name].replace(":", "_") for name in GRIDS_NAMES]
                )

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        images = [Image.open(img_path) for img_path in self.images_paths[idx]]
        depths = [
            self.depth_transform(
                self.depth_model(sample)["predicted_depth"].unsqueeze(0)
            )
            for sample in images
        ]
        images = [self.transform(sample) for sample in images]
        intrinsics = [np.load(intr_path) for intr_path in self.intrinsics_paths[idx]]
        car2cams = [np.load(car2cam_path) for car2cam_path in self.car2cam_paths[idx]]

        if self.mode != "test":
            static_grids = [
                np.load(grid_path) for grid_path in self.static_grids_paths[idx]
            ]
            return images, depths, intrinsics, car2cams, static_grids

        return images, depths, intrinsics, car2cams


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = BaseDataset(DATA_DIR / "autonomy_yandex_dataset_train", "train")
    img_num = 10
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=img_num)
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
