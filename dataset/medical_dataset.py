import csv
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass
class DomainConfig:
    name: str
    root: str
    has_masks: bool


def _load_labels(csv_path: str) -> dict:
    """
    点对点读取我们生成的 labels.csv。
    我们确信列名就是 'image_id' 和 'grade'。
    """
    labels = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    with open(csv_path, newline="", encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        # 简单检查一下表头，防止拿错文件
        if 'image_id' not in reader.fieldnames or 'grade' not in reader.fieldnames:
            raise ValueError(
                f"CSV format error in {csv_path}. Expected columns ['image_id', 'grade'], found {reader.fieldnames}")

        for row in reader:
            img_id = row['image_id'].strip()
            grade = int(row['grade'])
            labels[img_id] = grade

    return labels


def _apply_sync_augmentation(
        image: Image.Image,
        mask: Image.Image,
        rng: torch.Generator,
) -> Tuple[Image.Image, Image.Image]:
    if torch.rand((), generator=rng) < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if torch.rand((), generator=rng) < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    angle = float(torch.randint(-20, 21, (), generator=rng))
    # 注意：mask 旋转必须用 Nearest 防止产生插值杂色
    image = TF.rotate(image, angle=angle)
    mask = TF.rotate(mask, angle=angle, interpolation=TF.InterpolationMode.NEAREST)
    return image, mask


class MedicalDataset(Dataset):
    def __init__(
            self,
            domain: DomainConfig,
            image_size: int = 512,
            augment: bool = True,
            transform: Optional[Callable] = None,
    ) -> None:
        self.domain = domain
        self.image_dir = os.path.join(domain.root, "images")
        self.mask_dir = os.path.join(domain.root, "masks")

        # 1. 精准加载标签
        self.labels = _load_labels(os.path.join(domain.root, "labels.csv"))

        # 2. 过滤掉不存在的图片 (双重保险)
        self.image_names = []
        for fname in sorted(list(self.labels.keys())):
            if os.path.exists(os.path.join(self.image_dir, fname)):
                self.image_names.append(fname)

        if len(self.image_names) == 0:
            raise RuntimeError(f"Dataset {domain.name} is empty! Check {self.image_dir}")

        self.image_size = image_size
        self.augment = augment
        self.transform = transform
        self._rng = torch.Generator().manual_seed(0)

    def __len__(self) -> int:
        return len(self.image_names)

    def _load_mask(self, image_name: str) -> Image.Image:
        if not self.domain.has_masks:
            return Image.new("L", (self.image_size, self.image_size), color=0)

        mask_path = os.path.join(self.mask_dir, image_name)
        # 我们的处理脚本生成的都是 png，所以这里也可以简化逻辑
        if not os.path.exists(mask_path):
            # 兼容一下 jpg 后缀的情况 (虽然处理后全是 png)
            mask_path = os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")

        if os.path.exists(mask_path):
            return Image.open(mask_path).convert("L")
        else:
            return Image.new("L", (self.image_size, self.image_size), color=0)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(image_name)

        # Resize
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        # Augmentation
        if self.augment:
            image, mask = _apply_sync_augmentation(image, mask, self._rng)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # ToTensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # [关键] 强制二值化：语义值(50,100...)全部转为1.0供U-Net分割
        mask = (mask > 0).float()

        label = torch.tensor(self.labels[image_name], dtype=torch.long)

        return image, mask, label