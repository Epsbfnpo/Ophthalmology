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
    labels = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name = row.get("image") or row.get("image_name") or row.get("filename")
            label_value = row.get("label") or row.get("grade") or row.get("dr_grade")
            if image_name is None or label_value is None:
                raise ValueError("labels.csv must contain image and label columns")
            labels[image_name] = int(label_value)
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
    image = TF.rotate(image, angle=angle)
    mask = TF.rotate(mask, angle=angle)
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
        self.labels = _load_labels(os.path.join(domain.root, "labels.csv"))
        self.image_names = sorted(self.labels.keys())
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
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")
        mask = Image.open(mask_path).convert("L")
        return mask

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(image_name)

        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            image, mask = _apply_sync_augmentation(image, mask, self._rng)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0).float()
        label = torch.tensor(self.labels[image_name], dtype=torch.long)
        return image, mask, label
