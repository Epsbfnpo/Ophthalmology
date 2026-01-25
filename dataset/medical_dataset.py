import csv
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DomainConfig:
    name: str
    root: str
    has_masks: bool


def _load_labels(csv_path: str, domain_name: str = "") -> dict:
    labels = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    with open(csv_path, newline="", encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames:
            fieldnames = [name.strip() for name in reader.fieldnames]
        else:
            raise ValueError(f"Empty CSV or header: {csv_path}")

        img_col = next((x for x in fieldnames if x in ['image_id', 'image', 'name', 'Image', 'filename']), None)
        grade_col = next((x for x in fieldnames if x in ['grade', 'level', 'dr_grade', 'Retinopathy grade', 'label']),
                         None)

        if not img_col or not grade_col:
            raise ValueError(f"CSV Header Error in {csv_path}. Found: {fieldnames}")

        for row in reader:
            img_id = row[img_col].strip()
            grade_str = row[grade_col].strip()
            if img_id and grade_str:
                labels[img_id] = int(grade_str)

    print(f"[Init] Loaded {len(labels)} labels for {domain_name}.")
    return labels


def apply_clahe(image: Image.Image) -> Image.Image:
    """
    Apply CLAHE normalization to align brightness and contrast distributions across domains.
    This ensures the hard gating threshold works more consistently.
    """
    img_np = np.array(image)

    # 1. RGB -> LAB
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 2. Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 3. Merge and convert back to RGB
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return Image.fromarray(final_img)


def _apply_sync_augmentation(image, mask, rng):
    # Geometric transformations
    if torch.rand((), generator=rng) < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if torch.rand((), generator=rng) < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    angle = float(torch.randint(-30, 31, (), generator=rng))
    image = TF.rotate(image, angle=angle)
    mask = TF.rotate(mask, angle=angle, interpolation=TF.InterpolationMode.NEAREST)

    # Color jitter
    if torch.rand((), generator=rng) < 0.5:
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        image = jitter(image)

    return image, mask


class MedicalDataset(Dataset):
    def __init__(self, domain: DomainConfig, image_size: int = 512, augment: bool = True,
                 transform: Optional[Callable] = None) -> None:
        self.domain = domain
        self.image_dir = os.path.join(domain.root, "images")
        self.mask_dir = os.path.join(domain.root, "masks")
        self.labels = _load_labels(os.path.join(domain.root, "labels.csv"), domain.name)
        self.image_names = []

        for fname in sorted(list(self.labels.keys())):
            if os.path.exists(os.path.join(self.image_dir, fname)):
                self.image_names.append(fname)

        if len(self.image_names) == 0:
            raise RuntimeError(f"Dataset {domain.name} is empty!")

        self.image_size = image_size
        self.augment = augment
        self.transform = transform
        self._rng = torch.Generator().manual_seed(42)
        # ImageNet normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.image_names)

    def _load_mask(self, image_name: str) -> Image.Image:
        if not self.domain.has_masks:
            return Image.new("L", (self.image_size, self.image_size), color=0)

        mask_path = os.path.join(self.mask_dir, image_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")

        if os.path.exists(mask_path):
            return Image.open(mask_path).convert("L")
        else:
            return Image.new("L", (self.image_size, self.image_size), color=0)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 1. Load
        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(image_name)

        # 2. Apply CLAHE
        image = apply_clahe(image)

        # 3. Resize
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        # 4. Augment
        if self.augment:
            image, mask = _apply_sync_augmentation(image, mask, self._rng)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # 5. ToTensor & Normalize
        tensor_img = TF.to_tensor(image)
        image = self.normalize(tensor_img)

        mask = torch.tensor(np.array(mask), dtype=torch.long)
        label = torch.tensor(self.labels[image_name], dtype=torch.long)

        return image, mask, label