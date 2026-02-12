import os
import os.path as osp
import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from dataset.spmix_aug import SPMixAugmentation


class GDRBench(Dataset):
    def __init__(self, root, source_domains, target_domains, mode, trans_basic=None, trans_mask=None,
                 trans_fundus=None):
        if root is None:
            raise ValueError("Dataset root is None! Please check your config or args.")

        root = osp.abspath(osp.expanduser(root))
        self.mode = mode

        self.dataset_dir = osp.join(root, "images")
        self.mask_dir = osp.join(root, "masks")
        self.split_dir = osp.join(root, "splits")

        print(f"[{mode.upper()}] Dataset Root: {root}")
        print(f"[{mode.upper()}] Loading splits from: {self.split_dir}")

        self.data = []
        self.label = []
        self.domain = []
        self.masks = []

        self.trans_basic = trans_basic
        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask

        self.spmix = SPMixAugmentation(prob=0.5, alpha=1.0) if mode == 'train' else None

        if mode == "train":
            self._read_data(source_domains, "train")
        elif mode == "val":
            self._read_data(source_domains, "crossval")
        elif mode == "test":
            self._read_data(target_domains, "test")

        if len(self.data) == 0:
            print(f"❌ [ERROR] No images loaded for mode '{mode}'!")
            raise RuntimeError(f"Found 0 images for {mode} set.")
        else:
            print(f"✅ [{mode.upper()}] Successfully loaded {len(self.data)} images.")

    def _read_data(self, input_domains, split):
        for domain_idx, dname in enumerate(input_domains):
            files_to_try = []
            if split == "test":
                files_to_try.append(osp.join(self.split_dir, dname + "_test.txt"))
                files_to_try.append(osp.join(self.split_dir, dname + "_train.txt"))
                files_to_try.append(osp.join(self.split_dir, dname + "_crossval.txt"))
            else:
                files_to_try.append(osp.join(self.split_dir, dname + "_" + split + ".txt"))

            for file in files_to_try:
                if osp.exists(file):
                    self._read_split(file, domain_idx)
                elif split != "test":
                    print(f"⚠️ [WARNING] Split file not found: {file}")

    def _read_split(self, split_file, domain_idx):
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue

                impath_in_txt = parts[0]
                label = int(parts[1])
                final_image_path = osp.join(self.dataset_dir, impath_in_txt)

                final_mask_path = None
                base_rel_path, original_ext = osp.splitext(impath_in_txt)
                search_exts = list(dict.fromkeys(['.png', original_ext] + mask_extensions))
                for ext in search_exts:
                    try_path = osp.join(self.mask_dir, base_rel_path + ext)
                    if osp.exists(try_path):
                        final_mask_path = try_path
                        break
                if final_mask_path is None:
                    final_mask_path = osp.join(self.mask_dir, base_rel_path + ".png")

                self.data.append(final_image_path)
                self.masks.append(final_mask_path)
                self.label.append(label)
                self.domain.append(domain_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            image_path = self.data[index]
            img1 = Image.open(image_path).convert("RGB")
            label1 = self.label[index]
            domain = self.domain[index]

            if self.mode == "train":
                mask_path = self.masks[index]
                if osp.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                else:
                    mask = Image.new('L', img1.size, 255)
            else:
                mask = Image.new('L', img1.size, 255)

            if self.mode == "train":
                # --- [Fix] MASK_SIAM 兼容性修正 ---
                # 为了保证对比学习（Siamese）的逻辑正确性，我们不能混合不同身份的图片。
                # 这里我们强制 img2 = img1，利用 SPMix 做 Saliency-guided Self-CutPaste 增强
                # 或者如果不想用 CutPaste，可以直接跳过 SPMix，但为了保留 SP 增强特性，我们做 Self-Mix。

                img2 = img1  # 强制使用同一张图
                label2 = label1  # 标签保持一致

                # 执行 SPMix (现在变成了 Saliency-based Self-Augmentation)
                # lam 虽然会被计算，但在同图混合下，语义标签实际上是 100% label1
                img_strong_pil, lam = self.spmix(img1, img2)

                # 强制修正 lam，因为是同图混合，标签完全没有变
                lam = 1.0

                if self.trans_basic is not None:
                    img_weak = self.trans_basic(img1)
                    img_strong = self.trans_basic(img_strong_pil)
                else:
                    img_weak = img1
                    img_strong = img_strong_pil

                if self.trans_mask is not None:
                    mask = self.trans_mask(mask)

                # 返回值保持 8 个元素以兼容接口，但 label2=label1, lam=1.0
                return img_weak, img_strong, mask, label1, domain, index, label2, np.float32(lam)

            else:
                if self.trans_basic is not None:
                    data = self.trans_basic(img1)
                else:
                    data = img1

                return data, data, mask, label1, domain, index, label1, 1.0

        except Exception as e:
            print(f"Error loading index {index}, path: {self.data[index]}")
            raise e