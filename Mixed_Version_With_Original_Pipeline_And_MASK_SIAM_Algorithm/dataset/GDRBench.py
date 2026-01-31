import os.path as osp
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch
from guided_filter_pytorch.ModifiedHFCFilter import FourierButterworthHFCFilter


# Helper function for Frequency Domain Augmentation
def hfc_mul_mask(hfc_filter, image, mask, do_norm=False):
    # image and mask are expected to be Tensors here (C, H, W)
    # HFC Filter expects input in range [0, 1] usually, assuming standard normalization
    # Ensure inputs are on the correct device (CPU here usually)
    hfc = hfc_filter((image / 2 + 0.5), mask)
    if do_norm:
        hfc = 2 * hfc - 1
    return (hfc + 1) * mask - 1


# Dataset for fundus images including APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR (and DDR, Eyepacs for ESDG)
class GDRBench(Dataset):

    def __init__(self, root, source_domains, target_domains, mode, cfg=None, trans_basic=None, trans_mask=None,
                 trans_fundus=None):

        root = osp.abspath(osp.expanduser(root))
        self.mode = mode
        self.dataset_dir = osp.join(root, "images")
        self.split_dir = osp.join(root, "splits")
        self.cfg = cfg

        self.data = []
        self.label = []
        self.domain = []
        self.masks = []

        self.trans_basic = trans_basic
        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask

        # Frequency Augmentation Setup
        self.use_freq = False
        self.hfc_filter = None
        if self.mode == "train" and self.cfg is not None:
            if hasattr(self.cfg, 'TRANSFORM') and getattr(self.cfg.TRANSFORM, 'FREQ', False):
                self.use_freq = True
                # Initialize HFC Filter (CPU side for DataLoader)
                # Assuming image size 256x256 based on typical config
                self.hfc_filter = FourierButterworthHFCFilter(
                    butterworth_d0_ratio=0.003,
                    butterworth_n=1,
                    do_median_padding=False,
                    image_size=(256, 256)
                )

        if mode == "train":
            self._read_data(source_domains, "train")
        elif mode == "val":
            self._read_data(source_domains, "crossval")
        elif mode == "test":
            self._read_data(target_domains, "test")

    def _read_data(self, input_domains, split):
        items = []
        for domain, dname in enumerate(input_domains):
            if split == "test":
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval.txt")
                impath_label_list += self._read_split(file_val)
            else:
                file = osp.join(self.split_dir, dname + "_" + split + ".txt")
                impath_label_list = self._read_split(file)

            for impath, label in impath_label_list:
                self.data.append(impath)
                self.masks.append(impath.replace("images", "masks"))

                self.label.append(label)
                self.domain.append(domain)

    def _read_split(self, split_file):
        items = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))

        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = Image.open(self.data[index]).convert("RGB")

        if self.mode == "train":
            mask = Image.open(self.masks[index]).convert("L")

        label = self.label[index]
        domain = self.domain[index]

        # Apply Transforms
        if self.trans_basic is not None:
            data = self.trans_basic(data)  # Returns Tensor

        if self.trans_mask is not None:
            mask = self.trans_mask(mask)  # Returns Tensor

        if self.mode == "train":
            # [Updated Logic] Handle Frequency Domain Augmentation
            if self.use_freq:
                # Apply HFC filter to generate frequency enhanced image
                # Note: data and mask are already Tensors here
                data_freq = hfc_mul_mask(self.hfc_filter, data, mask, do_norm=True)

                # Return 5 items matching algorithms.py strict unpacking:
                # image, image_feq, mask, label, domain = minibatch
                return data, data_freq, mask, label, domain
            else:
                # Fallback for standard algorithms (if index is needed by them, add it back,
                # but for MASK_SIAM strict mode, we stick to 5 inputs or handle robustness in algo)
                # To be safe for mixed usage, MASK_SIAM algo expects 5.
                # If we are running ERM, it unpacks 4.
                # Ideally, collate_fn or dict return is better, but following your structure:
                return data, mask, label, domain, index
        else:
            # Validation/Test mode
            return data, label, domain, index