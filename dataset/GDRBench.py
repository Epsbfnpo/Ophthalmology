import os.path as osp
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

import random
# Dataset for fundus images including APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR (and DDR, Eyepacs for ESDG)
class GDRBench(Dataset):

    def __init__(self, root, source_domains, target_domains, mode, trans_basic=None, trans_mask = None, trans_fundus=None,trans_basic_freq=None, trans_basic_freq2=None, class_balance=False):

        root = osp.abspath(osp.expanduser(root))
        self.mode = mode
        self.dataset_dir = osp.join(root, "images")
        self.split_dir = osp.join(root, "splits")

        self.data = []
        self.label = []
        self.domain = []   ##
        self.masks = []   ## 
        
        self.trans_basic = trans_basic
        self.trans_freq = trans_basic_freq
        self.trans_freq2 = trans_basic_freq2

        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask
        
        self.class_balance = class_balance
        self.num_classes=5
        self.band = 'all'
        
        if mode == "train":
            self._read_data(source_domains, "train")
        elif mode == "val":
            self._read_data(source_domains, "crossval")
        elif mode == "test":
            self._read_data(target_domains, "test")
        
        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.label)):
            y = self.label[i]
            self.class_data[y].append(i)
            
            
            
    def _read_data(self, input_domains, split):
        items = []
        # print(input_domains)
        for domain, dname in enumerate(input_domains):
            if split == "test":
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval.txt")
                impath_label_list += self._read_split(file_val)
                # file_train = osp.join(self.split_dir, dname + "_crossval.txt")
                # impath_label_list = self._read_split(file_train)
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
        
        
        if self.class_balance:
            label = random.randint(0, self.num_classes - 1)
            index_b = random.choice(self.class_data[label])
            data_b = Image.open(self.data[index_b]).convert("RGB")
            mask_b = Image.open(self.masks[index_b]).convert("L")
            label_b = self.label[index_b]
            domain_b = self.domain[index_b]

        
            # print(data.size)
            if self.trans_freq is not None:
                data_freq_b = self.trans_freq(data_b)
                data_freq_c = self.trans_freq(data_b)


            if self.trans_basic is not None:
                data_b = self.trans_basic(data_b)
            # print(data.shape)  # torch.Size([3, 256, 256])
            

            if self.trans_mask is not None:
                mask_b = self.trans_mask(mask_b)
            
            # path = self.data[index]
            
            
        data = Image.open(self.data[index]).convert("RGB")
        
        # masked_image=self.transform(data)

        if self.mode == "train":
            mask = Image.open(self.masks[index]).convert("L")

        label = self.label[index]
        domain = self.domain[index]
        
        # print(data.size)
        if self.trans_freq is not None:
            data_freq = self.trans_freq(data)
            data_freq2 = self.trans_freq2(data)

        if self.trans_basic is not None:
            data = self.trans_basic(data)
        # print(data.shape)  # torch.Size([3, 256, 256])
        

        if self.trans_mask is not None:
            mask = self.trans_mask(mask)

        if self.mode == "train":
            if self.trans_freq is not None:
                return data, data_freq, data_freq2, mask, label, domain, index
            elif self.class_balance:
                return [data, mask, label, domain, index], [data_b, mask_b, label_b, domain_b, index_b]
            else:
                return data, mask, label, domain, index
        else:
            return data, label, domain, index
    

