# ade20k_dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class ADE20KDataset(Dataset):
    def __init__(self, root, split="training", img_size=224):
        self.root = root
        self.split = split
        self.img_size = img_size

        self.img_dir = os.path.join(root, "images", split)
        self.mask_dir = os.path.join(root, "annotations", split)

        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        # image
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0

        # --- MASQUE ---
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        # Nettoyage ADE20K : labels invalides → 0
        mask[mask < 0] = 0
        mask[mask >= 150] = 0


        return img, mask
