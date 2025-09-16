import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class WormDataset(Dataset):
    def __init__(self, images_dir, masks_dir, feature_extractor):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".png")])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".png")])
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        # Convert mask to binary {0,1}
        mask = (np.array(mask) > 127).astype(np.uint8)

        # Hugging Face preprocessing
        encoded = self.feature_extractor(images=image, return_tensors="pt")
        encoded = {k: v.squeeze() for k, v in encoded.items()}  # remove batch dim
        encoded["labels"] = torch.tensor(mask, dtype=torch.long)

        return encoded
